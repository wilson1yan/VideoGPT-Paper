import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from videogpt.utils import deepclone
from videogpt.layers.pos_embd import BroadcastPosEmbedND
from videogpt.layers.utils import view_range, shift_dim
from videogpt.layers.attention.full_attention import FullAttention
from videogpt.layers.attention.sparse_attention import SparseAttention


class EncoderAttention(nn.Module):
    def __init__(self, layer_idx, shape, dec_len, enc_len, dec_dim, enc_dim,
                 proj_dim, n_head, n_layer, attn_type,
                 attn_kwargs):
        super().__init__()
        assert attn_type == 'full', "currently only supports full attn"
        self.attn = MultiHeadAttention(layer_idx=layer_idx, shape=shape,
                                       len_q=dec_len, len_kv=enc_len,
                                       dim_q=dec_dim, dim_kv=enc_dim,
                                       proj_qk_dim=proj_dim, proj_v_dim=proj_dim,
                                       n_head=n_head, n_layer=n_layer,
                                       causal=False, # see all conditioning info,
                                       attn_type=attn_type, attn_kwargs=attn_kwargs)

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, dec_input, enc_output):
        return self.attn(dec_input, enc_output, enc_output,
                         decode_step=None, decode_idx=None)


class MultiHeadAttention(nn.Module):
    def __init__(self, layer_idx, shape, len_q, len_kv, dim_q, dim_kv,
                 proj_qk_dim, proj_v_dim,
                 n_head, n_layer, causal,
                 attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = proj_qk_dim
        self.d_v = proj_v_dim
        self.n_head = n_head
        self.attn_type = attn_type

        self.w_qs = nn.Linear(dim_q, n_head * proj_qk_dim, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * proj_qk_dim, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * proj_v_dim, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * proj_v_dim, dim_q, bias=False) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, n_head, len_q, len_kv, causal, **attn_kwargs)
        elif attn_type == 'sparse':
            self.attn = SparseAttention(shape=shape, n_head=n_head, causal=causal,
                                        **attn_kwargs)
        else:
            raise Exception('Invalid attn_type', attn_type)

        self.cache = None

    def clear_cache(self):
        self.cache = None

    def forward(self, q, k, v, decode_step, decode_idx):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None
            decode_step: an integer representing the current sampling index in AR ordering
            decode_idx: a tuple representing the current tensor index being sampled

        Returns
            The output after performing attention and any auxiliary losses if relevant
            (aux_loss != 0 only for routing attention)
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # axial transformer does not use this caching
        if decode_step is not None:
            # create cache if first iter of sampling
            if self.cache is None:
                if self.causal:
                    k_shape = (q.shape[0], n_head,) + self.shape + (self.d_k,)
                    v_shape = (q.shape[0], n_head,) + self.shape + (self.d_v,)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                      v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # in the non-causal case only need to cache once
                    self.cache = dict(k=k.clone(),
                                      v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None)) + \
                    tuple([slice(i, i + 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a
