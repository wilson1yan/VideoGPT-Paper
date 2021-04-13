import os
import numpy as np

import torch.nn as nn
import torch

from videogpt.utils import deepclone
from videogpt.layers.norm import LayerNorm
from videogpt.layers.utils import GeLU2, shift_dim, WrapPrePostProcess, checkpoint_layer
from videogpt.layers.attention.attention import MultiHeadAttention, EncoderAttention
from videogpt.layers.right_shift import RightShiftSequence
from videogpt.layers.pos_embd import BroadcastPosEmbedND
from videogpt.layers.utils import view_range, shift_dim, tensor_slice, LambdaModule


class SelfAttentionLayer(nn.Module):
    def __init__(self, layer_idx, shape, seq_len, embd_dim, proj_dim,
                 n_head, n_layer, causal, attn_type, attn_kwargs):
        super().__init__()
        self.attn = MultiHeadAttention(layer_idx=layer_idx,
                                       shape=shape, len_q=seq_len, len_kv=seq_len,
                                       proj_qk_dim=proj_dim, proj_v_dim=proj_dim,
                                       dim_kv=embd_dim, dim_q=embd_dim,
                                       n_head=n_head, n_layer=n_layer,
                                       causal=causal, attn_type=attn_type,
                                       attn_kwargs=attn_kwargs)

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, decode_step, decode_idx):
        return self.attn(x, x, x, decode_step, decode_idx)


class SelfAttentionBlock(nn.Module):
    def __init__(self, layer_idx, shape, seq_len, embd_dim, proj_dim, cond_dim,
                 n_head, n_layer, causal, ff_mult, dropout,
                 checkpoint, attn_type, attn_kwargs):
        super().__init__()

        norm_constr = lambda: LayerNorm(embd_dim=embd_dim, cond_dim=cond_dim)
        kwargs = dict(norm_constr=norm_constr, dropout=dropout)

        attn = SelfAttentionLayer(
            layer_idx=layer_idx,
            shape=shape, seq_len=seq_len, embd_dim=embd_dim, proj_dim=proj_dim,
            n_head=n_head, n_layer=n_layer, causal=causal,
            attn_type=attn_type, attn_kwargs=attn_kwargs)
        fn = lambda x, cond, decode_step, decode_idx: checkpoint_layer(attn, (x, decode_step, decode_idx),
                                              use_checkpoint=checkpoint and self.training)
        self.attn = WrapPrePostProcess(attn, fn=fn, **kwargs)

        if 'enc_attn' in cond_dim:
            cat_res_out_size = cond_dim['enc_attn']  # ((t), h, w, c)
            enc_len = np.prod(cat_res_out_size[:-1])
            enc_attn = EncoderAttention(layer_idx=layer_idx,
                                        shape=shape, dec_len=seq_len, enc_len=enc_len,
                                        dec_dim=embd_dim, enc_dim=cat_res_out_size[-1],
                                        proj_dim=proj_dim, n_head=n_head,
                                        n_layer=n_layer, attn_type='full',
                                        attn_kwargs=dict(attn_dropout=attn_kwargs['attn_dropout']))
            enc_attn_wrapper = lambda x, cond, decode_step, decode_idx: enc_attn(x, cond['enc_attn'])
            self.enc_attn = WrapPrePostProcess(enc_attn, fn=enc_attn_wrapper, **kwargs)
        else:
            self.enc_attn = None

        fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * ff_mult),
            GeLU2(),
            nn.Linear(in_features=embd_dim * ff_mult, out_features=embd_dim),
        )
        fc_block_wrapper = lambda x, cond, decode_step, decode_idx: checkpoint_layer(fc_block, (x,),
                                                                                        use_checkpoint=checkpoint and self.training)

        self.fc_block = WrapPrePostProcess(fc_block, fn=fc_block_wrapper, **kwargs)

    def clear_cache(self):
        self.attn.clear_cache()
        if self.enc_attn is not None:
            self.enc_attn.clear_cache()

    def forward(self, x, cond, decode_step, decode_idx):
        x = self.attn(x, cond, decode_step, decode_idx)
        if self.enc_attn is not None:
            x = self.enc_attn(x, cond, decode_step, decode_idx)
        x = self.fc_block(x, cond, decode_step, decode_idx)
        return x


class SelfAttentionModel(nn.Module):
    def __init__(self, *, shape, n_vocab, embd_dim, proj_dim, cond_dim,
                 n_head, n_layer, causal, ff_mult, cond_types, dropout,
                 checkpoint, attn_type, attn_kwargs):
        super().__init__()
        self.shape = shape
        self.embd_dim = embd_dim
        self.cond_dim = cond_dim

        self.pre_pos_embd = RightShiftSequence(embd_dim)
        pos_embd = BroadcastPosEmbedND(
            shape=shape, embd_dim=embd_dim
        )
        self.pos_embd = pos_embd
        if 'enc_attn' in cond_dim:
            cond_pos_embd = BroadcastPosEmbedND(
                shape=cond_dim['enc_attn'][:-1], embd_dim=cond_dim['enc_attn'][-1])
            self.cond_pos_embd = cond_pos_embd

        self.attn_nets = nn.ModuleList(
            [
                SelfAttentionBlock(
                    layer_idx=i,
                    shape=shape,
                    seq_len=np.prod(shape),
                    embd_dim=embd_dim,
                    proj_dim=proj_dim,
                    cond_dim=cond_dim,
                    n_head=n_head,
                    n_layer=n_layer,
                    causal=causal,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    checkpoint=checkpoint,
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
                for i in range(n_layer)
            ]
        )

    def forward(self, x, cond, decode_step, decode_idx):
        """
        Args
        ------
            x: (b, d1, d2, ..., dn, embd_dim), where dn-2 = height, dn-1 = width, dn = n_codebooks. n = 4 => (b, t, h, w, l, d) => dim_h = 2, dim_w = 3
            cond: a tuple of conditioning tensors
            decode_step: a scalar representing the ith index being sampled
            decode_idx: the idx representing the location of the element being sampled
        """
        x = self.pre_pos_embd(x, decode_step, decode_idx)
        pos_embd = self.pos_embd(x=x)

        if decode_step is not None:
            pos_embd = tensor_slice(pos_embd, [0, *decode_idx, 0],
                                    [x.shape[0], *(1,) * len(self.shape), x.shape[-1]])

        x = x + pos_embd.type_as(x)

        if 'enc_attn' in cond:
            if decode_step is not None:
                # just a hack to fix the sampling issue and cond
                # doesn't point to the cond object in gpt.py
                cond = deepclone(cond)
            c = cond['enc_attn']
            cond['enc_attn'] = c + self.cond_pos_embd(c).type_as(c)

        for i, net in enumerate(self.attn_nets):
            x = net(x, cond, decode_step, decode_idx)

        return x

    def clear_cache(self):
        for net in self.attn_nets:
            net.clear_cache()
