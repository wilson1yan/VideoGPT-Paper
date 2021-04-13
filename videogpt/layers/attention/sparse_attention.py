import sys
from collections import namedtuple
import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import *

try:
    from deepspeed.ops.sparse_attention import MatMul, Softmax, SparsityConfig
except:
    print('No Sparse Attention installed!')
    class SparsityConfig(object):
        def __init__(self):
            pass
from videogpt.layers.utils import view_range
from videogpt.layers.attention.attention_ops import scaled_dot_product_attention


# Only supports self attention
class SparseAttention(nn.Module):

    block_layout = dict()

    def __init__(self, shape, n_head, causal, attn_dropout=0.,
                 num_local_blocks=4, block=32):
        super().__init__()

        assert attn_dropout == 0, 'Dropout not supported'

        self.causal = causal
        self.attn_dropout = attn_dropout

        self.sparsity_config = StridedSparsityConfig(shape=shape, num_heads=n_head,
                                                     causal=self.causal, block=block,
                                                     num_local_blocks=num_local_blocks)

        self.seq_len = np.prod(shape)
        self.attn = SparseSelfAttention(seq_len=self.seq_len, causal=causal,
                                        sparsity_config=self.sparsity_config)

        if self.seq_len not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.seq_len] = self.sparsity_config.make_layout()

    def collect_stats_and_update(self):
        return dict()

    def forward(self, q, k, v, decode_step, decode_idx):
        SparseAttention.block_layout[self.seq_len] = SparseAttention.block_layout[self.seq_len].to(q)

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(SparseAttention.block_layout[self.seq_len], decode_step)
            out = scaled_dot_product_attention(q, k, v, mask=mask, training=self.training)
        else:
            out = self.attn(q, k, v)

        return view_range(out, 2, 3, old_shape)


class SparseSelfAttention(nn.Module):
    """Implements an efficient Sparse Self Attention of Transformer layer based on `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """
    ops = dict()
    attn_mask = dict()

    def __init__(
        self,
        seq_len,
        causal,
        sparsity_config=SparsityConfig(num_heads=4),
        key_padding_mask_mode='add'):
        """Initialize the sparse self attention layer.
        Arguments:
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class.
            key_padding_mask_mode: optional: a string determining if key padding mask needs to be added, `add`, or be multiplied, `mul`.
        """
        super().__init__()

        self.seq_len = seq_len
        self.causal = causal
        # sparsity information
        self.sparsity_config = sparsity_config

        self.get_ops()

        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode

        if causal and seq_len not in SparseSelfAttention.attn_mask:
            SparseSelfAttention.attn_mask[seq_len] = self.sparsity_config.make_sparse_attn_mask()

    # add to cache
    def get_ops(self):
        if self.seq_len not in SparseSelfAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseSelfAttention.ops[self.seq_len] = (sparse_dot_sdd_nt,
                                                     sparse_dot_dsd_nn,
                                                     sparse_softmax)
        return SparseSelfAttention.ops[self.seq_len]

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self,
                query,
                key,
                value,
                rpe=None,
                key_padding_mask=None):
        """Applies forward phase of sparse self attention

        Arguments:
            query: required: query tensor
            key: required: key tensor
            value: required: value tensor
            rpe: optional: a tensor same dimension as x that is used as relative position embedding
            key_padding_mask: optional: a mask tensor of size (BatchSize X SequenceLength)
            key_padding_mask_mode: optional: a boolean determining if key_padding_mask needs to be added or multiplied

        Return:
             attn_output: a dense tensor containing attnetion context
        """
        if self.causal:
            SparseSelfAttention.attn_mask[self.seq_len] = SparseSelfAttention.attn_mask[self.seq_len].to(query).type_as(query)
        attn_mask = SparseSelfAttention.attn_mask[self.seq_len] if self.causal else None
        bsz, num_heads, tgt_len, head_dim = query.size()

        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype,
                                                              key_padding_mask,
                                                              is_key_padding_mask=True)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()

        scaling = float(head_dim)**-0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)

        if attn_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_output_weights = sparse_softmax(
            attn_output_weights,
            scale=scaling,
            rpe=rpe,
            key_padding_mask=key_padding_mask,
            key_padding_mask_mode=self.key_padding_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output


class StridedSparsityConfig(SparsityConfig):
    def __init__(self, shape, num_heads, block, causal, num_local_blocks):
        super().__init__(num_heads=num_heads, block=block,
                         different_layout_per_head=False)
        self.shape = shape
        self.seq_len = np.prod(shape)
        self.num_local_blocks = num_local_blocks
        self.causal = causal

        assert self.num_local_blocks >= 0
        assert np.prod(shape) > block

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)

    def set_local_layout(self, h, layout):
        num_blocks = layout.shape[1]
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                    max(0, row - self.num_local_blocks),
                    (row + 1 if self.causal else end)):
                layout[h, row, col] = 1
        return layout

    def set_global_layout(self, h, layout):
        num_blocks = layout.shape[1]
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[h, row, col] = 1

        return layout

    def make_layout(self): # seq_len only there to fit override method args
        seq_len = np.prod(self.shape)
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_local_layout(h, layout)
            layout = self.set_global_layout(h, layout)

        layout = self.check_and_propagate_first_head_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        # returns attn_mask: [n_dense_blocks, block_size, block_size]

        block_layout = self.make_layout()

        num_heads = block_layout.shape[0]
        seq_len = np.prod(self.shape)
        assert seq_len % self.block == 0
        num_blocks = seq_len // self.block
        assert block_layout.shape[1] == block_layout.shape[2] == num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(num_heads):
            for i in range(num_blocks):
                for j in range(num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j: # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]] # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1:] = 0.
        return block_row

