import torch
import torch.nn as nn

from videogpt.layers.attention.attention_ops import scaled_dot_product_attention
from videogpt.layers.utils import view_range


class FullAttention(nn.Module):
    attn_mask = dict()

    def __init__(self, shape, n_head, len_qk, len_v, causal, attn_dropout):
        super().__init__()
        self.shape = shape
        self.causal = causal
        self.attn_dropout = attn_dropout
        if self.causal and shape not in FullAttention.attn_mask:
            # compute mask for attention weights
            i = torch.arange(len_qk).view(len_qk, 1)
            j = torch.arange(len_qk)
            FullAttention.attn_mask[shape] = i.ge(j).float()

    def collect_stats_and_update(self):
        return dict()

    def forward(self, q, k, v, decode_step, decode_idx):
        if self.causal:
            FullAttention.attn_mask[self.shape] = FullAttention.attn_mask[self.shape].to(q).type_as(q)
        mask = FullAttention.attn_mask[self.shape] if self.causal else None
        if decode_step is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)

