import torch.nn as nn
import torch
import numpy as np

from videogpt.layers.utils import shift_dim

class ChannelLayerNorm(nn.Module):
    # layer norm on channels
    def __init__(self, in_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.norm(x)
        x = shift_dim(x, -1, 1)
        return x

class LayerNorm(nn.Module):
    def __init__(self, embd_dim, cond_dim):
        super().__init__()
        self.conditional = 'affine_norm' in cond_dim

        if self.conditional:
            vec_dim, = cond_dim['affine_norm']
            self.w = nn.Linear(vec_dim, embd_dim, bias=False)
            nn.init.constant_(self.w.weight.data, 1. / np.sqrt(vec_dim))
            self.wb = nn.Linear(vec_dim, embd_dim, bias=False)
        else:
            self.g = nn.Parameter(torch.ones(embd_dim, dtype=torch.float32), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(embd_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, cond):
        if self.conditional:  # (b, cond_dim)
            g = 1 + self.w(cond['affine_norm']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1]) # (b, ..., embd_dim)
            b = self.wb(cond['affine_norm']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1])
        else:
            g = self.g  # (embd_dim,)
            b = self.b

        x_float = x.float()

        mu = x_float.mean(dim=-1, keepdims=True)
        s = (x_float - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x_float - mu) * (1e-5 + s.rsqrt())  # (b, ..., embd_dim)
        x_float = x_float * g + b

        x = x_float.type_as(x)
        return x
