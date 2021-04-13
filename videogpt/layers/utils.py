import os
import math
from typing import Union, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.checkpoint import checkpoint


def identity(x):
    return x

class Checkpoint(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self.module, *args)
        else:
            return self.module(*args)


def checkpoint_layer(fn, args, use_checkpoint):
    if use_checkpoint:
        return checkpoint(fn, *args)
    else:
        return fn(*args)

def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# Wraps pre and postprocess around a given module
# pre-process: norm
# post-process: dropout -> residual
class WrapPrePostProcess(nn.Module):
    def __init__(self, module, norm_constr, dropout, fn):
        super().__init__()
        self.norm = norm_constr()
        self.module = module
        self.dropout = nn.Dropout(dropout)
        self.fn = fn

    def collect_stats_and_update(self):
        if isinstance(self.module, nn.ModuleList):
            rtn_dict = dict()
            for m in self.module:
                d = m.collect_stats_and_update()
                for k, v in d.items():
                    if k not in rtn_dict:
                        rtn_dict[k] = []
                    rtn_dict[k].append(v)
            return rtn_dict

        else:
            return self.module.collect_stats_and_update()

    def clear_cache(self):
        if isinstance(self.module, nn.ModuleList):
            for m in self.module:
                m.clear_cache()
        else:
            self.module.clear_cache()

    def forward(self, x, cond, decode_step, decode_idx):
        h = self.norm(x, cond)
        h = self.fn(h, cond, decode_step, decode_idx)
        h = self.dropout(h)

        out = x + h
        return out


# Does not support dilation
class SamePadConvNd(nn.Module):
    def __init__(self, n_dim, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, weight_norm=False):
        super().__init__()
        if n_dim == 1:
            Conv = nn.Conv1d
        elif n_dim == 2:
            Conv = nn.Conv2d
        elif n_dim == 3:
            Conv = nn.Conv3d
        else:
            raise NotImplementedError

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dim
        if isinstance(stride, int):
            stride = (stride,) * n_dim

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias
        )
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        self.conv = conv

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTransposeNd(nn.Module):
    def __init__(self, n_dim, in_channels, out_channels, kernel_size, stride: Union[Tuple[int, ...], int] = 1, bias=True):
        super().__init__()

        if n_dim == 1:
            ConvTranspose = nn.ConvTranspose1d
        elif n_dim == 2:
            ConvTranspose = nn.ConvTranspose2d
        elif n_dim == 3:
            ConvTranspose = nn.ConvTranspose3d
        else:
            raise NotImplementedError

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dim
        if isinstance(stride, int):
            stride = (stride,) * n_dim

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = ConvTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # for VALID padding (see nn.ConvTranspose documentation to see how padding works)
            padding=tuple([k - 1 for k in kernel_size]) ,
            bias=bias
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


class LambdaModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x
