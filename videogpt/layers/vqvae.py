import os
import math
import functools

import numpy as np
from axial_attention import AxialAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.utils import SamePadConvNd, LambdaModule, shift_dim, SamePadConvTransposeNd
from videogpt.dist_ops import broadcast, allreduce


class AttentionResidualBlock(nn.Module):
    def __init__(self, input_shape, num_hiddens, num_residual_hiddens,
                 attn_n_heads, use_attn, pos_embd):
        super().__init__()
        self.use_attn = use_attn
        n_dim = len(input_shape)

        block = [
            nn.SyncBatchNorm(num_hiddens),
            nn.ReLU(inplace=True),
            SamePadConvNd(n_dim=n_dim, in_channels=num_hiddens,
                          out_channels=num_residual_hiddens,
                          kernel_size=3, stride=1, bias=False),
            nn.SyncBatchNorm(num_residual_hiddens),
            nn.ReLU(inplace=True),
            SamePadConvNd(n_dim=n_dim, in_channels=num_residual_hiddens,
                          out_channels=num_hiddens, kernel_size=1,
                          stride=1, bias=False)
        ]
        if use_attn:
            block.extend([
                nn.SyncBatchNorm(num_hiddens),
                nn.ReLU(inplace=True),
                LambdaModule(functools.partial(shift_dim, src_dim=1, dest_dim=-1)),
                LambdaModule(lambda x: x + pos_embd(x)),
                LambdaModule(functools.partial(shift_dim, src_dim=-1, dest_dim=1)),
                AxialAttention(dim=num_hiddens, dim_index=1, heads=attn_n_heads,
                               num_dimensions=n_dim)
            ])
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class AttentionResidualStack(nn.Module):
    def __init__(self, input_shape, num_hiddens,
                 num_residual_layers, num_residual_hiddens,
                 attn_n_heads, use_attn, pos_embd=None):
        super().__init__()

        self.blocks = nn.ModuleList([
            AttentionResidualBlock(input_shape, num_hiddens, num_residual_hiddens,
                                   attn_n_heads, use_attn, pos_embd=pos_embd)
            for _ in range(num_residual_layers)
        ])
        self.bn = nn.SyncBatchNorm(num_hiddens)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return F.relu(self.bn(h))


class Quantize(nn.Module):
    def __init__(self, n_codebooks, codes_per_book, embedding_dim, commitment_cost,
                 decay=0.99, threshold=1.0):
        super(Quantize, self).__init__()
        self.register_buffer('embeddings', torch.randn(n_codebooks, codes_per_book,
                                                       embedding_dim))
        self.register_buffer('N', torch.zeros(n_codebooks, codes_per_book))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self._need_init = True

        self.n_codebooks = n_codebooks
        self.codes_per_book = codes_per_book
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.threshold = threshold # when usage < threshold replace codebook element

    def _tile(self, x):
        d, ew = x.shape[1:]
        if d < self.codes_per_book:
            n_repeats = (self.codes_per_book + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(1, n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        # z: (b, c, t, h, w)
        c = z.shape[1]
        flat_inputs = shift_dim(z, 1, -1).view(-1, self.n_codebooks, self.embedding_dim) # (bthw, n_c, c)
        flat_inputs = flat_inputs.permute(1, 0, 2).contiguous() # (n_c, bthw, c)
        y = self._tile(flat_inputs)

        d = y.shape[1]
        idx0 = torch.cat([torch.zeros(self.codes_per_book) + i
                          for i in range(self.n_codebooks)]).long()
        idx1 = torch.cat([torch.randperm(d)[:self.codes_per_book]
                          for _ in range(self.n_codebooks)]).long()
        _k_rand = y[idx0, idx1].view(self.n_codebooks, self.codes_per_book, self.embedding_dim)
        _k_rand = broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codebooks, self.codes_per_book))

    def forward(self, z, no_flatten=False):
        if self._need_init and self.training:
            self._init_embeddings(z)
        # z: (b, c, t, h, w)
        # flat_inputs: (n_c, bthw, c)
        flat_inputs = shift_dim(z, 1, -1).view(-1, self.n_codebooks, self.embedding_dim)
        flat_inputs = flat_inputs.permute(1, 0, 2).contiguous() # (n_codebooks, bthw, c)
        # (n_c, bthw, 1) - 2 * (n_c, bthw, n_embeddings) + (n_c, 1, n_embeddings)
        distances = (flat_inputs ** 2).sum(dim=2, keepdim=True) \
                    - 2 * torch.bmm(flat_inputs, self.embeddings.transpose(1, 2)) \
                    + (self.embeddings.transpose(1, 2) ** 2).sum(dim=1, keepdim=True)
        # distances: (n_c, bthw, n_embeddings)
        encoding_indices = torch.argmin(distances, dim=2) # (n_c, bthw)
        # (n_c, bthw, n_embeddings)
        encode_onehot = F.one_hot(encoding_indices, self.codes_per_book).type(flat_inputs.dtype)
        # (n_c, b, t, h, w)
        encoding_indices = encoding_indices.view(self.n_codebooks, z.shape[0], *z.shape[2:])
        enc_idxs_return = shift_dim(encoding_indices, 0, -1) # (b, t, h, w, n_c)

        # (b, t, h, w, n_c, c)
        quantized = torch.stack([F.embedding(encoding_indices[i], self.embeddings[i])
                                 for i in range(self.n_codebooks)], dim=-2)
        if no_flatten:
            z = shift_dim(z, 1, -1)
            z = z.view(*z.shape[:-1], self.n_codebooks, self.embedding_dim)
            z = shift_dim(z, -1, 1)
        else:
            quantized = quantized.flatten(start_dim=-2) # (b, t, h, w, n_c*c)
        quantized = shift_dim(quantized, -1, 1)
        # if no_flatten: (b, c, t, h, w, n_c)
        # else: (b, n_c*c, t, h, w)

        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        loss = commitment_loss

        if self.training:
            n_total = encode_onehot.sum(1) # (n_c, n_embeddings)
            n_total = allreduce(n_total)
            encode_sum = torch.bmm(flat_inputs.transpose(1, 2), encode_onehot)
            encode_sum = allreduce(encode_sum)

            self.N.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            self.z_avg.data.mul_(self.decay).add_(encode_sum.transpose(1, 2),
                                                  alpha=1 - self.decay)

            n = self.N.sum(1, keepdim=True) # (n_c, 1)
            weights = (self.N + 1e-7) / (n + self.codes_per_book * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(-1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            d = y.shape[1]
            idx0 = torch.cat([torch.zeros(self.codes_per_book) + i
                              for i in range(self.n_codebooks)]).long()
            idx1 = torch.cat([torch.randperm(d)[:self.codes_per_book]
                              for _ in range(self.n_codebooks)]).long()
            _k_rand = y[idx0, idx1].view(self.n_codebooks, self.codes_per_book, self.embedding_dim)
            _k_rand = broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codebooks, self.codes_per_book, 1) >= self.threshold).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        quantized_st = (quantized - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=1) # (n_c, n_embeddings)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1))
        perplexity = perplexity.mean()

        return dict(quantized=quantized_st, encodings=enc_idxs_return,
                    commitment_loss=commitment_loss, perplexity=perplexity,
                    loss=loss)

    def dictionary_lookup(self, encodings, no_flatten=False):
        # encodings: (b, t, h, w, n_c)
        encodings = shift_dim(encodings, -1, 0) # (n_c, b, t, h, w)
        # (b, t, h, w, n_c, c)
        quantized = torch.stack([F.embedding(encodings[i], self.embeddings[i])
                                 for i in range(self.n_codebooks)], dim=-2)
        if not no_flatten:
            quantized = quantized.flatten(start_dim=-2) # flatten n_c * c
        quantized = shift_dim(quantized, -1, 1) # (b, n_c * c, t, h, w) or (b, c, t, h, w, n_c)
        return quantized


class Encoder(nn.Module):
    def __init__(self, input_shape, num_hiddens,
                 num_residual_layers, num_residual_hiddens,
                 attn_n_heads, downsample, use_attn, pos_embd):
        super().__init__()
        n_dim = len(input_shape)
        self.downsample = downsample

        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        in_channels = 3
        self.ds_convs = nn.ModuleList()
        while np.any(n_times_downsample > 0):
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConvNd(
                n_dim=n_dim,
                in_channels=in_channels,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=stride
            )
            self.ds_convs.append(conv)
            n_times_downsample -= 1
            in_channels = num_hiddens

            input_shape = tuple([t // s for t, s in zip(input_shape, stride)])

        self.conv_last = SamePadConvNd(
            n_dim=n_dim,
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
        )

        self.res_stack = AttentionResidualStack(input_shape, num_hiddens,
                                                num_residual_layers,
                                                num_residual_hiddens,
                                                attn_n_heads, use_attn,
                                                pos_embd=pos_embd)

    def forward(self, x):
        h = x
        for conv in self.ds_convs:
            h = F.relu(conv(h))

        h = self.conv_last(h)
        h = self.res_stack(x=h)

        return h


class Decoder(nn.Module):
    def __init__(self, input_shape, in_channels, num_hiddens,
                 num_residual_layers, num_residual_hiddens,
                 attn_n_heads, upsample, use_attn, pos_embd):
        super().__init__()
        n_dim = len(input_shape)
        self.conv1 = SamePadConvNd(
            n_dim=n_dim,
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
        )

        self.res_stack = AttentionResidualStack(input_shape, num_hiddens,
                                                num_residual_layers,
                                                num_residual_hiddens,
                                                attn_n_heads,use_attn,
                                                pos_embd=pos_embd)

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.us_convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else num_hiddens

            convt = SamePadConvTransposeNd(
                n_dim=n_dim,
                in_channels=num_hiddens,
                out_channels=out_channels,
                kernel_size=4,
                stride=tuple([2 if d > 0 else 1 for d in n_times_upsample])
            )
            self.us_convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.res_stack(x=h)
        for i, convt in enumerate(self.us_convts):
            h = convt(h)
            if i < len(self.us_convts) - 1:
                h = F.relu(h)
        return h
