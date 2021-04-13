from collections import OrderedDict
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.pos_embd import BroadcastPosEmbedND
from videogpt.layers.vqvae import Encoder, Decoder, Quantize
from videogpt.layers.utils import SamePadConvNd
from videogpt.layers.utils import shift_dim


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int, codes_per_book: int, n_codebooks: int,
                 input_shape: tuple, downsample: tuple, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int, use_attn: bool, attn_n_heads: int,
                 commitment_cost: float, decay: float, cond_types): # cond_types only to match GPT args
        super().__init__()
        assert len(input_shape) == len(downsample), ('input shape', input_shape, 'ds', downsample)

        assert all([int(math.log2(d)) == math.log2(d)] for d in downsample), f'downsample must be powers of 2'
        ds_shape = tuple([s // d for s, d in zip(input_shape, downsample)])

        if use_attn:
            # share embedding layer between encoder and decoder
            self.pos_embd = BroadcastPosEmbedND(
                shape=ds_shape, embd_dim=num_hiddens
            )
        else:
            self.pos_embd = None
        n_dim = len(input_shape)

        embedding_channels = embedding_dim * n_codebooks
        self.encoder = Encoder(input_shape, num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens,
                               attn_n_heads, downsample,
                               use_attn, pos_embd=self.pos_embd)
        self.decoder = Decoder(ds_shape, embedding_channels,
                               num_hiddens, num_residual_layers,
                               num_residual_hiddens,
                               attn_n_heads, downsample,
                               use_attn, pos_embd=self.pos_embd)

        self.pre_vq_conv1 = SamePadConvNd(
            n_dim,
            in_channels=num_hiddens,
            out_channels=embedding_channels,
            kernel_size=1,
            stride=1)

        self.codebook = Quantize(n_codebooks, codes_per_book,
                                 embedding_dim, commitment_cost,
                                 decay=decay)
        self.input_shape = input_shape

        self.latent_shape = (*ds_shape, n_codebooks)
        self.quantized_shape = (embedding_dim, *ds_shape, n_codebooks)

    @property
    def metrics(self):
        return ['loss', 'commitment', 'perplexity', 'recon']

    @property
    def metrics_fmt(self):
        return [':6.4f'] * len(self.metrics)

    def no_need_init(self):
        assert self.codebook._need_init
        self.codebook._need_init = False

    def forward(self, x):
        """
        :param x: torch.Tensor with shape (b, c, t, h, w)
        """
        return_dict = OrderedDict()
        z = self.pre_vq_conv1(self.encoder(x=x))

        vq_output = self.codebook(z, no_flatten=True)
        dec_inp = vq_output['quantized']
        dec_inp = shift_dim(dec_inp, -1, 1).flatten(1, 2)  # -> (b, l, d, t', h', w') -> (b, l*d, t', h', w')
        x_recon = self.decoder(x=dec_inp)

        commitment_loss = vq_output['commitment_loss']
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        loss = commitment_loss + recon_loss

        return_dict.update(loss=loss,
                           commitment=commitment_loss,
                           recon=recon_loss,
                           perplexity=vq_output['perplexity'])

        return return_dict

    def encode(self, x, no_flatten=False):
        """
        Must be in eval mode.
        :param x: (b, c, t, h, w)
        :param no_flatten:
        :return:
            quantize: (b, d*l, t', h', w') if no_flatten = False (default);
            else: (b, d, t', h', w', l)
            encodings: (b, t', h', w', l)
        """
        z = self.pre_vq_conv1(self.encoder(x=x))
        vq_output = self.codebook(z, no_flatten=no_flatten)
        return vq_output['quantized'], vq_output['encodings']

    def decode(self, x):
        x = self.codebook.dictionary_lookup(x)
        return self.decoder(x)

    def get_reconstruction(self, x):
        _, encodings = self.encode(x=x)
        return self.decode(encodings)
