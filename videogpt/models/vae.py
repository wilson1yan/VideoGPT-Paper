from collections import OrderedDict, namedtuple
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.pos_embd import BroadcastAbsolutePositionalEmbeddingND
from videogpt.layers.vae import Encoder, Decoder, Quantize
from videogpt.layers.utils import SamePadConvNd, LambdaModule
from videogpt.layers.utils import shift_dim
import videogpt.logger as logger


class VQVAE(nn.Module):
    def __init__(
        self,
            channel_in: int,
            channel_out: int,
            embedding_dim: int,  # codebook vector dim
            codes_per_book: int,  # codebook n_vocab
            n_codebooks: int,
            mse_norm: float,
            input_shape: tuple,
            downsample: tuple,
            # encoder / decoder / vq architecture
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            use_attn,
            attn_type,
            attn_kwargs,
            attn_n_heads,
            # training
            commitment_cost,
            decay,
            cond_types, # not used, just to match GPT inputs
            checkpoint=False,
    ):
        super().__init__()
        assert len(input_shape) == len(downsample), ('input shape', input_shape, 'ds', downsample)

        assert all([int(math.log2(d)) == math.log2(d)] for d in downsample), f'downsample must be powers of 2'
        ds_shape = tuple([s // d for s, d in zip(input_shape, downsample)])

        if use_attn:
            # share embedding layer between encoder and decoder
            self.pos_embd = BroadcastAbsolutePositionalEmbeddingND(
                shape=ds_shape, embd_dim=num_hiddens
            )
        else:
            self.pos_embd = None
        n_dim = len(input_shape)

        embedding_channels = embedding_dim * n_codebooks
        self.encoder = Encoder(input_shape, channel_in, num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens,
                               attn_n_heads, downsample,
                               use_attn, attn_type, attn_kwargs,
                               pos_embd=self.pos_embd,
                               checkpoint=checkpoint)
        self.decoder = Decoder(ds_shape, embedding_channels,
                               num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens, channel_out,
                               attn_n_heads, downsample,
                               use_attn, attn_type, attn_kwargs,
                               pos_embd=self.pos_embd,
                               checkpoint=checkpoint)

        self.pre_vq_conv1 = SamePadConvNd(
            n_dim,
            in_channels=num_hiddens,
            out_channels=embedding_channels,
            kernel_size=1,
            stride=1)

        self.codebook = Quantize(n_codebooks, codes_per_book,
                                    embedding_dim, commitment_cost,
                                    decay=decay)
        self.mse_norm = mse_norm
        self.input_shape = input_shape

        self.latent_shapes = ((*ds_shape, n_codebooks),)  # encodings shape
        self.quantized_sizes = ((embedding_dim, *ds_shape, n_codebooks),)
        self.feature_size = (embedding_dim, *ds_shape, n_codebooks)

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
        :return:
            quantize has shape (b, l*d, t', h', w'), encodings (b, t', h', w', l)
            feats: shape (b, latent_seq_len, latent_dim)
        """
        return_dict = OrderedDict()
        z = self.pre_vq_conv1(self.encoder(x=x))

        vq_output = self.codebook(z, no_flatten=True)
        dec_inp = vq_output['quantize']
        dec_inp = shift_dim(dec_inp, -1, 1).flatten(1, 2)  # -> (b, l, d, t', h', w') -> (b, l*d, t', h', w')
        x_recon = self.decoder(x=dec_inp)

        commitment_loss = vq_output['commitment_loss']
        recon_loss = F.mse_loss(x_recon, x) / self.mse_norm
        loss = commitment_loss + recon_loss

        return_dict.update(loss=loss,
                           commitment=commitment_loss,
                           recon=recon_loss,
                           perplexity=vq_output['perplexity'])

        return return_dict

    # Outputs a (b, embedding_dim, t, h, w) tensor of quantized latents
    # and a (b, t, h, w) tensor of codebook encodings in {0, ..., code_size}
    def _encode(self, x, no_flatten=False):
        """
        Must be in eval mode.
        :param x: (b, c, t, h, w)
        :param no_flatten:
        :param layer_idx:
        :return:
            quantize: (b, d*l, t', h', w') if no_flatten = False (default);
            else: (b, d, t', h', w', l)
            encodings: (b, t', h', w', l)
        """
        z = self.pre_vq_conv1(self.encoder(x=x))
        vq_output = self.codebook(z, no_flatten=no_flatten)
        return vq_output['quantize'], vq_output['encodings']

    def encode(self, x, no_flatten=False):
        return self._encode(x, no_flatten)

    # Outputs a (b, c, t, h, w) tensor of reconstructions for codebook encodings
    def _decode(self, encodings):
        quantized = self.codebook.dictionary_lookup(encodings)
        return self.decoder(x=quantized)

    def decode(self, x):
        return self._decode(x)

    def get_reconstruction(self, x):
        _, encodings = self._encode(x=x)
        return self._decode(encodings)
