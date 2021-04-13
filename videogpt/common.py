from collections import namedtuple
from typing import Tuple


# TODO: remove LatentShapeIndices, InputShapeIndices
LatentShapeIndices = namedtuple('LatentShapeIndices', ('t', 'h', 'w', 'l'))
# encodings (b, t, h, w, l)  quantized (b, d, t, h, w, l) or (b, d*l, t, h, w, l)
latent_shape_indices = LatentShapeIndices(t=0, h=1, w=2, l=3)  # this is for encodings, not for quantized

InputShapeIndices = namedtuple('InputShapeIndices', ('t', 'h', 'w', 'c'))
input_shape_indices = InputShapeIndices(c=0, t=1, h=2, w=3)  # assume batches are channel first

D_SHAPE_THW = Tuple[int, int, int, int]  # flatten quantized, (embd_dim*n_codebooks, t, h, w)
D_SHAPE_THWL = Tuple[int, int, int, int, int]  # unflatten quantized, (embd_dim, t, h, w, n_codebooks)
SHAPE_THWL = Tuple[int, int, int, int]  # unflatten latents, chennel-less, l = n_codebooksk
SHAPE_THW = Tuple[int, int, int]

# cond_type = 'affine_norm', 'enc_attn'
# cond_size: a tuple of the shape of the cond tensor
# preprocess_op: a lambda function that take in the initial cond and preprocess it
# model: 'identity', 'resnet', 'transformer', 'pretrained'
#   'identity': feed cond tensor directly to the transformer
#   'resnet': feed cond tensor into resnet
#   'transformer': feed cond tensor into a transformer
#   'pretrained': only works for hier VQ-VAE latents - feed an output layer
#                 of upper-level hierarchy as conditioning
# agg_cond: a boolean specifying whether conds should be concatenated or not
# CondType = namedtuple('CondType', ('cond_type', 'cond_size', 'preprocess_op',
#                                    'model', 'agg_cond'))

CondType = namedtuple(
    'CondType',
    ('name', 'type', 'out_size', 'preprocess_op', 'net', 'agg_cond'),
)
