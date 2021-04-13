from collections import namedtuple
from typing import Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn

from videogpt.layers.utils import identity
from videogpt.models.resnet import resnet_v1

CondType = namedtuple(
    'CondType',
    ('name', 'type', 'out_size', 'preprocess_op', 'net'),
)


def config_cond_types(
        *, cond_hp: Union[Dict[str, Any], None], dset,
) -> Tuple[Tuple[CondType, ...], Dict[str, Any]]:
    # inputs are first pre-processed by preprocess_op,
    # then cond-nets take in channel-first image / features, output channel-last, N-D features,
    # as well as auxillary loss (as a tuple)
    # output features will be flattened and fed into attention network
    # when enc_attn is True

    cond_types = []

    """ Conditioning on initial frames """

    if cond_hp['n_cond_frames'] > 0:
        cond_init_configs = cond_hp['cond_init_configs']
        cond_net_input_size = (  # channel first
            3, cond_hp['n_cond_frames'], *dset.input_shape[1:])  # (3, n_cond_frame, h, w)
        if cond_init_configs['model'] == 'resnet_v1':  # model for initial frames
            assert len(cond_net_input_size[1:]) == len(cond_init_configs['resnet_output_shape']) == 3
            preprocess_op = identity
            cond_net = resnet_v1(
                input_size=cond_net_input_size,  # if using lambda, value of cond_net_input_size is dynamically determined when it's called
                output_shape=cond_init_configs['resnet_output_shape'],
                resnet_dim=cond_init_configs['resnet_dim'],
                resnet_depth=cond_init_configs['resnet_depth'],
                width_multiplier=cond_init_configs['width_multiplier'],
                cifar_stem=cond_net_input_size[-1] < 128,
                norm_type='ln',
                pool=False,
            )

            cond_net_output_size = (  # channel last
                *cond_init_configs['resnet_output_shape'], cond_init_configs['resnet_dim'])
        else:
            raise NotImplementedError

        cond_types.append(
            CondType(
                name='cond_init',
                type=cond_init_configs['type'],
                out_size=cond_net_output_size,
                preprocess_op=preprocess_op,
                net=cond_net
            )
        )

    """ Conditioning on action sequence or class labels """

    if cond_hp['class_cond']:
        cond_size = (dset.n_classes,)
        cond_types.append(CondType(name='cond_class', type='affine_norm', out_size=cond_size,
                                   preprocess_op=identity, net=None))

    cond_types = tuple(cond_types)

    return cond_types, cond_hp
