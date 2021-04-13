from typing import Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn

from videogpt.common import CondType, D_SHAPE_THWL
from videogpt.layers.utils import shift_dim, identity
from videogpt.train_utils import load_model, get_ckpt
import videogpt.logger as logger
from videogpt.models.resnet import resnet_v1
from videogpt.layers.utils import shift_dim, LambdaModule
from videogpt.datasets.abstract_dataset import AbstractDataset

cond_model_resgistry = {
    'prior_resnet_v1': dict(
        model_cls='resnet_v1',
        # TODO
    )
}


def config_cond_types(
        *,
        cond_hp: Union[Dict[str, Any], None],
        dset: AbstractDataset,
        device,
) -> Tuple[Tuple[CondType, ...], Dict[str, Any]]:
    # inputs are first pre-processed by preprocess_op,
    # then cond-nets take in channel-first image / features, output channel-last, N-D features,
    # as well as auxillary loss (as a tuple)
    # output features will be flattened and fed into attention network
    # when enc_attn is True

    # Make compatible with old checkpoints
    cond_types = []

    """ Conditioning on initial frames """

    if cond_hp['n_cond_frames'] > 0:
        cond_init_configs = cond_hp['cond_init_configs']
        cond_net_input_size = (  # channel first
            dset.n_channels, cond_hp['n_cond_frames'], *dset.input_shape[1:])  # (3, n_cond_frame, h, w)
        if cond_init_configs['model'] == 'resnet_v1':  # model for initial frames
            assert len(cond_net_input_size[1:]) == len(cond_init_configs['resnet_output_shape']) == 3
            preprocess_op = identity
            cond_net = resnet_v1(
                input_size=cond_net_input_size,  # if using lambda, value of cond_net_input_size is dynamically determined when it's called
                output_shape=cond_init_configs['resnet_output_shape'],
                resnet_dim=cond_init_configs['resnet_dim'],
                resnet_depth=cond_init_configs['resnet_depth'],
                width_multiplier=cond_init_configs['width_multiplier'],
                cifar_stem=cond_net_input_size[-1] < 128,  # TODO: ?
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
                net=cond_net,
                agg_cond=cond_hp['agg_cond'],
            )
        )

    """ Conditioning on action sequence or class labels """

    if cond_hp['include_actions']:
        assert dset.action_dim > 0

        cond_size = (dset.action_dim,)
        cond_types.append(CondType(name='cond_action', type='affine_norm', out_size=cond_size,
                                   preprocess_op=identity, model=None, # shouldn't be used
                                   agg_cond=True))

    cond_types = tuple(cond_types)

    return cond_types, cond_hp
