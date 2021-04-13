import os
import os.path as osp
from .bair_pushing import BairPushing
from .tgif import TGIF
from .robonet.robonet import Robonet


__all__ = [
    'config_tgif',
    'config_bair_pushing_dataset',
    'config_robonet',
]

                                # not used args and just to maintain compatibility
def config_bair_pushing_dataset(resolution=64, crop_mode='per_video', extra_scale=1.,
                                **dset_configs):
    data_dir = '/your/path/here'
    root = os.path.join(data_dir, 'bair_robot_push')
    train_dset = BairPushing(root, split='train', **dset_configs)
    test_dset = BairPushing(root, split='test', **dset_configs)  # TODO: do 2 test dset, use_aug = True (for visualization), False (for comparing llh)

    ds_info = dict()

    return train_dset, test_dset, ds_info


def config_tgif(**dset_configs):
    data_dir = '/your/path/here'
    train_dset = TGIF(root=data_dir, split='train', **dset_configs)
    test_dset = TGIF(root=data_dir, split='test', **dset_configs)

    ds_info = dict()
    return train_dset, test_dset, ds_info

def config_robonet(**dset_configs):
    data_dir = '/your/path/here'
    train_dset = Robonet(root=data_dir, split='train', **dset_configs)
    test_dset = Robonet(root=data_dir, split='test', **dset_configs)
    ds_info = dict()
    return train_dset, test_dset, ds_info
