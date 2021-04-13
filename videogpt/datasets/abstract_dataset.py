import abc
import os
from typing import Tuple, Dict, Union
import torch
import torch.utils.data as data
from videogpt.common import SHAPE_THW


class AbstractDataset(abc.ABC, data.Dataset):
    # def __init__(self, root, split, include_actions, mode, use_aug, self_gen_types, extra_scale, resolution):
    def __init__(self):
        data.Dataset.__init__(self)

    @property
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @property
    @classmethod
    @abc.abstractmethod
    def mse_norm(cls) -> float:
        ...

    @property
    @classmethod
    @abc.abstractmethod
    def n_channels(cls) -> int:
        ...

    # @property
    # @classmethod
    # @abc.abstractmethod
    # def video_shape(cls) -> SHAPE_THW:
    #     ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def input_shape(self) -> SHAPE_THW:
        ...

    @abc.abstractmethod
    def _get_data(self, index) -> Dict[str, torch.Tensor]:
        ...

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return_dict = self._get_data(index)

        if os.environ.get('DEBUG') == '1':
            seq = return_dict['seq']
            assert seq.shape == (self.n_channels, *self.input_shape), f"{seq.shape}, {(self.n_channels, *self.input_shape)}"
            assert torch.all(seq.ge(-0.5 - 1e-4) & seq.le(0.5 + 1e-4)), f"{torch.min(seq)}, {torch.max(seq)}"

            if 'actions' in return_dict:
                actions = return_dict['actions']
                assert actions.shape == (self.action_dim,), ('got', actions.shape, 'expected', self.action_dim)

        return return_dict

    def __repr__(self):
        return f"{self.name}: {self.n_channels, self.input_shape}"
