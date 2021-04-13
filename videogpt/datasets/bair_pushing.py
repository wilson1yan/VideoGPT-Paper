import os
import h5py
import numpy as np
import torch
from videogpt.datasets.abstract_dataset import AbstractDataset


class BairPushing(AbstractDataset):
    name = 'bair_pushing'
    mse_norm = 0.059899278
    n_channels = 3
    video_shape = (16, 64, 64)

    def __init__(self, root, split, include_actions=False):
        super().__init__()
        assert split in ['train', 'test', 'val'], split
        self.split = split

        # train to reconstruct images or videos
        # to see how the vqvae is when reconstructing single frames
        # assert mode in ['video', 'image']
        mode = self.video_shape[0]

        if split == 'train':
            fname = 'bair_pushing.hdf5'
        else:
            fname = 'bair_pushing_test.hdf5'
        self.fpath = os.path.join(root, fname)

        f = h5py.File(self.fpath, 'r')
        start = 0
        end = len(f['frames'])
        f.close()
        self.start, self.end = start, end

        self._need_init = True
        self._mode = mode
        self._include_actions = include_actions

    @property
    def input_shape(self):
        return (self._mode, *self.video_shape[1:])

    def _init_dset(self):
        f = h5py.File(self.fpath, 'r')
        self.frames = f['frames']
        if self._include_actions:
            self.actions = f['actions']

        self._need_init = False

    @property
    def action_dim(self):
        return (self.input_shape[0] - 1) * 4

    def sample_actions(self, n):
        start_idxs = np.random.randint(0, len(self.actions) - 1, size=n)
        actions = np.stack([self.actions[idx:idx + self.input_shape[0] - 1] for idx in start_idxs], axis=0) # (n, T, 4)
        actions = torch.FloatTensor(actions).view(n, -1) # (n, self.action_dim)
        return actions

    def _get_data(self, idx):
        idx += self.start
        assert self.start <= idx <= self.end
        if self._need_init:
            self._init_dset()

        return_dict = dict()

        images = self.frames[idx]
        start = np.random.randint(low=0, high=images.shape[1] - self.video_shape[0] + 1) # in range {0, ..., 14}
        n_frames = max(self._mode, 1)

        images = torch.FloatTensor(images[:, start:start + n_frames]) / 255. - 0.5

        if self._include_actions:
            actions = self.actions[idx]
            actions = torch.FloatTensor(actions[start:start + n_frames - 1, :]).view(self.action_dim,)
            return_dict.update(actions=actions)

        return_dict.update(seq=images)

        return return_dict

    def __len__(self):
        return self.end - self.start

