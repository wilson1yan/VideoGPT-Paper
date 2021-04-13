import h5py
import numpy as np
import glob
import pickle
import warnings
import os
import os.path as osp
import random
import torch

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from videogpt.datasets.abstract_dataset import AbstractDataset
from videogpt.utils import resize_crop


class TGIF(AbstractDataset):
    mse_norm = 0.06
    n_channels = 3
    video_shape = (16, None, None)
    action_dim = None
    name = 'tgif'

    def __init__(self, root, split, include_actions,
                 resolution=64, crop_mode='per_video', extra_scale=1.):
        super().__init__()
        # merge train and val train since we never use val anyways
        # and just use train / test
        assert split in ['train', 'test', 'toy'], split
        assert not include_actions

        split_folder = osp.join(root, split)

        TGIF.video_shape = (16, resolution, resolution)
        self.split = split
        self.resolution = resolution
        self.crop_mode = crop_mode
        self.extra_scale = extra_scale

        self._need_init = True

        self._step_between_clips = 1 if split == 'train' else 16
        self._video_clips_fname = osp.join(split_folder, f"video_clips_{self._step_between_clips}_md.pkl")

        metadata = None
        if osp.exists(self._video_clips_fname):
            with open(self._video_clips_fname, 'rb') as f:
                metadata = pickle.load(f)
            video_list = metadata['video_paths']
        else:
            video_list = glob.glob(osp.join(split_folder, '**', '*.gif'), recursive=True)
            if split == 'train':
                video_list += glob.glob(osp.join(root, 'val', '**', '*.gif'), recursive=True)
            print(f"Found {len(video_list)} videos in {root}")

        video_clips = VideoClips(
           video_paths=video_list,
           clip_length_in_frames=self.video_shape[0],
           frames_between_clips=self._step_between_clips,
           frame_rate=None,
           _precomputed_metadata=metadata,
           num_workers=16,
           _video_width=0,
           _video_height=0,
           _video_min_dimension=0,
           _audio_samples=0
        )

        if not osp.exists(self._video_clips_fname):
            metadata = video_clips.metadata
            with open(self._video_clips_fname, 'wb') as f:
                pickle.dump(metadata, f)

        self._len = video_clips.num_clips()

    def _init_dset(self):
        with open(self._video_clips_fname, 'rb') as f:
            metadata = pickle.load(f)

        video_clips = VideoClips(
            video_paths=metadata['video_paths'],
            clip_length_in_frames=self.video_shape[0],
            frames_between_clips=self._step_between_clips,
            frame_rate=None,
            _precomputed_metadata=metadata,
            num_workers=16,
            _video_width=0,
            _video_height=0,
            _video_min_dimension=0,
            _audio_samples=0
        )

        self._video_clips = video_clips
        self._need_init = False

        warnings.filterwarnings('ignore')

    @property
    def input_shape(self):
        return (self.seq_len, self.resolution, self.resolution)

    @property
    def seq_len(self):
        return self.video_shape[0]

    @property
    def action_dim(self):
        raise NotImplementedError

    def _get_data(self, idx):
        assert 0 <= idx < self._len
        if self._need_init:
            self._init_dset()

        video, _, _, _ = self._video_clips.get_clip(idx)
        video = resize_crop(video, self.resolution, crop_mode=self.crop_mode,
                            extra_scale=self.extra_scale)

        if self.split == 'train' and random.random() < 0.5:
            video = torch.flip(video, [3])

        video = video.contiguous() / 255 - 0.5

        return dict(seq=video)

    def __len__(self):
        return self._len

