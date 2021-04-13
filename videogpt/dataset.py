import os
import os.path as osp
import math
import random
import warnings
import pickle
from collections import namedtuple

import h5py
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

DATA_DIR = os.environ.get("DATA_DIR", '/home/wilson/data/datasets')


def get_datasets(dataset, **dset_configs):
    name_to_dataset = {
        'bair_pushing': BairPushing,
        'ucf101':  UCF101Wrapper
    }

    Dataset = name_to_dataset[dataset]
    root = osp.join(DATA_DIR, dataset)
    train_dset = Dataset(root=root, train=True, **dset_configs)
    test_dset = Dataset(root=root, train=False, **dset_configs)
    return train_dset, test_dset


class BairPushing(data.Dataset):
    def __init__(self, root, train, resolution, n_frames):
        super().__init__()
        self.root = root
        self.train = train
        self.resolution = resolution
        self.n_frames = n_frames

        assert resolution == 64, 'BAIR only supports 64 x 64 video'

        fname = 'bair_pushing.hdf5' if train else 'bair_pushing_test.hdf5'
        self.fpath = osp.join(root, fname)

        f = h5py.File(self.fpath, 'r')
        self.size = len(f['frames'])
        f.close()

        self._need_init = True

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        raise Exception('BairPushing does not support class conditioning')

    def __len__(self):
        return self.size

    def _init_dset(self):
        f = h5py.File(self.fpath, 'r')
        self.frames = f['frames']
        self._need_init = False

    def __getitem__(self, idx):
        if self._need_init:
            self._init_dset()

        video = self.frames[idx]
        start = np.random.randint(low=0, high=video.shape[1] - self.n_frames + 1)
        video = torch.FloatTensor(video[:, start:start + self.n_frames]) / 255. - 0.5

        return dict(video=video)


class UCF101Wrapper(UCF101):
    def __init__(self, root, train, resolution, n_frames, fold=1):
        video_root = osp.join(root, 'UCF-101')
        super(UCF101, self).__init__(video_root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.n_frames = n_frames
        self.annotation_path = os.path.join(root, 'ucfTrainTestlist')
        self.classes = list(sorted(p for p in os.listdir(video_root) if osp.isdir(osp.join(video_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(video_root, class_to_idx, ('avi',), is_valid_file=None)
        video_list = [x[0] for x in self.samples]

        frames_between_clips = 1 if train else 16
        self.video_clips_fname = os.path.join(root, f'ucf_video_clips_{frames_between_clips}_{n_frames}.pkl')
        if not osp.exists(self.video_clips_fname):
            video_clips = VideoClips(
                video_paths=video_list,
                clip_length_in_frames=n_frames,
                frames_between_clips=1,
                num_workers=4
            )
            with open(self.video_clips_fname, 'wb') as f:
                pickle.dump(video_clips, f)
        else:
            with open(self.video_clips_fname, 'rb') as f:
                video_clips = pickle.load(f)
        indices = self._select_fold(video_list, self.annotation_path,
                                    fold, train)
        self.size = video_clips.subset(indices).num_clips()
        self._need_init = True

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self.size

    def _init_dset(self):
        with open(self.video_clips_fname, 'rb') as f:
            video_clips = pickle.load(f)
        video_list = [x[0] for x in self.samples]
        self.video_clips_metadata = video_clips.metadata
        self.indices = self._select_fold(video_list, self.annotation_path,
                                         self.fold, self.train)
        self.video_clips = video_clips.subset(self.indices)

        self._need_init = False
        # filter out the pts warnings
        warnings.filterwarnings('ignore')

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution,
                            'random' if self.train else 'center')

        if self.train and random.random() < 0.5:
            video = torch.flip(video, [3])

        video = video.float() / 255
        video = video - 0.5
        return video

    def __getitem__(self, idx):
        if self._need_init:
            self._init_dset()

        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]
        video = self._preprocess(video)
        one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        one_hot[label] = 1.

        return dict(video=video, label=one_hot)


def resize_crop(video, resolution, crop_mode):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop

    Args
        video: a tensor of shape [t, h, w, c] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'

    Returns
        a processed video of shape [c, t, h, w]
    """
    # [t, h, w, c] -> [t, c, h, w]
    video = video.permute(0, 3, 1, 2).float()
    _, _, h, w = video.shape

    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear')
    t, _, h, w = video.shape

    if crop_mode == 'center':
        w_start = (w - resolution) // 2
        h_start = (h - resolution) // 2
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    elif crop_mode == 'random':
        if w - resolution + 1 <= 0 or h - resolution + 1 <= 0:
            print(video.shape)
        w_start = np.random.randint(low=0, high=w - resolution + 1)
        h_start = np.random.randint(low=0, high=h - resolution + 1)
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    else:
        raise Exception(f"Invalid crop_mode:", crop_mode)

    # [t, c, h, w] -> [c, t, h, w]
    video = video.permute(1, 0, 2, 3).contiguous()
    return video
