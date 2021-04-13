import os
import random
import warnings
import pickle
from collections import namedtuple
import os.path as osp

import h5py
import numpy as np

import torch
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

DATA_DIR = os.environ.get("DATA_DIR", '/home/wilson/data/datasets')

InputShapeIndices = namedtuple('InputShapeIndices', ('t', 'h', 'w', 'c'))


def get_config(dset_name, **dset_configs):
    if dset_name == 'bair_pushing':
        return config_bair_pushing(**dset_configs)
    elif dset_name == 'ucf101':
        return config_ucf101(**dset_configs)
    elif dset_name == 'tgif':
        return config_tgif(**dset_configs)
    else:
        raise Exception("Invalid dataset dset_name:", dset_name)


def config_ucf101(resolution, n_frames):
    root = osp.join(DATA_DIR, 'UCF-101')
    annotation_path = os.path.join(DATA_DIR, 'ucfTrainTestlist')

    extensions = ('avi',)
    classes = list(sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    samples = make_dataset(root, class_to_idx, extensions, is_valid_file=None)
    video_list = [x[0] for x in samples]

    kwargs = dict(
        video_paths=video_list,
        clip_length_in_frames=n_frames,
        num_workers=4
    )
    fname = os.path.join(root, f'ucf_video_clips_1_{n_frames}.pkl')
    if not os.path.exists(fname):
        video_clips = VideoClips(
            frames_between_clips=1,
            **kwargs
        )
        with open(fname, 'wb') as f:
            pickle.dump(video_clips, f)

    fname = os.path.join(root, f'ucf_video_clips_16_{n_frames}.pkl')
    if not os.path.exists(fname):
        video_clips = VideoClips(
            frames_between_clips=16,
            **kwargs
        )
        with open(fname, 'wb') as f:
            pickle.dump(video_clips, f)

    train_dset = UCF101Wrapper(root=root, annotation_path=annotation_path, train=True,
        classes=classes, samples=samples, resolution=resolution, n_frames=n_frames
    )
    test_dset = UCF101Wrapper(root=root, annotation_path=annotation_path, train=False,
        classes=classes, samples=samples, resolution=resolution, n_frames=n_frames
    )

    return train_dset, test_dset


class UCF101Wrapper(UCF101):
    input_shape_indices = InputShapeIndices(c=0, t=1, h=2, w=3)
    name = 'ucf101'

    def __init__(self, root, annotation_path, samples, classes, train,
                 resolution, n_frames, fold=1):
        super(UCF101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.fold = fold
        self.train = train

        self.samples = samples
        self.classes = classes

        if train:
            self.video_clips_fname = os.path.join(DATA_DIR, f'ucf101/ucf_video_clips_1_{n_frames}.pkl')
        else:
            self.video_clips_fname = os.path.join(DATA_DIR, f'ucf101/ucf_video_clips_16_{n_frames}.pkl')

        with open(self.video_clips_fname, 'rb') as f:
            video_clips = pickle.load(f)
        video_list = [x[0] for x in self.samples]
        indices = self._select_fold(video_list, annotation_path,
                                    fold, train)
        self.size = video_clips.subset(indices).num_clips()

        self.annotation_path = annotation_path
        self.resolution = resolution
        self.n_frames = n_frames

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