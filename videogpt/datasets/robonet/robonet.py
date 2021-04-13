import os
import h5py
import numpy as np
import torch
import random
from videogpt.datasets.abstract_dataset import AbstractDataset
from videogpt.datasets.robonet.metadata_helper import load_metadata
from videogpt.datasets.robonet.hdf5_loader import load_data, load_camera_imgs, load_actions, load_annotations, load_states
from videogpt.datasets.robonet.dataset_utils import color_augment, split_train_val_test
import hashlib
import io


RNG = 11381294392481135266  # random seed to initialize data loader rng
SPLITS = (0.9, 0.05, 0.05)             # train, val, test
#
# class HParams(object):
#     def __init__(self, **d):
#         self.d = d
#
#     def __getattr__(self, item):
#         # print(f"--- fetch item {item}")
#         if item not in self.d:
#             print('trying to get', item)
#             raise
#         return self.d[item]
#
#     def __getstate__(self):
#         return self.d
#
#     def __setstate__(self, state):
#         self.d = state


def _load_data(inputs):
    # TODO: keep h5 file open in Robonet class
    if len(inputs) == 3:
        f_name, file_metadata, hparams = inputs
        return load_data(f_name, file_metadata, hparams)
    elif len(inputs) == 4:
        f_name, file_metadata, hparams, rng = inputs
        return load_data(f_name, file_metadata, hparams, rng)
    raise ValueError


class Robonet(AbstractDataset):
    name = 'robonet'
    mse_norm = 0.059899278  # TODO
    n_channels = 3
    # video_shape = (16, 64, 64)  # TODO
    action_dim = None

    def __init__(self, root, split, use_aug=False, data_aug=None, extra_scale=1, mode=16, resolution=64,
                 include_actions=False,
                 ):
        super().__init__()

        if mode == 0:
            mode = 16

        fpath = os.path.join(root, 'hdf5')
        database = load_metadata(fpath)
        database = database[database['img_T'] >= mode]  # some have 15 frames
        self._metadata = [database]
        self.mode = split
        self._include_actions = include_actions

        self._input_shape = (mode, resolution, resolution) if mode != -1 else (resolution, resolution)

        # from videogpt.datasets.robonet.hdf5_loader import STATE_MISMATCH, ACTION_MISMATCH
        default_dict = {
            'RNG': 11381294392481135266,             # random seed to initialize data loader rng
            'use_random_train_seed': False,          # use a random seed for training objects
            # 'sub_batch_size': 1,                     # sub batch to sample from each source
            'splits': (0.9, 0.05, 0.05),             # train, val, test
            # 'num_epochs': None,                      # maximum number of epochs (None if iterate forever)
            # 'ret_fnames': False,                     # return file names of loaded hdf5 record
            # 'buffer_size': 100,                      # examples to prefetch
            # 'all_modes_max_workers': True,           # use multi-threaded workers regardless of the mode
            # 'load_random_cam': True,                 # load a random camera for each example
            # 'same_cam_across_sub_batch': False,      # same camera across sub_batches
            # 'pool_workers': 0,                       # number of workers for pool (if 0 uses batch_size workers)
            # 'color_augmentation':0.0,                # std of color augmentation (set to 0 for no augmentations)
            # 'train_ex_per_source': [-1],             # list of train_examples per source (set to [-1] to rely on splits only)
            # 'pool_timeout': 10,                      # max time to wait to get batch from pool object
            # 'MAX_RESETS': 10,                        # maximum number of pool resets before error is raised in main thread
        # }
        # default_loader_hparams = {
        #     'target_adim': 4,
        #     'target_sdim': 5,
        #     'state_mismatch': STATE_MISMATCH.ERROR,     # TODO make better flag parsing
        #     'action_mismatch': ACTION_MISMATCH.ERROR,   # TODO make better flag parsing
        #     'img_size': [48, 64],
        #     'img_size': [64, 64],
            # 'cams_to_load': [0],
            # 'impute_autograsp_action': True,
            # 'load_annotations': False,
            # 'zero_if_missing_annotation': False,
            # 'load_T': 0,                              # TODO implement error checking here for jagged reading
            # 'load_T': mode,
        }

        # for k, v in default_loader_hparams.items():
        #     assert k not in default_dict
        #     default_dict[k] = v
        # self._hparams = HParams(**default_dict)
        self._mode = mode

        assert mode >= 0, 'not supporting image mode'
        # self._hparams.d['load_T'] = mode
        # print(self._hparams.load_T)

        self._init_rng()
        self._init_dataset()

    @property
    def input_shape(self):
        return self._input_shape

    def _init_rng(self):
        # if RNG is not supplied then initialize new RNG
        self._random_generator = {}

        seeds = [None for _ in range(len(self.modes) + 1)]
        # if self._hparams.RNG:
        seeds = [i + RNG for i in range(len(seeds))]

        for k, seed in zip(self.modes + ['base'], seeds):
            # if k == 'train' and self._hparams.use_random_train_seed:  # False
            #     seed = None
            self._random_generator[k] = random.Random(seed)
        self._np_rng = np.random.RandomState(self._random_generator['base'].getrandbits(32))

    def _init_dataset(self):
        # if self._hparams.load_random_cam and self._hparams.same_cam_across_sub_batch:
        #     for s in self._metadata:
        #         min_ncam = min(s['ncam'])
        #         if sum(s['ncam'] != min_ncam):
        #             print('sub-batch has data with different ncam but each same_cam_across_sub_batch=True! Could result in un-even cam sampling')
        #             break

        # check batch_size
        # assert self._batch_size % self._hparams.sub_batch_size == 0, "sub batches should evenly divide batch_size!"
        # assert np.isclose(sum(self._hparams.splits), 1) and all([0 <= i <= 1 for i in self._hparams.splits]), "splits is invalid"
        # assert self._hparams.load_T >=0, "load_T should be non-negative!"

        # # set output format
        # output_format = [tf.uint8, tf.float32, tf.float32]
        # if self._hparams.load_annotations:
        #     output_format = output_format + [tf.float32]
        #
        # if self._hparams.ret_fnames:
        #     output_format = output_format + [tf.string]
        # output_format = tuple(output_format)

        # smallest max step length of all dataset sources
        # for m in self._metadata:
        #     print('metadata frame config', min(m.frame['img_T']), min(m.frame['state_T']), min(m.frame['action_T']) + 1)
        min_steps = min([min(min(m.frame['img_T']), min(m.frame['state_T'])) for m in self._metadata])
        # print('min_steps', min_steps)
        # if not self._hparams.load_T:
        #     self._hparams.load_T = min_steps
        # else:
        #     assert self._hparams.load_T <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams.min_T, min_steps)
        # print('load T', self._hparams.load_T)
        assert 0 <= self._mode <= min_steps, f"ask to load {self._mode} but some records have {min_steps} steps"

        # self._n_workers = min(self._batch_size, multiprocessing.cpu_count())
        # if self._hparams.pool_workers:
        #     self._n_workers = min(self._hparams.pool_workers, multiprocessing.cpu_count())
        # self._n_pool_resets = 0
        # self._pool = multiprocessing.Pool(self._n_workers)

        # n_train_ex = 0
        mode_sources = [[] for _ in range(len(self.modes))]
        for m_ind, metadata in enumerate(self._metadata):
            files_per_source = self._split_files(m_ind, metadata)
            assert len(files_per_source) == len(self.modes), "files should be split into {} sets (it's okay if sets are empty)".format(len(self.modes))
            for m, fps in zip(mode_sources, files_per_source):
                if len(fps):
                    self._random_generator['base'].shuffle(fps)
                    m.append((fps, metadata))

        # self._place_holder_dict = self._get_placeholders()
        self._mode_generators = {}

        for name, m in zip(self.modes, mode_sources):
            if m:
                # rng = self._random_generator[name]
                if name == self.mode:
                    mode_source_files, mode_source_metadata = [[t[j] for t in m] for j in range(2)]
                    self._source = mode_source_files, mode_source_metadata
                    # print(name, 'len mode source files', len(mode_source_files))
                    self._n_files = sum([len(f) for f in mode_source_files])
                    # print(name, 'n files', self._n_files)
        #
        #         gen_func = self._wrap_generator(mode_source_files, mode_source_metadata, rng, name)
        #         # if name == self.primary_mode:
        #         #     dataset = tf.data.Dataset.from_generator(gen_func, output_format)
        #         #     dataset = dataset.map(self._get_dict)
        #         #     dataset = dataset.prefetch(self._hparams.buffer_size)
        #         #     self._data_loader_dict = dataset.make_one_shot_iterator().get_next()
        #         # else:
        #         self._mode_generators[name] = gen_func()
        #     else:
        #         print(f"missing mode source for mode {name}")
        #
        # return n_train_ex

    def __len__(self):
        return self._n_files

    @property
    def modes(self):
        return ['train', 'val', 'test']

    @property
    def primary_mode(self):
        return 'train'

    def _split_files(self, source_number, metadata):
        # if self._hparams.train_ex_per_source != [-1]:
        #     return split_train_val_test(metadata, train_ex=self._hparams.train_ex_per_source[source_number], rng=self._random_generator['base'])
        return split_train_val_test(metadata, splits=SPLITS, rng=self._random_generator['base'])

    def _get_data(self, index):
        # frames = self.frames[index][None, :, :, :] - 0.5  # c, t, h, w, bool -> [-0.5, 0.5]
        #
        # # need low < high i.e. len(frames) >= self.seq_len
        # start = np.random.randint(low=0, high=frames.shape[1] - self._seq_len + 1)  # start + seq_len <= len(frames)
        # frames = torch.FloatTensor(frames[:, start:start + self._seq_len])
        #
        # if self._include_actions:
        #     actions = self.actions[index]  # (t, 3) -> (3, t)
        #     actions = torch.FloatTensor(actions[:, start:start + self._seq_len - 1])  # action for the first frame is not included

        # rng = random.Random(rng)
        return_dict = dict()

        f_name, file_metadata = self._source
        assert len(f_name) == len(file_metadata) == 1
        f_name = f_name[0][index]
        file_metadata = file_metadata[0].get_file_metadata(f_name)
        # hparams = self._hparams

        # random camera
        # cams_to_load = [np.random.randint(0, file_metadata['ncam'])]

        assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
        with open(f_name, 'rb') as f:
            buf = f.read()
        assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

        with h5py.File(io.BytesIO(buf), 'r') as hf:
            start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
            assert n_states > 1, "must be more than one state in loaded tensor!"
            target_n_states = max(self._mode, 1)
            assert target_n_states <= n_states
            start_time = np.random.randint(0, n_states - target_n_states + 1)
            n_states = target_n_states

            # assert all([0 <= i < file_metadata['ncam'] for i in cams_to_load]), "cams_to_load out of bounds!"
            # images, selected_cams = [], []
            # for cam_index in cams_to_load:
            #     images.append(load_camera_imgs(cam_index, hf, file_metadata, self.input_shape[1:], start_time, n_states)[None])
            #     selected_cams.append(cam_index)
            # images = np.swapaxes(np.concatenate(images, 0), 0, 1)  # (31, ncam, h, w, c)
            cam_index = np.random.randint(0, file_metadata['ncam'])
            images = load_camera_imgs(
                cam_index=cam_index, file_pointer=hf, file_metadata=file_metadata,
                target_dims=self.input_shape[1:], start_time=start_time, n_load=n_states)

            images = torch.FloatTensor(images / 255.0 - 0.5).permute(3, 0, 1, 2)  # -> (c, t, h, w)
            # print('images shape', images.shape)
            y_gen = torch.zeros(0).long()
            return_dict.update(seq=images, y_gen=y_gen)

            if self._include_actions:
                actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states-1]
                return_dict.update(actions=actions)
            if False:
                states = load_states(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states]
                return_dict.update(states=states)

            if False:
                annotations = load_annotations(hf, file_metadata, hparams, selected_cams)[start_time:start_time + n_states]
                return_dict.update(annotations=annotation)

        return return_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('root', help='path to files containing hdf5 dataset')
    # parser.add_argument('--robots', type=str, nargs='+', default=None, help='will construct a dataset with batches split across given robots')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size for test loader (should be even for non-time test demo to work)')
    # parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    # parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    # parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    # fpath = '~/projects/RoboNet/hdf5'
    # database = load_metadata(fpath)
    # print(database)

    self = Robonet(split='train', root=args.root)

    # RobonNet init
    # source_probs = hparams.pop('source_selection_probabilities', None)
    # source_probs = None
    # self._batch_size = 4
    # self._metadata = [database]

    # initialize hparams and store metadata_frame
    # self._hparams = self._get_default_hparams()#.override_from_dict(hparams)

    # self._init_rng()

    # #initialize dataset
    # self._num_ex_per_epoch = self._init_dataset()
    # print('loaded {} train files'.format(self._num_ex_per_epoch))
    #
    # self._hparams.source_probs = source_probs

    # print(len(next(self._mode_generators['train'])))
    self._get_data(0)

    all_img_T = self._source[1][0]['img_T']
    print('episode length', min(all_img_T), max(all_img_T))

    from videogpt.config_dataset import get_config
    import torch.utils.data as data

    train_dset, test_dset, ds_info = get_config(dset_name='robonet', data_aug=None)
    train_loader = data.DataLoader(train_dset, batch_size=3, num_workers=1,
                                   pin_memory=True, sampler=None)

    print(train_loader)

    # hparams = {'RNG': 0, 'ret_fnames': True, 'load_T': args.load_steps, 'sub_batch_size': 8, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'same_cam_across_sub_batch':True}
    # if args.robots:
    #     from robonet.datasets import load_metadata
    #     meta_data = load_metadata(args.path)
    #     hparams['same_cam_across_sub_batch'] = True
    #     loader = RoboNetDataset(args.batch_size, [meta_data[meta_data['robot'] == r] for r in args.robots], hparams=hparams)
    # else:
    #     loader = RoboNetDataset(args.batch_size, args.path, hparams=hparams)
    #
    # if args.time_test:
    #     _timing_test(args.time_test, loader)
    #     exit(0)
    #
    # tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'f_names']]
    # s = tf.Session()
    # out_tensors = s.run(tensors, feed_dict=loader.build_feed_dict(args.mode))
    #
    # import imageio
    # writer = imageio.get_writer('test_frames.gif')
    # for t in range(out_tensors[0].shape[1]):
    #     writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    # writer.close()
    # import pdb; pdb.set_trace()
    # print('loaded tensors!')
