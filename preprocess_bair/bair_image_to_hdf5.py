import glob
import argparse
import h5py
import numpy as np
from PIL import Image
import os.path as osp
from tqdm import tqdm
import sys

def convert_data(split):
    root_dir = opt.data_dir
    path = osp.join(root_dir, 'processed_data', split)
    traj_paths = glob.glob(osp.join(path, '*', '*'))
    trajs, actions = [], []
    for traj_path in tqdm(traj_paths):
        image_paths = glob.glob(osp.join(traj_path, '*.png'))
        image_paths.sort(key=lambda x: int(osp.splitext(osp.basename(x))[0]))
        traj = []
        for img_path in image_paths:
            img = Image.open(img_path)
            arr = np.array(img) # H, W, C
            traj.append(arr)
        traj = np.stack(traj, axis=0) # T, H, W, C
        traj = np.transpose(traj, (3, 0, 1, 2)) # C, T, H, W
        trajs.append(traj)

        actions.append(np.load(osp.join(traj_path, 'actions.npy')))

    trajs = np.stack(trajs, axis=0) # N, C, T, H, W
    actions = np.stack(actions, axis=0) # N, T, act_dim

    if split == 'train':
        fname = osp.join(root_dir, 'bair_pushing.hdf5')
    else:
        fname = osp.join(root_dir, 'bair_pushing_test.hdf5')

    f = h5py.File(fname, 'a')
    f.create_dataset('frames', data=trajs)
    f.create_dataset('actions', data=actions)
    f.close()

    f = h5py.File(fname, 'r')
    print(f['frames'].shape, f['frames'].dtype)
    print(f['actions'].shape, f['actions'].dtype)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='base directory to save processed data')
opt = parser.parse_args()
convert_data('train')
convert_data('test')
