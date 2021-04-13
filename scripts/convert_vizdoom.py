import h5py
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob
import argparse
import os
import os.path as osp


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help="folder where ep trajectories are stored")
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--max_eps', type=int, default=1000)
args = parser.parse_args()

ep_paths = glob.glob(osp.join(args.data_dir, 'ep_*'))[:args.max_eps]
print(f"Found {len(ep_paths)} episode folders")

img_size = (64, 64)

idx = 0
ep_start_idxs = []
actions = []
frames = []
for ep_path in tqdm(ep_paths):
    img_paths = glob.glob(osp.join(ep_path, '*.png'))
    if len(img_paths) == 0:
        print('empty', ep_path, 'skipping...')
        continue
    img_paths.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
    ep_frames = []
    for img_path in img_paths:
        img = Image.open(img_path).resize(img_size)
        img = np.array(img).transpose(2, 0, 1) # (c, h, w)
        ep_frames.append(img)
    ep_frames = np.stack(ep_frames, axis=1) # (c, t, h, w)
    frames.append(ep_frames)

    with open(osp.join(ep_path, 'actions.json')) as f:
        ep_actions = json.load(f)
        actions.append(ep_actions) # (t, *)

    assert len(ep_actions) == ep_frames.shape[1], f"{len(ep_actions)} != {ep_frames.shape[1]}"

    ep_start_idxs.append(idx)
    idx += ep_frames.shape[1]

frames = np.concatenate(frames, axis=1)
actions = np.concatenate(actions, axis=0)
ep_start_idxs = np.array(ep_start_idxs).astype('int64')

os.makedirs(args.save_dir, exist_ok=True)

f = h5py.File(osp.join(args.save_dir, 'vizdoom.hdf5'), 'a')
f.create_dataset('frames', data=frames)
f.create_dataset('actions', data=actions)
f.create_dataset('ep_start_idxs', data=ep_start_idxs)
f.close()

f = h5py.File(osp.join(args.save_dir, 'vizdoom.hdf5'), 'r')
print(f['frames'].shape, f['frames'].dtype)
print(f['actions'].shape, f['actions'].dtype)
print(f['ep_start_idxs'].shape, f['ep_start_idxs'].dtype)

