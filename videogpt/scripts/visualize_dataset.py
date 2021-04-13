import argparse
import os
import os.path as osp
import numpy as np
import skvideo.io
import torch.utils.data as data
from videogpt.config_dataset import get_config

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='moving_mnist|bair_pushing|vizdoom')
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('-n', '--n_examples', type=int, default=4)
parser.add_argument('-r', '--resolution', type=int, default=64)
parser.add_argument('-f', '--n_frames', type=int, default=16)
args = parser.parse_args()

train_dset, test_dset, _ = get_config(args.dataset,
                                      resolution=args.resolution,
                                      n_frames=args.n_frames)

data_loader = data.DataLoader(train_dset, batch_size=args.n_examples // 2, shuffle=True)
images = next(iter(data_loader))['video'] # b, c, t, h, w, [-0.5, 0.5]
images = ((images + 0.5) * 255).numpy().astype(np.uint8)
train_images = np.transpose(images, (0, 2, 3, 4, 1))

data_loader = data.DataLoader(test_dset, batch_size=args.n_examples // 2, shuffle=True)
images = next(iter(data_loader))['video']
images = ((images + 0.5) * 255).numpy().astype(np.uint8)
test_images = np.transpose(images, (0, 2, 3, 4, 1))

images = np.concatenate((train_images, test_images))
print(images.shape)

os.makedirs(args.output_dir, exist_ok=True)
np.save(osp.join(args.output_dir, 'videos.npy'), images)


for i in range(args.n_examples):
    fname = osp.join(args.output_dir, f'ex_{i}.mp4')
    skvideo.io.vwrite(fname, images[i], inputdict={'-r': '5'})
print('Outputted example videos to', args.output_dir)

