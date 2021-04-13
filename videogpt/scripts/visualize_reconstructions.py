import os
from pprint import pprint
import os.path as osp
import argparse

import torch
import torch.utils.data as data
import numpy as np
import skvideo.io

from videogpt.models.vae import VQVAE
from videogpt.config_dataset import get_config
from videogpt.config_model import config_model
from videogpt.train_utils import seed_all, load_model, get_ckpt, config_device

MAX_N_TO_VIDEO = 16
MAX_BATCH = 16

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--vqvae_ckpt', type=str, required=True)
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-r', '--resolution', type=int, default=64)
parser.add_argument('-f', '--n_frames', type=int, default=16)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('-n', '--n_examples', type=int, default=256)
parser.add_argument('-s', '--seed', type=int, default=0)
args = parser.parse_args()

seed = args.seed
seed_all(seed)

dset_configs = dict(resolution=args.resolution, n_frames=args.n_frames)
_, test_dset, _ = get_config(args.dataset, **dset_configs)
test_loader = data.DataLoader(test_dset, batch_size=args.n_examples, shuffle=True)

device = config_device()
batch = next(iter(test_loader))
images = batch['video'][:args.n_examples].to(device)

vae, _ = load_model(
    ckpt=torch.load(get_ckpt(args.vqvae_ckpt), map_location=device), device=device,
    freeze_model=True, cond_types=())
vae.eval()

images_recon = []
with torch.no_grad():
    for i in range(0, args.n_examples, MAX_BATCH):
        images_recon.append(vae.get_reconstruction(images[i:i + MAX_BATCH]))
images_recon = torch.cat(images_recon, dim=0)

def _post_process(imgs):
    imgs = np.clip(imgs.cpu().numpy(), -0.5, 0.5) + 0.5
    imgs = (imgs * 255).astype(np.uint8)
    imgs = np.transpose(imgs, (0, 2, 3, 4, 1))
    return imgs

images, images_recon = _post_process(images), _post_process(images_recon)
os.makedirs(args.output_dir, exist_ok=True)
for i in range(min(args.n_examples, MAX_N_TO_VIDEO)):
    fname_real = osp.join(args.output_dir, f'real_{i}.mp4')
    fname_recon = osp.join(args.output_dir, f'recon_{i}.mp4')

    skvideo.io.vwrite(fname_real, images[i], inputdict={'-r': '5'})
    skvideo.io.vwrite(fname_recon, images_recon[i], inputdict={'-r': '5'})
np.save(osp.join(args.output_dir, 'samples.npy'), images_recon)
np.save(osp.join(args.output_dir, 'originals.npy'), images)

print(images.shape, images_recon.shape)  # (n_examples, t, h, w, 3)
images_comp = []
for i in range(args.n_examples):
    images_comp.append(images[i])
    images_comp.append(images_recon[i])
images_comp = np.stack(images_comp, axis=0)
np.save(osp.join(args.output_dir, 'comparisons.npy'), images_comp)

print('Outputted videos to', args.output_dir)
