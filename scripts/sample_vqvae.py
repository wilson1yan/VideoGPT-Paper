import argparse
from pprint import pprint
import os
import os.path as osp
import functools

from PIL import Image
import numpy as np
from tqdm import tqdm
import numpy as np
import skvideo.io

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.utils import save_image

from videogpt.dist_ops import allgather, DistributedDataParallel
from videogpt.train_utils import seed_all, get_distributed_loaders, config_device, \
    load_model, get_ckpt, sample
from videogpt.config_model import config_model
from videogpt.config_cond import config_cond_types
from videogpt.layers.utils import shift_dim

# In the case the script is being used to generate samples
# to calculate FVD, there's no need to save so many videos
MAX_N_SAMPLES_TO_VIDEO = 32


def main():
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=size, rank=rank)

    assert args.n_samples % size == 0, f'n_samples {args.n_samples} not divisible by size {size}'

    seed = args.seed + rank
    seed_all(seed)
    device = config_device()

    prior_ckpt = torch.load(get_ckpt(args.prior_ckpt), map_location=device)
    vqvae_ckpt = torch.load(get_ckpt(prior_ckpt['vqvae_ckpt']), map_location=device)

    """ Load datasets """
    dset_configs = prior_ckpt['dset_configs']
    cond_hp = prior_ckpt['cond_hp']

    train_loader, test_loader, dset = get_distributed_loaders(
        dset_configs=dset_configs,
        batch_size=args.n_samples,
        seed=seed
    )

    loader = test_loader
    # shuffle the dataset according to some fixed seed
    loader.sampler.set_epoch(seed)
    batch = next(iter(loader)) # get batch as early as possible for fixed seed sampling examples

    vqvae, vq_hp = load_model(ckpt=vqvae_ckpt, device=device, freeze_model=True,
                              cond_types=())

    def load_layer_prior(ckpt):
        # must use the same self_gen_types for vae and all prior layers
        cond_types, cond_hp = config_cond_types(
            cond_hp=ckpt['cond_hp'], dset=dset)
        # freeze all previous priors, not the current one
        prior, hp = load_model(
            ckpt=ckpt, device=device, freeze_model=True,
            cond_types=cond_types)
        codebook = vqvae.codebook
        return prior, hp, codebook


    latent_shape = vqvae.latent_shape
    quantized_shape = vqvae.quantized_shape

    if is_root:
        print('latent shapes', latent_shape)
        print('quantized shape', quantized_shape)
        print('total latents', np.prod(latent_shape))

    prior, prior_hp, codebook = load_layer_prior(prior_ckpt)
    if is_root:
        print(f"Loaded vqvae at iteration {vqvae_ckpt['iteration']}, loss = {vqvae_ckpt['best_loss']}")
        print(f"Loaded GPT at iteration {prior_ckpt['iteration']}, loss {prior_ckpt['best_loss']}")

    """ Generate samples """
    sample_fn = functools.partial(
        sample,
        cond_hp=cond_hp,
        vae=vqvae,
        prior=prior,
        codebook=codebook,
        device=device,
        temperature=args.temperature,
        rank=rank,
        size=size
    )

    gathered_samples, gathered_cond = sample_fn(n_samples=args.n_samples, batch=batch,
                                                gather=True)

    if is_root:
        os.makedirs(args.output_dir, exist_ok=True)
        # (n, c, t, h, w) -> (n, t, c, h, w)
        samples = shift_dim(gathered_samples, 2, 1)
        T = samples.shape[1]
        save_image(samples.flatten(end_dim=1), osp.join(args.output_dir, "samples.png"),
                   nrow=T)

        # (n, t, c, h, w) -> (n, t, h, w, c)
        samples = shift_dim(samples, 2, -1)
        samples = (samples.cpu().numpy() * 255).astype('uint8')

        for i in range(min(args.n_samples, MAX_N_SAMPLES_TO_VIDEO)):
            skvideo.io.vwrite(osp.join(args.output_dir, f'samples_{i}.mp4'),
                              samples[i], inputdict={'-r': '5'})
        np.save(osp.join(args.output_dir, 'samples.npy'), samples)

        print('outputted videos to:', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prior_ckpt', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--n_samples', type=int, default=16)
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=23456)

    args = parser.parse_args()

    main()
