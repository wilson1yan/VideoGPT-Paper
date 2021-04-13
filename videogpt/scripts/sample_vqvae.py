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
    config_logger, load_model, get_ckpt, sample
from videogpt.config_model import config_model
from videogpt.config_cond import config_cond_types
from videogpt.layers.utils import shift_dim
import videogpt.logger as logger
from videogpt.common import LatentShapeIndices

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

    config_logger(is_root, output_dir=args.output_dir)
    seed = args.seed + rank
    seed_all(seed)
    device = config_device(rank=rank)

    prior_ckpt = torch.load(get_ckpt(args.prior_ckpt), map_location=device)
    vae_ckpt = torch.load(get_ckpt(prior_ckpt['vae_ckpt']), map_location=device)

    """ Load datasets """
    dset_configs = prior_ckpt['dset_configs']
    cond_hp = prior_ckpt['cond_hp']

    train_loader, test_loader, dset, _ = get_distributed_loaders(
        dset_configs=dset_configs,
        batch_size=args.n_samples,
        size=size, rank=rank, seed=seed
    )

    if args.split == 'train':
        loader = train_loader
    else:
        loader = test_loader
    # shuffle the dataset according to some fixed seed
    loader.sampler.set_epoch(seed)
    batch = next(iter(loader)) # get batch as early as possible for fixed seed sampling examples

    vqvae, vq_hp = load_model(ckpt=vae_ckpt, device=device, freeze_model=True,
                              cond_types=())

    def load_layer_prior(ckpt):
        # must use the same self_gen_types for vae and all prior layers
        cond_types, cond_hp = config_cond_types(
            cond_hp=ckpt['cond_hp'], dset=dset,
            device=device)
        # freeze all previous priors, not the current one
        prior, hp = load_model(
            ckpt=ckpt, device=device, freeze_model=True,
            cond_types=cond_types)
        codebook = vqvae.codebook
        return prior, hp, codebook


    latent_shapes = vqvae.latent_shapes
    quantized_sizes = vqvae.quantized_sizes

    if is_root:
        logger.info('latent shapes', latent_shapes)
        logger.info('quantized sizes', quantized_sizes)
        logger.info('total latents', sum([np.prod(latent_shape) for latent_shape in latent_shapes]))


    prior, prior_hp, codebook = load_layer_prior(prior_ckpt)
    if is_root:
        logger.info(f"Loaded vqvae at iteration {vae_ckpt['iteration']}, loss = {vae_ckpt['best_loss']}")
        logger.pretty_info(vq_hp)
        logger.info(f"Loaded GPT at iteration {prior_ckpt['iteration']}, loss {prior_ckpt['best_loss']}")
        logger.pretty_info(prior_hp)

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
        print(gathered_samples.shape)
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
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--port', type=int, default=23456)

    args = parser.parse_args()
    assert args.split in ['train', 'test']

    main()
