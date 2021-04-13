import functools
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from videogpt.train_utils import get_distributed_loaders, seed_all, get_ckpt, \
    load_model, config_device, sample
from videogpt.fvd.fvd import FVD_SAMPLE_SIZE, get_fvd_logits, frechet_distance, \
    load_fvd_model
from videogpt.config_cond import config_cond_types
from videogpt.dist_ops import allgather


def main():
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    assert FVD_SAMPLE_SIZE % ngpus == 0

    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=size, rank=rank)

    seed = args.seed + rank
    seed_all(seed)
    device = config_device()

    prior_ckpt = torch.load(args.prior_ckpt, map_location=device)
    #################### Load VQ-VAE ########################################
    vqvae_ckpt = torch.load(prior_ckpt['vqvae_ckpt'], map_location=device)

    dset_configs = vqvae_ckpt['dset_configs']

    _, test_loader, dset = get_distributed_loaders(
        dset_configs=dset_configs,
        batch_size=FVD_SAMPLE_SIZE, seed=seed,
    )

    vqvae, vq_hp = load_model(
        ckpt=vqvae_ckpt,
        device=device,
        freeze_model=True,
        cond_types=tuple()
    )

    #################### Load Prior ########################################
    dset_configs = prior_ckpt['dset_configs']
    cond_types, cond_hp = config_cond_types(
        cond_hp=prior_ckpt['cond_hp'], dset=dset
    )

    latent_shape = vqvae.latent_shape
    quantized_shape = vqvae.quantized_shape
    if is_root:
        print('latent shape', latent_shape)
        print('quantized shape', quantized_shape)
        print('total latents', np.prod(latent_shape))

    prior, hp = load_model(ckpt=prior_ckpt, device=device, freeze_model=True,
                           cond_types=cond_types)
    codebook = vqvae.codebook

    if is_root:
        print(f"Loaded vqvae {prior_ckpt['vqvae_ckpt']} at iteration {vqvae_ckpt['iteration']}, loss = {vqvae_ckpt['best_loss']}")
        print(f"Loaded GPT {args.prior_ckpt} at iteration {prior_ckpt['iteration']}, loss {prior_ckpt['best_loss']}")

    #################### Load I3D ########################################
    i3d = load_fvd_model(device)

    #################### Compute FVD ###############################
    sample_fn = functools.partial(
        sample,
        cond_hp=prior_ckpt['cond_hp'],
        vae=vqvae,
        prior=prior,
        codebook=codebook,
        device=device,
        temperature=args.temperature,
        rank=rank if not is_root else 1, # Just so it won't tqdm during sampling
        size=size,
        n_samples=FVD_SAMPLE_SIZE,
        gather=False
    )

    fvds = []
    fvds_star = []
    if is_root:
        pbar = tqdm(total=args.n_trials)
    for _ in range(args.n_trials):
        fvd, fvd_star = eval_fvd(sample_fn, i3d, vqvae, test_loader, device, rank, size, is_root)
        fvds.append(fvd)
        fvds_star.append(fvd_star)

        if is_root:
            pbar.update(1)
            fvd_mean = np.mean(fvds)
            fvd_std = np.std(fvds)

            fvd_star_mean = np.mean(fvds_star)
            fvd_star_std = np.std(fvds_star)

            pbar.set_description(f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/0 {fvd_star_std:.2f}")
    if is_root:
        pbar.close()
        print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/- {fvd_star_std:.2f}")



def eval_fvd(sample_fn, i3d, vqvae, loader, device, rank, size, is_root):
    batch = next(iter(loader))

    fake, _ = sample_fn(batch=batch)
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy()
    fake = (fake * 255).astype('uint8')
    fake_embeddings = get_fvd_logits(fake, i3d=i3d, device=device)

    # real, fake (b, c, t, h, w)
    real = batch['video'].to(device)
    with torch.no_grad():
        real_recon = vqvae.get_reconstruction(x=real)
        real_recon = (real_recon + 0.5).clamp(0, 1) # [-0.5, 0.5] -> [0, 1]
    real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
    real_recon = (real_recon * 255).astype('uint8')
    real_recon_embeddings = get_fvd_logits(real_recon, i3d=i3d, device=device)

    real = real + 0.5 # [-0.5, 0.5] -> [0, 1]
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # (b, t, h, w, c)
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)


    fake_embeddings = allgather(fake_embeddings, rank, size)
    real_recon_embeddings = allgather(real_recon_embeddings, rank, size)
    real_embeddings = allgather(real_embeddings, rank, size)

    assert fake_embeddings.shape[0] == real_recon_embeddings.shape[0] == real_embeddings.shape[0] == FVD_SAMPLE_SIZE

    if is_root:
        fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
        fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)
        return fvd.item(), fvd_star.item()

    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-p', '--prior_ckpt', type=str, required=True)
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    parser.add_argument('-n', '--n_trials', type=int, default=10)
    parser.add_argument('--port', type=int, default=23455)

    args = parser.parse_args()
    args.prior_ckpt = get_ckpt(args.prior_ckpt)

    main()
