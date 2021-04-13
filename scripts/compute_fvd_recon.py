import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from videogpt.train_utils import get_distributed_loaders, seed_all, get_ckpt, \
    load_model, config_device
from videogpt.fvd.fvd import FVD_SAMPLE_SIZE, get_fvd_logits, frechet_distance, \
    load_fvd_model
from videogpt.config_cond import config_cond_types


def main():
    assert torch.cuda.is_available()
    seed = args.seed
    seed_all(seed)
    device = config_device()

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"Loading VQ-VAE from {args.ckpt}, iteration {ckpt['iteration']}, best_loss {ckpt['best_loss']}")

    dset_configs = ckpt['dset_configs']

    # Each process has full dataloader
    train_loader, test_loader, dset = get_distributed_loaders(
        dset_configs=dset_configs,
        batch_size=FVD_SAMPLE_SIZE, seed=seed
    )

    vqvae, _ = load_model(
        ckpt=ckpt,
        device=device,
        freeze_model=True,
        cond_types=tuple()
    )

    i3d = load_fvd_model(device)

    eval_fvd(i3d, vqvae, test_loader, device)


def eval_fvd(i3d, vqvae, loader, device):
    real = next(iter(loader))['video'].to(device)
    with torch.no_grad():
        fake = []
        b = 32
        for i in range(0, real.shape[0], b):
            fake.append(vqvae.get_reconstruction(x=real[i:i + b]))
        fake = torch.cat(fake, dim=0)
        fake = (fake + 0.5).clamp(0, 1)

    # real, fake (b, c, t, h, w)
    real = real + 0.5 # [-0.5, 0.5] -> [0, 1]
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # (b, t, h, w, c)
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy()
    fake = (fake * 255).astype('uint8')
    fake_embeddings = get_fvd_logits(fake, i3d=i3d, device=device)

    fvd = frechet_distance(fake_embeddings, real_embeddings)
    print(f"FVD: {fvd}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--ckpt', type=str, required=True)

    args = parser.parse_args()
    args.ckpt = get_ckpt(args.ckpt)

    main()
