from math import ceil
from typing import Tuple, Dict, Any
import shutil
import os
import random
import time

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torch.distributed as dist

from videogpt.config_model import config_model
from videogpt.dataset import get_datasets
from videogpt.utils import chunk
from videogpt.dist_ops import allgather
from videogpt.layers.utils import shift_dim


def get_rank_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def get_ckpt(ckpt):
    if ckpt is not None:
        assert os.path.exists(ckpt), f'invalid checkpoint path: {ckpt}'
        if os.path.isdir(ckpt):
            if os.path.basename(ckpt) == 'checkpoints':
                ckpt = os.path.join(ckpt, 'model_best.pth.tar')
            else:
                ckpt = os.path.join(ckpt, 'checkpoints/model_best.pth.tar')
        else:
            assert os.path.isfile(ckpt), f'invalid ckpt {ckpt}'
    else:
        pass
    return ckpt


def get_output_dir(output_dir):
    output_dir = f'{output_dir}_{time.time()}'
    if os.environ.get('DEBUG') == '1':
        output_dir = output_dir + '_DEBUG'
    return output_dir


def get_distributed_loaders(dset_configs, batch_size, seed):
    rank, size = get_rank_size()
    # batch size is total batch size for all ranks
    assert batch_size % size == 0

    train_dset, test_dset = get_datasets(**dset_configs)
    train_sampler = data.distributed.DistributedSampler(train_dset, num_replicas=size, rank=rank, seed=seed)
    train_loader = data.DataLoader(train_dset, batch_size=batch_size // size, num_workers=4,
                                   pin_memory=True, sampler=train_sampler)

    test_sampler = data.distributed.DistributedSampler(test_dset, num_replicas=size, rank=rank, seed=seed)
    test_loader = data.DataLoader(test_dset, batch_size=batch_size // size, num_workers=4,
                                  pin_memory=True, sampler=test_sampler)

    return train_loader, test_loader, train_dset

class InfDataLoader:

    def __init__(self, data_loader, init_epoch=0):
        self.epoch = init_epoch
        self.data_loader = data_loader

        self.iter = iter(self)

    def __iter__(self):
        # warning: calling next(iter(loader)) gives the same batch
        # because self.epoch += 1 is not reached
        while True:
            self.data_loader.sampler.set_epoch(self.epoch)
            for batch in self.data_loader:
                yield batch
            self.epoch += 1

    def __next__(self):
        return next(self.iter)


def config_summary_writer(is_root, output_dir):
    if is_root:
        log_folder = os.path.join(output_dir, 'logs')
        os.makedirs(log_folder)
        writer = SummaryWriter(log_folder)

        ckpt_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(ckpt_dir)
    else:
        writer = None
    return writer


def config_device():
    rank, _ = get_rank_size()
    device = torch.device('cuda:{}'.format(rank))
    torch.cuda.set_device(rank)

    return device


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_vqvae(ckpt, device, is_root, freeze_model) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    # load checkpoint if ckpt is a path
    if isinstance(ckpt, str):
        if is_root:
            print(f'Loading VQ-VAE from {ckpt}')
        ckpt = torch.load(ckpt, map_location=device)
    elif isinstance(ckpt, dict):
        pass
    else:
        raise RuntimeError(f'ckpt must be str or dict, but of type {type(ckpt)}')

    if is_root:
        print(f"VQ-VAE checkpoint iteration {ckpt['iteration']} with best loss {ckpt['best_loss']}")

    model, hp = config_model(
        configs_str='', **ckpt['hp'], cond_types=None)
    model = model.to(device=device)
    model.load_state_dict(ckpt['state_dict'])

    # only perform data-dependent init when training from scratch, broadcast otherwise
    model.no_need_init()

    if freeze_model:
        # disable gradients
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        ckpt = None  # no need to return checkpoint dict

    return model, hp, ckpt


def save_checkpoint(state, is_best, is_root, output_dir, filename='checkpoint.pth.tar'):
    if is_root:
        ckpt_dir = os.path.join(output_dir, 'checkpoints')
        filename = os.path.join(ckpt_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(ckpt_dir, 'model_best.pth.tar'))

        print(f'saved checkpoints to {filename}, is_best = {is_best}')
    else:
        # only save checkpoints in root to save writing overhead
        # params from all processes are the same for torch DDP
        pass


def load_model(
        ckpt: Dict[str, Any],
        device: torch.device,
        freeze_model: bool,
        cond_types,
) -> Tuple[Any, Dict[str, Any]]:
    model, hp = config_model(
        configs_str='', **ckpt['hp'], cond_types=cond_types)  # hp will be overwritten
    model = model.to(device=device)

    model.load_state_dict(ckpt['state_dict'])

    if freeze_model:
        # disable gradients
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    return model, hp


def sample(n_samples, batch, cond_hp, vae, prior, codebook,
           device, temperature, rank, size, gather):
    # the following might cause uneven chunks, then gathering hangs
    n_samples_per = max(min(ceil(n_samples / size), batch['video'].shape[0]), 1)
    if rank == 0 and os.environ.get('VERBOSE') == '1':
        print(f'need local samples {n_samples_per}')
    batch = {k: v[:n_samples_per] for k, v in batch.items()}

    # batch for each process should be different
    # assuming seeding is configured correctly
    vae.eval()
    prior.eval()

    with torch.no_grad():
        cond = []
        if cond_hp['n_cond_frames'] > 0:
            images = batch['video'].to(device, non_blocking=True)
            cond_frames = images[:, :, :cond_hp['n_cond_frames']]
            cond.append(cond_frames)
        if cond_hp['class_cond']:
            cond.append(batch['label'].to(device, non_blocking=True))

        _, encodings = prior.sample(
            n=n_samples_per,
            codebook=codebook,
            cond=cond,
            no_flatten=True,
            device=device,
            temperature=temperature,
            is_root=rank == 0
        )

        samples = vae.decode(x=encodings)  # (n, t, h, w, l) -> (n, c, t, h, w)
        samples = (samples + 0.5).clamp(0, 1)  # in [0, 1]
        samples = samples.detach().contiguous()

    if gather:
        samples = allgather(samples, rank, size)  # n_samples = batch_size * world_size
        cond = tuple(allgather(c.clone(), rank, size) for c in cond)

    return samples, cond
