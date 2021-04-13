import functools
import json
import gc
import math
import time
import os
import os.path as osp
import shutil

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from videogpt.utils import ProgressMeter, AverageMeter
from videogpt.train_utils import get_distributed_loaders, seed_all, get_ckpt, \
    get_output_dir, save_checkpoint, InfDataLoader, load_model, \
    config_summary_writer, config_device
from videogpt.dist_ops import allreduce_avg, allgather, DistributedDataParallel
from videogpt.config_model import config_model


def main():
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    print('Executing script on', ngpus, 'gpus')

    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=size, rank=rank)

    """ Config writer, seed and device """
    writer = config_summary_writer(is_root=is_root, output_dir=args.output_dir)
    seed = args.seed + rank
    seed_all(seed)
    device = config_device()

    if is_root:
        print(f"rank {rank} / size {size}, device {torch.cuda.current_device()}, seed {seed}")

    torch.backends.cudnn.benchmark = True

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        if is_root:
            print(f"Loading VQ-VAE from {args.ckpt}, iteration {ckpt['iteration']}, best_loss {ckpt['best_loss']}")
    else:
        ckpt = None

    """ Build datasets """
    if ckpt is not None:
        dset_configs = ckpt['dset_configs']

        # overwrite args
        args.dataset = dset_configs['dataset']
        args.resolution = dset_configs['resolution']
        args.n_frames = dset_configs['n_frames']
    else:
        dset_configs = dict(dataset=args.dataset,
                            resolution=args.resolution,
                            n_frames=args.n_frames)

    train_loader, test_loader, dset = get_distributed_loaders(
        dset_configs=dset_configs, batch_size=args.batch_size, seed=seed
    )

    if is_root:
        print(
            f"dset loader n_batch: train = {len(train_loader)}, test = {len(test_loader)}")

    """ Build networks """

    if ckpt is not None:
        model, hp = load_model(
            ckpt=ckpt,
            device=device,
            freeze_model=False,
            cond_types=tuple(),
        )
    else:
        model, hp = config_model(
            configs_str=args.cfg,
            input_shape=dset.input_shape,
            cond_types=tuple(),
        )
        model = model.to(device=device)

    # find_unused_parameters needs to be False for gradient checkpointing to work
    model = DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False
    )

    if is_root:
        total_parameters = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        print('Total Parameters {}'.format(total_parameters))

    def inputs_fn(batch):
        x = batch['video']
        x = x.to(device, non_blocking=True)

        return dict(x=x)

    if is_root:
        total_latents = np.prod(model.latent_shape)
        print('total latents', total_latents)
        print('input shape', model.input_shape,
              'latent shape', model.latent_shape,
              'not flattened quantized shape', model.quantized_shape)

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch_start = ckpt['epoch']
        iteration_start = ckpt['iteration'] + 1
        best_loss = ckpt['best_loss']
        if is_root:
            print(f"Loaded VQ-VAE ckpt at Iteration {ckpt['iteration']}, loss {best_loss}")
    else:
        best_loss = float('inf')
        epoch_start = 0
        iteration_start = 1

    train_loader = InfDataLoader(train_loader, epoch_start)

    train_for = functools.partial(train, train_loader=train_loader,
                                  model=model, optimizer=optimizer,
                                  writer=writer, is_root=is_root,
                                  device=device, size=size, inputs_fn=inputs_fn)
    validate_for = functools.partial(validate, test_loader=test_loader,
                                     model=model, writer=writer,
                                     is_root=is_root, device=device,
                                     size=size, inputs_fn=inputs_fn)
    save_reconstructions_for = functools.partial(save_reconstructions,
                                                 test_loader=test_loader, model=model,
                                                 writer=writer, device=device,
                                                 is_root=is_root, inputs_fn=inputs_fn)

    default_ckpt_dict = {
        'dset_configs': dset_configs,
        'hp': hp,
    }

    """ Training loop """
    print('start training loop', rank, '/', size)

    iteration = iteration_start
    while iteration <= args.total_iters:
        iteration = train_for(iteration=iteration)

        if iteration % args.test_every == 0:
            test_loss = validate_for(iteration=iteration)  # reconstruction loss

            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            save_checkpoint({
                'epoch': train_loader.epoch,
                'iteration': iteration,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                **default_ckpt_dict,
            }, is_best=is_best, is_root=is_root, output_dir=args.output_dir)

        if iteration % args.recon_every == 0:
            save_reconstructions_for(iteration=iteration)
        iteration += 1

    if is_root:
        print(f'Logs saved under {args.output_dir}')
        writer.close()


def need_to_return(iteration):
    return any([
        iteration % args.test_every == 0,
        iteration % args.recon_every == 0,
        iteration >= args.total_iters,
        os.environ.get('DEBUG') == '1',
    ])


def train(*, train_loader, model, optimizer, iteration,
          writer, is_root, device, size, inputs_fn):
    batch_time = AverageMeter('time/train', ':6.3f')
    data_time = AverageMeter('time/data', ':6.3f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(model.metrics, model.metrics_fmt)}

    progress = ProgressMeter(
        args.total_iters,
        [batch_time, data_time,] + list(avg_meters.values()),
    )

    model.train()

    end = time.time()
    while True:
        batch = next(train_loader)

        if is_root:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
        data_time.update(time.time() - end)

        bs = batch['video'].shape[0]
        output_dict = model(**inputs_fn(batch))

        for k in model.metrics:
            v = output_dict[k]
            avg_meters[k].update(v.item(), bs)
            reduced_v = allreduce_avg(v, size).item()
            if is_root:
                writer.add_scalar(f"train/{k}", reduced_v, iteration)

        optimizer.zero_grad()
        output_dict['loss'].backward() # backward automatically syncs gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if is_root and iteration % args.log_interval == 0:
            progress.display(iteration)

        if need_to_return(iteration):
            return iteration

        iteration += 1


def validate(*, test_loader, model, iteration, writer, is_root, device, size, inputs_fn):
    batch_time = AverageMeter('time/eval', ':6.3f')
    data_time = AverageMeter('time/data', ':6.3f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(model.metrics, model.metrics_fmt)}

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time] + list(avg_meters.values()),
        prefix="Test:"
    )

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(test_loader):
            bs = batch['video'].shape[0]
            output_dict = model(**inputs_fn(batch))  # no aug for val?

            for k in model.metrics:
                v = output_dict[k]
                avg_meters[k].update(v.item(), bs)
            batch_time.update(time.time() - end)
            end = time.time()

            if is_root and i % args.log_interval == 0:
                progress.display(i)

            if os.environ.get('DEBUG') == '1':
                break
    for k in model.metrics:
        reduced_v = allreduce_avg(torch.tensor(avg_meters[k].avg, device=device), size).item()
        if is_root:
            writer.add_scalar(f"test/{k}", reduced_v, iteration)

    return allreduce_avg(torch.tensor(avg_meters['recon'].avg, device=device), size).item()


def save_reconstructions(test_loader, model, iteration, writer, device, is_root, inputs_fn):
    model.eval()

    inps = inputs_fn(next(iter(test_loader)))
    images = inps['x'][:8]
    T = images.shape[2]

    with torch.no_grad():
        images_recon = model.get_reconstruction(x=images)

    # (b, 2, c, t, h, w) -> (b, 2, t, c, h, w) -> (-1, c, h, w)
    all_images = torch.stack((images, images_recon), dim=1).permute(0, 1, 3, 2, 4, 5).contiguous().cpu()
    all_images = torch.flatten(all_images, end_dim=2)

    all_images = all_images + 0.5

    img_grid = torch.clamp(make_grid(all_images, nrow=T), 0, 1)
    if is_root:
        writer.add_image(f'iter {iteration} reconstruction', img_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='bair_pushing',
                        help='bair_pushing|ucf101 (default: bair_pushing), ignored if ckpt is provided')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-c', '--ckpt', type=str, default=None)

    # Training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='default: 32')
    parser.add_argument('-e', '--total_iters', type=int, default=100000, help='default: 100000')
    parser.add_argument('--lr', type=float, default=3e-4, help='default: 3e-4')
    parser.add_argument('-i', '--log_interval', type=int, default=100, help='default: 100')
    parser.add_argument('-t', '--test_every', type=int, default=10000)
    parser.add_argument('-r', '--recon_every', type=int, default=10000)

    parser.add_argument('-p', '--port', type=int, default=23456,
                        help='tcp port for distributed training (default: 23456)')

    parser.add_argument('--cfg', type=str, default=None,
                        help='model configs, ignored if ckpt is provided')

    args = parser.parse_args()

    args.ckpt = get_ckpt(args.ckpt)
    args.output_dir = get_output_dir(args.output_dir)

    main()
