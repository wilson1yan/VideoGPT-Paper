import argparse
import functools
import time
import os
import os.path as osp
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler

from videogpt.utils import ProgressMeter, AverageMeter, chunk
from videogpt.config_model import config_model
from videogpt.config_cond import config_cond_types
from videogpt.layers.utils import shift_dim
from videogpt.train_utils import get_distributed_loaders, seed_all, get_output_dir, \
    get_ckpt, save_checkpoint, InfDataLoader, load_model, \
    config_summary_writer, config_device, sample
from videogpt.dist_ops import DistributedDataParallel, allreduce_avg_list, allgather
from videogpt.layers.checkpoint import CheckpointFunction


def main():
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0

    dist.init_process_group(backend='nccl', init_method=f"tcp://localhost:{args.port}",
                            world_size=size, rank=rank)

    """ Config writer, seed and device """
    CheckpointFunction.use_amp = args.amp

    writer = config_summary_writer(is_root=is_root, output_dir=args.output_dir)
    seed = args.seed + rank
    seed_all(seed)
    device = config_device()

    torch.backends.cudnn.benchmark = True

    """ Load Dataloaders  """

    if args.ckpt is not None:
        gpt_ckpt = torch.load(args.ckpt, map_location=device)

        if is_root:
            print(f"Loading GPT from checkpoint {args.ckpt} with loss {gpt_ckpt['best_loss']}")
        dset_configs = gpt_ckpt['dset_configs']

        # overwrite
        args.dataset = dset_configs['dataset']
        args.resolution = dset_configs['resolution']
    else:
        gpt_ckpt = None
        dset_configs = dict(dataset=args.dataset, resolution=args.resolution,
                            n_frames=args.n_frames)

    train_loader, test_loader, dset = get_distributed_loaders(
        dset_configs=dset_configs, batch_size=args.batch_size, seed=seed
    )
    if is_root:
        print(f"dset loader n_batch: train = {len(train_loader)}, test = {len(test_loader)}")

    """ Load VQ-VAE """
    vqvae_ckpt = args.vqvae_ckpt if gpt_ckpt is None else gpt_ckpt['vqvae_ckpt']
    if is_root:
        print(f'Loading VQ-VAE from {vqvae_ckpt}')

    vqvae_ckpt_loaded = torch.load(vqvae_ckpt, map_location=device)
    vqvae, vq_hp = load_model(
        ckpt=vqvae_ckpt_loaded,
        device=device, freeze_model=True, cond_types=tuple()
    )
    del vqvae_ckpt_loaded

    latent_shape = vqvae.latent_shape
    quantized_shape = vqvae.quantized_shape
    if is_root:
        print('latent shape', latent_shape)
        print('quantized shape', quantized_shape)
        print('total latents', np.prod(latent_shape))

    """ Config cond_types"""

    if gpt_ckpt is not None:
        cond_hp = gpt_ckpt['cond_hp']
    else:
        cond_hp = dict(
            n_cond_frames=args.n_cond_frames,
            class_cond=args.class_cond,
            cond_init_configs=dict(
                type='enc_attn',
                model='resnet_v1',
                resnet_dim=576,
                resnet_depth=34,
                resnet_output_shape=(1, 16, 16),
                width_multiplier=1,
            ),
        )

    def load_prior(layer_ckpt):
        """ Check consistency """
        layer_cond_types, _ = config_cond_types(
            cond_hp=layer_ckpt['cond_hp'], dset=dset)
        # freeze all previous priors, not the current one
        layer_prior, layer_hp = load_model(
            ckpt=layer_ckpt, device=device, freeze_model=False,
            cond_types=layer_cond_types)
        layer_codebook = vqvae.codebook
        return layer_prior, layer_hp, layer_codebook

    def inputs_fn(batch):
        with torch.no_grad():
            videos = batch['video'].to(device, non_blocking=True)  # (b, c, t, h, w)

            cond = []
            if cond_hp['n_cond_frames'] > 0:
                cond_frames = videos[:, :, :cond_hp['n_cond_frames']]
                cond.append(cond_frames)
            if cond_hp['class_cond']:
                cond.append(batch['label'].to(device, non_blocking=True))

            quantized, encodings = vqvae.encode(x=videos, no_flatten=True)

            # latent_shape = (t, h, w, l)
            quantized = shift_dim(quantized, 1, -1)  # (b, d, t, h, w, l) -> (b, t, h, w, l, d)  # channel first -> last
            encodings = encodings.long()

            cond = tuple(cond)
            return dict(encodings=encodings, quantized=quantized, cond=cond,
                        decode_step=None, decode_idx=None)

    cond_types, cond_hp = config_cond_types(
        cond_hp=cond_hp, dset=dset
    )

    if is_root:
        print('cond_types', [(c.name, c.type, c.out_size) for c in cond_types])

    """ Load GPT snapshot, if any """
    if gpt_ckpt is not None:
        prior, hp, codebook = load_prior(layer_ckpt=gpt_ckpt)

        best_loss = gpt_ckpt['best_loss']

        optimizer = optim.Adam(prior.parameters(), lr=args.lr)
        optimizer.load_state_dict(gpt_ckpt['optimizer'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.total_iters)
        scheduler.load_state_dict(gpt_ckpt['scheduler'])
        scaler = GradScaler()
        scaler.load_state_dict(gpt_ckpt['scaler'])

        epoch_start = gpt_ckpt['epoch']
        iteration_start = gpt_ckpt['iteration'] + 1

        del gpt_ckpt
    else:
        # TODO: use (self_gen_n_embd*num_self_gen_in_use,) i.e. concat, or use below i.e. sum up y_gen?
        prior, hp = config_model(
            configs_str=args.cfg,
            shape=latent_shape,
            in_features=vq_hp['embedding_dim'],
            n_vocab=vq_hp['codes_per_book'],
            cond_types=cond_types,
        )
        prior = prior.to(device)
        codebook = vqvae.codebook

        optimizer = optim.Adam(prior.parameters(), lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.total_iters)
        scaler = GradScaler()
        best_loss = float('inf')

        epoch_start = 0
        iteration_start = 1
    # find_unused_parameters needs to be False for gradient checkpointing to work
    prior = DistributedDataParallel(prior, device_ids=[rank], find_unused_parameters=False,
                                    broadcast_buffers=False)

    if is_root:
        for cond_net in prior.cond_nets:
            print('cond_net size with grad', sum(p.numel() for p in cond_net.parameters() if p.requires_grad))
            print('cond_net size', sum(p.numel() for p in cond_net.parameters()))

    if is_root:
        if args.amp:
            print('Training with AMP')

    # to be saved to model checkpoints
    default_ckpt_dict = {
        'dset_configs': dset_configs,
        'cond_hp': cond_hp,
        'hp': hp,
        'vqvae_ckpt': vqvae_ckpt,
    }

    def get_ckpt_dict(**ckpt_dict):
        return {**ckpt_dict, **default_ckpt_dict}

    if is_root:
        total_parameters = sum([np.prod(p.shape) for p in prior.parameters() if p.requires_grad])
        print('model size: prior params count with grads = {}'.format(total_parameters))

    train_loader = InfDataLoader(train_loader, epoch_start)

    # training and validation, all in latent space
    train_for = functools.partial(
        train,
        train_loader=train_loader,
        inputs_fn=inputs_fn,
        prior=prior,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        is_root=is_root,
        size=size,
        device=device,
    )
    validate_for = functools.partial(
        validate,
        test_loader=test_loader,
        inputs_fn=inputs_fn,
        prior=prior,
        writer=writer,
        is_root=is_root,
        size=size,
        device=device,
    )

    # end to end sampling in pixel space
    sample_fn = functools.partial(
        sample,
        cond_hp=cond_hp,
        vae=vqvae,
        prior=prior,
        codebook=codebook,
        device=device,
        temperature=args.temperature,
        rank=rank,
        size=size,
    )  # takes in n_samples, batch, returns samples of size min(n_samples, batch_size * size (roughly, not verified))
    # tensor (n, c, t, h, w) in [0, 1]

    save_samples_for = functools.partial(
        save_samples,
        sample_fn=sample_fn,
        loader=test_loader,
        writer=writer,
        is_root=is_root,
        size=size,
    )

    iteration = iteration_start
    log_mem_usage, log_time_usage = True, True
    time_start = time.time()

    while iteration <= args.total_iters:
        train_loss, iteration = train_for(iteration=iteration)  # average gen_loss

        if iteration % args.test_every == 0:
            test_loss = validate_for(iteration=iteration)
            if is_root:
                writer.add_scalar('test/gen_loss_gap', test_loss - train_loss, iteration * args.batch_size)
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)

            ckpt_dict = get_ckpt_dict(
                epoch=train_loader.epoch,
                iteration=iteration,
                n_obs=iteration * args.batch_size,
                state_dict=prior.module.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                best_loss=best_loss,
            )
            save_checkpoint(ckpt_dict, is_best=is_best, is_root=is_root,
                            output_dir=args.output_dir)

        if iteration % args.generate_every == 0 and save_samples_for:
            save_samples_for(iteration=iteration)

        iteration += 1

    if is_root:
        print(f'Final iteration: {iteration}, best loss: {best_loss}')
        print(f'Logs saved under {args.output_dir}')
        writer.close()


def need_to_return(iteration):
    return any([
        iteration % args.test_every == 0,
        iteration % args.generate_every == 0,
        os.environ.get('DEBUG') == '1',
    ])


def train(train_loader, inputs_fn, prior, optimizer, scheduler, scaler,
          iteration, writer, is_root, size, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    avg_loss = AverageMeter('Loss', ':6.3f')

    progress = ProgressMeter(
        args.total_iters,
        [batch_time, data_time, avg_loss]
    )

    prior.train()

    end = time.time()
    while True:
        batch = next(train_loader)

        if is_root:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration * args.batch_size)
        data_time.update(time.time() - end)

        bs = batch['video'].shape[0]
        inp = inputs_fn(batch)
        with autocast(enabled=args.amp):
            return_dict = prior(**inp)

            loss = return_dict['loss']

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1)
                optimizer.step()

        scheduler.step(iteration)

        avg_loss.update(loss.item(), bs)
        vals = [loss]
        names = ['loss']

        vals = [v.to(device) for v in vals]
        stats = allreduce_avg_list(vals, size)

        if is_root:
            for name, reduced_stat in zip(names, stats):
                writer.add_scalar(f"train/{name}", reduced_stat, iteration * args.batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if is_root and iteration % args.log_interval == 0:
            progress.display(iteration)

        if need_to_return(iteration):
            return avg_loss.avg, iteration

        iteration += 1


def validate(test_loader, inputs_fn, prior, iteration, writer,
             is_root, size, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    avg_loss = AverageMeter('Loss', ':6.3f')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, avg_loss],
        prefix="Test:"
    )

    prior.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(test_loader):
            bs = batch['video'].shape[0]
            inp = inputs_fn(batch)  # no aug for eval for now, can add separate metrics
            return_dict = prior(**inp) # evaluate with full precision
            loss = return_dict['loss']

            avg_loss.update(loss.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if is_root and i % args.log_interval == 0:
                progress.display(i)

            if i == 1 and os.environ.get('DEBUG') == '1':
                break

    losses = [torch.tensor(avg_loss.avg, device=device)]

    reduced_loss, = allreduce_avg_list(losses, size)
    if is_root:
        writer.add_scalar('test/loss', reduced_loss, iteration * args.batch_size)

    return reduced_loss


def save_samples(sample_fn, loader, iteration, writer, is_root, size):
    # uncomment if you want a different set of conditioning every time
    # loader.sampler.set_epoch(iteration)
    gathered_samples, gathered_cond = sample_fn(n_samples=size, batch=next(iter(loader)), gather=True)

    if is_root:
        samples = gathered_samples.permute(0, 2, 1, 3, 4)  # -> (n, t, c, h, w)
        cond = gathered_cond

        writer.add_video(f'iter {iteration} samples', samples)
        for i, c in enumerate(cond):
            if len(c.shape) == 5: # images
                c = c + 0.5 # (b, c, t, h, w)
                c = c.permute(0, 2, 1, 3, 4).contiguous()
                T = c.shape[1]
                c = c.flatten(end_dim=1) # (b * t, c, h, w)
                img = make_grid(c, nrow=T)
                writer.add_image(f'iter {iteration} cond_{i} image', img)
            elif len(c.shape) == 2: # vector
                dirname = osp.join(args.output_dir, f'cond_{i}_vector')
                os.makedirs(dirname, exist_ok=True)
                fname = osp.join(dirname, f'iter_{iteration}_cond_vector.npy')
                np.save(fname, c.cpu().numpy())
            else:
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--ckpt', type=str, required=False,
                        help='path to GPT checkpoint')
    parser.add_argument('--cfg', type=str, help='ignored when ckpt is provided')
    parser.add_argument('--vqvae_ckpt', type=str, required=False, help='path to VAE checkpoint, ignored when ckpt provided', default=None)
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature when sampling')

    # Dataset parameters
    parser.add_argument('-d', '--dataset', type=str, default='bair_pushing', help='defult: bair_pushing')
    parser.add_argument('-r', '--resolution', type=int, default=64, help='default: 64')
    parser.add_argument('-f', '--n_frames', type=int, default=16, help='default: 16')
    parser.add_argument('--n_cond_frames', type=int, default=0,
                        help='number of frames to condition on')
    parser.add_argument('--class_cond', action='store_true', help='condition on actions')

    # Training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size total for all gpus (default: 128)')
    parser.add_argument('--lr', type=float, default=3e-4, help='default: 3e-4')
    parser.add_argument('-e', '--total_iters', type=int, default=200000,
                        help='default: 200000')
    parser.add_argument('--amp', action='store_true', help='Use AMP training')

    # Logging Parameters
    parser.add_argument('--test_every', type=int, default=10000, help='default: 5000')
    parser.add_argument('--generate_every', type=int, default=10000, help='default: 10000')
    parser.add_argument('-i', '--log_interval', type=int, default=100, help='default: 100')

    parser.add_argument('-p', '--port', type=int, default=23455,
                        help='tcp port for distributed training (default: 23455)')

    args = parser.parse_args()


    args.ckpt = get_ckpt(args.ckpt)
    args.vqvae_ckpt = get_ckpt(args.vqvae_ckpt)
    args.output_dir = get_output_dir(args.output_dir)

    main()
