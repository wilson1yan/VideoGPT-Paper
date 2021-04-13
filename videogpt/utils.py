import os
import os.path as osp
import math
import time
import contextlib
import numpy as np
import json

import torch
import torch.nn.functional as F

def deepclone(x):
    if isinstance(x, list):
        return [deepclone(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple([deepclone(elem) for elem in x])
    elif isinstance(x, dict):
        return {k: deepclone(v) for k, v in x.items()}
    else:
        return x.clone()


def safe_serialize(obj):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dumps(obj, default=default)


@contextlib.contextmanager
def timer(key, results=None, verbose=False):
    start = time.time()
    yield
    elapsed = time.time() - start

    if verbose:
        print('time/{key} {elapsed:0.3f}'.format(key=key, elapsed=elapsed))
    if results is None:
        return

    if key in results and isinstance(results[key], list):
        results[key].append(elapsed)
    else:
        results[key] = elapsed


def pprint_timer_results(results, prefix='time/'):
    return [(f'{prefix}{k}', np.mean(np.asarray(v))) for k, v in results.items()]


def make_if_not_exist(folder_name):
    if not osp.exists(folder_name):
        os.makedirs(folder_name)


def chunk(total, n_chunks, rank):
    subset = total // n_chunks
    remainder = total % n_chunks
    return subset + (rank < remainder)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meters, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = meters
        self.prefix = prefix

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'
