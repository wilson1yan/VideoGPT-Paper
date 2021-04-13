import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def allreduce(tensor, op=dist.ReduceOp.SUM):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor

def allreduce_list(tensor_list, op=dist.ReduceOp.SUM):
    tensor_list = [t.clone() for t in tensor_list]
    handles = [dist.all_reduce(t, op=op, async_op=True)
               for t in tensor_list]
    for h in handles:
        h.wait()
    return tensor_list


def allreduce_avg(tensor, size):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= size
    return tensor


def allreduce_avg_list(tensor_list, size):
    tensor_list = [t.clone() for t in tensor_list]
    handles = [dist.all_reduce(t, op=dist.ReduceOp.SUM,
                               async_op=True)
               for t in tensor_list]
    for h in handles:
        h.wait()
    return [t / size for t in tensor_list]

def allgather(tensor, rank, size, post_op='cat', sizes=None):
    # sizes a list specifying size of first dim for each tensor for
    # each rank. sizes = None assumes that all tensors on all ranks
    # have the same first dim. Needed for special case when gathering
    # tensors that have different batch dims on different ranks

    if sizes is None:
        tensor_list = [torch.zeros_like(tensor) for _ in range(size)]
        tensor_list[rank].copy_(tensor)
        dist.all_gather(tensor_list, tensor)
    else:
        assert len(sizes) == size, f"{len(sizes)} != {size}"
        assert tensor.shape[0] == sizes[rank], f"{tensor.shape[0]} != {sizes[rank]}"
        max_size = max(sizes)
        tensor_list = [torch.zeros((max_size, *tensor.shape[1:]), dtype=tensor.dtype,
                                   device=tensor.device)
                       for s in sizes]
        tensor_list[rank][:sizes[rank]] = tensor
        dist.all_gather(tensor_list, tensor_list[rank])
        tensor_list = [t[:s] for t, s in zip(tensor_list, sizes)]

    if post_op == 'cat':
        return torch.cat(tensor_list, dim=0)
    elif post_op == 'stack':
        return torch.stack(tensor_list, dim=0)
    else:
        raise NotImplementedError

def broadcast(tensor, src):
    tensor = tensor.clone()
    dist.broadcast(tensor, src)
    return tensor
