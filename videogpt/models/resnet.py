import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

from videogpt.layers.norm import ChannelLayerNorm
from videogpt.layers.utils import SamePadConvNd, shift_dim


class NormReLU(nn.Module):

    def __init__(self, channels, relu=True, affine=True, norm_type='bn'):
        super().__init__()

        self.relu = relu
        if norm_type == 'bn':
            self.norm = nn.SyncBatchNorm(channels)
        elif norm_type == 'ln':
            self.norm = ChannelLayerNorm(channels)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        x_float = x.float()
        x_float = self.norm(x_float)
        x = x_float.type_as(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, n_dim, in_channels, filters,
                 stride, use_projection=False, norm_type='bn'):
        super().__init__()

        if use_projection:
            self.proj_conv = SamePadConvNd(n_dim, in_channels, filters,
                                           kernel_size=1, stride=stride, bias=False)
            self.proj_bnr = NormReLU(filters, relu=False, norm_type=norm_type)

        self.conv1 = SamePadConvNd(n_dim, in_channels, filters,
                                   kernel_size=3, stride=stride, bias=False)
        self.bnr1 = NormReLU(filters, norm_type=norm_type)

        self.conv2 = SamePadConvNd(n_dim, filters, filters,
                                   kernel_size=3, stride=1, bias=False)
        self.bnr2 = NormReLU(filters, norm_type=norm_type)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))

        return F.relu(x + shortcut, inplace=True)


class BottleneckBlock(nn.Module):

    def __init__(self, n_dim, in_channels, filters,
                 stride, use_projection=False, norm_type='bn'):
        super().__init__()

        if use_projection:
            filters_out = 4 * filters
            self.proj_conv = SamePadConvNd(n_dim, in_channels, filters_out,
                                           kernel_size=1, stride=stride, bias=False)
            self.proj_bnr = NormReLU(filters_out, relu=False, norm_type=norm_type)

        self.conv1 = SamePadConvNd(n_dim, in_channels, filters,
                                   kernel_size=1, stride=1, bias=False)
        self.bnr1 = NormReLU(filters, norm_type=norm_type)

        self.conv2 = SamePadConvNd(n_dim, filters, filters,
                                   kernel_size=3, stride=stride, bias=False)
        self.bnr2 = NormReLU(filters, norm_type=norm_type)

        self.conv3 = SamePadConvNd(n_dim, filters, 4 * filters,
                                   kernel_size=1, stride=1, bias=False)
        self.bnr3 = NormReLU(4 * filters, norm_type=norm_type)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))
        x = self.bnr3(self.conv3(x))

        return F.relu(x + shortcut, inplace=True)


class BlockGroup(nn.Module):

    def __init__(self, n_dim, in_channels, filters, block_fn, blocks, stride, norm_type='bn'):
        super().__init__()

        self.start_block = block_fn(n_dim, in_channels, filters, stride,
                                    use_projection=True, norm_type=norm_type)
        in_channels = filters * 4 if block_fn == BottleneckBlock else filters

        self.blocks = []
        for _ in range(1, blocks):
            self.blocks.append(block_fn(n_dim, in_channels, filters, 1, norm_type=norm_type))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.start_block(x)
        x = self.blocks(x)
        return x


class ResNet(nn.Module):

    def __init__(self, n_dim, in_channels, block_fn, layers,
                 width_multiplier, stride, cifar_stem=False,
                 norm_type='bn', resnet_dim=256, pool=True):
        super().__init__()
        self.pool = pool
        self.n_dim = n_dim
        self.width_multiplier = width_multiplier
        self.resnet_dim = resnet_dim

        assert all([int(math.log2(d)) == math.log2(d) for d in stride]), stride
        n_times_downsample = np.array([int(math.log2(d)) for d in stride])

        if cifar_stem:
            self.stem = nn.Sequential(
                SamePadConvNd(n_dim, in_channels, 64 * width_multiplier,
                              kernel_size=3, stride=1, bias=False),
                NormReLU(64 * width_multiplier, norm_type=norm_type)
            )
        else:
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            n_times_downsample -= 1  # conv
            n_times_downsample[-2:] = n_times_downsample[-2:] - 1  # pooling
            MaxPoolNd = nn.MaxPool3d if n_dim == 3 else nn.MaxPool2d
            self.stem = nn.Sequential(
                SamePadConvNd(n_dim, in_channels, 64 * width_multiplier,
                              kernel_size=7, stride=stride, bias=False),
                NormReLU(64 * width_multiplier, norm_type=norm_type),
                MaxPoolNd(kernel_size=3,
                          stride=(1, 2, 2) if n_dim == 3 else 2,
                          padding=1)
            )

        scalar = 4 if block_fn == BottleneckBlock else 1

        self.group1 = BlockGroup(n_dim, 64 * width_multiplier, 64 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[0], stride=1,
                                 norm_type=norm_type)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group2 = BlockGroup(n_dim, 64 * width_multiplier * scalar, 128 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[1], stride=stride,
                                 norm_type=norm_type)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group3 = BlockGroup(n_dim, 128 * width_multiplier * scalar, 256 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[2], stride=stride,
                                 norm_type=norm_type)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group4 = BlockGroup(n_dim, 256 * width_multiplier * scalar,
                                 resnet_dim,
                                 block_fn=block_fn, blocks=layers[3], stride=stride,
                                 norm_type=norm_type)
        assert all([d <= 0 for d in n_times_downsample]), f'final downsample {n_times_downsample}'

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        if self.pool:
            # (b, resnet_dim)
            dim = [2 + i for i in range(self.n_dim)]
            x = torch.mean(x, dim=dim).view(x.shape[0], -1)
        else:
            # (b, t, h, w, resnet_dim)
            x = shift_dim(x, 1, -1)

        return x


def resnet_v1(input_size, resnet_depth, width_multiplier, output_shape=None, stride=None, cifar_stem=False,
                    norm_type='bn', resnet_dim=256, pool=True):
    if stride is None:
        assert output_shape is not None
        # compute stride from output_shape
        stride = tuple(
            dim_in // dim_out for dim_in, dim_out in
            zip(input_size[1:], output_shape)
        )
        assert all(dim_in % dim_out == 0 for dim_in, dim_out in zip(input_size[1:], output_shape))
    model_params = {
        18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
        101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
        152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
        200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    in_channels = input_size[0]
    n_dim = len(input_size[1:])

    params = model_params[resnet_depth]
    return ResNet(n_dim, in_channels, params['block'], params['layers'], width_multiplier,
                  stride, cifar_stem=cifar_stem, norm_type=norm_type,
                  resnet_dim=resnet_dim, pool=pool)
