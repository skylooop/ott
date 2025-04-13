from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax.numpy as jnp
import numpy as np

from flax import linen as nn

ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = partial(nn.leaky_relu, negative_slope=0.2)
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            mutable = self.is_mutable_collection('batch_stats')
            x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x
    

def rsoftmax(x, radix, cardinality):
    # (batch_size, features) -> (batch_size, features)
    batch = x.shape[0]
    if radix > 1:
        x = x.reshape((batch, cardinality, radix, -1)).swapaxes(1, 2)
        return nn.softmax(x, axis=1).reshape((batch, -1))
    else:
        return nn.sigmoid(x)


class SplAtConv2d(nn.Module):
    channels: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    groups: int = 1
    radix: int = 2
    reduction_factor: int = 4

    conv_block_cls: ModuleDef = ConvBlock
    cardinality: int = groups

    # Match extra bias here:
    # github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py#L39
    match_reference: bool = False

    @nn.compact
    def __call__(self, x):
        inter_channels = max(x.shape[-1] * self.radix // self.reduction_factor, 32)

        conv_block = self.conv_block_cls(self.channels * self.radix,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         groups=self.groups * self.radix,
                                         padding=self.padding)
        conv_cls = conv_block.conv_cls  # type: ignore
        x = conv_block(x)

        if self.radix > 1:
            # torch split takes split_size: int(rchannel//self.radix)
            # jnp split takes num sections: self.radix
            split = jnp.split(x, self.radix, axis=-1)
            gap = sum(split)
        else:
            gap = x

        gap = gap.mean((1, 2), keepdims=True)  # type: ignore # global average pool

        # Remove force_conv_bias after resolving
        # github.com/zhanghang1989/ResNeSt/issues/125
        gap = self.conv_block_cls(inter_channels,
                                  kernel_size=(1, 1),
                                  groups=self.cardinality,
                                  force_conv_bias=self.match_reference)(gap)

        attn = conv_cls(self.channels * self.radix,
                        kernel_size=(1, 1),
                        feature_group_count=self.cardinality)(gap)  # n x 1 x 1 x c
        attn = attn.reshape((x.shape[0], -1))
        attn = rsoftmax(attn, self.radix, self.cardinality)
        attn = attn.reshape((x.shape[0], 1, 1, -1))

        if self.radix > 1:
            attns = jnp.split(attn, self.radix, axis=-1)
            out = sum(a * s for a, s in zip(attns, split))
        else:
            out = attn * x

        return out
    

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(64,
                                   kernel_size=(7, 7),
                                   strides=(2, 2),
                                   padding=[(3, 3), (3, 3)])(x)


class ResNetDStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 32

    # If True, n_filters for first conv is (input_channels + 1) * 8
    adaptive_first_width: bool = False

    @nn.compact
    def __call__(self, x):
        cls = partial(self.conv_block_cls, kernel_size=(3, 3), padding=((1, 1), (1, 1)))
        first_width = (8 * (x.shape[-1] + 1)
                       if self.adaptive_first_width else self.stem_width)
        x = cls(first_width, strides=(2, 2))(x)
        x = cls(self.stem_width, strides=(1, 1))(x)
        x = cls(self.stem_width * 2, strides=(1, 1))(x)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x)
        return x


class ResNetDSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        if x.shape[-1] != out_shape[-1]:
            x = self.conv_block_cls(out_shape[-1], (1, 1), activation=lambda y: y)(x)
        return x


class ResNeStSkipConnection(ResNetDSkipConnection):
    # Inheritance to ensures our variables dict has the right names.
    pass


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = partial(nn.leaky_relu, negative_slope=0.2)
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y)
        return self.activation(y * 0.1 + skip_cls(self.strides)(x, y.shape))


class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1  # cardinality
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        # Downsampling strides in 3x3 conv instead of 1x1 conv, which improves accuracy.
        # This variant is called ResNet V1.5 (matches torchvision).
        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=((1, 1), (1, 1)))(y)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNetDBlock(ResNetBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNetDBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNeStBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNeStSkipConnection
    avg_pool_first: bool = False
    radix: int = 2

    splat_cls: ModuleDef = SplAtConv2d

    @nn.compact
    def __call__(self, x):
        assert self.radix == 2  # TODO: implement radix != 2

        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)

        if self.strides != (1, 1) and self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.splat_cls(group_width,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding=[(1, 1), (1, 1)],
                           groups=self.groups,
                           radix=self.radix)(y)

        if self.strides != (1, 1) and not self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)

        return self.activation(y + skip_cls(self.strides)(x, y.shape))


def ResNet(
    block_cls: ModuleDef,
    *,
    stage_sizes: Sequence[int],
    n_classes: int,
    hidden_sizes: Sequence[int] = (64, 128, 256, 512),
    conv_cls: ModuleDef = nn.Conv,
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9),
    conv_block_cls: ModuleDef = ConvBlock,
    stem_cls: ModuleDef = ResNetStem,
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1))),
) -> nn.Sequential:
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    layers = [stem_cls(), pool_fn]

    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for b in range(n_blocks):
            strides = (1, 1) if i == 0 or b != 0 else (2, 2)
            layers.append(block_cls(n_hidden=hsize, strides=strides))

    layers.append(partial(jnp.mean, axis=(1, 2)))  # global average pool
    layers.append(nn.Dense(n_classes))
    return nn.Sequential(layers)


# yapf: disable
ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                   stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)

WideResNet50 = partial(ResNet50, hidden_sizes=(128, 256, 512, 1024),
                       block_cls=partial(ResNetBottleneckBlock, expansion=2))
WideResNet101 = partial(ResNet101, hidden_sizes=(128, 256, 512, 1024),
                        block_cls=partial(ResNetBottleneckBlock, expansion=2))

ResNeXt50 = partial(ResNet50,
                    block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=4))
ResNeXt101 = partial(ResNet101,
                     block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=8))

ResNetD18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)

ResNeSt50Fast = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                        stem_cls=ResNetDStem,
                        block_cls=partial(ResNeStBottleneckBlock, avg_pool_first=True))
ResNeSt50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNeStBottleneckBlock)
ResNeSt101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt269 = partial(ResNet, stage_sizes=STAGE_SIZES[269],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)


class NegAbs(nn.Module):
    @nn.compact
    def __call__(self, x):
        return -jnp.abs(x)


class ResNet_D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"

    size: int = 64 
    nlayers: int = 4
    nc: int = 3
    nfilter: int = 64
    nfilter_max: int = 512
    activation = partial(nn.leaky_relu, negative_slope=0.2)
   
    @nn.compact
    def __call__(self, x, train=True):

        x = x.reshape(-1, self.nc, self.size, self.size).transpose(0, 2, 3, 1)

        s0 = 4
        nf = self.nfilter
        nf_max = self.nfilter_max
        
        nf0_last = min(nf_max, nf * 2**self.nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, conv_block_cls=partial(ConvBlock, norm_cls=None)),
            ResNetBlock(nf1, conv_block_cls=partial(ConvBlock, norm_cls=None))
        ]

        for i in range(1, self.nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                partial(nn.avg_pool, window_shape=(3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)]),
                ResNetBlock(nf0, conv_block_cls=partial(ConvBlock, norm_cls=None)),
                ResNetBlock(nf1, conv_block_cls=partial(ConvBlock, norm_cls=None)),
            ]

        batch_size = x.shape[0]

        out = nn.leaky_relu(
            nn.Conv(nf, (3, 3), padding=[(1, 1), (1, 1)])(x),
            0.2
        )

        for block in blocks:
            out = block(out)

        out = out.reshape(batch_size, nf0_last * s0 * s0)
        out = nn.Dense(1)(out)
        
        # out = self.negabs(out)

        return out