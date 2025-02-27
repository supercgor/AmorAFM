import math
import numpy as np
import torch

from abc import abstractmethod
from functools import partial
from itertools import chain
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Any
from torch import nn, Tensor
from torch.nn import functional as F

# =============================================
# Adapted from github repo: https://github.com/AlexGraikos/diffusion_priors
def conv_nd(dims, in_channels, out_channels, kernel_size, stride: Any = 1, padding = 0, dilation = 1,
            groups = 1, bias = True, padding_mode = "zeros", *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, *args,
                         **kwargs)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, *args,
                         **kwargs)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, *args,
                         **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def max_pool_nd(dims, kernel_size, stride = None, padding=0, dilation=1, return_indices=False,
                ceil_mode=False, *args, **kwargs):
    if dims == 1:
        return nn.MaxPool1d(kernel_size, stride, padding, dilation,
                            return_indices, ceil_mode, *args, **kwargs)
    elif dims == 2:
        return nn.MaxPool2d(kernel_size, stride, padding, dilation,
                            return_indices, ceil_mode, *args, **kwargs)
    elif dims == 3:
        return nn.MaxPool3d(kernel_size, stride, padding, dilation,
                            return_indices, ceil_mode, *args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, kernel_size, stride: Any = None, padding=0, ceil_mode=False,
                count_include_pad=True, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(kernel_size, stride, padding, ceil_mode,
                            count_include_pad, *args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(kernel_size, stride, padding, ceil_mode,
                            count_include_pad, *args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(kernel_size, stride, padding, ceil_mode,
                            count_include_pad, *args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def max_adt_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AdaptiveMaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AdaptiveMaxPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AdaptiveMaxPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_adt_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AdaptiveAvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AdaptiveAvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AdaptiveAvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def normalization(channels):
    return nn.GroupNorm(min(32, channels), channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GatedConvNd(nn.Module):

    def __init__(self,
                 dims,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode="zeros",
                 num_heads=8,
                 *args,
                 **kwargs):
        super().__init__()
        out_channels = out_channels or in_channels
        if num_heads is None:
            num_heads = out_channels
        self.num_head = num_heads
        self.conv = conv_nd(dims, in_channels, num_heads + out_channels,
                            kernel_size, stride, padding, dilation, groups,
                            bias, padding_mode, *args, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        mask = torch.sigmoid(x[:, :self.num_head])  # B, H, ...
        x = x[:, self.num_head:]  # B, C, ...
        x = x.view(x.shape[0], self.num_head, -1,
                   *x.shape[2:])  # B, H, C/H, ...
        x = x * mask[:, :, None]
        x = x.view(x.shape[0], -1, *x.shape[3:])  # B, C, ...
        return x


class ReferenceBlock(nn.Module):
    """
    A block that takes a reference tensor as an input.
    """

    @abstractmethod
    def forward(self, x, ref):
        """
        Apply the module to `x` given `ref` as a reference tensor.
        """


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock, ReferenceBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, ref_emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ReferenceBlock):
                x = layer(x, ref_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 z_down=False,
                 out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.z_down = z_down
        if isinstance(out_size, np.ndarray):
            out_size = out_size.tolist()
        self.out_size = out_size

        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=1)

    def forward(self, x):
        assert x.shape[
            1] == self.channels, f"input channel({x.shape[1]}) must be equal to {self.channels}"
        if self.dims == 3:
            if self.out_size is None:
                if self.z_down:
                    x = F.interpolate(x, scale_factor=2, mode="nearest")
                else:
                    x = F.interpolate(
                        x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                        mode="nearest")
            else:
                x = F.interpolate(x, self.out_size, mode="nearest")
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, self.out_size, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 z_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.z_down = z_down
        if dims == 3 and not z_down:
            stride = (1, 2, 2)
        else:
            stride = 2
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class MixBlock(ReferenceBlock):

    def __init__(self,
                 channels,
                 ref_channels,
                 out_channels: None | int = None,
                 mode="concat",
                 dims=2):
        super().__init__()
        self.channels = channels
        self.ref_channels = ref_channels
        self.out_channels = out_channels or channels
        self.mode = mode
        if mode == "concat":
            self.mix = lambda x, y: torch.cat([x, y], dim=1)
            self.op = conv_nd(dims, channels + ref_channels, out_channels, 1)
        elif mode == "add":
            assert channels == ref_channels, f"channels({channels}) must be equal to ref_channels({ref_channels}) when mode is add"
            self.mix = torch.add
            self.op = conv_nd(dims, channels, out_channels, 1)
        elif mode == "dot":
            assert channels == ref_channels, f"channels({channels}) must be equal to ref_channels({ref_channels}) when mode is dot"
            self.mix = torch.mul
            self.op = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x, ref):
        x = self.mix(x, ref)
        x = self.op(x)
        return x


class GatedResBlock(TimestepBlock):

    def __init__(self,
                 channels,
                 dropout,
                 emb_channels=None,
                 out_channels: None | int = None,
                 dims=3,
                 pos_emb=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.dropout = dropout

        # self.pos_emb = pos_emb
        # self.posemb = PositionalEncoding(channels, flatten=False)
        # self._cache_emb = None

        self.in_layers = nn.Sequential()
        self.in_layers.add_module("norm", normalization(channels))
        self.in_layers.add_module("act", nn.SiLU(True))
        self.in_layers.add_module(
            "conv", GatedConvNd(dims, channels, out_channels, 3, padding=1))

        self.out_layers = nn.Sequential()
        self.out_layers.add_module("norm", normalization(channels))
        self.out_layers.add_module("act", nn.SiLU(True))
        self.out_layers.add_module(
            "conv", GatedConvNd(dims, out_channels, out_channels, 3,
                                padding=1))
        self.out_layers.add_module("drop", nn.Dropout(p=dropout))

        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                conv_nd(dims, emb_channels, self.out_channels, 1),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = GatedConvNd(dims, channels, out_channels, 1)

    def forward(self, x, emb=None):
        h = self.in_layers(x)
        if emb is not None:
            emb = self.emb_layers(emb)
            h = h + emb
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        z_down=False,
        padding_mode="reflect",
        use_gated_conv=False,
        gated_conv_heads=16,
        activation='silu',
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.z_down = z_down
        self.use_gated_conv = use_gated_conv
        self.gated_conv_heads = gated_conv_heads

        _acti = get_activation(activation)
        _conv = partial(
            GatedConvNd,
            num_heads=gated_conv_heads) if use_gated_conv else conv_nd

        self.in_layers = nn.Sequential()
        self.in_layers.add_module("norm", normalization(channels))
        self.in_layers.add_module("act", _acti)
        self.in_layers.add_module(
            "conv",
            _conv(dims,
                  channels,
                  self.out_channels,
                  3,
                  padding=1,
                  padding_mode=padding_mode))

        self.out_layers = nn.Sequential()
        self.out_layers.add_module("norm", normalization(self.out_channels))
        self.out_layers.add_module("act", _acti)
        self.out_layers.add_module("drop", nn.Dropout(p=dropout))
        self.out_layers.add_module(
            "conv",
            _conv(dims, self.out_channels, self.out_channels, 3, padding=1))

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, z_down=z_down)
            self.x_upd = Upsample(channels, False, dims, z_down=z_down)
        elif down:
            self.h_upd = Downsample(channels, False, dims, z_down=z_down)
            self.x_upd = Downsample(channels, False, dims, z_down=z_down)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                _acti,
                nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = _conv(dims,
                                         channels,
                                         self.out_channels,
                                         3,
                                         padding=1)
        else:
            self.skip_connection = _conv(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.emb_channels is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        position_encode=True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.attention = F.scaled_dot_product_attention

        self.position_encode = PositionalEncoding(
            channels) if position_encode else None

        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.norm(x).permute(0, 2, 1)
        q = k = self._pos_emb(h, False)

        q, k, v = self.q(q), self.k(k), self.v(h)

        h = self.attention(query=q, key=k, value=v)
        h = self.proj_out(h).permute(0, 2, 1)
        return (x + h).reshape(b, c, *spatial)

    def _pos_emb(self, x, channel_first=True):
        if self.position_encode is None:
            return x
        else:
            if channel_first:
                b, c, *spatial = x.shape
                shape = (b, *spatial, c)
                pos = self.position_encode(shape, device=x.device)
                pos.transpose_(1, 2)
            else:
                shape = x.shape
                pos = self.position_encode(shape, device=x.device)
            return x + pos


# =====================================
# copied and adapted from github: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py


def positional_encoding(x: torch.Tensor | tuple,
                        channels: int | None = None,
                        temperture: int = 10000,
                        flatten: bool = True,
                        scale: float = 2 * math.pi) -> torch.Tensor:
    # x: (B, x, y, z, ch)
    if isinstance(x, tuple):
        b, *axis, c = x
    else:
        b, *axis, c = x.shape
    channels = channels or c
    axis_space = tuple([torch.linspace(0, 1, i) for i in axis])
    axis_dim = (channels // len(axis_space)) + 1

    dim_t = torch.arange(axis_dim).float()
    dim_t = temperture**(dim_t / axis_dim)  # (axis_dim)

    axis_embed = torch.stack(torch.meshgrid(*axis_space, indexing="ij"),
                             dim=-1) * scale  # (x, y, z, 3)
    axis_embed = axis_embed.unsqueeze(-1) / dim_t  # (x, y, z, 3, axis_dim)
    axis_embed[..., 0::2].sin_()
    axis_embed[..., 1::2].cos_()
    axis_embed = axis_embed.transpose(
        -1, -2).flatten(-2)[..., :channels]  # x, y, z, channels
    if flatten:
        axis_embed = axis_embed.flatten(0, -2)  # (x * y * z, channels)
    return axis_embed.unsqueeze(0)  # (1, x * y * z, c or 1, x, y, z, c)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        channels: int | None = None,
        temperture: int = 10000,
        flatten: bool = True,
        scale: float = 2 * math.pi,
    ):
        super().__init__()
        self.channels = channels
        self.temperture = temperture
        self.flatten = flatten
        self.scale = scale
        self._cache_shape = None
        self._cache = None

    def forward(
            self, x: torch.Tensor | tuple, device=torch.device("cpu")):
        if isinstance(x, tuple):
            xshape = x
        else:
            xshape = x.shape[1:-1]
            
        if xshape == self._cache_shape:
            return self._cache
        else:
            self._cache = positional_encoding(x, self.channels,
                                              self.temperture, self.flatten,
                                              self.scale).to(device)
            self._cache_shape = self._cache.shape[1:-1]
            return self._cache


def get_activation(activation: str):
    if activation == 'silu':
        return nn.SiLU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"activation {activation} is not supported")


class UNetND(nn.Module):
    def __init__(self,
                 in_size: tuple[int, ...] = (10, 100, 100),
                 in_channels: int = 1,
                 out_size: tuple[int, ...] = (4, 32, 32),
                 out_channels: tuple[int] | int = 8,
                 out_conv_blocks=1,
                 model_channels: int = 32,
                 embedding_input: int = 0,  # 0 for bulk water # 1 for cluster water
                 embedding_channels: int = 128,
                 num_res_blocks: tuple[int, ...] = (2, 2),
                 attention_resolutions: tuple[int, ...] = (4, 8),
                 dropout: float = 0.1,
                 channel_mult: tuple[int, ...] = (1, 2, 4, 8),
                 out_mult: int = 1,
                 z_down: tuple[int, ...] = (1, 2, 4),
                 conv_resample: bool = True,
                 num_heads: int = 8,
                 activation: str = 'relu',
                 use_gated_conv: bool = False,
                 gated_conv_heads: int = 16,
                 cls_weight=None, 
                 xy_weight=None, 
                 z_weight=None, 
                 pos_weight=None,
                 **kwargs
                 ):
        super().__init__()
        self.in_size = np.asarray(in_size)
        self.out_size = np.asarray(out_size)
        self.dims = len(in_size)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.embedding_input = embedding_input
        self.embedding_channels = embedding_channels
        self.up_blocks = num_res_blocks[0]
        self.down_blocks = num_res_blocks[1]
        self.dropout = dropout
        self.channel_mult = np.asarray(channel_mult)
        self.out_mult = out_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float
        self.inp_transform = lambda x: x
        self.out_transform = lambda x: x.permute(0, 3, 4, 2, 1).sigmoid()
        
        self.cls_weight = cls_weight or 1.0
        self.xy_weight = xy_weight or 0.5
        self.z_weight = z_weight or 0.5
        
        pos_weight = pos_weight or [5.0, 5.0]

        self.register_buffer("pos_weight", torch.as_tensor(pos_weight))
        self.pos_weight: torch.Tensor
        
        _conv = partial(
            GatedConvNd, num_heads=gated_conv_heads) if use_gated_conv else conv_nd

        if embedding_input > 0:
            self.embedding = nn.Sequential(
                nn.Linear(embedding_input, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
            )
        else:
            self.embedding = None
            self.embedding_channels = None
            self.embedding_input = None

        down_size = []
        up_size = []

        cur_size = self.in_size
        for i in range(len(self.channel_mult)):
            down_size.append(cur_size)
            ds = 2 ** i
            if ds >= out_mult and i < len(channel_mult) - 1:
                up_size.insert(0, cur_size)
            if ds in z_down:
                cur_size = np.ceil(cur_size / 2).astype(int)
            else:
                cur_size = np.ceil(cur_size / [1, 2, 2]).astype(int)

        down_ch = model_channels * self.channel_mult
        up_ch = np.flip(down_ch, axis=0)[:len(up_size)+1]

        self.inp = _conv(self.dims, in_channels, model_channels, 3, padding=1)

        ds = 1
        skip_chs = []
        self.enc = nn.Sequential()
        for level, (in_ch, out_ch) in enumerate(zip(down_ch[:-1], down_ch[1:])):
            for n in range(self.down_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(in_ch,
                                                  self.embedding_channels,
                                                  dropout=dropout,
                                                  dims=self.dims,
                                                  use_gated_conv=use_gated_conv,
                                                  activation=activation,
                                                  gated_conv_heads=gated_conv_heads
                                                  ))
                if ds in attention_resolutions:
                    layer.add_module(f"attn", AttentionBlock(
                        in_ch, num_heads=num_heads))
                skip_chs.append(in_ch)
                self.enc.add_module(f"layer{level}-{n}", layer)
            self.enc.add_module(f"down{level}", Downsample(
                in_ch, conv_resample, dims=self.dims, out_channels=out_ch, z_down=ds in z_down))
            ds *= 2

        in_ch = out_ch

        self.mid = TimestepEmbedSequential()
        self.mid.add_module("res0", ResBlock(in_ch,
                                             self.embedding_channels,
                                             dropout=dropout,
                                             dims=self.dims,
                                             use_gated_conv=use_gated_conv,
                                             activation=activation,
                                             gated_conv_heads=gated_conv_heads
                                             ))
        if ds in attention_resolutions:
            self.mid.add_module("attn", AttentionBlock(
                in_ch, num_heads=num_heads))
            self.mid.add_module("res1", ResBlock(in_ch,
                                                 self.embedding_channels,
                                                 dropout=dropout,
                                                 dims=self.dims,
                                                 use_gated_conv=use_gated_conv,
                                                 activation=activation,
                                                 gated_conv_heads=gated_conv_heads
                                                 ))

        self.dec = nn.Sequential()
        for level, (in_ch, out_ch, ups) in enumerate(zip(up_ch[:-1], up_ch[1:], up_size)):
            self.dec.add_module(f"up{level}", Upsample(
                in_ch, True, self.dims, out_ch, out_size=ups.tolist()))
            ds //= 2
            for n in range(self.up_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(skip_chs.pop() + out_ch,
                                                  self.embedding_channels,
                                                  dropout=dropout,
                                                  out_channels=out_ch,
                                                  dims=self.dims,
                                                  use_gated_conv=use_gated_conv,
                                                  activation=activation,
                                                  gated_conv_heads=gated_conv_heads
                                                  ))
                if ds in attention_resolutions:  # and n < self.up_blocks - 1:
                    layer.add_module(f"attn", AttentionBlock(
                        out_ch, num_heads=num_heads))
                self.dec.add_module(f"layer{level}-{n}", layer)
            if out_mult == ds:
                break

        if (out_size == up_size[-1]).all():
            self.resample = nn.Identity()
        else:
            self.resample = TimestepEmbedSequential(
                avg_adt_pool_nd(self.dims, list(out_size)),
                ResBlock(out_ch, None,
                         dropout=dropout,
                         dims=self.dims,
                         use_gated_conv=use_gated_conv,
                         activation=activation,
                         gated_conv_heads=gated_conv_heads
                         )
            )

        self.out = TimestepEmbedSequential()
        if isinstance(out_channels, int):
            out_channels = (out_channels,)
        for i, ch in enumerate(out_channels):
            layer = TimestepEmbedSequential()
            for j in range(out_conv_blocks):
                layer.add_module(f"conv{j}", _conv(
                    self.dims, out_ch, out_ch, 1))
                layer.add_module(f"act{j}", nn.ReLU())
            layer.add_module(
                f"conv{j+1}", _conv(self.dims, out_ch, ch, 1, bias=True))
            self.out.add_module(f"out{i}", layer)

    def forward(self, x: Tensor, emb: Tensor | None = None) -> Tensor:
        x = self.inp_transform(x)

        xs = []
        x = self.inp(x)
        if self.embedding is not None:
            emb = self.embedding(emb)
        else:
            emb = None
        ds = 1
        
        for i, module in enumerate(self.enc):
            if isinstance(module, Downsample):
                x = module(x)
                ds *= 2
            else:
                x = module(x, emb)
                if ds >= self.out_mult:
                    xs.append(x)

        x = self.mid(x, emb)

        for i, module in enumerate(self.dec):
            if isinstance(module, Upsample):
                x = module(x)
            else:
                y = xs.pop()
                x = torch.cat([x, y], dim=1)
                x = module(x, emb)

        x = self.resample(x)
        xs = []

        for i, module in enumerate(self.out):
            xs.append(module(x))
            
        x = torch.cat(xs, dim=1)
        x = self.out_transform(x)

        return x

    def compute_loss(self, x: torch.Tensor, y):
        B, X, Y, Z, C = x.shape
        pred = x.reshape(B, X * Y * Z, 2, C // 2)
        y = y.reshape(B, X * Y * Z, 2, C // 2)

        mask = y[..., 0] > 0.5

        loss_c = F.binary_cross_entropy_with_logits(pred[..., 0].logit(
        ), y[..., 0], pos_weight=self.pos_weight, reduction='mean')

        loss_xy = F.mse_loss(pred[..., (1, 2)][mask], y[..., (1, 2)][mask])
        loss_z = F.mse_loss(pred[..., 3][mask], y[..., 3][mask])

        total_loss = self.cls_weight * loss_c + \
                     self.xy_weight * loss_xy + \
                     self.z_weight * loss_z

        return total_loss, {'conf': loss_c, 'xy': loss_xy, 'z': loss_z}


class CVAE3D(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels,
                 latent_channels=8,
                 in_size=(12, 25, 25),  # Z, X, Y
                 channel_mult: tuple[int, ...] = (1, 2, 4, 4),
                 z_down: tuple[int, ...] = (1, 2),
                 cond_in_size=(4, 25, 25),
                 cond_channel_mult: tuple[int, ...] = (1, 2, 4, 4),
                 cond_z_down: tuple[int, ...] = (1, 2),
                 attention_resolutions: tuple[int, ...] = (2, 4),
                 dropout=0.0,
                 num_res_blocks=1,
                 use_gated_conv=False,
                 gated_conv_heads=16,
                 conf_weight=None, 
                 offset_weight=None, 
                 vae_weight=None, 
                 pos_weight=None
                 ):
        super().__init__()
        self.in_ch = in_channels
        self.ch = model_channels
        self.z_ch = model_channels if latent_channels is ... else latent_channels
        self.dp = dropout
        self.in_size = np.asarray(in_size)
        self.cond_in_size = in_size if cond_in_size is None else np.asarray(
            cond_in_size)

        in_size = self.in_size
        cond_in_size = self.cond_in_size
        cond_channel_mult = channel_mult if cond_channel_mult is ... else cond_channel_mult
        cond_z_down = z_down if cond_z_down is None else cond_z_down

        self.conf_weight = conf_weight or 1.0
        self.offset_weight = offset_weight or 1.0
        self.vae_weight = vae_weight or 1.0
        pos_weight = pos_weight or [1.0]
        
        self.pos_weight: torch.Tensor
        self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float))

        channels = [self.ch * ch_mult for ch_mult in channel_mult]

        self.in_conv = conv_nd(3, self.in_ch, self.ch, 1)
        
        in_sizes = [in_size]
        cond_in_sizes = [cond_in_size]
        
        for i in range(len(channel_mult) - 1):
            if 2 ** i in z_down:
                in_size = np.ceil(in_size / 2).astype(int)    
            else:
                in_size = np.ceil(in_size / [1, 2, 2]).astype(int)
            if 2 ** i in cond_z_down:
                cond_in_size = np.ceil(cond_in_size / 2).astype(int)
            else:
                cond_in_size = np.ceil(cond_in_size / [1, 2, 2]).astype(int)
                
            in_sizes.append(in_size)
            cond_in_sizes.append(cond_in_size)
        
        self.vae_enc = TimestepEmbedSequential()
        self.cond_enc = TimestepEmbedSequential()
        
        for i in range(len(channels)):
            ds = 2 ** i
            layer = TimestepEmbedSequential()
            clayer = TimestepEmbedSequential()
            
            for j in range(num_res_blocks):
                layer.add_module(f'res{j}',  ResBlock(channels[i], None, 
                                                      out_channels=channels[i], 
                                                      dropout = self.dp, 
                                                      dims = 3, 
                                                      padding_mode='zeros', 
                                                      use_gated_conv = use_gated_conv,
                                                      gated_conv_heads=gated_conv_heads
                                                      ))
                
                clayer.add_module(f'res{j}', ResBlock(channels[i], None, 
                                                      out_channels=channels[i], 
                                                      dropout = self.dp, 
                                                      dims = 3, 
                                                      padding_mode='zeros', 
                                                      use_gated_conv = use_gated_conv,
                                                      gated_conv_heads=gated_conv_heads
                                                      ))
                if ds in attention_resolutions:
                    layer.add_module(f'att{j}', AttentionBlock(channels[i], num_heads = 8))
                    clayer.add_module(f'att{j}', AttentionBlock(channels[i], num_heads = 8))
            
            if i < len(channels) - 1:
                layer.add_module(f'down',  Downsample(channels[i], True, 3, channels[i + 1], z_down = ds in z_down))
                clayer.add_module(f'down', Downsample(channels[i], True, 3, channels[i + 1], z_down = ds in cond_z_down))
            else:
                layer.add_module(f"pool", avg_adt_pool_nd(3, (1, 1, 1)))
                layer.add_module(f"flat", nn.Flatten(1))
                layer.add_module(f"lin", nn.Linear(channels[i], 2 * self.z_ch))
                
                clayer.add_module(f"pool", avg_adt_pool_nd(3, (1, 1, 1)))
                clayer.add_module(f"flat", nn.Flatten(1))
                clayer.add_module(f"lin", nn.Linear(channels[i], self.z_ch))
                
            self.vae_enc.add_module(f'layer{i}', layer)
            self.cond_enc.add_module(f'layer{i}', clayer)
        
        in_sizes = in_sizes[::-1]
        cond_in_sizes = cond_in_sizes[::-1]
        channels = channels[::-1]
        
        self.dec = TimestepEmbedSequential()
        for i in range(len(channels)):
            ds = 2 ** (len(channels) - 1 - i)
            layer = TimestepEmbedSequential()
            if i == 0:
                self.dec.add_module(f"lin", nn.Linear(self.z_ch, channels[i] * np.prod(in_sizes[i])))
                self.dec.add_module(f"unflat", nn.Unflatten(1, (channels[i], *in_sizes[i].tolist())))
            for j in range(num_res_blocks):
                layer.add_module(f'res{j}', ResBlock(channels[i], None, 
                                                     out_channels=channels[i], 
                                                     dropout = self.dp, 
                                                     dims = 3, 
                                                     padding_mode='zeros',
                                                     use_gated_conv = use_gated_conv,
                                                     gated_conv_heads=gated_conv_heads
                                                     ))
                if ds in attention_resolutions:
                    layer.add_module(f'att{j}', AttentionBlock(channels[i], num_heads = 8))
            if i < len(channels) - 1:
                layer.add_module(f'up', Upsample(channels[i], True, 3, channels[i + 1], out_size = in_sizes[i + 1]))
            
            self.dec.add_module(f'layer{i}', layer)
        
        layer = TimestepEmbedSequential(
            nn.SiLU(),
            conv_nd(3, channels[-1], self.in_ch, 1),
            nn.Sigmoid()
        )
        
        self.dec.add_module('out', layer)

    def forward(self, pred):
        pred = self.inp_transform(pred)
        x, cond = torch.split(
            pred, [self.in_size[0], self.cond_in_size[0]], dim=2)

        z_enc = self.in_conv(x)
        c_enc = self.in_conv(cond)

        z_enc = self.vae_enc(z_enc)
        c_enc = self.cond_enc(c_enc)

        x = self.reparameterize(z_enc)

        x = self.dec(x)

        x = torch.cat([x, cond], dim=2)  # B C Z X Y

        x = self.out_transform(x)  # B X Y Z C

        return x, (z_enc, ), (c_enc, )

    def compute_loss(self, pred, targ, zs, cs):
        pred_x, pred_c = torch.split(
            pred, [self.in_size[0], self.cond_in_size[0]], dim=-2)  # B X Y Z C
        targ_x, targ_c = torch.split(
            targ, [self.in_size[0], self.cond_in_size[0]], dim=-2)

        mask = targ_x[..., 0] > 0.5
        mask_w = pred_x[..., 0].detach().clamp(min=0.5) * mask

        num_elem = (mask.sum() + pred_x[..., 0].numel()) / mask.shape[0]

        conf = F.binary_cross_entropy_with_logits(pred_x[..., 0].logit(
            eps=1E-6), targ_x[..., 0], pos_weight=self.pos_weight)
        off = F.binary_cross_entropy(pred_x[..., 1:4][mask], targ_x[..., 1:4][mask], reduction='none') - \
            F.binary_cross_entropy(
                targ_x[..., 1:4][mask], targ_x[..., 1:4][mask], reduction='none')
        # print(off.shape, mask_w[mask].shape)
        off = (off * mask_w[mask][:, None]).mean()

        kls = []
        for z, c in zip(zs, cs):
            mu, logvar = torch.chunk(z, 2, dim=1)
            kl = -0.5 * (1 + logvar - (mu - c).pow(2) - logvar.exp())
            kl = kl.flatten(1).sum(1) / num_elem
            kls.append(kl.mean())

        total_loss = conf * self.conf_weight + off * \
            self.offset_weight + sum(kls) * self.vae_weight

        return total_loss, {'vae': sum(kls), 'conf': conf, 'offset': off}

    def inp_transform(self, x):
        # B X Y Z C -> B C Z X Y
        x = x.permute(0, 4, 3, 1, 2)
        return x

    def out_transform(self, x):
        # B C Z X Y -> B X Y Z C
        x = x.permute(0, 3, 4, 2, 1)
        return x

    def reparameterize(self, x):
        mu, logvar = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def conditional_sample(self, cond, resample=False):
        cond = self.inp_transform(cond)
        if not np.allclose(cond.shape[2:], self.cond_in_size):
            cond = cond[:, :, self.in_size[0]:self.in_size[0] +
                        self.cond_in_size[0], :self.cond_in_size[1], :self.cond_in_size[2]]
        c_enc = self.in_conv(cond)
        c_enc = self.cond_enc(c_enc)
        if resample:
            x = torch.randn_like(c_enc) + c_enc
        else:
            x = c_enc

        x = self.dec(x)

        x = torch.cat([x, cond], dim=2)  # B C Z X Y

        x = self.out_transform(x)  # B X Y Z C

        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class GANDiscriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels=16,
                 channels_mult=(1, 2, 4),
                 z_down=(),
                 ):
        super().__init__()

        channels = [in_channels] + [model_channels * i for i in channels_mult]
        self.block = nn.Sequential()

        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            stride = 2 if (2 ** i) in z_down else (1, 2, 2)
            layer = nn.Sequential()
            layer.add_module(f"conv{i}", conv_nd(
                3, in_ch, out_ch, kernel_size=4, stride=stride))
            layer.add_module(f"bn{i}", nn.InstanceNorm3d(out_ch))
            layer.add_module(f"act{i}", nn.LeakyReLU(0.2, True))
            self.block.add_module(f"layer{i}", layer)
        self.out = conv_nd(3, channels_mult[-1] * model_channels, 1, kernel_size=1)
        
        self.inp_transform = lambda x: x
        self.out_transform = lambda x: x.sigmoid()

    def forward(self, inputs):
        x = self.inp_transform(inputs)
        for module in self.block:
            x = module(x)
        x = self.out(x)
        return self.out_transform(x)

class CycleGAN(nn.Module):
    def __init__(self,
                 in_size=(10, 100, 100),
                 channels=1,
                 out_conv_blocks=1,
                 model_channels=16,
                 num_res_blocks=(1, 1),
                 attention_resolutions=(),
                 dropout=0.0,
                 gen_channel_mult=(1, 2, 2, 4),
                 disc_channel_mult=(1, 2, 4),
                 out_mult=1,
                 gen_z_down=(1, 2, 4),
                 disc_z_down=(),
                 conv_resample=True,
                 num_heads=8,
                 activation='relu',
                 ):
        super().__init__()

        self.G_to_A = UNetND(
            in_size=in_size,
            in_channels=channels,
            out_size=in_size,
            out_channels=channels,
            out_conv_blocks=out_conv_blocks,
            model_channels=model_channels,
            embedding_input=0,
            embedding_channels=0,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=gen_channel_mult,
            out_mult=out_mult,
            z_down=gen_z_down,
            conv_resample=conv_resample,
            num_heads=num_heads,
            activation=activation
        )

        self.G_to_B = UNetND(
            in_size=in_size,
            in_channels=channels,
            out_size=in_size,
            out_channels=channels,
            out_conv_blocks=out_conv_blocks,
            model_channels=model_channels,
            embedding_input=0,
            embedding_channels=0,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=gen_channel_mult,
            out_mult=out_mult,
            z_down=gen_z_down,
            conv_resample=conv_resample,
            num_heads=num_heads,
            activation=activation
        )

        self.D_A = GANDiscriminator(
            in_channels=channels,
            model_channels=model_channels,
            channels_mult=disc_channel_mult,
            z_down=disc_z_down,
        )

        self.D_B = GANDiscriminator(
            in_channels=channels,
            model_channels=model_channels,
            channels_mult=disc_channel_mult,
            z_down=disc_z_down,
        )

        self.G_params = chain(self.G_to_A.parameters(), self.G_to_B.parameters())
        self.D_params = chain(self.D_A.parameters(), self.D_B.parameters())
        self.G_to_A.out_transform = lambda x: x.sigmoid()
        self.G_to_B.out_transform = lambda x: x.sigmoid()
        
        self.cls_weight = 1.0
        self.cyc_weight = 10.0
        self.idt_weight = 0.5

    @torch.no_grad()
    def to_A(self, target):
        self.G_to_A.eval().requires_grad_(False)
        return self.G_to_A(target)

    @torch.no_grad()
    def to_B(self, source):
        self.G_to_B.eval().requires_grad_(False)
        return self.G_to_B(source)

    def forward_gen(self, real_A, real_B):        
        self.G_to_A.train().requires_grad_(True)
        self.G_to_B.train().requires_grad_(True)
        self.D_A.eval().requires_grad_(False)
        self.D_B.eval().requires_grad_(False)

        fake_A = self.G_to_A(real_B)
        fake_A_score = self.D_A(fake_A)
        cycle_B = self.G_to_B(fake_A)
        
        cls_A_loss = F.binary_cross_entropy(fake_A_score, torch.ones_like(fake_A_score))
        cycle_B_loss = F.l1_loss(real_B, cycle_B)
        
        (cls_A_loss * self.cls_weight + cycle_B_loss * self.cyc_weight).backward()
        
        fake_B = self.G_to_B(real_A)
        fake_B_score = self.D_B(fake_B)
        cycle_A = self.G_to_A(fake_B)
        
        cls_B_loss = F.binary_cross_entropy(fake_B_score, torch.ones_like(fake_B_score))
        cycle_A_loss = F.l1_loss(real_A, cycle_A)
        
        (cls_B_loss * self.cls_weight + cycle_A_loss * self.cyc_weight).backward()
        
        idt_A = self.G_to_A(real_A)
        
        idt_A_loss = F.l1_loss(real_A, idt_A)
        
        (idt_A_loss * self.idt_weight).backward()
        
        idt_B = self.G_to_B(real_B)
        
        idt_B_loss = F.l1_loss(real_B, idt_B)
        
        (idt_B_loss * self.idt_weight).backward()
        
        self.real_A = real_A.detach().cpu().numpy()
        self.real_B = real_B.detach().cpu().numpy()
        self.fake_A = fake_A.detach().cpu().numpy()
        self.fake_B = fake_B.detach().cpu().numpy()
        self.cycle_A = cycle_A.detach().cpu().numpy()
        self.cycle_B = cycle_B.detach().cpu().numpy()
        self.idt_A = idt_A.detach().cpu().numpy()
        self.idt_B = idt_B.detach().cpu().numpy()
        
        gan_A_loss = self.cls_weight * cls_A_loss + self.cyc_weight * cycle_B_loss + self.idt_weight * idt_A_loss
        gan_B_loss = self.cls_weight * cls_B_loss + self.cyc_weight * cycle_A_loss + self.idt_weight * idt_B_loss
        
        total_loss = gan_A_loss + gan_B_loss

        return total_loss, {"G_to_A_loss": gan_A_loss.item(),
                            "G_to_A_cls":  cls_A_loss.item(),
                            "G_to_A_cyc":  cycle_A_loss.item(),
                            "G_to_A_idt":  idt_A_loss.item(),
                            "G_to_B_loss": gan_B_loss.item(),
                            "G_to_B_cls":  cls_B_loss.item(),
                            "G_to_B_cyc":  cycle_B_loss.item(),
                            "G_to_B_idt":  idt_B_loss.item()
                            }

    def forward_disc(self, real_A, real_B):
        self.G_to_A.eval().requires_grad_(False)
        self.G_to_B.eval().requires_grad_(False)
        self.D_A.train().requires_grad_(True)
        self.D_B.train().requires_grad_(True)

        fake_B = self.G_to_B(real_A)
        fake_A = self.G_to_A(real_B)

        fake_A_score = self.D_A(fake_A)
        fake_B_score = self.D_B(fake_B)
        real_A_score = self.D_A(real_A)
        real_B_score = self.D_B(real_B)

        fake_A_score = F.binary_cross_entropy(fake_A_score, torch.zeros_like(fake_A_score))
        fake_B_score = F.binary_cross_entropy(fake_B_score, torch.zeros_like(fake_B_score))
        real_A_score = F.binary_cross_entropy(real_A_score, torch.ones_like(real_A_score))
        real_B_score = F.binary_cross_entropy(real_B_score, torch.ones_like(real_B_score))
        
        D_A_loss = (fake_A_score + real_A_score) / 2
        D_B_loss = (fake_B_score + real_B_score) / 2
        
        loss = D_A_loss + D_B_loss
        loss.backward()
        
        return loss, {"D_A_loss": D_A_loss.item(),
                      "D_B_loss": D_B_loss.item()
                      }

    def plot(self, path, title):
        fig = plt.figure()
        font_size = 12
        
        H, W = self.real_A.shape[3], self.real_A.shape[4]
        font_height = (font_size + 2) * fig.dpi / 72
        
        real_A = self.real_A[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        real_B = self.real_B[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        fake_A = self.fake_A[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        fake_B = self.fake_B[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        cycle_A = self.cycle_A[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        cycle_B = self.cycle_B[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        idt_A = self.idt_A[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        idt_B = self.idt_B[0, 0, :9].reshape(3, 3, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 3 * W)
        
        # First row: real_a | fake_b | cycle_a | idt_b
        # Create a grid of images
        images = [
            [real_A, fake_B, cycle_A, idt_B],
            [fake_A, real_B, cycle_B, idt_A]
        ]
        
        titles = [
            ["Real A", "Fake B", "Cycle A", "IDT B"],
            ["Fake A", "Real B", "Cycle B", "IDT A"] 
        ]
        
        # Concatenate images horizontally and vertically
        top_row = np.concatenate(images[0], axis=-1)
        bottom_row = np.concatenate(images[1], axis=-1)

        fig.suptitle(title, fontsize=font_size)
        # Plot the combined grid
        ax1, ax2 = fig.subplots(2, 1)
        im1 = ax1.imshow(top_row)
        ax1.set_axis_off()
        im2 = ax2.imshow(bottom_row)
        ax2.set_axis_off()
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        # Add titles
        for j in range(4):
            ax1.text(j*images[0][0].shape[1] + images[0][0].shape[1]/2, 0, titles[0][j], 
                     horizontalalignment='center', color='black', fontsize=12)
            ax2.text(j*images[1][0].shape[1] + images[1][0].shape[1]/2, 0, titles[1][j],
                     horizontalalignment='center', color='black', fontsize=12)
        
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)