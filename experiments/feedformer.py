# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.models.utils import *
# import attr
import math
from timm.models.layers import DropPath, trunc_normal_

# from IPython import embed


import warnings
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import init


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm._BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """
    Efficient Conv-BN forward during evaluation.
    """
    weight_on_the_fly = conv.weight
    bias_on_the_fly = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_var)

    bn_weight = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
    bn_bias = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_var)

    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape([-1] + [1] * (len(conv.weight.shape) - 1))
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


class ConvModule(nn.Module):
    """
    A streamlined Conv-Norm-Activation block without external dependencies.
    Supports efficient Conv-BN evaluation and various padding modes.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act'),
                 efficient_conv_bn_eval: bool = False):
        super().__init__()
        self.conv_cfg = conv_cfg or {}
        self.norm_cfg = norm_cfg or {}
        self.act_cfg = act_cfg or {}
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        self.efficient_conv_bn_eval_forward = None

        # Validate and parse configurations
        assert len(order) == 3 and set(order) == {'conv', 'norm', 'act'}
        official_padding_modes = ['zeros', 'circular']
        self.with_explicit_padding = padding_mode not in official_padding_modes

        # Handle automatic bias setting
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # Build padding layer if needed
        if self.with_explicit_padding:
            if padding_mode == 'reflect':
                self.padding_layer = nn.ReflectionPad2d(padding)
            elif padding_mode == 'replicate':
                self.padding_layer = nn.ReplicationPad2d(padding)
            else:
                raise NotImplementedError(f"Padding mode {padding_mode} not supported")
            conv_padding = 0
        else:
            self.padding_layer = None
            conv_padding = padding

        # Build convolution layer
        conv_type = self.conv_cfg.get('type', 'Conv2d')
        conv_cls = getattr(nn, conv_type)
        conv_kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': conv_padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            **{k: v for k, v in self.conv_cfg.items() if k != 'type'}
        }
        self.conv = conv_cls(**conv_kwargs)

        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # Build normalization
        self.norm = None
        if self.with_norm:
            norm_type = self.norm_cfg.get('type', 'BatchNorm2d')
            norm_cls = getattr(nn, norm_type)
            
            # Determine input channels for norm
            if order.index('norm') > order.index('conv'):
                num_features = out_channels
            else:
                num_features = in_channels

            norm_kwargs = {
                'num_features': num_features,
                **{k: v for k, v in self.norm_cfg.items() if k != 'type'}
            }
            self.norm = norm_cls(**norm_kwargs)

        # Build activation
        self.activate = None
        if self.with_activation:
            act_type = self.act_cfg.get('type', 'ReLU')
            act_cls = getattr(nn, act_type)
            act_kwargs = {k: v for k, v in self.act_cfg.items() if k != 'type'}
            
            # Handle common activation parameters
            if act_type in ['ReLU', 'LeakyReLU']:
                act_kwargs.setdefault('inplace', inplace)
            if act_type == 'LeakyReLU':
                act_kwargs.setdefault('negative_slope', 0.01)
                
            self.activate = act_cls(**act_kwargs)

        # Initialize weights
        self._init_weights()
        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

    def _init_weights(self):
        # Conv layer initialization
        if hasattr(self.conv, 'init_weights'):
            self.conv.init_weights()
        else:
            if self.with_activation:
                act_type = self.act_cfg.get('type', 'ReLU')
                if act_type == 'LeakyReLU':
                    a = self.act_cfg.get('negative_slope', 0.01)
                    nonlinearity = 'leaky_relu'
                else:
                    a = 0
                    nonlinearity = 'relu'
                init.kaiming_normal_(self.conv.weight, a=a, mode='fan_out', nonlinearity=nonlinearity)
            if self.conv.bias is not None:
                init.constant_(self.conv.bias, 0)

        # Norm layer initialization
        if self.norm is not None:
            init.constant_(self.norm.weight, 1)
            init.constant_(self.norm.bias, 0)

    def turn_on_efficient_conv_bn_eval(self, enable=True):
        if enable and self.norm is not None and isinstance(self.norm, nn.BatchNorm2d) and self.norm.track_running_stats:
            self.efficient_conv_bn_eval_forward = efficient_conv_bn_eval_forward
        else:
            self.efficient_conv_bn_eval_forward = None

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        for layer in self.order:
            if layer == 'conv':
                if self.padding_layer is not None:
                    x = self.padding_layer(x)
                # Handle efficient conv-bn evaluation
                next_layer = self.order[self.order.index(layer)+1] if self.order.index(layer)+1 < len(self.order) else None
                if next_layer == 'norm' and self.efficient_conv_bn_eval_forward is not None and norm and not self.norm.training:
                    self.conv.forward = partial(self.efficient_conv_bn_eval_forward, self.norm, self.conv)
                    x = self.conv(x)
                    self.conv.forward = self.conv._conv_forward  # Restore original forward
                else:
                    x = self.conv(x)
            elif layer == 'norm' and self.norm is not None and norm:
                x = self.norm(x)
            elif layer == 'act' and self.activate is not None and activate:
                x = self.activate(x)
        return x

    @classmethod
    def create_from_conv_bn(cls, conv: nn.modules.conv._ConvNd,
                            bn: nn.modules.batchnorm._BatchNorm,
                            efficient_conv_bn_eval=True) -> 'ConvModule':
        """Create from existing Conv and BN layers."""
        module = cls.__new__(cls)
        super(cls, module).__init__()
        
        # Configure basic parameters
        module.conv = conv
        module.norm = bn
        module.activate = None
        module.order = ('conv', 'norm', 'act')
        
        # Configure parameters
        module.in_channels = conv.in_channels
        module.out_channels = conv.out_channels
        module.kernel_size = conv.kernel_size
        module.stride = conv.stride
        module.padding = conv.padding
        module.dilation = conv.dilation
        module.groups = conv.groups
        
        # Configure norm parameters
        module.with_norm = True
        module.with_activation = False
        module.with_bias = conv.bias is not None
        module.padding_layer = None
        
        # Configure efficient eval
        module.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)
        return module

class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 dw_norm_cfg: Union[Dict, str] = 'default',
                 dw_act_cfg: Union[Dict, str] = 'default',
                 pw_norm_cfg: Union[Dict, str] = 'default',
                 pw_act_cfg: Union[Dict, str] = 'default',
                 **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    """
    CrossAttention with optional local windowing in the query feature map.
    Falls back to global attention if self.window_size is None.
    """
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., pool_ratio=16, window_size=7):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} must be divisible by num_heads {num_heads}."

        self.dim1 = dim1  # dimension for 'x'
        self.dim2 = dim2  # dimension for 'y'
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        # Pool+conv for reducing the y feature map spatially
        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    @staticmethod
    def window_partition(x, window_size):
        """
        x: (B, nh, H, W, C)
        return windows: (B*nh, num_windows, window_size^2, C)
        """
        B, nh, H, W, C = x.shape
        assert H % window_size == 0 and W % window_size == 0, \
            f"H({H}) or W({W}) not divisible by window_size({window_size})"

        h_tiles = H // window_size
        w_tiles = W // window_size

        # Reshape into windows
        x = x.view(B, nh, h_tiles, window_size, w_tiles, window_size, C)
        # Permute to gather each window
        # (B, nh, h_tiles, w_tiles, window_size, window_size, C) => (B, nh, h_tiles*w_tiles, window_size*window_size, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        windows = x.view(B * nh, h_tiles * w_tiles, window_size * window_size, C)
        return windows

    @staticmethod
    def window_reverse(windows, window_size, H, W):
        """
        windows: (B*nh, num_windows, window_size^2, C)
        returns x: (B*nh, H, W, C)
        """
        Bnh, num_windows, _, C = windows.shape
        h_tiles = H // window_size
        w_tiles = W // window_size
        # Reshape back into the tiled form
        windows = windows.view(Bnh, h_tiles, w_tiles, window_size, window_size, C)
        # Now permute back: (Bnh, h_tiles, window_size, w_tiles, window_size, C) => (Bnh, H, W, C)
        x = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bnh, H, W, C)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H1, W1, H2, W2):
        """
        x: [B1, N1, C1], N1=H1*W1
        y: [B2, N2, C2], N2=H2*W2
        For cross-attn, typically B1 == B2. We pool y -> shape ~ (B2, C2, H2//pool, W2//pool).
        """
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        # Project query
        q = self.q(x)  # => [B1, N1, dim1]
        # Reshape to [B1, H1, W1, num_heads, head_dim], then permute => [B1, num_heads, H1, W1, head_dim]
        q = q.reshape(B1, H1, W1, self.num_heads, C1 // self.num_heads).permute(0, 3, 1, 2, 4)

        # Process y -> k,v
        #  1) reshape y => (B2, C2, H2, W2)
        #  2) pool => (B2, C2, Hp, Wp)
        #  3) sr conv => (B2, C2, Hp, Wp)
        #  4) flatten => (B2, Hp*Wp, C2)
        y_2d = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        y_2d = self.pool(y_2d)     # => [B2, C2, Hp, Wp]
        y_2d = self.sr(y_2d)       # => [B2, C2, Hp, Wp]
        Hp, Wp = y_2d.shape[2], y_2d.shape[3]

        y_2d = y_2d.reshape(B2, C2, Hp * Wp).permute(0, 2, 1)  # => [B2, Hp*Wp, C2]
        y_2d = self.norm(y_2d)    # LN
        y_2d = self.act(y_2d)     # GELU

        # Now get k,v => shape (2, B2, num_heads, Hp*Wp, head_dim)
        kv = self.kv(y_2d)        # => [B2, Hp*Wp, 2*dim1]
        kv = kv.reshape(B2, Hp * Wp, 2, self.num_heads, C1 // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]       # => [B2, num_heads, Hp*Wp, head_dim]

        if self.window_size is not None:
            # Local (windowed) attention

            # 1) reshape k,v to [B2, num_heads, Hp, Wp, head_dim]
            k = k.reshape(B2, self.num_heads, Hp, Wp, C1 // self.num_heads)
            v = v.reshape(B2, self.num_heads, Hp, Wp, C1 // self.num_heads)

            # 2) window partition
            q_windows = self.window_partition(q, self.window_size)  # (B1*nh, #wins, win^2, head_dim)
            k_windows = self.window_partition(k, self.window_size)  # (B2*nh, #wins, win^2, head_dim)
            v_windows = self.window_partition(v, self.window_size)

            # Check or assume B1 == B2 for cross-attn in same batch
            # 3) attention
            attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v_windows  # => (B1*nh, #wins, win^2, head_dim)

            # 4) window reverse for query shape
            out = self.window_reverse(out, self.window_size, H1, W1)  # => [B1*nh, H1, W1, head_dim]
            out = out.reshape(B1, self.num_heads, H1 * W1, C1 // self.num_heads)
            out = out.transpose(1, 2).reshape(B1, H1 * W1, C1)

        else:
            # Fallback to global attention:
            # q,k,v are [B, heads, H?, W?, head_dim]. Flatten (H?, W?) -> single token dimension.
            # For cross-attn, H?W? can differ for q vs k. We'll just do standard cross attention with
            # q => [B, heads, N_q, head_dim], k => [B, heads, N_k, head_dim], etc.

            # Flatten (H1,W1) in q, and (Hp,Wp) in k,v:
            q_2d = q.flatten(start_dim=2, end_dim=3)  # => [B1, heads, H1*W1, head_dim]
            k_2d = k.reshape(B2, self.num_heads, Hp, Wp, C1 // self.num_heads)
            v_2d = v.reshape(B2, self.num_heads, Hp, Wp, C1 // self.num_heads)

            k_2d = k_2d.flatten(start_dim=2, end_dim=3)  # => [B2, heads, Hp*Wp, head_dim]
            v_2d = v_2d.flatten(start_dim=2, end_dim=3)  # => [B2, heads, Hp*Wp, head_dim]

            # Usually we assume B1 == B2, so let's name them B for clarity
            # [B, heads, H1*W1, head_dim] x [B, heads, head_dim, Hp*Wp]
            attn = (q_2d @ k_2d.transpose(-2, -1)) * self.scale  # => [B, heads, H1*W1, Hp*Wp]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v_2d  # => [B, heads, H1*W1, head_dim]

            # Finally, reshape to [B, H1*W1, C1]
            out = out.permute(0, 2, 1, 3).reshape(B1, H1 * W1, C1)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    """
    One cross-attention + MLP block.
    """
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, pool_ratio=16, window_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1,
                                   dim2=dim2,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   pool_ratio=pool_ratio,
                                   window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        # x: [B, H1*W1, dim1], y: [B, H2*W2, dim2]
        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                self.norm2(y),
                H1, W1,    # shape of x
                H2, W2     # shape of y
            )
        )
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))
        return x


class FeedFormerHead(BaseDecodeHead):
    """
    Example segmentation head that uses the above CrossAttention blocks to fuse features.
    """
    def __init__(self, feature_strides, **kwargs):
        super(FeedFormerHead, self).__init__(channels=48,
                                             input_transform='multiple_select',
                                             **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        # Cross-attn blocks
        self.attn_c4_c1 = Block(dim1=c4_in_channels,
                                dim2=c1_in_channels,
                                num_heads=8, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3_c1 = Block(dim1=c3_in_channels,
                                dim2=c1_in_channels,
                                num_heads=4, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=4)
        # Windowed cross-attn for c2->c1
        self.attn_c2_c1 = Block(dim1=c2_in_channels,
                                dim2=c1_in_channels,
                                num_heads=2, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=2,
                                window_size=8)
        

        self.linear_fuse = ConvModule(
            in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BatchNorm2d')
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs: list of 4 feature maps: c1, c2, c3, c4
        x = self._transform_inputs(inputs)  # c1->1/4, c2->1/8, c3->1/16, c4->1/32
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape   # c4 is [B, c4_in_channels, h4, w4]
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # Flatten each to [B, N, C]
        c1f = c1.flatten(2).transpose(1, 2)  # => [B, H1*W1, C1]
        c2f = c2.flatten(2).transpose(1, 2)  # => [B, H2*W2, C2]
        c3f = c3.flatten(2).transpose(1, 2)
        c4f = c4.flatten(2).transpose(1, 2)

        # Cross attention c4->c1, output shape => [B, H4*W4, c4_in_channels]
        _c4 = self.attn_c4_c1(c4f, c1f, h1, w1, h4, w4)
        # revert shape => [B, c4_in_channels, h4, w4]
        _c4 = _c4.permute(0, 2, 1).reshape(n, -1, h4, w4)
        _c4 = resize(_c4, size=(h1, w1), mode='bilinear', align_corners=False)

        _c3 = self.attn_c3_c1(c3f, c1f, h1, w1, h3, w3)
        _c3 = _c3.permute(0, 2, 1).reshape(n, -1, h3, w3)
        _c3 = resize(_c3, size=(h1, w1), mode='bilinear', align_corners=False)

        _c2 = self.attn_c2_c1(c2f, c1f, h1, w1, h2, w2)
        _c2 = _c2.permute(0, 2, 1).reshape(n, -1, h2, w2)
        _c2 = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=False)

        # c1 is not cross-attended in these blocks, so just reshape
        _c1 = c1f.permute(0, 2, 1).reshape(n, -1, h1, w1)

        # Fuse all four
        fused = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(fused)  # => [B, embedding_dim, H1, W1]

        x = self.dropout(_c)
        x = self.linear_pred(x)       # => [B, num_classes, H1, W1]
        return x