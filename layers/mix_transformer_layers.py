# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:mix_transformer_layers
    author: 12718
    time: 2022/4/29 9:45
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from layers.dropout import DropPath
from comm.helper import _pair

__all__ = ["OverlapPatchEmbed", "Block"]

class DWConv(nn.Module):
    def __init__(self, dim=768):
        """
        Implementation of the depth-wise conv in SegFormer.
        "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        <https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf>
        Args:
            dim (int): number of dimensions of the input
        """
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        bs, N, ch = x.shape
        x = x.transpose(1, 2).view(bs, ch, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        """
        Implementation of MixFFN in SegFormer.
        "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        <https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf>
        Args:
            in_dim (int): number of dimensions for input tensor, default:768
            hidden_dim (int): number of dimensions for hidden state, default:in_dim
            out_dim (int): number of dimensions for output, default:in_dim
            act_layer (nn.Module): activation function layer
            drop (float): dropout rate
        """
        super(MixFFN, self).__init__()
        hidden_dim = in_dim or hidden_dim
        out_dim = in_dim or out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
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
        net = self.fc1(x)
        net = self.dwconv(net, H, W)
        net = self.act(net)
        net = self.drop(net)
        net = self.fc2(net)
        net = self.drop(net)
        return net

MLP = MixFFN

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0., proj_drop=0., reduce_ratio=1):
        """
        Implementation of the efficient attention in SegFormer.
        "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        <https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf>
        Args:
            dim (int): dimension of input
            num_heads (int): number of heads
            qk_scale (float): scale value of kv pair
            qkv_bias (bool): whether use bias at q,k,v calculate
            attn_drop (float): dropout prob of attention
            proj_drop (float): dropout probability of linear project
            reduce_ratio (int): size reduce rate
        """
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divided by the num_heads {num_heads}"
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = reduce_ratio
        self.att_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if reduce_ratio > 1:
            # I think it is different from the original paper.
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduce_ratio, stride=reduce_ratio)
            self.norm = nn.LayerNorm(dim)
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
        bs, n, ch = x.shape
        q = self.q(x).view(bs, n, self.num_heads, ch // self.num_heads).permute(0, 2, 1, 3) #[bs, num_heads, n, head_dim]

        if self.sr_ratio > 1:
            _x  = x.transpose(1, 2).view(bs, ch, H, W)
            _x = self.sr(_x).view(bs, ch, -1).transpose(1, 2)
            _x = self.norm(_x)
            kv = self.kv(_x)
            kv = kv.view(bs, -1, 2, self.num_heads, ch // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).view(bs, -1, 2, self.num_heads, ch//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] #[bs, num_heads, n', head_dim]
        att = (q @ k.transpose(-2, -1)) * self.scale #[bs, num_heads, n, n']
        att = torch.softmax(att, dim=-1)
        att = self.att_drop(att)

        net = (att@v).transpose(1, 2).reshape(bs, n, ch) #[bs, n, dim]
        net = self.proj(net)
        net = self.proj_drop(net)
        return net

class Block(nn.Module):
    def __init__(self, dim, num_heads=8,  qk_scale=None, qkv_bias=False, act_layer=nn.GELU,
                 att_drop=0., drop=0., reduce_ratio=1, norm_layer=nn.LayerNorm, mlp_ratio=4.,
                 drop_path=0.):
        """
        Implementation of MixTransformer Block in SegFormer.
        "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        <https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf>
        Args:
            dim (int): number of dimension for input
            num_heads (int): number of heads for attention
            qk_scale (int): scale rate of the query-key pair
            qkv_bias (bool): whether use bias in query, key, value map projecter
            act_layer (nn.Module): activation module
            att_drop (float): dropout rate for the attention
            drop (float): dropout rate
            reduce_ratio (int): rate of the size reduce
            norm_layer (nn.Module): normalization layer
            mlp_ratio (float): expansion ratio for mlp layer hidden dimension
            drop_path (float): dropout path rate
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qk_scale=qk_scale, qkv_bias=qkv_bias,
                                   attn_drop=att_drop, proj_drop=drop, reduce_ratio=reduce_ratio)
        hidden_dim = int(mlp_ratio*dim)
        self.mlp = MLP(dim, hidden_dim=hidden_dim, drop=drop, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
        net = self.drop_path(self.attn(self.norm1(x), H, W)) + x
        net = self.drop_path(self.mlp(self.norm2(net), H, W)) + net
        return net


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7,  stride=4, in_chans=3, embed_dim=76):
        """
        Implementation of the Overlap Patch Merging in SegFormer.
        "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        <https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf>
        Args:
            img_size (Union[int, tuple]): size of image
            patch_size (Union[int, tuple]): size of patch
            stride (Union[int,tuple]): strides
            in_chans (int): number of channels for input
            embed_dim (int): dimension of embedding tensor
        """
        super(OverlapPatchEmbed, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patchs = self.H * self.W
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, x):
        net = self.proj(x)
        H, W = net.shape[2:]
        net = self.norm(net.flatten(2).transpose(1, 2))
        return net, H, W