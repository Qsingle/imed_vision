# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:mha
    author: 12718
    time: 2022/8/14 15:30
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .mbconv import MBConv
from .dropout import DropPath
from imed_vision.comm.helper import _pair


__all__ = ["MHA", "TransformerBlock", "MaxSA", "window_partion", "window_reverse"]

def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

class RelPosBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super(RelPosBias, self).__init__()
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self.attn_area: int = window_size[0] * window_size[1]
        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size[0],
                                                                                    window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        net = self._get_relative_positional_bias() + x
        return net

class MLP(nn.Module):
    def __init__(self, dim, expansion_rate=4., act_layer=nn.GELU(), drop_rate=0.):
        super(MLP, self).__init__()
        drop_rate = _pair(drop_rate)
        expansion_dim = int(dim*expansion_rate)
        self.fc1 = nn.Linear(dim, expansion_dim)
        self.drop1 = nn.Dropout(drop_rate[0])
        self.act = act_layer
        self.fc2 = nn.Linear(expansion_dim, dim)
        self.drop2 = nn.Dropout(drop_rate[1])

    def forward(self, x):
        net = self.fc1(x)
        net = self.drop1(net)
        net = self.fc2(net)
        net = self.drop2(net)
        return net

class MHA(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False,
                 att_drop=0., proj_drop=0., rel_position=None):
        """
        Implementation of multi head attention
        "Attention is all you need"
        <https://arxiv.org/abs/1706.03762>
        Args:
            dim (int): number of dimension
            num_head (int): number of head
            qkv_bias (bool): whether use the bias
            att_drop (float): dropout rate of the attention linear layer
            proj_drop (float): dropout rate of the linear layer
        """
        super(MHA, self).__init__()
        assert dim % num_head == 0, "Number of dimension {} must divided by number of head {}".\
            format(dim, num_head)
        hidden_dim = dim // num_head
        self.num_head = num_head
        self.scale = hidden_dim**0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rel_postion = rel_position
        self.att_drop = nn.Dropout(att_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, D // self.num_head).permute(2, 0, 3, 1, 4) #3, B, num_head, N, D // num_head
        q, k, v = qkv.unbind(0) #B, num_head, N, D // num_head
        atten = (q @ k.transpose(-2, -1)) * self.scale #B, num_head, N, N
        if self.rel_postion is not None:
            atten = self.rel_postion(atten)
        atten = torch.softmax(atten, dim=-1)
        atten = self.att_drop(atten)
        out = (atten @ v).transpose(1, 2) #B, N, num_head, D // num_head
        out = out.reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, att_drop=0., proj_drop=0., ffn_expansion_rate=4.,
                 drop_path_rate=0., rel_position=None):
        super(TransformerBlock, self).__init__()
        self.attention = MHA(dim, num_head=num_head, qkv_bias=qkv_bias, att_drop=att_drop, proj_drop=proj_drop,
                             rel_position=rel_position)
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path_att = DropPath(dropout_rate=drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.ffn = MLP(dim, expansion_rate=ffn_expansion_rate)
        self.drop_path_ffn = DropPath(dropout_rate=drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        net = self.attention(x)
        net = self.drop_path_att(net) + x
        net = self.norm1(net)
        shortcut = net
        net = self.ffn(net)
        net = self.drop_path_ffn(net) + shortcut
        return net

def window_partion(x:torch.Tensor, window_size):
    bs, ch, h, w = x.size()
    window_h, window_w = window_size
    if w % window_w != 0 or h % window_h != 0:
        raise ValueError("The feature size {} must divided by window size {}".format((h, w), window_size))
    x = x.reshape(bs, ch, h // window_h, window_h, w // window_w, window_w).permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(-1, window_w, window_h, ch)
    return x

def unbind(x:torch.Tensor, window_size, img_size):
    ch = x.size()[-1]
    h, w = img_size
    x = x.reshape(-1, h // window_size[0], w // window_size[1], window_size[0], window_size[1], ch).permute(0, 5, 1, 3, 4, 2)
    x = x.reshape(-1, h, w, ch)
    return x

window_reverse = unbind

class MaxSA(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size=(7, 7), grid_size=(7, 7), num_head=2, reduction=0.25,
                 ksize=3, expansion=4., stride=1, att_drop=0., proj_drop=0., qkv_bias=False,
                 droppath_rate=0., ffn_expansion_rate=4, img_size=(224, 224)):
        super(MaxSA, self).__init__()
        self.mbconv = MBConv(in_ch, out_ch, ksize=ksize, stride=stride, has_skip=True, exp_ratio=expansion,
                             drop_path_rate=droppath_rate, se_layer=True, act_layer=nn.ReLU(),
                             reduction=reduction)
        block_rel_pos = RelPosBias(window_size=patch_size, num_heads=num_head)
        self.block_attention = TransformerBlock(out_ch, num_head, qkv_bias=qkv_bias, att_drop=att_drop, proj_drop=proj_drop,
                                                ffn_expansion_rate=ffn_expansion_rate, drop_path_rate=droppath_rate,
                                                rel_position=block_rel_pos)
        grid_patch = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
        grid_rel_pos = RelPosBias(window_size=grid_patch, num_heads=num_head)
        self.grid_attention = TransformerBlock(out_ch, num_head, qkv_bias=qkv_bias, att_drop=att_drop, proj_drop=proj_drop,
                                                ffn_expansion_rate=ffn_expansion_rate, drop_path_rate=droppath_rate,
                                               rel_position=grid_rel_pos)
        self.patch_size = patch_size
        self.grid_size = grid_size

    def forward(self, x):
        net = self.mbconv(x)
        img_size = net.shape[2:]
        patch_size = self.patch_size
        net = window_partion(net, window_size=patch_size)
        _bs, h, w, ch = net.shape
        block_x = net.reshape(_bs, -1, ch)
        net = self.block_attention(block_x).reshape(_bs, h, w, ch)
        net = unbind(net, patch_size, img_size).permute(0, 3, 1, 2)
        grid_size = self.grid_size
        patch_size = [img_size[0] // grid_size[0], img_size[1] // grid_size[1]]
        net = window_partion(net, patch_size)
        _bs, h, w, ch = net.shape
        grid_x = net.reshape(_bs, -1, ch)
        net = self.grid_attention(grid_x).reshape(_bs, h, w, ch)
        net = unbind(net, patch_size, img_size).permute(0, 3, 1, 2)
        return net

class Attention4D(nn.Module):
    def __init__(self, num_head, dim, dim_k, att_ratio=4.,qkv_bias=True, proj_bias=True,
                 rel_position=None, act_layer=None, downsample=False):
        """
        The efficient Multi-Head Attention in
        "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>
        Args:
            num_head (int): Number of heads
            dim (int): dim of the input
            dim_k (int): dim of each head
            att_ratio (float): expansion rate for the value tensor. (I don't know whether the official code add this)
            qkv_bias (bool): option to switch the bias in qkv linear
            proj_bias (bool): option to switch the bias in proj linear
            rel_position (nn.Module): the relation position module
            act_layer: nonlinear activation function
            downsample(bool): whether downsample the key and value
        """
        super(Attention4D, self).__init__()
        self.dim_k = dim_k
        self.num_head = num_head
        self.scale = dim_k ** -0.5
        act_layer = nn.ReLU if act_layer is None else act_layer
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim_k*num_head, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(dim_k*num_head)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim_k * num_head, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(dim_k * num_head)
        )
        self.d = int(dim_k*att_ratio)
        self.dh = self.num_head * self.d
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(self.dh)
        )
        self.rel_pos = rel_position if rel_position else nn.Identity()
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, 3, 1, 1, groups=self.dh),
            nn.BatchNorm2d(self.dh)
        )

        self.talking_head1 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.talking_head2 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1, bias=proj_bias),
            nn.BatchNorm2d(dim)
        )
        self.downsample = nn.Conv2d(dim, dim, 3, 2, 1) if downsample else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x).reshape(B, self.num_head, self.dim_k, -1).transpose(2, 3) #B, Num_Head, N, head_dim
        x = self.downsample(x)
        k = self.k(x).reshape(B, self.num_head, self.dim_k, -1) #B, Num_Head, head_dim, N
        v = self.v(x)
        v = self.v_local(v)
        v = v.reshape(B, self.num_head, self.d, -1).transpose(2, 3) #B, Num_Head, N, head_dim
        att = q @ k #B, Num_Head, N, N
        att = att * self.scale
        att = self.rel_pos(att)
        att = self.talking_head1(att)
        att = torch.softmax(att, dim=-1)
        att = self.talking_head2(att)
        net = att @ v #B, Num_Head, N, Head_Dim
        net = net.transpose(2, 3).reshape(B, -1, H, W)
        net = self.proj(net)
        return net

class QueryDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Used to downsample the Query as mentioned in Section 3.5
        "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>
        Args:
            in_ch:
            out_ch:
        """
        super(QueryDown, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch)
        self.gpool = nn.AvgPool2d(3, 2, 1)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.gpool(x)
        net = conv + pool
        net = self.proj(net)
        return net
class AttentionDownsample(nn.Module):
    def __init__(self, num_head, dim, dim_k, att_ratio=4.,qkv_bias=True, proj_bias=True,
                 rel_position=None, act_layer=None):
        """
        The attention based downsample in
         "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>
        Args:
            num_head (int): Number of heads
            dim (int): dim of the input
            dim_k (int): dim of each head
            att_ratio (float): expansion rate for the value tensor. (I don't know whether the official code add this)
            qkv_bias (bool): option to switch the bias in qkv linear
            proj_bias (bool): option to switch the bias in proj linear
            rel_position (nn.Module): the relation position module
            act_layer: nonlinear activation function
        """
        super(AttentionDownsample, self).__init__()
        self.dim_k = dim_k
        self.num_head = num_head
        self.scale = dim_k ** -0.5
        act_layer = nn.ReLU if act_layer is None else act_layer
        self.q = QueryDown(dim, num_head*dim_k)
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim_k * num_head, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(dim_k * num_head)
        )
        self.d = int(dim_k * att_ratio)
        self.dh = self.num_head * self.d
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(self.dh)
        )
        self.rel_pos = rel_position if rel_position else nn.Identity()
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, 3, 1, 1, groups=self.dh),
            nn.BatchNorm2d(self.dh)
        )

        self.talking_head1 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.talking_head2 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1, bias=proj_bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        H = H // 2
        W = W // 2
        q = self.q(x).reshape(B, self.num_head, self.dim_k, -1).transpose(2, 3)  # B, Num_Head, N, head_dim
        k = self.k(x).reshape(B, self.num_head, self.dim_k, -1)  # B, Num_Head, head_dim, N
        v = self.v(x)
        v = self.v_local(v)
        v = v.reshape(B, self.num_head, self.d, -1).transpose(2, 3)  # B, Num_Head, N, head_dim
        att = q @ k  # B, Num_Head, N, N
        att = att * self.scale
        att = self.rel_pos(att)
        att = self.talking_head1(att)
        att = torch.softmax(att, dim=-1)
        att = self.talking_head2(att)
        net = att @ v  # B, Num_Head, N, Head_Dim
        net = net.transpose(2, 3).reshape(B, -1, H, W)
        net = self.proj(net)
        return net

class UFFN(nn.Module):
    def __init__(self, dim, out_dim, expansion_rate=4.,  act_layer=nn.GELU, use_mid_conv=True):
        """
        Unified FFN introduced at Section 3.1 in
         "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>

        Args:
            dim (int):
            out_dim (int): dimension for output
            expansion_rate (float):  expansion rate
            act_layer (nn.Module): nonlinear function for activation
            use_mid_conv (bool): option to switch the middle depthwise convolution
        """
        super(UFFN, self).__init__()
        if act_layer is None:
            act_layer = nn.GELU
        hidden_dim = int(dim*expansion_rate)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        ) if use_mid_conv else nn.Identity()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )
    def forward(self, x):
        net = self.fc1(x)
        net = self.mid_conv(net)
        net = self.fc2(net)
        return net

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32).cuda()
    # model = MaxSA(3, 64, patch_size=(7, 7), grid_size=(7, 7), num_head=2, ksize=3, stride=2).cuda()
    model = AttentionDownsample(8, 64, 32).cuda()
    print(model(x).shape)

