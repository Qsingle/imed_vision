# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:task_fusion
    author: 12718
    time: 2022/5/9 9:52
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layernorm import LayerNorm
from .cross_attention import CrossAttentionConv

class RCAB(nn.Module):
    def __init__(self, ch, reduction=4):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.se = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch // reduction, ch, 1, 1),
            nn.Sigmoid()
        )
        #self.conv3 = nn.Conv2d(ch, ch, 1, 1)

    def forward(self, x):
        identity = x
        net = self.conv1(x)
        # net = F.leaky_relu(net)
        net = self.conv2(net)
        net = self.se(net)*net
        #net = self.conv3(net)
        net = net + identity
        return net

class RCAB3D(nn.Module):
    def __init__(self, ch, reduction=4):
        super(RCAB3D, self).__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, 1, 1, groups=ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, 1, 1, groups=ch)
        self.se = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv3d(ch, ch // reduction, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(ch // reduction, ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        net = self.conv1(x)
        # net = F.leaky_relu(net)
        net = self.conv2(net)
        net = self.se(net) * net
        # net = self.conv3(net)
        net = net + identity
        return net

class ImagePool(nn.Module):
    def __init__(self, in_ch):
        super(ImagePool, self).__init__()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=False)
        return net


class ImagePool3D(nn.Module):
    def __init__(self, in_ch):
        super(ImagePool3D, self).__init__()
        self.gpool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = F.interpolate(net, size=x.size()[2:], mode="trilinear", align_corners=False)
        return net
    
class MSConv3d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv3d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv3d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups - 1):
            self.convs.append(
                nn.Conv3d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        self.convs.append(ImagePool3D(group_ch))
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm3d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net

class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups-1):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        self.convs.append(ImagePool(group_ch))
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net

class GlobalSpatialAttention(nn.Module):
    def __init__(self, dim, down_rate=16):
        super(GlobalSpatialAttention, self).__init__()
        self.norm = LayerNorm(dim)
        self.down = nn.Conv2d(dim, dim, kernel_size=down_rate, stride=down_rate)
        self.qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.scale = dim**-0.5
        self.deconv = nn.ConvTranspose2d(dim, dim, down_rate, down_rate)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        net = self.down(x)
        net = self.norm(net)
        bs, ch, h, w = net.size()
        qkv = self.qkv(net)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(bs, ch, h*w)
        k = k.reshape(bs, ch, h*w)
        v = v.transpose(bs, ch, h*w)
        atten = q.transpose(1, 2) @ k
        atten = torch.softmax(atten*self.scale, dim=-1)
        out = v @ atten
        out = out.reshape(bs, ch, h, w)
        #out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        out = self.deconv(out)
        out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        out = self.proj(out) + x
        return out

class Gate(nn.Module):
    def __init__(self, in_ch, reduction=4, down_rate=32):
        super(Gate, self).__init__()
        self.rcab = RCAB(in_ch, reduction)
        self.msconv = MSConv2d(in_ch)
        # self.msconv = ASPP(in_ch, in_ch, rates=[1, 12, 24, 36])
        #self.global_spatial = GlobalSpatialAttention(in_ch, down_rate=down_rate)

    def forward(self, x):
        net = self.rcab(x)
        net = self.msconv(net)
        #net = self.global_spatial(net)
        net = net + x #long range
        return net
class Gate3D(nn.Module):
    def __init__(self, in_ch, reduction=4, down_rate=32):
        super(Gate3D, self).__init__()
        self.rcab = RCAB3D(in_ch, reduction)
        self.msconv = MSConv3d(in_ch)

    def forward(self, x):
        net = self.rcab(x)
        net = self.msconv(net)
        # net = self.global_spatial(net)
        net = net + x  # long range
        return net

class CrossGL(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32, reduction=4, down_rate=32):
        super(CrossGL, self).__init__()
        #fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, hidden_state, 1, 1),
            nn.BatchNorm2d(hidden_state),
            nn.GELU(),
            Gate(hidden_state, reduction=reduction, down_rate=down_rate)
        )
        #cross
        self.linear_cross_sr = nn.Sequential(
            nn.Conv2d(hidden_state, sr_ch, 1, 1),
            nn.GELU()
        )
        self.linear_cross_seg = nn.Sequential(
            nn.Conv2d(hidden_state, seg_ch, 1, 1),
            nn.GELU()
        )
        self.gate_sr = nn.Conv2d(sr_ch, sr_ch, 1, 1)
        self.gate_seg = nn.Conv2d(seg_ch, seg_ch, 1, 1)

    def forward(self, sr_fe, seg_fe):
        fusion = torch.cat([sr_fe, seg_fe], dim=1)
        fusion = self.fusion_conv(fusion)
        cross_sr = self.gate_sr(self.linear_cross_sr(fusion)*sr_fe) + sr_fe
        cross_seg = self.gate_seg(self.linear_cross_seg(fusion)*seg_fe) + seg_fe
        return cross_sr, cross_seg

class CrossGL3D(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32, reduction=4, down_rate=32):
        super(CrossGL3D, self).__init__()
        #fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(sr_ch+seg_ch, hidden_state, 1, 1),
            nn.BatchNorm3d(hidden_state),
            nn.GELU(),
            Gate3D(hidden_state, reduction=reduction, down_rate=down_rate)
        )
        #cross
        self.linear_cross_sr = nn.Sequential(
            nn.Conv3d(hidden_state, sr_ch, 1, 1),
            nn.GELU()
        )
        self.linear_cross_seg = nn.Sequential(
            nn.Conv3d(hidden_state, seg_ch, 1, 1),
            nn.GELU()
        )
        self.gate_sr = nn.Conv3d(sr_ch, sr_ch, 1, 1)
        self.gate_seg = nn.Conv3d(seg_ch, seg_ch, 1, 1)

    def forward(self, sr_fe, seg_fe):
        fusion = torch.cat([sr_fe, seg_fe], dim=1)
        fusion = self.fusion_conv(fusion)
        cross_sr = self.gate_sr(self.linear_cross_sr(fusion)*sr_fe) + sr_fe
        cross_seg = self.gate_seg(self.linear_cross_seg(fusion)*seg_fe) + seg_fe
        return cross_sr, cross_seg

class CrossTaskAttention(nn.Module):
    def __init__(self, x_ch, y_ch, dim=32, num_head=8, qkv_bias=False, patch_size=32):
        super(CrossTaskAttention, self).__init__()
        # self.cross_att = CrossAttention(x_ch, y_ch, dim=dim, num_head=num_head, qkv_bias=qkv_bias)
        self.cross_attention = CrossAttentionConv(x_ch, y_ch, dim)
        # self.x_patch_conv = nn.Conv2d(x_ch, x_ch, patch_size, patch_size, groups=1)
        # self.y_patch_conv = nn.Conv2d(y_ch, y_ch, patch_size, patch_size, groups=1)
        # self.x_unpatch = nn.ConvTranspose2d(x_ch, x_ch, patch_size, patch_size)
        # self.y_unpatch = nn.ConvTranspose2d(y_ch, y_ch, patch_size, patch_size)

    def forward(self, x, y):
        # x_patch = self.x_patch_conv(x)
        # y_patch = self.y_patch_conv(y)
        # x_patch, y_patch = self.cross_att([x_patch, y_patch])
        # x_patch = F.interpolate(self.x_unpatch(x_patch), size=x.size()[2:], mode="bilinear", align_corners=True)
        # y_patch = F.interpolate(self.y_unpatch(y_patch), size=y.size()[2:], mode="bilinear", align_corners=True)
        x, y = self.cross_attention([x, y])
        return x, y

class DownAttention4D(nn.Module):
    def __init__(self, dim, down_rate=32, num_head=4, sclae_init=1e-6) -> None:
        super().__init__()
        self.down_conv = nn.Conv2d(dim, dim, kernel_size=down_rate, stride=down_rate, groups=dim)
        self.head_dim = dim // num_head
        self.num_head = num_head
        self.gamma = self.head_dim ** -0.5
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.layer_scaler = nn.Parameter(torch.ones(dim)*sclae_init, requires_grad=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1, 1),
            nn.GELU()
        )
        self.ffn_scaler = nn.Parameter(torch.ones(dim)*sclae_init, requires_grad=True)
    
    def forward(self, x):
        down_x = self.down_conv(x)
        bs, ch, h, w = down_x.size()
        q_x = self.q_proj(down_x).reshape(bs, self.num_head, self.head_dim, h*w).transpose(2, 3) #bs, num_head, N, head_num
        k_x = self.k_proj(down_x).reshape(bs, self.num_head, self.head_dim, h*w) #bs, num_head, head_num, N
        v_x = self.v_proj(down_x).reshape(bs, self.num_head, self.head_dim, h*w).transpose(2, 3) #bs, num_head, N, head_num
        att = q_x @ k_x
        att = torch.softmax(att*self.gamma, dim=-1)
        net = att @ v_x
        net = net.transpose(2, 3).reshape(bs, ch, h, w)
        net = self.layer_scaler.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * net + down_x
        net = self.ffn(net)*self.ffn_scaler.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + net
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=True)
        return net

class CAGL(nn.Module):
    def __init__(self, in_ch1, in_ch2, hidden_state=32, reduction=4, down_rate=32, num_head=4) -> None:
        super(CAGL, self).__init__()
        fu_ch = in_ch1 + in_ch2
        self.fu_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fu_ch, fu_ch//reduction, 1, 1),
            nn.ReLU(),
            nn.Conv2d(fu_ch//reduction, fu_ch, 1, 1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(fu_ch, hidden_state, 1, 1)
        self.global_att = DownAttention4D(hidden_state, down_rate=down_rate, num_head=num_head)
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, 3, 1, 1, groups=hidden_state),
            nn.GELU(),
            nn.Conv2d(hidden_state, hidden_state, 1, 1),
            nn.GELU()
        )
        self.out_x = nn.Sequential(
            nn.Conv2d(hidden_state, in_ch1, 1, 1),
            nn.Softmax(dim=1)
        )
        self.out_y = nn.Sequential(
            nn.Conv2d(hidden_state, in_ch2, 1, 1),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x, y):
        fuse = torch.cat([x, y], dim=1)
        fuse = self.fu_se(fuse)*fuse
        fuse = self.fuse(fuse)
        fuse_g = self.global_att(fuse)
        fuse_l = self.local_conv(fuse)
        fuse = fuse + fuse_g + fuse_l
        o_x = self.out_x(fuse)*x + x
        o_y = self.out_y(fuse)*y + y
        return o_x, o_y