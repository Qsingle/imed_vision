# -*- coding:utf-8 -*-
"""
    FileName: cross_attention
    Author: 12718
    Create Time: 2023-04-07 09:59
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, x_ch, y_ch, dim=32, num_head=8, qkv_bias=False):
        """
        Implementation of the cross attention for two features
        Args:
            x_ch(int): number of channels for input x
            y_ch(int): number of channels for input y
            dim(int): number of dimensions for the hidden state
            num_head(int): number of heads
        """
        super(CrossAttention, self).__init__()
        self.head_dim = dim // num_head
        self.num_head = num_head
        self.gamma = self.head_dim ** -0.5
        #branch x
        self.x_conv = nn.Sequential(
            nn.Conv2d(x_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.q_x = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        self.k_x = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        self.v_x = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        #branch y
        self.y_conv = nn.Sequential(
            nn.Conv2d(y_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.q_y = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        self.k_y = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        self.v_y = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias),
            nn.BatchNorm2d(dim)
        )
        #after
        self.x_after_conv = nn.Sequential(
            nn.Conv2d(dim, x_ch, 1, 1, bias=False),
            nn.BatchNorm2d(x_ch)
        )
        self.y_after_conv = nn.Sequential(
            nn.Conv2d(dim, y_ch, 1, 1, bias=False),
            nn.BatchNorm2d(y_ch)
        )

    def forward(self, fes):
        x, y = fes
        bs, x_ch, x_h, x_w = x.size()
        _, y_ch, y_h, y_w = y.size()
        _x = self.x_conv(x)
        _y = self.y_conv(y)

        #qkv for x
        q_x = self.q_x(_x).reshape(bs, self.num_head, self.head_dim, x_h*x_w).transpose(-2, -1)
        k_x = self.k_x(_x).reshape(bs, self.num_head, self.head_dim, x_h*x_w)
        v_x = self.v_x(_x).reshape(bs, self.num_head, self.head_dim, x_h*x_w).transpose(-2, -1)

        #qkv for y
        q_y = self.q_y(_y).reshape(bs, self.num_head, self.head_dim, y_h * y_w).transpose(-2, -1)
        k_y = self.k_x(_y).reshape(bs, self.num_head, self.head_dim, y_h*y_w)
        v_y = self.v_x(_y).reshape(bs, self.num_head, self.head_dim, y_h*y_w).transpose(-2, -1)

        #cross attention forward phase
        cross_att_x = q_x @ k_y #bs, num_head, h_x*w_x, h_y*w*y
        cross_att_x = cross_att_x * self.gamma
        cross_att_x = torch.softmax(cross_att_x, dim=-1)
        cross_att_x = cross_att_x @ v_y #bs, num_head, h_x*w_x, head_dim

        cross_att_y = q_y @ k_x
        cross_att_y = cross_att_y * self.gamma
        cross_att_y = torch.softmax(cross_att_y, dim=-1)
        cross_att_y = cross_att_y @ v_x
        cross_att_x = cross_att_x.transpose(-2, -1).reshape(bs, -1, x_h, x_w)
        cross_att_y = cross_att_y.transpose(-2, -1).reshape(bs, -1, y_h, y_w)

        #after cross

        cross_att_x = self.x_after_conv(cross_att_x) + x
        cross_att_y = self.y_after_conv(cross_att_y) + y
        return cross_att_x, cross_att_y


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

class SplitSpatialConv(nn.Module):
    def __init__(self, ch, cards):
        super(SplitSpatialConv, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(cards):
            self.convs.append(
                nn.Conv2d(ch, ch, 3, 1, padding=i+1, dilation=i+1, groups=ch, bias=False)
            )
        # self.convs.append(
        #     ImagePool(ch)
        # )
        self.fusion = nn.Conv2d(ch*cards, ch, 1, 1, 0)
    def forward(self, x):
        nets = []
        for conv in self.convs:
            nets.append(conv(x))
        return self.fusion(torch.cat(nets, dim=1))

class CrossAttentionConv(nn.Module):
    def __init__(self, x_ch, y_ch, dim=64):
        super(CrossAttentionConv, self).__init__()
        self.x_map_conv = nn.Sequential(
            nn.Conv2d(x_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.y_map_conv = nn.Sequential(
            nn.Conv2d(y_ch, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        #spatial
        self.x_spatial = nn.Sequential(
            SplitSpatialConv(2, cards=4),
            nn.Conv2d(2, 1, 1, 1, bias=False),
            # nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )

        self.y_spatial = nn.Sequential(
            SplitSpatialConv(2, cards=4),
            nn.Conv2d(2, 1, 1, 1, bias=False),
            # nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )
        #channel
        self.x_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, 1),
            nn.Conv2d(dim//4, dim, 1, 1),
            nn.Sigmoid()
        )

        self.y_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, 1),
            nn.Conv2d(dim // 4, dim, 1, 1),
            nn.Sigmoid()
        )

        #
        self.x_out = nn.Conv2d(dim, x_ch, 1, 1)
        self.y_out = nn.Conv2d(dim, y_ch, 1, 1)

    def forward(self, fes):
        x, y = fes
        x_hidden = self.x_map_conv(x)
        y_hidden = self.y_map_conv(y)

        #channel
        x_channel = self.x_channel(x_hidden)
        y_channel = self.y_channel(y_hidden)
        x_hidden = y_channel * x_hidden
        y_hidden = x_channel * y_hidden
        #spatial
        x_max = torch.max(x_hidden, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x_hidden, dim=1, keepdim=True)
        x_spatial = torch.cat([x_max, x_avg], dim=1)

        x_spatial = self.x_spatial(x_spatial)

        y_max = torch.max(y_hidden, dim=1, keepdim=True)[0]
        y_avg = torch.mean(y_hidden, dim=1, keepdim=True)
        y_spatial = torch.cat([y_max, y_avg], dim=1)
        y_spatial = self.y_spatial(y_spatial)
        x_hidden = x_hidden * y_spatial
        y_hidden = y_hidden * x_spatial

        x = self.x_out(x_hidden) + x
        y = self.y_out(y_hidden) + y
        return x, y

def cross_att_test(model, x_sizes, y_sizes, device=torch.device("cpu")):
    model = model.to(device)
    x = torch.randn(*x_sizes).to(device)
    y = torch.randn(*y_sizes).to(device)
    cross_x, cross_y = model([x, y])
    print(cross_x.shape, cross_y.shape)
    from pytorch_benchmark.benchmark import measure_flops
    from pytorch_benchmark.format import format_num
    print("Flops for", model.__class__.__name__, " :", format_num(measure_flops(model, [x, y])))

if __name__ == "__main__":
    model = CrossAttention(64, 64, 64, num_head=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_att_test(model, [1, 64, 64, 64], [1, 64, 64, 64], device=device)
    model = CrossAttentionConv(64, 64, 64)
    cross_att_test(model, [1, 64, 64, 64], [1, 64, 64, 64], device=device)

