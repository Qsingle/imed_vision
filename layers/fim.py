# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:fim
    author: 12718
    time: 2022/8/6 16:30
    tool: PyCharm
"""
import torch
import torch.nn as nn

from .channel_shuffle import ChannelShuffle

class SpatialModule(nn.Module):
    def __init__(self, in_ch):
        super(SpatialModule, self).__init__()
        out_ch = in_ch // 3
        self.out_ch = out_ch
        self.branch1 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, dilation=1, groups=out_ch)
        self.branch2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=2, dilation=2, groups=out_ch)
        self.branch3 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=4, dilation=4, groups=out_ch)
        self.shuffle = ChannelShuffle(3)

    def forward(self, x):
        splits = torch.split(x, self.out_ch, dim=1)
        branch1 = self.branch1(splits[0])
        branch2 = self.branch2(splits[1])
        branch3 = self.branch3(splits[2])
        net = torch.cat([branch1, branch2, branch3], dim=1)
        net = self.shuffle(net)
        return net

class FIM(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=8):
        super(FIM, self).__init__()
        out_ch = hidden_state * 3
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, out_ch, 1, 1),
            nn.ReLU()
        )
        self.spatial_conv = SpatialModule(out_ch)
        self.att = nn.Sequential(
            nn.Conv2d(out_ch, hidden_state, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_state, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        concat = torch.cat([sr_fe, seg_fe], dim=1)
        net = self.fusion_conv(concat)
        net = self.spatial_conv(net)
        net = self.att(net)
        return net