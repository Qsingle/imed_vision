# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:epp
    author: 12718
    time: 2023/1/8 14:42
    tool: PyCharm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv2d
from .channel_shuffle import ChannelShuffle


class EPP(nn.Module):
    def __init__(self,in_ch, proj_ch, out_ch, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        """
        Efficient Pyramid Module
        https://github.com/sacmehta/EdgeNets/blob/2b232d3f7fb60658755dad1ebca0ffc895cc795e/nn_layers/efficient_pyramid_pool.py#L12
        Args:
            in_ch (int): number of channels for input
            proj_ch (int): number of filters for proj layer
            out_ch (int): number of channels for the output
            scales (list): list of the scales
            last_layer_br (bool): whether use the bn and activation at last
        """
        super(EPP, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)
        self.proj_layer = Conv2d(in_ch, proj_ch, 1, 1, activation=nn.PReLU(proj_ch))
        for _ in enumerate(scales):
            self.stages.append(
                nn.Conv2d(proj_ch, proj_ch, 3, 1, 1, bias=False, groups=proj_ch)
            )
        self.merge_layer = nn.Sequential(
            nn.BatchNorm2d(proj_ch*len(scales)),
            nn.PReLU(proj_ch*len(scales)),
            ChannelShuffle(groups=len(scales)),
            Conv2d(proj_ch*len(scales), proj_ch, 3, 1, groups=proj_ch),
            nn.Conv2d(proj_ch, out_ch, 1, 1, bias=not last_layer_br)
        )
        self.br = nn.Identity()
        if last_layer_br:
            self.br = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.PReLU(out_ch)
            )
        self.scales = scales

    def forward(self, x):
        outs = []
        x = self.proj_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height*self.scales[i]))
            w_s = int(math.ceil(width*self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                s_o = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                s_o = stage(s_o)
                s_o = F.interpolate(s_o, size=(height, width), mode="bilinear", align_corners=True)
            elif self.scales[i] > 1.0:
                s_o = F.interpolate(x, (h_s, w_s), mode="bilinear", align_corners=True)
                s_o = stage(s_o)
                s_o = F.adaptive_avg_pool2d(s_o, output_size=(height, width))
            else:
                s_o = stage(x)
            outs.append(s_o)
        out = torch.cat(outs, dim=1)
        out = self.merge_layer(out)
        out = self.br(out)
        return out