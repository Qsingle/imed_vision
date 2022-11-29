# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pyramid_spatial_pooling
    author: 12718
    time: 2022/11/18 10:54
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class PSP(nn.Module):
    def __init__(self, in_ch, sizes=(1, 2, 3, 6)):
        """
        The Pyramid Spatial Pooling module.
        References:
         "Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>
        Args:
            in_ch (int): number of channels for input
            sizes (list): list for the size of the output of pooling
        """
        super(PSP, self).__init__()
        assert in_ch % len(sizes) == 0, "The dimension"
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_ch, in_ch // len(sizes), 1, 1, 0)
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        feas = []
        for pool in self.pools:
            feas.append(F.interpolate(pool(x), size=x.size()[2:], mode="bilinear", align_corners=True))
        out = torch.cat(feas, dim=1)
        return out

