# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:espcn
    author: 12718
    time: 2022/6/6 9:32
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, in_ch,  upscale_factor=2):
        """
        Implementation of the ESPCN model
        "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
        <https://arxiv.org/abs/1609.05158>
        Args:
            in_ch (int): number of input channels
            upscale_factor (int): upscale factor
        """
        super(ESPCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, 1, 2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, (upscale_factor**2)*in_ch, 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        return net