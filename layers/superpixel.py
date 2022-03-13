# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/27 9:33
    @filename: superpixel.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


from .utils import *

class SuperResolutionModule(nn.Module):
    def __init__(self, in_ch):
        super(SuperResolutionModule, self).__init__()

        self.conv = nn.Sequential(
            Conv2d(in_ch, 64, 5, stride=1, padding=2, norm_layer=None, activation=nn.Tanh()),
            Conv2d(64, 32, 3, stride=1, padding=1, norm_layer=None, activation=nn.Tanh()),
            nn.Conv2d(32, 3*4, 3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        net = self.conv(x)
        net = torch.sigmoid(net)
        return net


if __name__ == "__main__":
    x = torch.randn(size=(2, 3, 112, 112))
    model = SuperResolutionModule(3)
    out = model(x)
    print(out.shape)