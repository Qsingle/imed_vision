# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:eca
    author: 12718
    time: 2021/9/23 16:05
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1,
                 activation=nn.Sigmoid()):
        """
                Implementation of Efficient Channel Attention Module,
                References:
                    "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
                    <https://arxiv.org/abs/1910.03151v3>
                Args:
                    channels (int): number of channels
                    gamma (int): gamma introduced in paper
                    b (int): b introduced in paper
                    activation (nn.Module): attention activation function
                """
        super(ECA, self).__init__()

        t = int(abs(math.log2(channels)/gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.activation =activation

    def forward(self, x):
        net = self.avgpool(x)
        net = net.squeeze(3)
        net = net.transpose(-1, -2)
        net = self.conv(net)
        net = net.transpose(-1, -2).unsqueeze(-1)
        net = torch.sigmoid(net)
        return x*net.expand_as(x)