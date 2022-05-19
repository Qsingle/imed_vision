# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:bisenetv1_layers
    author: 12718
    time: 2022/5/19 11:07
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

from .utils import Conv2d

class ARM(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Implementation of Attention refinement module in BiseNet
        "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation"
        <https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
        """
        super(ARM, self).__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3, 1, 1)
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        # self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        net = self.conv1(x)
        attn = self.gpool(net)
        attn = self.linear(attn)
        attn = torch.sigmoid(attn)
        net = attn * net
        return net

class FFM(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        """
        Implementation of Feature Fusion Module in BiseNet.
        "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation"
        <https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
        """
        super(FFM, self).__init__()
        self.conv1 = Conv2d(in_ch1+in_ch2, out_ch, 3, 1, 1)
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(out_ch//4, out_ch, 1, 1, 0)

    def forward(self, fea1, fea2):
        feat = torch.cat([fea1, fea2], dim=1)
        feat = self.conv1(feat)
        attn = self.gpool(feat)
        attn = self.fc2(self.fc1(attn))
        attn = torch.sigmoid(attn)
        out = feat*attn + feat
        return out