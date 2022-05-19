# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:stdc
    author: 12718
    time: 2022/5/16 19:49
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import math
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=0, groups=1, dilation=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        net = self.relu(self.bn(self.conv(x)))
        return net

ConvX = ConvBNReLU

class STDC(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, block_num=4):
        """
        Implementation of  Short-Term Dense Concatenate module
        "Rethinking BiSeNet For Real-time Semantic Segmentation"
        <https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            stride (Union[int,tuple]): stride for the block
            block_num (int): number of blocks
        """
        super(STDC, self).__init__()
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_ch // 2, out_ch // 2, kernel_size=3, stride=2, padding=1,
                          groups=out_ch // 2, bias=False),
                nn.BatchNorm2d(out_ch // 2),
            )
            self.skip = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        else:
            self.skip = nn.Identity()
            self.avd_layer = nn.Identity()

        for i in range(block_num):
            if i == block_num - 1:
                self.conv_list.append(ConvX(out_ch // int(math.pow(2, i)), out_ch // int(math.pow(2, i)), 3, 1, 1))
            elif i == 0:
                self.conv_list.append(ConvX(in_ch, out_ch // 2, 1))
            elif i == 1:
                self.conv_list.append(ConvX(out_ch // int(math.pow(2, i)), out_ch // int(math.pow(2, i+1)), 3, stride, 1))
            else:
                self.conv_list.append(ConvX(out_ch // int(math.pow(2, i)), out_ch // int(math.pow(2, i+1)), 3, 1, 1))

    def forward(self, x):
        outputs = []
        out = self.conv_list[0](x)
        out_identity = self.skip(out)
        outputs.append(out_identity)
        for i in range(1, len(self.conv_list)):
            if i == 1:
                out = self.conv_list[i](self.avd_layer(out))
            else:
                out = self.conv_list[i](out)
            outputs.append(out)
        output = torch.cat(outputs, dim=1)
        return output