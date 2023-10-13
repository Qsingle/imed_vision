# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 10:52
    @filename: splat.py.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SplAtConv2d"]

class rSoftmax(nn.Module):
    def __init__(self, cardinality=1, radix=2):
        '''
            The r-Softmax in ResNest.
            Args:
                cardinality (int): the number of card
                radix (int): the radix index
        '''
        super(rSoftmax, self).__init__()
        self.cardinality = cardinality
        self.radix = radix

    def forward(self, x):
        bs = x.size(0)
        if self.radix > 1:
            net = x.reshape(bs, self.cardinality, self.radix, -1)
            net = F.softmax(net, dim=1)
            net = net.reshape(bs, -1)
        else:
            net = torch.sigmoid(x)
        return net


class SplAtConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1, bias=False, groups=1,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=None, nolinear=None, **kwargs):
        '''
            Split Attention Conv2d from
            "ResNeSt: Split-Attention Networks"<https://hangzhang.org/files/resnest.pdf>
            Args:
                in_ch (int): the number of channels for input
                ksize (Union[int, tuple]): the kernel size)
                stride (Union[int, tuple]): the stride of slide for conv)
                dilation (int): the dilation rate
                bias (int): whether use the bias
                groups (int): the number of groups for conv kernels
                radix (int): the radix indexes
                drop_prob (float): the droup out keep rate
                reduction (int): the reduction factor for channel reduction
                norm_layer (nn.BatchNorm2d): the normalization layer
                nolinear (nn.ReLU or other activation layer): the nolinear function to activate the output
        '''
        super(SplAtConv2d, self).__init__()
        self.radix = radix
        self.reduction = reduction
        self.use_bn = norm_layer is not None
        inter_channels = max(radix * out_ch // reduction, 32)
        self.drop_prob = drop_prob
        self.conv = nn.Conv2d(in_ch, out_ch * radix, kernel_size=ksize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups * radix, bias=bias, **kwargs)
        self.cardinality = groups

        if self.use_bn:
            self.bn0 = norm_layer(out_ch * radix)

        self.relu = nolinear if nolinear is not None else nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(out_ch, inter_channels, kernel_size=1, stride=1, padding=0, groups=self.cardinality,)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)

        self.fc2 = nn.Conv2d(inter_channels, out_ch*radix, 1, stride=1, padding=0, groups=self.cardinality)
        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)
        self.rsoftmax = rSoftmax(cardinality=self.cardinality, radix=self.radix)

    def forward(self, x):
        net = self.conv(x)

        if self.use_bn:
            net = self.bn0(net)

        net = self.relu(net)

        batch, rchannels = net.size()[:2]
        if self.radix > 1:
            splited = torch.split(net, rchannels // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = net

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)

        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, rchannels // self.radix, dim=1)
            out = sum([att * split for att, split in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()