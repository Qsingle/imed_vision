# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 10:58
    @filename: utils.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

from .splat import SplAtConv2d

__all__ = ["Conv2d",  "SEModule", "DepthWiseSeparableConv2d"]



class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=16, norm_layer=None, sigmoid=None, activation=None):
        """
        SEModule of SENet and MobileNetV3
        Args:
            in_ch (int): the number of input channels
            reduction (int): the reduction rate
            norm_layer (nn.Module): the normalization module
            sigmoid （nn.Module): the sigmoid activation function for the last of fc
            activation (nn.Module): the middle activation function
        """
        super(SEModule, self).__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if sigmoid is None:
            sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        inter_channel = in_ch // reduction
        self.fc = nn.Sequential(
            Conv2d(in_ch, inter_channel, ksize=1, stride=1, norm_layer=norm_layer, activation=activation),
            Conv2d(inter_channel, in_ch, ksize=1, stride=1, norm_layer=norm_layer, activation=sigmoid)
        )
    def forward(self, x):
        net = self.avg_pool(x)
        net = self.fc(net) * x
        return net

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), dropout_rate=0.0, gn_groups=32,
                 **kwargs):
        """
        The conv2d with normalization layer and activation layer.
        Args:
            in_ch (int): the number of channels for input
            out_ch (int): the number of channels for output
            ksize (Union[int,tuple]): the size of conv kernel, default is 1
            stride (Union[int,tuple]): the stride of the slide window, default is 1
            padding (Union[int, tuple]): the padding size, default is 0
            dilation (Union[int,tuple]): the dilation rate, default is 1
            groups (int): the number of groups, default is 1
            bias (bool): whether use bias, default is False
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the nonlinear module
            dropout_rate (float): dropout rate
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, **kwargs)
        self.norm_layer = norm_layer
        if not norm_layer is None:
            if isinstance(norm_layer, nn.GroupNorm):
                self.norm_layer = norm_layer(gn_groups, out_ch)
            else:
                self.norm_layer = norm_layer(out_ch)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        net = self.conv(x)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout_rate > 0:
            self.dropout(net)
        return net



class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1, bias=False, **kwargs):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depth_wise = nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation,
                                    bias=bias,**kwargs)
        self.point_wise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, bias=bias, **kwargs)

    def forward(self, x):
        net = self.depth_wise(x)
        net = self.point_wise(net)
        return net