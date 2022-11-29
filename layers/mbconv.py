# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:mbconv
    author: 12718
    time: 2022/9/19 10:56
    tool: PyCharm
"""
import torch
import torch.nn as nn

from .utils import SAMEConv2d
from .utils import SEModule
from .dropout import DropPath

SqueezeExcite = SEModule

def create_conv2d(in_ch, out_ch, ksize, stride, dilation=1, groups=1):
    if stride == 1 and (dilation*(ksize -1)) % 2 == 0:
        padding = ((stride-1) + dilation*(ksize-1)) // 2
        return nn.Conv2d(in_ch, out_ch, ksize, stride, dilation=dilation,
                         padding=padding, groups=groups, bias=False)
    else:
        return SAMEConv2d(in_ch, out_ch, ksize, stride, dilation=dilation,
                          groups=groups, bias=False)

class MBConv(nn.Module):
    """
        Inverted Bottleneck introduced in MobilenetV2
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"<https://arxiv.org/abs/1801.04381v4>
        In MobilenetV3, the block introduce the se layer.
        "Searching for MobileNetV3"<https://arxiv.org/abs/1905.02244>
    """
    def __init__(self, in_ch, out_ch, exp_ratio=4., ksize=3, stride=1, act_layer=nn.ReLU6(), norm_laryer=nn.BatchNorm2d,
                 se_layer=True, gate_layer=nn.Sigmoid(), has_skip=False, drop_path_rate=0., reduction=0.25):
        super(MBConv, self).__init__()
        exp_ch = round(in_ch*exp_ratio)
        self.pw = nn.Conv2d(in_ch, exp_ch, 1, 1, 0)
        self.act = act_layer
        self.bn1 = norm_laryer(exp_ch)
        self.dw = create_conv2d(exp_ch, exp_ch, ksize, stride, dilation=1, groups=exp_ch)
        self.bn2 = norm_laryer(exp_ch)
        self.pw1 = nn.Conv2d(exp_ch, out_ch, 1, 1, 0)
        self.bn3 = norm_laryer(out_ch)
        self.se = SqueezeExcite(exp_ch, reduction=reduction,activation=act_layer,
                                sigmoid=gate_layer) if se_layer else nn.Identity()
        self.skip = has_skip or (in_ch != out_ch)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.shortcut = nn.Identity() if in_ch == out_ch and self.skip else nn.Conv2d(in_ch, out_ch, 1, 1)
        if stride == 2 and self.skip:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(2, 2) if stride == 2 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, 1, 1)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        net = self.pw(x)
        net = self.bn1(net)
        net = self.act(net)
        net = self.dw(net)
        net = self.bn2(net)
        net = self.se(net)
        net = self.pw1(net)
        net = self.bn3(net)
        if self.skip:
            net = self.drop_path(net) + shortcut
        return net