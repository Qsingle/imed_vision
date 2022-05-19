# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:stdcnet
    author: 12718
    time: 2022/5/18 19:43
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

from layers.stdc import STDC, ConvX

from .create_model import BACKBONE_REGISTER

class STDCNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1000, layers=[2, 2, 2], use_conv_last=False, dropout=0.2):
        """
        Implementation of STDCNet.
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            layers (list): number of blocks for stage3-stage5
            use_conv_last (bool):whether use last conv layer
            dropout (float): dropout rate
        """
        super(STDCNet, self).__init__()
        self.conv1 = ConvX(in_ch, 32, 3, 2, 1)
        self.conv2 = ConvX(32, 64, 3, 2, 1)
        self.ch = 64
        self.stage3 = self._make_layers(layers[0], 256)
        self.stage4 = self._make_layers(layers[1], 512)
        self.stage5 = self._make_layers(layers[2], 1024)
        self.conv_last = ConvX(1024, 1024, 1)
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1024, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, num_classes, bias=False)
        self.use_conv_last = use_conv_last

    def _make_layers(self, num_blcoks, out_ch):
        blocks = []
        for i in range(num_blcoks):
            if i == 0:
                blocks.append(STDC(self.ch, out_ch, stride=2))
                self.ch = out_ch
            else:
                blocks.append(STDC(self.ch, self.ch))
        return nn.Sequential(*blocks)

    def forward_features(self, x):
        features = []
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.stage3(net)
        features.append(net)
        net = self.stage4(net)
        features.append(net)
        net = self.stage5(net)
        features.append(net)
        net = self.conv_last(net)
        if self.use_conv_last:
            features.append(net)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        net = self.gpool(features[-1])
        net = torch.flatten(net, 1)
        net = self.fc(net)
        net = self.bn(net)
        net = self.relu(net)
        net = self.drop(net)
        net = self.linear(net)
        return net

def _stdcnet(pretrained=False, checkpoint=None, **kwargs):
    model = STDCNet(**kwargs)
    if pretrained:
        assert checkpoint is not None, "Please provide the path to the pretrained weights"
        state = torch.load(checkpoint)
        model.load_state_dict(state)
    return model

@BACKBONE_REGISTER.register()
def stdcnet_1(pretrained=False, checkpoint=None, **kwargs):
    kwargs["layers"] = [2, 2, 2]
    model = _stdcnet(pretrained, checkpoint, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def stdcnet_2(pretrained=False, checkpoint=None, **kwargs):
    kwargs["layers"] = [4, 5, 3]
    model = _stdcnet(pretrained, checkpoint, **kwargs)
    return model


if __name__ == "__main__":
    import torch
    import sys
    import os
    import re
    from collections import OrderedDict
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 224, 224)
    model = stdcnet_2(pretrained=False)
    model.eval()
    out = model(x)
    print(out.shape)