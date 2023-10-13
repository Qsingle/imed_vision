# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/3/2 10:51
    @filename: genet.py
    @software: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from .create_model import BACKBONE_REGISTER
__all__ = ["GENet", "genet_small", "genet_normal", "genet_large"]

class XXBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        super(XXBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, expansion_out_ch,ksize, stride=stride, padding=ksize // 2),
            norm_layer(expansion_out_ch),
            activation,
            nn.Conv2d(expansion_out_ch, out_ch, ksize,stride=1, padding=(ksize-1) // 2, bias=bias),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )


    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class BottleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        super(BottleBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, expansion_out_ch, 1, stride=1, padding=0),
            norm_layer(expansion_out_ch),
            activation,
            nn.Conv2d(expansion_out_ch, expansion_out_ch, ksize, stride=stride, padding=(ksize-1) // 2, bias=bias),
            norm_layer(expansion_out_ch),
            activation,
            nn.Conv2d(expansion_out_ch, out_ch, 1, stride=1, padding=0),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )


    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class DwBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        super(DwBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, expansion_out_ch, 1, stride=1, padding=0),
            norm_layer(expansion_out_ch),
            activation,
            nn.Conv2d(expansion_out_ch, expansion_out_ch, ksize,stride=stride, padding=ksize // 2, bias=bias),
            norm_layer(expansion_out_ch),
            activation,
            nn.Conv2d(expansion_out_ch, out_ch, 1, stride=1, padding=0),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = nn.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                #official code not use avg downsample, we add it in there
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )

    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        assert (expansion - 1) < 1e-6, ValueError("The expansion of the conv block cannot greater than 1")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            norm_layer = nn.ReLU(inplace=True)
        expansion_out = int(round(out_ch*expansion))
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, expansion_out, ksize, stride=stride, bias=bias),
            norm_layer(expansion_out),
            activation
        )

    def forward(self, x):
        net = self.conv_block(x)
        return net

class GENet(nn.Module):
    def __init__(self, cfg, in_chs=3, num_classes=1000, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), features_only=False):
        super(GENet, self).__init__()
        self.current_channels = in_chs
        self.features = nn.ModuleDict()
        for i in range(len(cfg)):
            self.features[f"layer_{i}"] = self._make_layer(cfg[i], norm_layer, activation)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.current_channels, num_classes)
        )
        self.features_only = features_only

    def _make_layer(self, cfg, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        block_type = cfg[0]
        block = self._get_block(block_type)
        sub_layers = cfg[1]
        out_channels = cfg[2]
        current_stride = cfg[3]
        ksize = cfg[4]
        expansion = cfg[5]
        layers = []
        for i in range(sub_layers):
            current_stride = current_stride if i < 1 else 1
            layers.append(block(self.current_channels, out_channels, ksize, stride=current_stride,
                                expansion=expansion, norm_layer=norm_layer, activation=activation))
            self.current_channels = out_channels
        return nn.Sequential(*layers)

    def _get_block(self, block_type):
        if block_type.lower() == "conv":
            return ConvBlock
        elif block_type.lower() == "xx":
            return XXBlock
        elif block_type.lower() == "dw":
            return DwBlock
        elif block_type.lower() == "bl":
            return BottleBlock

    def forward_features(self, x):
        net = x
        features = []
        for i, layer in enumerate(self.features.values()):
            net = layer(net)
            features.append(net)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        if self.features_only:
            return features
        net = self.global_avg(features[-1])
        net = self.classifier(net)
        return net


cfg = {
    "genet_small":[
        ["conv", 1, 13, 2, 3, 1],
        ["xx",1, 48, 2, 3, 1],
        ["xx", 3, 48, 2, 3, 1],
        ["bl",7, 384, 2, 3, 0.25],
        ["dw", 2, 560, 2, 3, 3],
        ["dw", 1, 256, 1, 3, 3],
        ["conv", 1, 1920, 1, 1, 1]
        ],
    "genet_normal":[
        ["conv", 1, 32, 2, 3, 1],
        ["xx",1, 128 , 2, 3, 1],
        ["xx", 2, 192, 2, 3, 1],
        ["bl", 6, 640, 2, 3, 0.25],
        ["dw", 4, 640, 2, 3, 3],
        ["dw", 1, 640, 1, 3, 3],
        ["conv", 1, 2560, 1, 1, 1]
        ],
    "genet_large":[
        ["conv", 1, 32, 2, 3, 1],
        ["xx",1, 128, 2, 3, 1],
        ["xx", 2, 192, 2, 3, 1],
        ["bl",6, 640, 2, 3, 0.25],
        ["dw", 5, 640, 2, 3, 3],
        ["dw", 4, 640, 1, 3, 3],
        ["conv", 1, 2560, 1, 1, 1]
        ]
}


def get_cfg(model_name):
    return cfg[model_name]


def _create_model(model_name, **kwargs):
    cfg = get_cfg(model_name)
    model = GENet(cfg, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def genet_small(**kwargs):
    return _create_model("genet_small", **kwargs)

@BACKBONE_REGISTER.register()
def genet_normal(**kwargs):
    return _create_model("genet_normal", **kwargs)

@BACKBONE_REGISTER.register()
def genet_large(**kwargs):
    return _create_model("genet_large", **kwargs)

if __name__ == "__main__":
    model = genet_large(features_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x = torch.randn((1, 3, 224, 224)).cuda()
    with torch.no_grad():
        out = model(x)[-1]
        print(out.shape)