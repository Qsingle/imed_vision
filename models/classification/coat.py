# -*- coding:utf-8 -*-
"""
    FileName: coat
    Author: 12718
    Create Time: 2023-04-27 14:53
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
from typing import Dict, Union

from layers.mbconv import MBConv
from layers.attentions import Attention4D, AttentionDownsample
from .create_model import BACKBONE_REGISTER

cfgs = {
    "coat_0": {
        "lengths": [2, 3, 5, 2],
        "depths": [64, 96, 192, 384, 768],
        "num_head": 32
    },
    "coat_1": {
        "lengths": [2, 6, 14, 2],
        "depths": [64, 96, 192, 384, 768],
        "num_head": 32
    },
    "coat_2": {
        "lengths": [2, 6, 14, 2],
        "depths": [128, 128, 256, 512, 1024],
        "num_head": 32
    },
    "coat_3": {
        "lengths": [2, 6, 14, 2],
        "depths": [192, 192, 384, 768, 1536],
        "num_head": 32
    },
    "coat_4": {
        "lengths": [2, 12, 28, 2],
        "depths": [192, 192, 384, 768, 1536],
        "num_head": 32
    }
}

class CoAt(nn.Module):
    def __init__(self, cfg:Dict, in_ch=3, num_classes=1000):
        super(CoAt, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, cfg['depths'][0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cfg['depths'][0]),
            nn.GELU(),
            nn.Conv2d(cfg['depths'][0], cfg['depths'][0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(cfg['depths'][0], cfg['depths'][0]),
            nn.GELU()
        )
        self.out_features = cfg['depths']
        blocks = [MBConv(cfg['depths'][0], cfg['depths'][1], stride=2, act_layer=nn.GELU())]
        for _ in range(1, cfg['lengths'][0]):
            blocks.append(MBConv(cfg['depths'][1], cfg['depths'][1]))
        self.s1 = nn.Sequential(*blocks)
        blocks = [MBConv(cfg['depths'][1], cfg['depths'][2], stride=2, act_layer=nn.GELU())]
        for _ in range(1, cfg['lengths'][1]):
            blocks.append(MBConv(cfg['depths'][2], cfg['depths'][2]))
        self.s2 = nn.Sequential(*blocks)
        blocks = [nn.Conv2d(cfg['depths'][2], cfg['depths'][3], 1, 1), nn.BatchNorm2d(cfg['depths'][3]), nn.MaxPool2d(3, 2, 1),
                  Attention4D(num_head=cfg["num_head"], dim=cfg['depths'][3], dim_k=cfg['depths'][3], att_ratio=4.)]
        for i in range(1, cfg['lengths'][2]):
            blocks.append(Attention4D(num_head=cfg['num_head'], dim=cfg['depths'][3], dim_k=cfg['depths'][3], att_ratio=4., downsample=True))
        self.s3 = nn.Sequential(*blocks)
        blocks = [nn.Conv2d(cfg['depths'][3], cfg['depths'][4], 1, 1), nn.BatchNorm2d(cfg['depths'][4]),
                  AttentionDownsample(num_head=cfg["num_head"], dim=cfg['depths'][4], dim_k=cfg['depths'][4],
                                      att_ratio=4.)]
        for i in range(1, cfg['lengths'][3]):
            blocks.append(
                Attention4D(num_head=cfg['num_head'], dim=cfg['depths'][4], dim_k=cfg['depths'][4], att_ratio=4.,
                            downsample=True))
        self.s4 = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg['depths'][4], num_classes)
        )

    def forward_features(self, x):
        x = self.stem(x)
        festures = [x]
        x = self.s1(x)
        festures.append(x)
        x = self.s2(x)
        festures.append(x)
        x = self.s3(x)
        festures.append(x)
        x = self.s4(x)
        festures.append(x)
        return festures

    def forward(self, x):
        features = self.forward_features(x)
        pool = self.avgpool(features[-1])
        out = self.classifier(pool)
        return out

@BACKBONE_REGISTER.register()
def coat_0(**kwargs):
    kwargs['cfg'] = cfgs['coat_0']
    return CoAt(**kwargs)

@BACKBONE_REGISTER.register()
def coat_1(**kwargs):
    kwargs['cfg'] = cfgs['coat_1']
    return CoAt(**kwargs)

@BACKBONE_REGISTER.register()
def coat_2(**kwargs):
    kwargs['cfg'] = cfgs['coat_2']
    return CoAt(**kwargs)

@BACKBONE_REGISTER.register()
def coat_3(**kwargs):
    kwargs['cfg'] = cfgs['coat_3']
    return CoAt(**kwargs)

@BACKBONE_REGISTER.register()
def coat_4(**kwargs):
    kwargs['cfg'] = cfgs['coat_4']
    return CoAt(**kwargs)

if __name__ == "__main__":
    with torch.no_grad():
        model = coat_2()
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(out.shape)