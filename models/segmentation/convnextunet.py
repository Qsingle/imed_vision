# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:convnextunet
    author: 12718
    time: 2022/4/11 10:18
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvNeXtBlock
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=3, drop_path_rate=0.):
        super(Downsample, self).__init__()
        blcks = []
        for i in range(num_blocks):
            blcks.append(ConvNeXtBlock(in_ch, dropout_rate=drop_path_rate))
        self.conv1 = nn.Sequential(*blcks)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        feature = self.conv1(x)
        down = self.down(feature)
        return feature, down

class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, drop_path_rate=0.):
        super(Upsample, self).__init__()
        self.upsample_conv = nn.Conv2d(in_ch1, out_ch, 1, 1, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch+in_ch2, out_ch, 1, 1, 0),
            ConvNeXtBlock(out_ch, dropout_rate=drop_path_rate),
            ConvNeXtBlock(out_ch, dropout_rate=drop_path_rate),
        )

    def forward(self, fe1, fe2):
        up = F.interpolate(fe1, size=fe2.size()[2:], align_corners=True, mode="bilinear")
        up = self.upsample_conv(up)
        fusion = torch.cat([up, fe2], dim=1)
        out = self.conv(fusion)
        return out

class ConvNeXtUNet(nn.Module):
    def __init__(self, in_ch, num_classes=2, num_blocks=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super(ConvNeXtUNet, self).__init__()
        self.stem = nn.Conv2d(in_ch, dims[0], 4, stride=2, padding=1)
        self.down1 = Downsample(dims[0], dims[1], num_blocks[0], drop_path_rate)
        self.down2 = Downsample(dims[1], dims[2], num_blocks[1], drop_path_rate)
        self.down3 = Downsample(dims[2], dims[3], num_blocks[2], drop_path_rate)
        self.down4 = nn.Sequential(
            *[ConvNeXtBlock(dims[3], dropout_rate=drop_path_rate) for _ in range(num_blocks[3])]
        )
        self.up5 = Upsample(dims[3], dims[2], dims[2], drop_path_rate)
        self.up6 = Upsample(dims[2], dims[1], dims[1], drop_path_rate)
        self.up7 = Upsample(dims[1], dims[0], dims[0], drop_path_rate)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]*4, 1, 1, 0),
            nn.Conv2d(dims[0]*4, dims[0], 1, 1, 0),
            nn.Conv2d(dims[0], num_classes, 1, 1, 0)
        )

    def forward(self, x):
        fe1 = self.stem(x)
        down1_f, down1 = self.down1(fe1)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f = self.down4(down3)
        up5 = self.up5(down4_f, down3_f)
        up6 = self.up6(up5, down2_f)
        up7 = self.up7(up6, down1_f)
        up8 = F.interpolate(up7, size=x.size()[2:], mode="bilinear", align_corners=True)
        out = self.out_conv(up8)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = ConvNeXtUNet(3, 2)
    total = 0
    for params in model.parameters():
        param = 1
        for dim in params.size():
            param *= dim
        total += param
    total = total*4
    print("params:{} M".format(total/ (1024*1024)))
    out = model(x)
    print(out.shape)