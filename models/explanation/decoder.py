# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:decoder
    author: 12718
    time: 2022/4/7 15:19
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mbconv import MBConv
from typing import List

from models.classification import create_backbone

class MultiDecoder(nn.Module):
    def __init__(self, in_chs, out_ch=3):
        super(MultiDecoder, self).__init__()
        self.layer_list = nn.ModuleList()
        for i in range(len(in_chs)):
            self.layer_list.append(
                nn.Sequential(
                    MBConv(in_chs[i], 64, exp_ratio=4),
                    MBConv(64, 64, exp_ratio=4)
                )
            )
        self.up_conv = nn.ModuleList()
        for i in range(len(in_chs)-1):
            self.up_conv.append(nn.Conv2d(64, 64, 1, 1))
        self.fusion_out = nn.Sequential(
            nn.Conv2d(64*len(in_chs), 256, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        )
        self.out_conv = nn.Conv2d(256, out_ch, 1, 1)

    def forward(self, fes: List[torch.Tensor]) -> torch.Tensor:
        sam_fes = []
        for i in range(len(fes)):
            sam_fes.append(self.layer_list[i](fes[i]))
        for i in range(1, len(sam_fes)):
            sam_fes[i] = F.interpolate(sam_fes[i], size=sam_fes[0].size()[2:], mode="bilinear", align_corners=True)
            sam_fes[i] = self.up_conv[i-1](sam_fes[i])
        fusion = torch.cat(sam_fes, dim=1)
        fusion = self.fusion_out(fusion)
        out = self.out_conv(fusion)
        return out

class RGM(nn.Module):

    def __init__(self, class_model="resnet50", mlp=False, chs=[], pretrained=False):
        super(RGM, self).__init__()
        self.encode_model = create_backbone(class_model, pretrained=pretrained)

        self.chs = chs
