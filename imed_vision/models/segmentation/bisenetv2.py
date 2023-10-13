# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:bisenetv2
    author: 12718
    time: 2022/5/17 19:38
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import torch.nn as nn
import torch.nn.functional as F

from imed_vision.layers.bisenetv2_layers import *


class BiseNetV2(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, expansion=6, alpha=1, d=1, lambd=4, dchs=[64, 64, 128], boost=True,
                 control=64):
        """
        Implementation of BiseNetV2
         "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            expansion (int): expansion rate for the GE block
            alpha (int): channel expansion rate
            d (int): depth control
            lambd (int): lambda to control the number of channels for semantic branch
            dchs (list): channels for detail head
            boost (bool): whether use boost prediction
            control (int): number of channels for hidden state in segmentation head
        """
        super(BiseNetV2, self).__init__()
        dchs = [ch * alpha for ch in dchs]
        self.detail = DetailHead(in_ch, chs=dchs)
        ch = dchs[0] // lambd
        self.semantic_s1 = StemBlock(in_ch, out_ch=ch)
        depth_s3 = int(2*d)
        s3_blocks = [GE(ch, int(dchs[2] // lambd), stride=2, expansion=expansion)]
        ch = dchs[2] // lambd
        for i in range(1, depth_s3):
            s3_blocks.append(GE(ch, ch, expansion=expansion))
        self.semantic_s3 = nn.Sequential(*s3_blocks)
        s4_depth = int(2*d)
        s4_blocks = [GE(ch, int(64*alpha), stride=2, expansion=expansion)]
        ch = int(64*alpha)
        for i in range(1, s4_depth):
            s4_blocks.append(GE(ch, ch, expansion=expansion))
        self.semantic_s4 = nn.Sequential(*s4_blocks)
        s5_depth = int(4*d)
        s5_blocks = [GE(ch, dchs[2], stride=2, expansion=expansion)]
        ch = dchs[2]
        for i in range(1, s5_depth):
            s5_blocks.append(GE(ch, ch, expansion=expansion))
        self.semantic_s5 = nn.Sequential(*s5_blocks)
        self.ce = CE(ch)
        self.aggre = BGA(dchs[2], ch)
        self.out = BiseNetV2Head(dchs[2], control, num_classes)
        self.boost = boost
        if boost:
            self.s1_head = BiseNetV2Head((dchs[0] // lambd), control, num_classes)
            self.s3_head = BiseNetV2Head(int(dchs[2] // lambd), control, num_classes)
            self.s4_head = BiseNetV2Head(int(64*alpha), control, num_classes)
            self.s5_head = BiseNetV2Head(ch, control, num_classes)

    def forward(self, x):
        detail = self.detail(x)
        s1 = self.semantic_s1(x)
        s3 = self.semantic_s3(s1)
        s4 = self.semantic_s4(s3)
        s5 = self.semantic_s5(s4)
        ce = self.ce(s5)
        aggre = self.aggre(detail, ce)
        out = F.interpolate(self.out(aggre), size=x.size()[2:], mode="bilinear", align_corners=False)
        if self.boost and self.training:
            out_s1 = F.interpolate(self.s1_head(s1), size=x.size()[2:], mode="bilinear", align_corners=False)
            out_s3 = F.interpolate(self.s3_head(s3), size=x.size()[2:], mode="bilinear", align_corners=False)
            out_s4 = F.interpolate(self.s4_head(s4), size=x.size()[2:], mode="bilinear", align_corners=False)
            out_s5 = F.interpolate(self.s5_head(s5), size=x.size()[2:], mode="bilinear", align_corners=False)
            return out, out_s1, out_s3, out_s4, out_s5
        return out

def bisenetv2(**kwargs):
    return BiseNetV2(alpha=1, d=1, **kwargs)

def bisenetv2_l(**kwargs):
    return BiseNetV2(alpha=2, d=3, **kwargs)

if __name__ == "__main__":
    import torch
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 300, 300)
    model = bisenetv2_l(num_classes=2)
    model.eval()
    out = model(x)
    print(out.shape)