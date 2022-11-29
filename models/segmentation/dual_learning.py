# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dual_learning
    author: 12718
    time: 2022/9/19 14:59
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification import create_backbone
from layers.task_fusion import MSConv2d, CrossGL


class Decoder(nn.Module):
    def __init__(self, x_ch, y_ch, dim):
        super(Decoder, self).__init__()
        self.x_dense = nn.Sequential(
            nn.Conv2d(x_ch, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.y_dense = nn.Sequential(
            nn.Conv2d(y_ch, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.gate_y = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gate_x = nn.Conv2d(dim, dim ,1, 1, 0)

    def forward(self, high, low):
        x = F.interpolate(high, size=low.size()[2:], mode="bilinear", align_corners=True)
        x_dense = self.x_dense(x)
        y = self.y_dense(low)
        gate_y = self.gate_y(x_dense)*y + y
        gate_x = self.gate_x(y)*x_dense + x_dense + gate_y
        return gate_x

class DualLearning(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, arch="resnet50", sr_layer=3, upscale_rate=4, pretrained=True):
        super(DualLearning, self).__init__()
        self.backbone = create_backbone(arch, pretrained=pretrained, in_ch=in_ch)
        del self.backbone.fc
        # self.expand_down = create_conv2d(2048, 2048, ksize=3, stride=2)
        self.ms_conv = MSConv2d(2048, groups=8)
        self.sr_layer = sr_layer
        self.seg_decoder1 = Decoder(2048, 1024, 1024)
        self.seg_decoder2 = Decoder(1024, 512, 512)
        self.seg_decoder3 = Decoder(512, 256, 256)
        self.sr_decoder1 = Decoder(1024, 512, 512)
        self.sr_decoder2 = Decoder(512, 256, 256)
        self.sr_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, in_ch*(upscale_rate**2), 1, 1, 0),
            nn.PixelShuffle(upscale_rate)
        )
        self.fusion = CrossGL(256, 256, hidden_state=128, reduction=4)
        self.out = nn.Conv2d(256, num_classes, 1, 1)
        self.out_up = nn.Sequential(
            nn.Conv2d(num_classes, num_classes*(upscale_rate**2), 1, 1, 0),
            nn.PixelShuffle(upscale_rate)
        )


    def forward(self, x):
        features = self.backbone.forward_features(x)
        # net = self.expand_down(features[-1])
        net = self.ms_conv(features[-1])
        #segmentation part
        seg_up_1 = self.seg_decoder1(net, features[2])
        seg_up_2 = self.seg_decoder2(seg_up_1, features[1])
        seg_up_3 = self.seg_decoder3(seg_up_2, features[0])
        seg_up_3 = F.interpolate(seg_up_3, size=x.size()[2:], mode="bilinear", align_corners=True)
        out = self.out(seg_up_3)
        out = self.out_up(out)
        if not self.training:
            return out
        # super resolution part
        sr_up1 = self.sr_decoder1(features[2], features[1])
        sr_up2 = self.sr_decoder2(sr_up1, features[0])
        sr_up2 = F.interpolate(sr_up2, x.size()[2:], mode="bilinear", align_corners=True)
        sr = self.sr_head(sr_up2)
        #fusion part
        fusion_sr, fusion_seg = self.fusion(sr_up2, seg_up_3)
        fusion_sr = self.sr_head(fusion_sr)
        fusion_seg = self.out(fusion_seg)
        fusion_seg = self.out_up(fusion_seg)
        return out, sr, fusion_seg, fusion_sr




if __name__ == "__main__":
    from torchstat import stat
    # x = torch.randn(1, 3, 1024, 1024).cuda()
    model = DualLearning(pretrained=False)
    from models.segmentation import DeeplabV3Plus
    # model = DeeplabV3Plus(3, 4)
    # model(x)
    stat(model, (3, 512, 512))