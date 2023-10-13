# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pspnet
    author: 12718
    time: 2022/11/18 9:15
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from imed_vision.models.classification import create_backbone
from imed_vision.layers.psp import PSP
from imed_vision.layers.task_fusion import CrossGL

SUPPORTED_BACKBONES = ["resnet50", "resnest50", "resnet101",
                       "resnest101", "resnet152", "resnest200", "resnest269"]

class PSPNet(nn.Module):
    """
    PSPNet
    References:
         "Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>
    """
    def __init__(self, in_ch=3, num_classes=19, backbone="resnet50", aux=True, pretrain=False, bin_sizes=(1, 2, 3, 6),
                 dropout=0.1, super_reso=False, upscale_rate=2):
        """
        Initialize the PSPNet object
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            backbone (str): name of the backbone model, support (resnet50, resnet101, resnest50, resnest101, resnet152,
                                                                resnest200, resnest269)
            aux (bool): whether use the aux fcn
            pretrain (bool): whether use the pretrain weights
        """
        super(PSPNet, self).__init__()
        assert backbone in SUPPORTED_BACKBONES, "The name of backbone must in {}, but got {}".format(
            SUPPORTED_BACKBONES,
            backbone)
        self.backbone = create_backbone(backbone, pretrained=pretrain, strides=[1, 2, 2, 1], dilations=[1, 1, 2, 4],
                                        in_ch=in_ch)
        del self.backbone.fc
        self.psp = PSP(2048, sizes=bin_sizes)
        fea_dim = 2048
        fea_dim = fea_dim*2
        self.out_conv = nn.Sequential(
            nn.Conv2d(fea_dim, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(512, num_classes, 1, 1)
        )
        self.aux = aux
        if self.aux:
            self.aux_out = nn.Sequential(
                nn.Conv2d(1024, 256, 3, 1, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv2d(256, num_classes, 1, 1)
            )
        self.super_reso = super_reso
        if self.super_reso:
            self.sr_decoder_fea = nn.Sequential(
                nn.Conv2d(fea_dim, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.sr_branch = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.sisr = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            )
            self.fusion = CrossGL(in_ch, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        psp_out = self.psp(features[-1])
        psp_out = torch.cat([psp_out, features[-1]], dim=1)
        out = self.out_conv(psp_out)
        out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        if self.aux and self.training:
            aux_out = self.aux_out(features[-2])
            aux_out = F.interpolate(aux_out, size=x.size()[2:], mode="bilinear", align_corners=True)
            if self.super_reso:
                sr_out = self.sr_decoder_fea(psp_out)
                sr_out = F.interpolate(sr_out, size=features[0].size()[2:], mode="bilinear", align_corners=True)
                sr_out = torch.cat([sr_out, features[0]], dim=1)
                sr_out = self.sr_branch(sr_out)
                sr_out = self.sisr(sr_out)
                fusion_sr, fusion_seg = self.fusion(sr_out, out)
                return out, aux_out, sr_out, fusion_sr, fusion_seg
            return out, aux_out
        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    m = PSPNet(backbone="resnet101")
    out, aux_out = m(x)
    print(out.shape)