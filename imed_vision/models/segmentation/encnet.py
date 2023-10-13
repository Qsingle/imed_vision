# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/16 17:39
    @filename: encnet.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from imed_vision.models.classification.resnet import *

__all__ = ["EncNet"]

backbones = {
    "resnest50" : resnest50,
    "resnest101" : resnest101,
    "resnest14" : resnest14,
    "resnet50" : resnet50,
    "resnet101" : resnet101
}


class Encoding(nn.Module):
    def __init__(self, K, D):
        super(Encoding, self).__init__()
        self.k = K
        self.d = D
        self.codes = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / ((self.k * self.d) ** 0.5)
        self.codes.data.uniform_(-std, std)
        self.scale.data.uniform_(0, 1)

    @staticmethod
    def scale_l2(x, c, s):
        s = s.reshape(1, 1, c.size(0), 1)
        x = x.unsqueeze(2).expand(x.size(0), x.size(1), c.size(0), c.size(1))
        c = c.unsqueeze(0).unsqueeze(0)
        out = (x-c) * s
        out = out.pow(2).sum(3)
        return out

    @staticmethod
    def aggregate(a, x, c):
        a = a.unsqueeze(3)
        x = x.unsqueeze(2).expand(x.size(0), x.size(1), c.size(0), c.size(1))
        c = c.unsqueeze(0).unsqueeze(0)
        e = (x-c) * a
        e = e.sum(1)
        return e

    def forward(self, x):
        assert self.d == x.size(1)
        bs, d = x.size()[:2]
        if x.dim() == 3:
            x = x.transpose(1, 2).contiguous()
        elif x.dim() == 4:
            x = x.reshape(bs, d, -1).transpose(1, 2).contiguous()
        else:
            raise ValueError("Unknown dim of input")

        a = torch.softmax(self.scale_l2(x, self.codes, self.scale), dim=2)
        e = self.aggregate(a, x, self.codes)
        return e

class EncModule(nn.Module):
    def __init__(self, in_ch,n_classes, num_codes=32, se_loss=True, norm_layer=nn.BatchNorm2d):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(in_ch),
            nn.ReLU(inplace=True),
            Encoding(num_codes,in_ch),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.Sigmoid()
        )
        if self.se_loss:
            self.se_layer = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        bs, c = x.size()[:2]
        en = self.encoder(x).mean(1)
        gamma = self.fc(en)
        y = gamma.reshape(bs, c, 1, 1)
        outputs = [F.relu_(y*x + x)]
        if self.se_loss:
            outputs.append(self.se_layer(en))
        return outputs

class EncNet(nn.Module):
    def __init__(self, in_ch, num_classes, num_codes=32, backbone="resnet50",
                 se_loss=True, norm_layer=nn.BatchNorm2d, light_head=True,
                 laternel=True,
                 **kwargs):
        super(EncNet, self).__init__()
        dilations = [1, 1, 2, 4]
        strides = [1, 2, 1, 1]
        self.backbone = backbones[backbone](in_ch=in_ch, light_head=light_head,
                                            dilations=dilations, strides=strides,**kwargs)
        del self.backbone.fc
        del self.backbone.avg_pool
        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            norm_layer(512),
            nn.ReLU(inplace=True)
        )
        self.laternel = laternel
        if self.laternel:
            self.shortcut_c2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1),
                norm_layer(512),
                nn.ReLU(inplace=True)
            )
            self.shortcut_c3 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, stride=1),
                norm_layer(512),
                nn.ReLU(inplace=True)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(512*3, 512, kernel_size=1, stride=1),
                norm_layer(512),
                nn.ReLU(inplace=True)
            )
        self.enc_module = EncModule(512, num_classes, num_codes=num_codes,
                                    se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        )


    def forward(self, x):
        net = self.backbone.conv1(x)
        net = self.backbone.max_pool(net)
        c1 = self.backbone.layer1(net)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        feat = self.conv5(c4)
        if self.laternel:
            c2 = self.shortcut_c2(c2)
            c3 = self.shortcut_c3(c3)
            feat = self.fusion(torch.cat([feat, c2, c3], dim=1))

        outs = list(self.enc_module(feat))
        outs[0] = self.conv6(outs[0])
        # if self.layer_attention:
        #     outs[0] = outs[0] * torch.softmax(c4, dim=1) + outs[0]
        outs[0] = F.interpolate(outs[0], size=x.size()[2:], mode="bilinear", align_corners=True)
        return outs




if __name__ == "__main__":
    net = EncNet(3, 20)
    x = torch.randn((2, 3, 32, 32))
    out = net(x)
    for ou in out:
        print(ou.shape)