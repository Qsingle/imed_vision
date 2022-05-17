# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:bisnetv2_layers
    author: 12718
    time: 2022/5/17 16:48
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv2d, DepthWiseConv2d

__all__ = ["DetailHead", "GE", "BGA", "CE", "StemBlock", "BiseNetV2Head"]

class DetailHead(nn.Module):
    def __init__(self, in_ch, chs=[64, 64, 128]):
        """
        Implementation of DetailHead in BiseNetV2
        BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            in_ch (int): number of channels for input
            chs (list): list of channels for each stage
        """
        super(DetailHead, self).__init__()
        ch = chs[0]
        self.conv1 = nn.Sequential(
            Conv2d(in_ch, ch, ksize=3, stride=2, padding=1),
            Conv2d(ch, ch, ksize=3, stride=1, padding=1)
        )
        ch = chs[1]
        self.conv2 = nn.Sequential(
            Conv2d(chs[0], ch, ksize=3, stride=2, padding=1),
            *[Conv2d(ch, ch, ksize=3, stride=1, padding=1) for _ in range(2)]
        )
        ch = chs[2]
        self.conv3 = nn.Sequential(
            Conv2d(chs[1], ch, ksize=3, stride=2, padding=1),
            *[Conv2d(ch, ch, ksize=3, stride=1, padding=1) for _ in range(2)]
        )

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        return net



class StemBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Implementation of Stem Block in BiseNetV2
         "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
        """
        super(StemBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, 1, 1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.MaxPool2d(3, 2, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        net = self.conv1(x)
        branch1 = self.branch1(net)
        branch2 = self.branch2(net)
        concat = torch.cat([branch1, branch2], dim=1)
        net = self.fusion(concat)
        return net

class GE(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=1, expansion=6):
        """
            Implementation of Gather and Expansion Layer in BiseNetV2
            "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            stride (Union[int, tuple]): stride of the layer
            expansion (int): expansion rate
        """
        super(GE, self).__init__()
        hidden_state = int(in_ch*expansion)
        if out_ch is None:
            out_ch = in_ch
        #Gather Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_state, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_state),
            nn.ReLU(inplace=True)
        )
        if stride != 1:
            self.conv2 = nn.Sequential(
                DepthWiseConv2d(hidden_state, hidden_state, 3, stride, padding=1),
                nn.BatchNorm2d(hidden_state),
                DepthWiseConv2d(hidden_state, hidden_state, 3, 1, padding=1),
                nn.BatchNorm2d(hidden_state)
            )
        else:
            self.conv2 = nn.Sequential(
                DepthWiseConv2d(hidden_state, hidden_state, 3, stride, padding=1),
                nn.BatchNorm2d(hidden_state)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_state, out_ch, 1, 1),
            nn.BatchNorm2d(out_ch)
        )
        if stride != 1:
            self.skip = nn.Sequential(
                DepthWiseConv2d(in_ch, in_ch, 3, stride=stride, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = net + identity
        return net

class CE(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        """
        Implementation of the Context Embedding Block in BiseNetV2.
        "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>

        Args:
            in_ch (int): number of channels for input
        """
        super(CE, self).__init__()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv = Conv2d(in_ch, in_ch, 1, 1)
        self.gather = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = net + x
        net = self.gather(net)
        return net

class BGA(nn.Module):
    def __init__(self, dch, sch):
        """
        Implementation of the Bilateral Guided Aggregation Layer in BiseNetV2
        "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            dch (int): number of the channels for detail branch
            sch (int): number of the channels for semantic branch
        Notes:
            the dch and sch in the BiseNetV2 is equal to 128
        """
        super(BGA, self).__init__()
        #detatil branch
        self.ddown = nn.Sequential(
            nn.Conv2d(dch, sch, 3, 2, 1),
            nn.BatchNorm2d(sch),
            nn.AvgPool2d(3, 2, 1)
        )
        self.dpath = nn.Sequential(
            DepthWiseConv2d(dch, dch, 3, 1, 1),
            nn.BatchNorm2d(dch),
            nn.Conv2d(dch, dch, 1, 1)
        )
        #segmentic branch
        self.sup = nn.Sequential(
            nn.Conv2d(sch, dch, 3, 1, 1),
            nn.BatchNorm2d(dch)
        )
        self.spath = nn.Sequential(
            DepthWiseConv2d(sch, sch, 3, 1, 1),
            nn.BatchNorm2d(sch),
            nn.Conv2d(sch, sch, 1, 1)
        )
        self.aggregat = nn.Sequential(
            nn.Conv2d(dch, dch, 3, 1, 1),
            nn.BatchNorm2d(sch)
        )

    def forward(self, dfe, sfe):
        """
        Forward implementation
        Args:
            dfe (Tensor): feature for detail head
            sfe (Tensor): feature for semantic head

        Returns:
            Tensor:Aggregated Tensor
        """
        dpath = self.dpath(dfe)
        ddown = self.ddown(dfe)

        sup = torch.sigmoid(F.interpolate(self.sup(sfe), size=dfe.size()[2:], mode="bilinear", align_corners=False))
        spath = torch.sigmoid(self.spath(sfe))

        daggre = sup * dpath
        saggre = ddown*spath
        saggre = F.interpolate(saggre, size=daggre.size()[2:], mode="bilinear", align_corners=False)
        out = saggre + daggre
        out = self.aggregat(out)
        return out

class BiseNetV2Head(nn.Module):
    def __init__(self, in_ch, hidden_ch,num_classes):
        """
        Segmentation head of BiseNetV2
        "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
            <https://arxiv.org/pdf/2004.02147.pdf>
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
        """
        super(BiseNetV2Head, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.Conv2d(hidden_ch, num_classes, 1,  1)
        )

    def forward(self, x):
        return self.conv(x)