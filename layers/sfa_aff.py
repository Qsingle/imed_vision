# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:sfa_aff
    author: 12718
    time: 2022/1/17 10:55
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class SFA(nn.Module):
    def __init__(self, in_ch):
        """
            Implementation of Scale-aware feature aggregation module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>

            Parameters:
                in_ch (int): number of input channels
        """
        super(SFA, self).__init__()
        self.dilation_1 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=1, dilation=1, bias=False)
        self.dilation_3 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=3, dilation=3, bias=False)
        self.dilation_5 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=5, dilation=5, bias=False)

        self.fusion_12 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.fusion_23 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.att_fusion = nn.Conv2d(in_ch, in_ch, 1, 1)


    def forward(self, x):
        f1 = self.dilation_1(x)
        f2 = self.dilation_3(x)
        f3 = self.dilation_5(x)

        f12 = torch.cat([f1, f2], dim=1)
        f23 = torch.cat([f2, f3], dim=1)

        fusion_12 = self.fusion_12(f12)
        fusion_23 = self.fusion_23(f23)

        att_12 = torch.softmax(fusion_12, dim=1)
        w_alpha1, w_beta1 = torch.split(att_12, 1, dim=1)

        att_23 = torch.softmax(fusion_23, dim=1)
        w_alpha2, w_beta2 = torch.split(att_23, 1, dim=1)

        att_1 = w_alpha1*f1 + w_beta1*f2
        att_2 = w_alpha2*f2 + w_beta2*f3
        out = att_1 + att_2 + x
        out = self.att_fusion(out)
        return out

class AFF(nn.Module):
    def __init__(self, in_ch1, in_ch2, reduction=16):
        """
            Implementation of Adaptive feature fusion module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>
            Parameters
            ----------
            in_ch1 (int): number of channels of low-level feature
            in_ch2 (int): number of channels of high-level feature
            reduction (int): reduction rate for squeeze
        """
        super(AFF, self).__init__()
        hidden_ch = in_ch1 + in_ch2
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_ch, hidden_ch // reduction, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch // reduction, hidden_ch, 1, 1),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(hidden_ch, in_ch1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, low_fea, high_fea):
        """

        Parameters
        ----------
        low_fea (Tensor): low level feature, (n,c,h,w)
        high_fea (Tensor): high level feature, (n,c,h,w)

        Returns
        -------
            Tensor, fused feature
        """
        ch = low_fea.size()[1]
        fusion = torch.cat([high_fea, low_fea], dim=1)
        sq_ex = self.mlp(fusion)
        sq_ex = sq_ex * fusion
        sq_ex = self.conv1(sq_ex)
        gap = self.gap(sq_ex)
        att = torch.sigmoid(gap)
        low_fea = low_fea*weight
        out = low_fea + high_fea
        return out