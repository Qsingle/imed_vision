# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:maf
    author: 12718
    time: 2022/11/18 16:26
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSC(nn.Module):
    def __init__(self, in_ch, groups=4):
        """

        Args:
            in_ch (int): number of channels for input
            groups (int, Optional): Number of groups, Defatults to 4.
        """
        super(SSC, self).__init__()
        assert in_ch % groups == 0
        group_ch = in_ch // groups
        self.group_ch = group_ch
        self.conv = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1, 0)
        ])
        for i in range(1, groups):
            self.conv.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, bias=False)
            )
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        groups = torch.split(x, self.group_ch, dim=1)
        features = []
        for i, group in enumerate(groups):
            features.append(self.conv[i](group))
        features = torch.cat(features, dim=1)
        features = self.bn(features)
        features += x
        features = self.relu(features)
        return features

class MAF(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_dim=32, groups=4):
        """

        Args:
            sr_ch:
            seg_ch:
            hidden_dim:
            groups:
        """
        super(MAF, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sr_ch+seg_ch, hidden_dim, 1, 1),
            SSC(hidden_dim, groups=groups)
        )
        self.sr_att = nn.Sequential(
            nn.Conv2d(hidden_dim, sr_ch, 1, 1),
            nn.Sigmoid()
        )
        self.seg_att = nn.Sequential(
            nn.Conv2d(hidden_dim, seg_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        cat = torch.cat([sr_fe, seg_fe], dim=1)
        fusion = self.fusion(cat)
        sr_att = self.sr_att(fusion)
        seg_att = self.seg_att(fusion)
        sr_out = sr_att*sr_fe + sr_fe
        seg_out = seg_att*seg_fe + seg_fe
        return sr_out, seg_out