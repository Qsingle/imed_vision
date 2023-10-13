# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:spatial_fusion
    author: 12718
    time: 2022/1/15 17:45
    tool: PyCharm
"""

import torch
import torch.nn as nn



class SpatialFusion(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32):
        super(SpatialFusion, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(sr_ch + seg_ch, hidden_state, 1, 1),
            nn.ReLU()
        )
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, (7, 1), (1, 1), padding=(3, 0)),
        )

        self.fusion_2 = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, (1, 7), (1, 1), padding=(0, 3))
        )
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_state, seg_ch, 1, 1),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        proj = self.proj(torch.cat([sr_fe, seg_fe], dim=1))
        fusion_1 = self.fusion_1(proj)
        fusion_2 = self.fusion_2(proj)
        fusion = self.fusion(fusion_1+fusion_2)
        return fusion