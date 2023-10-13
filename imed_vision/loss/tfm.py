# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:tfm
    author: 12718
    time: 2022/1/30 11:06
    tool: PyCharm
"""

import torch.nn as nn
import torch.nn.functional as F

from .fa_loss import FALoss

class TFM(nn.Module):
    def __init__(self, subscale_rate=1/32):
        super(TFM, self).__init__()
        self.sub_scale_rate = subscale_rate

    def forward(self, f1, f2, gt):
        fusion = f1*f2
        tfl = F.mse_loss(fusion, gt)
        ssl = FALoss(self.sub_scale_rate)(fusion, gt)
        loss = tfl + ssl
        return loss