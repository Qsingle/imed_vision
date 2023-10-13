# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:h_activation
    author: 12718
    time: 2022/1/15 15:06
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        """
        H-Swish activation
        Math:
            x*\frac{relu6(x+3)}{6}
        Args:
            x (tensor): input value

        Returns:
            tensor
        """
        return x * F.relu6(x + 3, inplace=True) / 6


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)