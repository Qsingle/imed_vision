# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:ce
    author: 12718
    time: 2022/5/30 10:51
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBCE(nn.Module):
    def forward(self, output, target):
        ns = output.size()[1]
        with torch.no_grad():
            target_onehot = F.one_hot(target, ns).permute(0, 3, 1, 2)
            weights = torch.sum(target_onehot, dim=[0, 2, 3]).float()
        loss = F.cross_entropy(output, target, weight=weights)
        return loss