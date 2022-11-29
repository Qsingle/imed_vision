# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:mcb
    author: 12718
    time: 2022/6/28 20:19
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class MCBLoss(nn.Module):
    def __init__(self, activation=nn.Sigmoid()):
        super(MCBLoss, self).__init__()
        self.activation = activation

    def forward(self, output, target):
        bs, nc, h, w = output.shape
        output_activated = self.activation(output)
        pos_loss = torch.zeros((nc))
        neg_loss = torch.zeros((nc, 5))
        pos_count = torch.zeros((nc))
        neg_count = torch.zeros((nc, 5))
        for i in range(nc):
            pos_count[i] += torch.sum(target==(i+1))
            temp_output = output[:, i, :, :]
            temp_pos = temp_output[target==(i+1)]
            pos_loss[i] += -torch.sum(temp_pos*(0-(temp_pos>-0)) -
                                      torch.log(1+torch.exp(temp_pos-
                                    2*temp_pos*(temp_pos >= 0))))
