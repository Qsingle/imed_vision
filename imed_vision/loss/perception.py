# -*- coding:utf-8 -*-
"""
    FileName: perception
    Author: 12718
    Create Time: 2023-06-01 10:30
"""
import torch
import torch.nn as nn

class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()