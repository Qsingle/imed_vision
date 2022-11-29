# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:efficientnet
    author: 12718
    time: 2022/9/19 14:37
    tool: PyCharm
"""

import torch
import torch.nn as nn

from layers.mbconv import create_conv2d, MBConv

def make_divisible(v, divisor=8, min_value=None, low_limit=0.9):
    """
    Make the value divided by divisor.
    Args:
        v (int): value
        divisor (int): divide factor
        min_value (int): Minimal value
        low_limit (float): minimal scale rate

    Returns:
        int: value after modify
    """
    min_value = min_value or divisor
    new_v = max(min_value, (v+divisor//2)//divisor*divisor)
    if new_v < v*low_limit:
        new_v += divisor
    return new_v

def multiply_channel(ch, multiplier=1.0, divisor=8, min_value=None, low_limit=0.9):
    if not multiplier:
        return ch
    new_ch = make_divisible(int(ch*multiplier), divisor, min_value, low_limit)
    return new_ch

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()