# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:layernorm
    author: 12718
    time: 2022/4/11 9:17
    tool: PyCharm
    Implementation of layernorm introduce in ConvNeXt,
    ""<>
    References:https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Iterable

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, data_format="channel_first"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert data_format in ["channel_first", "channel_last"], "unsupported data format:{}, " \
                                                                 "only support channel_first " \
                                                                 "or channel_last".format(data_format)
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, Iterable) else (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channel_last":
            if hasattr(F, "layer_norm"):
                x = F.layer_norm(x, normalized_shape=self.normalized_shape, weight=self.weight,
                                 bias=self.bias, eps=self.eps)
            else:
                u = torch.mean(x, dim=-1, keepdim=True)
                v = torch.mean((x - u).pow(2), dim=-1, keepdim=True)
                x = (x - u) / (v + self.eps)
                x = x*self.weight[None, None, :] + self.bias[None, None, :]
            return x
        else:
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x