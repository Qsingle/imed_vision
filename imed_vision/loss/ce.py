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
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self,
                 size_average: Optional[bool] = None,
                 ignore_index: int = -100,
                 reduce: Optional[bool] = None,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0,
                 gamma=2,
                 alpha=0.25,
                 cross_entropy=F.cross_entropy
                 ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.cross_entropy = cross_entropy

    def forward(self, output, target, weight=None):
        logpt = - self.cross_entropy(output, target, reduction='none', weight=weight,
                             label_smoothing=self.label_smoothing, size_average=self.size_average,
                             ignore_index=self.ignore_index, reduce=self.reduce,
                             )
        pt = torch.exp(logpt)
        focal_loss = -((1-pt)**self.gamma) * logpt
        loss = self.alpha*focal_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            if self.reduction is None or self.reduction == "none":
                return loss
            else:
                raise ValueError("Unsupported reduction way {}".format(self.reduction))
        return loss


class CBCE(nn.Module):
    def __init__(self, beta=0.9999,
                 size_average: Optional[bool] = None,
                 ignore_index: int = -100,
                 reduce: Optional[bool] = None,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0,
                 ):
        super(CBCE, self).__init__()
        self.beta = beta
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        weights = []
        for ni in range(input.size(1)):
            nums_i = torch.sum(target == ni)
            weights.append((1-self.beta) / (1-torch.pow(self.beta, nums_i)))
        # input = torch.log_softmax(input, dim=1)
        # loss = F.nll_loss(input, target, torch.tensor(weights).to(target.device))
        loss = F.cross_entropy(input, target, weight=torch.tensor(weights).to(target.device),
                               size_average=self.size_average,
                               ignore_index=self.ignore_index,
                               reduce=self.reduce,
                               reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
        return loss


if __name__ == '__main__':
    a=torch.tensor([[1,1],[0,1]])
    print(a.view(1,-1).size())
    print(FocalLoss()(torch.randn(2, 2).to(torch.float32), a.to(torch.float32)))