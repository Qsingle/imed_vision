# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:ibloss
    author: 12718
    time: 2022/4/20 9:14
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

__all__ = ["ibloss", "IBLoss"]

def off_diagonal(x:torch.Tensor):
    """
    Get the off-diagonal
    Args:
        x (Tensor): a square matrix

    Returns:
        Tensor: off diagonal elements
    """
    n, m = x.shape
    assert n == m, "The matrix must be a square matrix, but got {} and {}".format(n, m)
    return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()

def ibloss(x1:torch.Tensor, x2:torch.Tensor, lambd=0.0051,eps=1e-6):
    """
    Implementation of the information bottleneck loss in Barlow Twins
    References:
        "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
        <https://arxiv.org/abs/2103.03230>
    code:
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    Args:
        x1 (Tensor): representation tensor
        x2 (Tensor): representation tensor
        lambd (float):  a constant value to make trade of invariance term and redundancy reduction term
        eps (float): a small constant value to avoid divide by zero

    Returns:
        loss scale
    """
    x1 = (x1 - x1.mean()) / (x1.std() + eps)
    x2 = (x2 - x2.mean()) / (x2.std() + eps)
    N = x1.size(0) #batch size
    c = torch.mm(x1.T, x2) / N #DxD
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # invariance term
    off_diag = off_diagonal(c).pow_(2).sum() # redundancy reduction term
    loss = on_diag + lambd * off_diag
    return loss

class IBLoss(nn.Module):
    def __init__(self, lambd=0.0051, eps=1e-6):
        super(IBLoss, self).__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, x1, x2):
        return ibloss(x1, x2, self.lambd, self.eps)