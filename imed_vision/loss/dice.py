# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dice
    author: 12718
    time: 2022/2/15 8:45
    tool: PyCharm
"""
import torch
import torch.nn as nn

def dice_coff(output:torch.Tensor, target:torch.Tensor, smooth=0.0, g_dice=False, eps=1e-9, p=1):
    """
    Calculate the dice coefficient
    Args:
        output (torch.Tensor): the model predict
        target (torch.Tensor): the ground truth for the image

    Returns:
        torch.Tensor: dice coefficient
    """
    ns = output.size()[1]
    if ns >= 2:
        mask = target < ns
        target = target*mask
        output = output*mask.unsqueeze(1)
        output = torch.softmax(output, dim=1)
        target_onehot = torch.zeros_like(output, device=output.device, dtype=torch.float32)
        with torch.no_grad():
            target_onehot.scatter_(1, target.long().unsqueeze(1), 1)
    else:
        target_onehot = target
        # output = torch.sigmoid(output)
    target_onehot = target_onehot.flatten(2)
    output = output.flatten(2)
    w = 1
    if g_dice:
        w = torch.sum(target_onehot, dim=-1)
        w = 1 / ((w) ** 2 + eps)
    # axis = [1, 2, 3]
    inter = w * torch.sum(output * target_onehot, dim=-1)
    if p > 1:
        union = w * torch.sum(output.pow(p)  + target_onehot.pow(1), dim=-1)
    else:
        union = w * torch.sum(output + target_onehot, dim=-1)
    _coff = (2*inter+smooth) / (union + smooth)
    return _coff

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, gdice=False, eps=1e-9, p=1):
        """
        Warp for the dice loss calculate
        Args:
            smooth (float): one factor to accelerate the convergence, default 1.0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gdice = gdice
        self.eps = eps
        self.p = p

    def forward(self, output, target):
        coff = dice_coff(output, target, smooth=self.smooth, g_dice=self.gdice, eps=self.eps, p=self.p)
        loss = 1 - coff
        return loss.mean()