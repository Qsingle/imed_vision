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

def dice_coff(output:torch.Tensor, target:torch.Tensor, smooth=0.0):
    """
    Calculate the dice coefficient
    Args:
        output (torch.Tensor): the model predict
        target (torch.Tensor): the ground truth for the image

    Returns:
        torch.Tensor: dice coefficient
    """
    ns = output.size()[1]
    if ns > 1:
        target_onehot = torch.zeros_like(output, device=output.device, dtype=torch.float32)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        output = torch.softmax(output, dim=1)
        axis = [1, 2, 3]
        inter = torch.sum(output * target_onehot, dim=axis)
        union = torch.sum(output + target_onehot, dim=axis)
    else:
        target_onehot = target.unsqueeze(1)
        output = torch.sigmoid(output)
        inter = target_onehot * output
        union = target_onehot + output

    _coff = (2*inter+smooth) / (union + smooth)
    return _coff

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Warp for the dice loss calculate
        Args:
            smooth (float): one factor to accelerate the convergence, default 1.0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        coff = dice_coff(output, target, smooth=self.smooth)
        loss = 1 - torch.mean(coff)
        return loss