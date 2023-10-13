# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:detail_loss
    author: 12718
    time: 2022/5/19 14:37
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import DiceLoss

def laplacianconv(x, stride=1):
    kernel = torch.tensor([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32)
    kernel.unsqueeze_(0).unsqueeze_(0)
    kernel.requires_grad_(False)
    kernel = kernel.to(x.device, dtype=x.dtype)
    out = F.conv2d(x, kernel, stride=stride, padding=1)
    return out

class LaplacianConv(nn.Module):
    def __init__(self, stride=1):
        super(LaplacianConv, self).__init__()
        self.stride = stride

    def forward(self, x):
        return laplacianconv(x, stride=self.stride)

def get_boundary(x, stride=1):
    boundary = laplacianconv(x, stride)
    boundary = torch.clamp(boundary, min=0)
    boundary = torch.where(boundary > 0.1, 1, 0).to(dtype=torch.float32)
    return boundary

class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.BCELoss()
        self.conv = nn.Conv2d(3, 1, 1, 1)

    def forward(self, output, target):
        with torch.no_grad():
            if target.ndim < 4:
                tmp_target = target.unsqueeze(1).clone().to(dtype=torch.float32)
            else:
                tmp_target = target.clone().to(dtype=torch.float32)
            boundary_1 = get_boundary(tmp_target)
            boundary_2 = get_boundary(tmp_target, 2)
            boundary_2 = F.interpolate(boundary_2, size=boundary_1.size()[2:])
            boundary_4 = get_boundary(tmp_target, 4)
            boundary_4 = F.interpolate(boundary_4, size=boundary_1.size()[2:])
            boundary = torch.cat([boundary_1, boundary_2, boundary_4], dim=1)
        target_boundary = self.conv(boundary)
        dice = self.dice(output, target_boundary)
        ce = self.dice(output, target_boundary)
        loss = dice + ce
        return loss



if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    out = laplacianconv(x, 1)
    print(out.shape)