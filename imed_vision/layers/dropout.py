# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dropout
    author: 12718
    time: 2022/3/23 10:16
    tool: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DropoutBlock", "DropPath", "drop_path"]

class DropoutBlock(nn.Module):
    def __init__(self, drop_prob, kernel_size, gamma_scale:float=1.0,
                 batch_wise:bool=False, with_noise:bool=False):
        """
        Implementation of the dropout block.
        "DropBlock: A regularization method for convolutional networks"<https://arxiv.org/pdf/1810.12890.pdf>
        <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py>
        Args:
            drop_prob (float): rate of dropout
            kernel_size (int): size of the kernel
            gamma_scale (float): rate of the scale
            batch_wise (bool): mode of the method, batch or sample
            with_noise (bool): whether use noise
        """
        super(DropoutBlock, self).__init__()
        self.drop_prob = drop_prob
        self.kernel_size = kernel_size
        self.gama_scale = gamma_scale
        self.batch_wise = batch_wise
        self.with_noise = with_noise

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        _, c, h, w = x.shape
        total_size = w * h
        clipped_block_size = min(self.kernel_size, min(w, h))
        gamma = self.gama_scale * self.drop_prob * total_size / clipped_block_size ** 2 / (
                (w - self.kernel_size + 1) * (h - self.kernel_size + 1)
        )
        if self.batch_wise:
            # one mask for whole batch, quite a bit faster
            block_mask = torch.rand((1, c, h, w), dtype=x.dtype, device=x.device) < gamma
        else:
            # mask per batch element
            block_mask = torch.rand_like(x) < gamma

        block_mask = F.max_pool2d(
            block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
        if self.with_noise:
            normal_noise = torch.randn(size=x.shape if not self.batch_wise else (1, c, h, w))
            y = x * (1. - block_mask) + normal_noise * block_mask
        else:
            block_mask = 1 - block_mask
            normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
            y = x * block_mask * normalize_scale
        return y

def drop_path(x, dropout_rate=0., training=False, scale_by_keep: bool = True):
    """
    Implementation drop path
    Args:
        x (Tensor): input tensor
        dropout_rate (float): rate of drop out
        training (bool): whether training
    References:
        <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py>
    Returns:
        Tensor
    """
    if not training or dropout_rate == 0.:
        return x
    keep_prob = 1 - dropout_rate
    random_shape = (x.shape[0], ) + (1,) * (x.ndim -1)
    if hasattr(torch, "bernoulli"):
        random_tensor = torch.empty(random_shape, device=x.device, dtype=x.dtype)
        random_tensor = torch.bernoulli(random_tensor, p=keep_prob)
    else:
        random_tensor = torch.rand(random_shape, device=x.device, dtype=x.dtype) + keep_prob
        random_tensor.floor_()
    if scale_by_keep and keep_prob > 0.:
        random_tensor = torch.div(random_tensor, keep_prob)
    x = random_tensor * x
    return x

class DropPath(nn.Module):
    def __init__(self, dropout_rate=0., scale_by_keep=True):
        """
        Class decorator of drop path. Stochastic Depth
        Args:
            dropout_rate (float): probability of drop
            scale_by_keep (bool): whether use keep probability to scale the tensor
        """
        super(DropPath, self).__init__()
        self.drop_rate = dropout_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_rate, self.training, self.scale_by_keep)