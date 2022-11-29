# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:patch_embd
    author: 12718
    time: 2022/8/28 11:32
    tool: PyCharm
"""

import torch.nn as nn
from comm.helper import to_tuple

__all__ = ["PatchEmbedding"]

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, embd_dim, img_size=(224, 224), patch_size=(16, 16)):
        super(PatchEmbedding, self).__init__()
        self.img_size = to_tuple(img_size, 2)
        self.patch_size = to_tuple(patch_size, 2)
        assert self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0, \
            "The image must be divided into {} patch size".format(patch_size)
        self.num_patchs = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_ch, embd_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patchs = self.proj(x)
        return patchs