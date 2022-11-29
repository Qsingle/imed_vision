# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:maxim
    author: 12718
    time: 2022/8/28 15:52
    tool: PyCharm
"""
import torch
import torch.nn as nn

from layers.gmlp import MAB, CGB
from layers.layernorm import LayerNorm

class RCAB(nn.Module):
    def __init__(self, ch, ksize, reduction=4):
        """
        Implementation of Residual Channel Attention Block
        Args:
            ch (int): number of channels
            reduction (int): reduction rate 
        """
        super(RCAB, self).__init__()
        self.norm = LayerNorm(ch)
        self.conv1 = nn.Conv2d(ch, ch, ksize, 1, padding=(ksize-1)//2)
        self.conv2 = nn.Conv2d(ch, ch, ksize, 1, padding=(ksize-1)//2)
        self.activate = nn.LeakyReLU(negative_slope=0.2)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1, 1),
            nn.ReLU(),
            nn.Conv2d(ch // reduction, ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        net = self.norm(x)
        net = self.conv1(net)
        net = self.activate(net)
        net = self.conv2(net)
        net = self.channel_attention(net)*net
        net = net + x
        return net


class Encoder(nn.Module):
    def __init__(self, img_size, in_ch, out_ch, patch_size=(16, 16), grid_size=(16, 16),
                 downsample=False, reduction=4, expansion_rate=2, gmlp_expansion_rate=2,
                 rcab_ksize=3, num_blocks=2):
        """

        Args:
            img_size:
            in_ch:
            out_ch:
            patch_size:
            grid_size:
            downsample:
            reduction:
            expansion_rate:
            gmlp_expansion_rate:
            rcab_ksize:
            num_blocks:
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        blcoks = []
        for i in range(num_blocks):
            blcoks.append(
                nn.Sequential(
                    MAB(out_ch, img_size, patch_size=patch_size, grid_size=grid_size,
                               expansion_rate=expansion_rate, gmlp_expansion_rate=gmlp_expansion_rate),
                    RCAB(out_ch, rcab_ksize, reduction)
                )
            )
        self.block = nn.Sequential(*blcoks)
        self.down_conv = nn.Conv2d(out_ch, out_ch, 3, 2, padding=1) if downsample else nn.Identity()

    def forward(self, x):
        net = self.conv1(x)
        net = self.block(net)
        net = self.down_conv(net)
        return net

class Decoder(nn.Module):
    def __init__(self, img_size, in_ch, out_ch, patch_size=(16, 16), grid_size=(16, 16),
                 reduction=4, expansion_rate=2, gmlp_expansion_rate=2,
                 rcab_ksize=3, num_blocks=2):
        """

        Args:
            img_size:
            in_ch:
            out_ch:
            patch_size:
            grid_size:
            reduction:
            expansion_rate:
            gmlp_expansion_rate:
            rcab_ksize:
            num_blocks:
        """
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.block = Encoder(img_size, out_ch, out_ch, patch_size=patch_size, grid_size=grid_size,
                             reduction=reduction, expansion_rate=expansion_rate, gmlp_expansion_rate=gmlp_expansion_rate,
                             rcab_ksize=rcab_ksize, num_blocks=num_blocks)

    def forward(self, x):
        net = self.up_conv(x)
        net = self.block(net)
        return net

class MAXIM(nn.Module):
    def __init__(self, in_ch, out_ch, img_size,
                 patch_sizes=[(16, 16), (16, 16), (8, 8)],
                 grid_sizes=[(16, 16), (16, 16), (8, 8)]
                 ):
        super(MAXIM, self).__init__()