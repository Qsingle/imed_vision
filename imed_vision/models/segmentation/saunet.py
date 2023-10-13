# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:saunet
    author: 12718
    time: 2021/12/19 16:17
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


from imed_vision.layers import DropoutBlock

class DConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=1.0,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(),
                 dropout=0.12, drop_block_size=11):
        super(DConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            DropoutBlock(dropout, kernel_size=drop_block_size),
            norm_layer(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            DropoutBlock(dropout, kernel_size=drop_block_size),
            norm_layer(out_ch),
            activation
        )

    def forward(self, x):
        net = self.conv(x)
        return net

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)

    def forward(self, x):
        bs, _, h, w = x.shape
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg, max], dim=1)
        net = self.conv(concat)
        net = torch.sigmoid(net) * x
        return net


class SAUnet(nn.Module):
    def __init__(self, in_ch, num_classes=2, keep_prob=0.82, drop_block_size=11):
        """
        Implementation of the SA-Unet.
        "SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation"<https://arxiv.org/abs/2004.03696v3>
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            keep_prob (float): keep rate for the conv layer, dropout rate = 1 - keep_prob
            drop_block_size (int): kernel size of the dropout block
        """
        super(SAUnet, self).__init__()
        drop_prob = 1 - keep_prob
        base_ch = 16
        self.encoder1 = DConv(in_ch, base_ch, dropout=drop_prob, drop_block_size=drop_block_size)
        self.encoder2 = DConv(base_ch, base_ch*2, dropout=drop_prob, drop_block_size=drop_block_size)
        self.encoder3 = DConv(base_ch*2, base_ch*4, dropout=drop_prob, drop_block_size=drop_block_size)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 3, 1, 1),
            DropoutBlock(drop_prob, kernel_size=drop_block_size),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True),
            SA(),
            nn.Conv2d(base_ch*8, base_ch*8, 3, 1, 1),
            DropoutBlock(drop_prob, kernel_size=drop_block_size),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv5 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder5 = DConv(base_ch*8, base_ch*4, dropout=drop_prob, drop_block_size=drop_block_size)

        self.deconv6 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder6 = DConv(base_ch*4, base_ch*2)

        self.deconv7 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder7 = DConv(base_ch*2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down1_0 = self.encoder1(x)
        down = self.maxpool(down1_0)
        down2_0 = self.encoder2(down)
        down = self.maxpool(down2_0)
        down3_0 = self.encoder3(down)
        down = self.maxpool(down3_0)
        down4 = self.encoder4(down)
        up5 = self.deconv5(down4)
        up5 = torch.cat([down3_0, up5], dim=1)
        up5 = self.decoder5(up5)
        up6 = self.deconv6(up5)
        up6 = torch.cat([down2_0, up6], dim=1)
        up6 = self.decoder6(up6)
        up7 = self.deconv7(up6)
        up7 = torch.cat([down1_0, up7], dim=1)
        up7 = self.decoder7(up7)
        out = self.out_conv(up7)
        return out

if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    model = SAUnet(3, 2)
    out = model(x)
    print(out.shape)