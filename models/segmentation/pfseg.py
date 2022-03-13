#-*- coding:utf-8 -*-
#!/usr/bin/env python
"""
    Author: Zhongxi Qiu
    Filename: pfseg.py
    Time: 2022.02.09 17:30
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm(out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm(out_ch)
        )
        self.relu = nn.LeakyReLU()
        # self.identity = M.Identity()
        # if in_ch != out_ch:
        self.identity = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm(out_ch)
        )

    def forward(self, x):
        identity = self.identity(x)
        net = self.double_conv(x) + identity
        net = self.relu(net)
        return net

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample, self).__init__()
        self.conv = ResidualDoubleConv(in_ch, out_ch)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down_pre = self.conv(x)
        down = self.down(down_pre)
        return down_pre, down

class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch,
                 bilinear=True
                 ):
        super(Upsample, self).__init__()
        self.conv = ResidualDoubleConv(in_ch2*2, out_ch)
        self.bilinear = bilinear
        if not bilinear:
            self.upsample = nn.ConvTranspose2d(in_ch1, in_ch2, kernel_size=4, stride=2)
        else:
            self.upsamle_conv = nn.Sequential(
                nn.Conv2d(in_ch1, in_ch2, 1, 1, bias=False),
                nn.LeakyReLU()
            )


    def forward(self, x1, x2):
        if self.bilinear:
            upsample = F.interpolate(x1, size=x2.shape[2:], align_corners=True)
            upsample = self.upsamle_conv(upsample)
        else:
            upsample = self.upsample(x1)
            # upsample = F.nn.pad(upsample, ((0, 0), (0, 0), (1, 0), (1,0)))
        cat = torch.cat([upsample, x2], axis=1)
        out = self.conv(cat)
        return out

class PFSeg(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(PFSeg, self).__init__()
        self.down1 = Downsample(in_ch, 32)
        self.down2 = Downsample(32, 32)
        self.down3 = Downsample(32, 64)
        self.down4 = Downsample(64, 128)
        self.down5 = ResidualDoubleConv(128, 256)

        self.up6 = Upsample(256+64, 128, 128)
        self.up7 = Upsample(128, 64, 64)
        self.up8 = Upsample(64, 32, 32)
        self.up9 = Upsample(32, 32, 32)
        self.up10_conv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.LeakyReLU()
        )
        self.up10 = ResidualDoubleConv(32, 16)
        self.out_conv = nn.Conv2d(16, num_classes, 1)

        self.sr_up6 = Upsample(256 + 64, 128, 128)
        self.sr_up7 = Upsample(128, 64, 64)
        self.sr_up8 = Upsample(64, 32, 32)
        self.sr_up9 = Upsample(32, 32, 32)
        self.sr_up10_conv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.LeakyReLU()
        )
        self.sr_up10 = ResidualDoubleConv(32, 16)
        self.out_sr = nn.Conv2d(
            16, in_ch, 1, 1
        )

        self.high_freq_extract = nn.Sequential(
            ResidualDoubleConv(in_ch, 16),
            nn.MaxPool2d(2, 2),
            ResidualDoubleConv(16, 32),
            nn.MaxPool2d(2, 2),
            ResidualDoubleConv(32, 64),
            nn.MaxPool2d(2, 2),
            ResidualDoubleConv(64, 64)
        )

    def forward(self, x, guidance):
        down1_0, down1 = self.down1(x)
        down2_0, down2 = self.down2(down1)
        down3_0, down3 = self.down3(down2)
        down4_0, down4 = self.down4(down3)
        down5 = self.down5(down4)
        hfe_seg = self.high_freq_extract(guidance)
        up6 = self.up6(torch.cat([down5, hfe_seg], dim=1), down4_0)
        up7 = self.up7(up6, down3_0)
        up8 = self.up8(up7, down2_0)
        up9 = self.up9(up8, down1_0)
        up9 = F.interpolate(up9, scale_factor=2, mode="bilinear", align_corners=True)
        up10 = self.up10_conv(up9)
        up10 = self.up10(up10)
        out = self.out_conv(up10)

        hfe_sr = self.high_freq_extract(guidance)
        hr_up6 = self.sr_up6(torch.cat([down5, hfe_sr], dim=1), down4_0)
        hr_up7 = self.sr_up7(hr_up6, down3_0)
        hr_up8 = self.sr_up8(hr_up7, down2_0)
        hr_up9 = self.sr_up9(hr_up8, down1_0)
        hr_up9 = F.interpolate(hr_up9, scale_factor=2, mode="bilinear", align_corners=True)
        hr_fe = self.sr_up10_conv(hr_up9)
        hr_fe = self.sr_up10(hr_fe)
        # up9, hr_fe = self.query_module(hr_fe, up9)
        hr = self.out_sr(hr_fe)
        return out, hr