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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
        )

        self.residual_upsampler = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.conv(input) + self.residual_upsampler(input))


class Deconv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                                  stride=(stride, stride, stride), padding=(padding, padding, padding),
                                  output_padding=0, bias=True),
            nn.LeakyReLU())

    def forward(self, x):
        return self.deconv(x)


class SubPixel_Block(nn.Module):
    def __init__(self, upscale_factor=2):
        super(SubPixel_Block, self).__init__()

        self.subpixel = nn.Sequential(
            PixelShuffle3d(upscale_factor),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.subpixel(x)


class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''

    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height,
                                             in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class PFSeg3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(PFSeg3D, self).__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.conv2 = DoubleConv(32, 32)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool3d((2, 2, 2))
        self.conv5 = DoubleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256 + 64, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.up10_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_seg = DoubleConv(32, 16)
        self.conv11_seg = nn.Conv3d(16, out_channels, 1)

        self.up6_sr = Deconv3D_Block(256 + 64, 128, 4, stride=2)
        self.conv6_sr = DoubleConv(256, 128)
        self.up7_sr = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_sr = DoubleConv(128, 64)
        self.up8_sr = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_sr = DoubleConv(64, 32)
        self.up9_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_sr = DoubleConv(64, 32)
        self.up10_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_sr = DoubleConv(32, 16)
        self.conv11_sr = nn.Conv3d(16, out_channels, 1)

        # SGM
        self.high_freq_extract = nn.Sequential(
            DoubleConv(in_channels, 16),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            DoubleConv(16, 32),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            DoubleConv(32, 64),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            DoubleConv(64, 64),
        )

    def forward(self, x, guidance):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        hfe_seg = self.high_freq_extract(guidance)
        up_6_seg = self.up6_seg(torch.cat([c5, hfe_seg], dim=1))
        merge6_seg = torch.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = torch.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = torch.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = torch.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        up_10_seg = self.up10_seg(c9_seg)
        c10_seg = self.conv10_seg(up_10_seg)
        # c11_seg = self.pointwise(c10_seg)
        c11_seg = self.conv11_seg(c10_seg)
        # out_seg = nn.Sigmoid()(c11_seg)
        out_seg = c11_seg

        hfe_sr = self.high_freq_extract(guidance)

        up_6_sr = self.up6_sr(torch.cat([c5, hfe_sr], dim=1))
        merge6_sr = torch.cat([up_6_sr, c4], dim=1)
        c6_sr = self.conv6_sr(merge6_sr)
        up_7_sr = self.up7_sr(c6_sr)
        merge7_sr = torch.cat([up_7_sr, c3], dim=1)
        c7_sr = self.conv7_sr(merge7_sr)
        up_8_sr = self.up8_sr(c7_sr)
        merge8_sr = torch.cat([up_8_sr, c2], dim=1)
        c8_sr = self.conv8_sr(merge8_sr)
        up_9_sr = self.up9_sr(c8_sr)
        merge9_sr = torch.cat([up_9_sr, c1], dim=1)
        c9_sr = self.conv9_sr(merge9_sr)
        up_10_sr = self.up10_sr(c9_sr)
        c10_sr = self.conv10_sr(up_10_sr)
        c11_sr = self.conv11_sr(c10_sr)
        out_sr = nn.ReLU()(c11_sr)

        return out_seg, out_sr