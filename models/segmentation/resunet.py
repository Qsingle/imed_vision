#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   resunet.py
    @Time    :   2023/07/05 14:42:23
    @Author  :   12718 
    @Version :   1.0
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from layers.unet_blocks import Downsample3D, ResidualDoubleConv3D, TransposeUpsample3D

class ResUNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=7, super_reso=False, fusion=False, upscale_rate=2):
        super(ResUNet3D, self).__init__()
        base_ch = 32
        self.modal1_down1 = Downsample3D(in_ch, base_ch, conv_op=ResidualDoubleConv3D)
        self.modal1_down2 = Downsample3D(base_ch, base_ch, conv_op=ResidualDoubleConv3D)
        self.modal1_down3 = Downsample3D(base_ch, base_ch*2, conv_op=ResidualDoubleConv3D)
        self.modal1_down4 = Downsample3D(base_ch*2, base_ch*4, conv_op=ResidualDoubleConv3D)
        self.modal1_down5 = ResidualDoubleConv3D(base_ch*4, base_ch*8)
        self.up5 = TransposeUpsample3D(base_ch*4, base_ch*8, base_ch*4, conv_op=ResidualDoubleConv3D)
        self.up6 = TransposeUpsample3D(base_ch*2, base_ch*4, base_ch*2, conv_op=ResidualDoubleConv3D)
        self.up7 = TransposeUpsample3D(base_ch, base_ch*2, base_ch, conv_op=ResidualDoubleConv3D)
        self.up8 = TransposeUpsample3D(base_ch, base_ch, base_ch, conv_op=ResidualDoubleConv3D)

        self.out_conv = nn.Conv3d(
            base_ch, num_classes, 1, 1
        )
        self.super_reso = super_reso
        if self.super_reso:
            self.seg_up = nn.Upsample(scale_factor=upscale_rate, mode="trilinear")
            self.upscale_rate = upscale_rate
            from layers.superpixel import PixelShuffle3d
            from layers.task_fusion import CrossGL3D
            self.sr_up5 = TransposeUpsample3D(base_ch*4, base_ch*8, base_ch*4, conv_op=ResidualDoubleConv3D)
            self.sr_up6 = TransposeUpsample3D(base_ch*2, base_ch*4, base_ch*2, conv_op=ResidualDoubleConv3D)
            self.sr_up7 = TransposeUpsample3D(base_ch, base_ch*2, base_ch, conv_op=ResidualDoubleConv3D)
            self.sr_up8 = TransposeUpsample3D(base_ch, base_ch, base_ch, conv_op=ResidualDoubleConv3D)
            self.sr_conv = nn.Sequential(
                nn.Conv3d(base_ch, (upscale_rate ** 3) * in_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

            self.sr_module = PixelShuffle3d(scale=upscale_rate)
            self.fusion = fusion
            if fusion:
                self.seg_sr_fusion = CrossGL3D(base_ch, base_ch)

    def forward(self, x):
        modal1_1, down = self.modal1_down1(x)
        modal1_2, down = self.modal1_down2(down)
        modal1_3, down = self.modal1_down3(down)
        modal1_4, down = self.modal1_down4(down)
        modal1_5 = self.modal1_down5(down)

        up5 = self.up5(modal1_4, modal1_5)
        up6 = self.up6(modal1_3, up5)
        up7 = self.up7(modal1_2, up6)
        up8 = self.up8(modal1_1, up7)
        out = self.out_conv(up8)
        if self.super_reso:
            out = self.seg_up(out)
            if self.training:
                sr_up5 = self.sr_up5(modal1_4, modal1_5)
                sr_up6 = self.sr_up6(modal1_3, sr_up5)
                sr_up7 = self.sr_up7(modal1_2, sr_up6)
                sr_up8 = self.sr_up8(modal1_1, sr_up7)
                sr = self.sr_conv(sr_up8)
                sr = self.sr_module(sr)
                fusion_seg = None
                fusion_sr = None
                if self.fusion:
                    fusion_sr, fusion_seg = self.seg_sr_fusion(sr_up8, up8)
                    fusion_sr = self.sr_conv(fusion_sr)
                    fusion_sr = self.sr_module(fusion_sr)
                    fusion_seg = self.out_conv(fusion_seg)
                    fusion_seg = self.seg_up(fusion_seg)
                    # fusion_seg = F.interpolate(fusion_seg, scale_factor=self.upscale_rate, mode="trilinear", align_corners=True)
                return out, sr, fusion_sr, fusion_seg
        return out
