# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 16:46
    @filename: unet.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from imed_vision.layers.unet_blocks import *
from imed_vision.layers.maf import MAF
from imed_vision.layers.spatial_fusion import SpatialFusion
from imed_vision.layers.fim import FIM
from imed_vision.layers.task_fusion import CrossGL

__all__ = ["Unet", "NestedUNet", "AttUnet", "MiniUnet"]

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, out_ch1=None, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 layer_attention=False, super_reso=False, upscale_rate=2, sr_layer=4,
                 sr_seg_fusion=False, fim=False, before=True, l1=False, ss_maf=False,
                 fa=False, **kwargs):
        """
        Unet with different block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            convblock (nn.Module):
            radix (int): number of groups, default 2
            drop_prob (float): dropout rate, default 0.0
            expansion (float): expansion rate for channels, default 1.0
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer before SplAtConv
            layer_attention (bool): whether use layer attention
            multi_head (bool): whether use multi-head attention
            num_head (int): number of head for multi-head attention
            fim (bool): whether use SuperVessel
        """
        super(Unet, self).__init__()
        base_ch = kwargs.pop("base_ch", 64)
        print("Input channel", in_ch)
        out_ch1 = out_ch1 or in_ch
        self.down1 = Downsample(in_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down2 = Downsample(base_ch, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down3 = Downsample(base_ch*2, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down4 = Downsample(base_ch*4, base_ch*8, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down5 = convblock(base_ch*8, base_ch*16, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.layer_attention = layer_attention
        self.super_reso = super_reso
        #
        self.up6 = Upsample(base_ch*16, base_ch*8, base_ch*8, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up7 = Upsample(base_ch*8, base_ch*4, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up8 = Upsample(base_ch*4, base_ch*2, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

        self.up9 = Upsample(base_ch*2, base_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.upscale_rate = upscale_rate
        self.upsample_way = kwargs.pop("upsample_way", 1)
        if self.upsample_way == 2:
            self.out_up = nn.Sequential(
                nn.Conv2d(out_ch, out_ch * (upscale_rate ** 2), kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            )
        self.l1 = l1
        if super_reso:
            self.sr_layer = sr_layer
            if sr_layer == 4:
                self.sr_up6 = Upsample(base_ch*8, base_ch*4, base_ch*4, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(base_ch*4, base_ch*2, base_ch*2,radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = Upsample(base_ch*2, base_ch, base_ch, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up9 = nn.Identity()
            elif sr_layer == 5:
                self.sr_up6 = Upsample(base_ch*16, base_ch*8, base_ch*8, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(base_ch*8, base_ch*4, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = Upsample(base_ch*4, base_ch*2, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

                self.sr_up9 = Upsample(base_ch*2, base_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

            self.sr_module = nn.Sequential(
                nn.Conv2d(base_ch, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.Conv2d(32, (upscale_rate ** 2) * out_ch1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            ) if upscale_rate > 1 else nn.Conv2d(base_ch, out_ch1, 1, 1, 0)
            self.sr_seg_fusion = sr_seg_fusion
            self.before = before and not fim
            self.fim = fim
            self.fa = nn.Sequential(
                nn.Conv2d(out_ch, in_ch, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU()
            ) if fa else None
            if fa:
                self.sssr = nn.Sequential(
                    nn.ConvTranspose2d(out_ch, out_ch, 2, 2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()
                )
            else:
                self.sssr = None
            if sr_seg_fusion:
                # self.sr_seg_fusion_module = SpatialFusion(in_ch, out_ch)
                if not self.before:
                    if fim:
                        self.sr_seg_fusion_module = FIM(out_ch1, out_ch)
                    else:
                        self.sr_seg_fusion_module = CrossGL(out_ch1, out_ch, 32)
                else:
                    # self.sr_seg_fusion_module = FeatureFusion(base_ch, base_ch, 32)
                    if ss_maf:
                        self.sr_seg_fusion_module = MAF(base_ch, base_ch)
                    else:
                        self.sr_seg_fusion_module = CrossGL(base_ch, base_ch)

                # self.sr_seg_fusion_module = LinearFusion(base_ch, base_ch, 32)
                # self.sr_seg_fusion_module = LinearFusion(in_ch, out_ch)
                # self.fusion_mlp = nn.Sequential(
                #     nn.Conv2d(out_ch, 32, 1, 1),
                #     nn.ReLU(),
                #     nn.Conv2d(32, out_ch, 1, 1),
                # )

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        up6 = self.up6(down5, down4_f)
        up7 = self.up7(up6, down3_f)
        up8 = self.up8(up7, down2_f)
        up9 = self.up9(up8, down1_f)
        out = self.out_conv(up9)
        if out.size() != x.size() and not self.super_reso:
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        else:
            h, w = x.size()[2:]
            if self.upsample_way == 1:
                if self.sssr is not None:
                    out = self.sssr(out)
                else:
                    out = F.interpolate(out, size=(h * self.upscale_rate, w * self.upscale_rate), mode="bilinear",
                                align_corners=True)
            elif self.upsample_way == 2:
                out = self.out_up(out)

        fusion_seg = None
        sr = None
        fusion_sr = None
        if self.super_reso and self.training:
            if self.sr_layer == 4:
                sr_up6 = self.sr_up6(down4_f, down3_f)
                sr_up7 = self.sr_up7(sr_up6, down2_f)
                sr_up8 = self.sr_up8(sr_up7, down1_f)
                sr_up9 = self.sr_up9(sr_up8, down1_f)
            elif self.sr_layer == 5:
                sr_up6 = self.sr_up6(down5, down4_f)
                sr_up7 = self.sr_up7(sr_up6, down3_f)
                sr_up8 = self.sr_up8(sr_up7, down2_f)
                sr_up9 = self.sr_up9(sr_up8, down1_f)
            else:
                raise ValueError("Unknown sr layer number")

            sr = self.sr_module(sr_up9)
            if self.sr_seg_fusion:
                # fusion = self.sr_seg_fusion_module(sr, out)
                # fusion_seg = fusion*out + out
                if self.upsample_way == 1:
                    out = F.interpolate(out, size=(h * self.upscale_rate, w * self.upscale_rate), mode="bilinear",
                                        align_corners=True)
                elif self.upsample_way == 2:
                    out = self.out_up(out)
                if self.before:
                    fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr_up9, up9)
                    fusion_sr = self.sr_module(fusion_sr)
                    fusion_seg = self.out_conv(fusion_seg)
                    h, w = x.size()[2:]

                    if self.upsample_way == 2:
                        fusion_seg = self.out_up(fusion_seg)
                    else:
                        fusion_seg = F.interpolate(fusion_seg, size=(h * self.upscale_rate, w * self.upscale_rate),
                                                   mode="bilinear",
                                                   align_corners=True)
                else:
                    if self.fim:
                        fusion_attention = self.sr_seg_fusion_module(sr, out)
                        fusion_seg = fusion_attention*out + out
                    else:
                        fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr, out)
                    # fusion_seg = fusion_seg*out + out
                    # fusion_sr = fusion_sr*sr + sr
                # fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr, out)
                # fusion_sr = fusion*sr

        # out = torch.max(out, dim=1)[1]
        if self.super_reso and self.training:
            if self.l1:
                l1_loss = nn.functional.l1_loss
                l1 = l1_loss(sr_up6, up6) + l1_loss(sr_up7, up7) + l1_loss(sr_up8, up8) + l1_loss(sr_up9, up9)
                return out, sr, l1
            if self.fa is not None:
                fa = self.fa(out)
                return out, sr, fa
            if self.sr_seg_fusion:
                return out, sr, fusion_seg, fusion_sr
            return out, sr
        return out

class MiniUnet(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 layer_attention=False, super_reso=False, upscale_rate=2, sr_layer=4,
                 sr_seg_fusion=False, **kwargs):
        """
        Unet with different block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            convblock (nn.Module):
            radix (int): number of groups, default 2
            drop_prob (float): dropout rate, default 0.0
            expansion (float): expansion rate for channels, default 1.0
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer before SplAtConv
            layer_attention (bool): whether use layer attention
            multi_head (bool): whether use multi-head attention
            num_head (int): number of head for multi-head attention
        """
        super(MiniUnet, self).__init__()
        base_ch = kwargs.pop("base_ch", 64)
        self.down1 = Downsample(in_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down2 = Downsample(base_ch, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down3 = Downsample(base_ch*2, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down4 = convblock(base_ch*4, base_ch*8, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.layer_attention = layer_attention
        self.super_reso = super_reso
        #
        self.up6 = Upsample(base_ch*8, base_ch*4, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up7 = Upsample(base_ch*4, base_ch*2, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up8 = Upsample(base_ch*2, base_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.upscale_rate = upscale_rate
        if super_reso:
            self.sr_layer = sr_layer
            if sr_layer == 3:
                self.sr_up6 = Upsample(base_ch*4, base_ch*2, base_ch*2, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(base_ch*2, base_ch, base_ch,radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = nn.Identity()
            elif sr_layer == 4:
                self.sr_up6 = Upsample(base_ch*8, base_ch*4, base_ch*4, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(base_ch*4, base_ch*2, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = Upsample(base_ch*2, base_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

            self.sr_module = nn.Sequential(
                nn.Conv2d(base_ch, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            )
            self.sr_seg_fusion = sr_seg_fusion
            self.upsample_way = kwargs.pop("upsample_way", 1)
            if self.upsample_way == 2:
                self.out_up = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch*(upscale_rate**2), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelShuffle(upscale_factor=upscale_rate)
                )
            if sr_seg_fusion:
                self.sr_seg_fusion_module = SpatialFusion(in_ch, out_ch)
                # self.sr_seg_fusion_module = LinearFusion(in_ch, out_ch)

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4= self.down4(down3)
        up6 = self.up6(down4, down3_f)
        up7 = self.up7(up6, down2_f)
        up8 = self.up8(up7, down1_f)
        out = self.out_conv(up8)
        if out.size() != x.size() and not self.super_reso:
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        else:
            h, w = x.size()[2:]
            if self.upsample_way == 1:
                out = F.interpolate(out, size=(h * self.upscale_rate, w * self.upscale_rate), mode="bilinear",
                                align_corners=True)
            elif self.upsample_way == 2:
                out = self.out_up(out)
                # out = self.out_conv(out)

        fusion_seg = None
        sr = None
        fusion_sr = None
        if self.super_reso and self.training:
            if self.sr_layer == 3:
                sr_up6 = self.sr_up6(down3_f, down2_f)
                sr_up7 = self.sr_up7(sr_up6, down1_f)
                sr_up8 = self.sr_up8(sr_up7)
            elif self.sr_layer == 5:
                sr_up6 = self.sr_up6(down4, down3_f)
                sr_up7 = self.sr_up7(sr_up6, down2_f)
                sr_up8 = self.sr_up8(sr_up7, down1_f)
            else:
                raise ValueError("Unknown sr layer number")

            sr = self.sr_module(sr_up8)
            if self.sr_seg_fusion:
                fusion = self.sr_seg_fusion_module(sr, out)
                fusion_seg = fusion*out + out
                # fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr, out)
                # fusion_sr = fusion*sr

        if self.super_reso and self.training:
            if self.sr_seg_fusion:
                return out, sr, fusion_seg, fusion_sr
            return out, sr
        return out

class Up(nn.Module):
    def forward(self, fe, size):
        return F.interpolate(fe, size=size, mode="bilinear", align_corners=True)

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, layer_attention=False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = Up()

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.layer_attention = layer_attention

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0.shape[2:])], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0.shape[2:])], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0.shape[2:])], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0.shape[2:])], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0.shape[2:])], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0.shape[2:])], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0.shape[2:])], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0.shape[2:])], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0.shape[2:])], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, x0_0.shape[2:])], 1))
        if self.layer_attention:
            x0_4 = torch.sigmoid(x0_0) * x0_4 + x0_4
            if self.deep_supervision:
                x0_3 = torch.softmax(x0_0, dim=1)*x0_3 + x0_3
                x0_2 = torch.softmax(x0_0, dim=1)*x0_2 + x0_2
                x0_1 = torch.softmax(x0_0, dim=1)*x0_1 + x0_1

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class AttUnet(nn.Module):
    def __init__(self, in_ch, num_classes, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 layer_attention=False):
        super(AttUnet, self).__init__()
        self.down1 = Downsample(in_ch, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down2 = Downsample(64, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down3 = Downsample(128, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down4 = Downsample(256, 512, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.down5 = convblock(512, 1024, radix=radix, drop_prob=drop_prob,
                               dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                               norm_layer=norm_layer, activation=activation, expansion=expansion)
        self.layer_attention = layer_attention

        self.att6 = AttentionBlock(512, 1024, 512)
        self.up6 = convblock(512, 512)
        self.att7 = AttentionBlock(256, 512, 256)
        self.up7 = convblock(256, 256)
        self.att8 = AttentionBlock(128, 256, 128)
        self.up8 = convblock(128, 128)
        self.att9 = AttentionBlock(64, 128, 64)
        self.up9 = convblock(64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        att6 = self.att6(down4_f, down5)
        up6 = self.up6(att6)
        att7 = self.att7(down3_f, up6)
        up7 = self.up7(att7)
        att8 = self.att8(down2_f, up7)
        up8 = self.up8(att8)
        att9 = self.att9(down1_f, up8)
        up9 = self.up9(att9)
        if self.layer_attention:
            up9 = F.softmax(down1_f, dim=1) * up9 + up9
        out = self.out_conv(up9)
        return out

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=7, super_reso=False, fusion=False, upscale_rate=2, max_ch=320):
        super(UNet3D, self).__init__()
        base_ch = 32
        self.modal1_down1 = Downsample3D(in_ch, base_ch, reduction=2)
        self.modal1_down2 = Downsample3D(base_ch, base_ch*2, reduction=2)
        self.modal1_down3 = Downsample3D(base_ch*2, base_ch*4, reduction=2)
        ch = min(base_ch*8, max_ch)
        self.modal1_down4 = Downsample3D(base_ch*4, ch, reduction=2)
        ch1 = min(base_ch*16, max_ch)
        self.modal1_down5 = DoubleConv3D(ch, ch1, reduction=2)
        self.up5 = Upsample3D(ch, ch1, ch)
        self.up6 = Upsample3D(base_ch*4, ch, base_ch*4)
        self.up7 = Upsample3D(base_ch*2, base_ch*4, base_ch*2)
        self.up8 = Upsample3D(base_ch, base_ch*2, base_ch)

        self.out_conv = nn.Conv3d(
            base_ch, num_classes, 1, 1
        )
        self.super_reso = super_reso
        if self.super_reso:
            self.upscale_rate = upscale_rate
            from layers.superpixel import PixelShuffle3d
            from layers.task_fusion import CrossGL3D
            self.sr_up5 = Upsample3D(ch, ch1, ch)
            self.sr_up6 = Upsample3D(base_ch*4, ch, base_ch*4)
            self.sr_up7 = Upsample3D(base_ch*2, base_ch*4, base_ch*2)
            self.sr_up8 = Upsample3D(base_ch, base_ch*2, base_ch)
            self.sr_conv = nn.Sequential(
                nn.Conv3d(base_ch, (upscale_rate ** 3) * in_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

            self.sr_module = PixelShuffle3d(scale=upscale_rate)
            self.fusion = fusion
            if fusion:
                hidden_state = 32
                # if num_classes < 32:
                #     hidden_state = 32
                # elif num_classes < 64:
                #     hidden_state = 64
                # elif num_classes < 128:
                #     hidden_state = 128
                # elif num_classes < 256:
                #     hidden_state = 256
                # else:
                #     hidden_state = max_ch
                self.seg_sr_fusion = CrossGL3D(base_ch, base_ch, hidden_state=hidden_state)

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
            out = F.interpolate(out, scale_factor=self.upscale_rate, mode="trilinear", align_corners=True)
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
                    fusion_seg = F.interpolate(fusion_seg, scale_factor=self.upscale_rate, mode="trilinear", align_corners=True)
                return out, sr, fusion_sr, fusion_seg
        return out


if __name__ == "__main__":
    #model = Unet(3, 3, convblock=SplAtBlock, expansion=4.0, avd=True, layer_attention=True)
    #model = MultiHeadLayerAttention(3, 64)
    # model = LayerAttentionModule(3, 3, expansion=1.0)
    from scipy.ndimage.interpolation import affine_transform
    model = UNet3D()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    x = torch.randn((2, 1, 20, 32, 32))
    x1 = torch.randn((2, 1, 20, 32, 32))
    with torch.no_grad():
        out = model(x, x1, x1)
        print("Out Shape:", out.shape)