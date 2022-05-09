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

from layers.unet_blocks import *
from layers.superpixel import SuperResolutionModule
from layers.spatial_fusion import SpatialFusion
from layers.task_fusion import LinearFusion

__all__ = ["Unet", "NestedUNet", "AttUnet"]

class Unet(nn.Module):
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
        super(Unet, self).__init__()
        self.down1 = Downsample(in_ch, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down2 = Downsample(64, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down3 = Downsample(128, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down4 = Downsample(256, 512, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.down5 = convblock(512, 1024, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.layer_attention = layer_attention
        self.super_reso = super_reso
        if super_reso:
            self.sup = SuperResolutionModule(64)
            self.sup_conv = convblock(3, 64, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
            self.sup_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.up6 = Upsample(1024, 512, 512, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up7 = Upsample(512, 256, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        self.up8 = Upsample(256, 128, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

        self.up9 = Upsample(128, 64, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

        self.upscale_rate = upscale_rate
        if super_reso:
            self.sr_layer = sr_layer
            if sr_layer == 4:
                self.sr_up6 = Upsample(512, 256, 256, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(256, 128, 128,radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = Upsample(128, 64, 64, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up9 = nn.Identity()
            elif sr_layer == 5:
                self.sr_up6 = Upsample(1024, 512, 512, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up7 = Upsample(512, 256, 256, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
                self.sr_up8 = Upsample(256, 128, 128, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

                self.sr_up9 = Upsample(128, 64, 64, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                    dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                    norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)

            self.sr_module = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            )
            self.sr_seg_fusion = sr_seg_fusion
            if sr_seg_fusion:
                # self.sr_seg_fusion_module = SpatialFusion(in_ch, out_ch)
                self.sr_seg_fusion_module = LinearFusion(64, 64)
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
            out = F.interpolate(out, size=(h * self.upscale_rate, w * self.upscale_rate), mode="bilinear",
                                align_corners=True)
        fusion_seg = None
        sr = None
        fusion_sr = None
        if self.super_reso and self.training:
            if self.sr_layer == 4:
                sr_up6 = self.sr_up6(down4_f, down3_f)
                sr_up7 = self.sr_up7(sr_up6, down2_f)
                sr_up8 = self.sr_up8(sr_up7, down1_f)
                sr_up9 = self.sr_up9(sr_up8)
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
                fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr_up9, up9)
                # fusion_sr = fusion*sr

        # out = torch.max(out, dim=1)[1]
        if self.super_reso and self.training:
            if self.sr_seg_fusion:
                return out, sr, fusion_seg, fusion_sr
            return out, sr
        return out



class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, layer_attention=False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
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




class UNet3DMultiModal(nn.Module):
    def __init__(self, in_ch=1, num_classes=7):
        super(UNet3DMultiModal, self).__init__()
        self.modal1_down1 = Downsample3D(in_ch, 64)
        self.modal1_down2 = Downsample3D(64, 128)
        self.modal1_down3 = Downsample3D(128, 256)
        self.modal1_down4 = DoubleConv3D(256, 512)
        self.up5 = Upsample3D(256, 512, 256)
        self.up6 = Upsample3D(128, 256, 128)
        self.up7 = Upsample3D(64, 128, 64)

        self.out_conv = nn.Conv3d(
            64, num_classes, 1, 1
        )

    def forward(self, x):
        modal1_1, modal1_down = self.modal1_down1(x)
        modal1_2, modal1_down = self.modal1_down2(modal1_down)
        modal1_3, modal1_down = self.modal1_down3(modal1_down)
        modal1_4 = self.modal1_down4(modal1_down)

        up5 = self.up5(modal1_3, modal1_4)
        up6 = self.up6(modal1_2, up5)
        up7 = self.up7(modal1_1, up6)
        out = self.out_conv(up7)
        return out





if __name__ == "__main__":
    #model = Unet(3, 3, convblock=SplAtBlock, expansion=4.0, avd=True, layer_attention=True)
    #model = MultiHeadLayerAttention(3, 64)
    # model = LayerAttentionModule(3, 3, expansion=1.0)
    model = UNet3DMultiModal()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    x = torch.randn((2, 1, 20, 32, 32))
    x1 = torch.randn((2, 1, 20, 32, 32))
    with torch.no_grad():
        out = model(x, x1, x1)
        print("Out Shape:", out.shape)