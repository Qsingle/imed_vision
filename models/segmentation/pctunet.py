# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gmlp_net
    author: 12718
    time: 2022/10/25 10:43
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.pct import PCTLayer
from layers.utils import Conv2d
from layers.unet_blocks import DoubleConv
from comm.helper import _pair

class PCTConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, **kwargs):
        super(PCTConv, self).__init__()
        self.conv = DoubleConv(in_ch=in_ch, out_ch=out_ch, norm_layer=norm_layer, activation=activation)
        img_size = kwargs.get("img_size", None)
        if img_size is None:
            raise ValueError("The size of image cannot be none")
        self.pct_layer = PCTLayer(img_size, out_ch)

    def forward(self, x):
        net = self.conv(x)
        net = self.pct_layer(net)
        return net

class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2,out_ch, convblock=DoubleConv,
                 radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2, **kwargs):
        """
        Upsample part with different conv block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch1 (int): number of channels for inputs1 (upsampled from last level)
            in_ch2 (int): number of channels for inputs2 (same level features)
            out_ch (int): number of channels for outputs
            convblock (nn.Module): block to extract features
            expansion (float):  expansion rate for channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (float): dilation rate for conv
            padding (Union[int, tuple]): the padding size
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(Upsample, self).__init__()
        self.upsample_conv = Conv2d(in_ch1, out_ch, norm_layer=norm_layer, activation=activation)
        self.conv = convblock(out_ch+in_ch2, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction, expansion=expansion, num_current=num_current, **kwargs)

    def forward(self, x, x1):
        net = F.interpolate(x, size=x1.size()[2:], mode="bilinear", align_corners=True)
        net = self.upsample_conv(net)
        net = torch.cat([net, x1], dim=1)
        net = self.conv(net)
        return net

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,radix=2, drop_prob=0.0,
                 dilation=1, padding=1, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True),avd=False, avd_first=False, num_current=2, **kwargs):
        """
        Downsample part with different block.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for inputs
            out_ch (int): number of channels for outputs
            convblock (nn.Module): block to extract features
            expansion (float):  expansion rate for channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (float): dilation rate for conv
            padding (Union[int, tuple]): the padding size
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """

        super(Downsample, self).__init__()
        self.conv = convblock(in_ch, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction, expansion=expansion, num_current=num_current, **kwargs)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feature = self.conv(x)
        downsample = self.down(feature)
        return feature, downsample

class PCTUnet(nn.Module):
    def __init__(self, img_size, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,
                 radix=2, drop_prob=0.0, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False,
                 layer_attention=False, super_reso=False, upscale_rate=2, sr_layer=4,
                 sr_seg_fusion=False, fim=False, before=True, l1=False, **kwargs):
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
        super(PCTUnet, self).__init__()
        base_ch = kwargs.pop("base_ch", 64)
        img_size = _pair(img_size)
        self.down1 = Downsample(in_ch, base_ch, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        img_size = [s // 2 for s in img_size]
        self.down2 = Downsample(base_ch, base_ch*2, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        img_size = [s // 2 for s in img_size]
        self.down3 = Downsample(base_ch*2, base_ch*4, convblock=convblock, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction,avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        img_size = [s // 2 for s in img_size]
        kwargs["img_size"] = img_size
        self.down4 = Downsample(base_ch*4, base_ch*8, convblock=PCTConv, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        img_size = [s // 2 for s in img_size]
        kwargs["img_size"] = img_size
        self.down5 = PCTConv(base_ch*8, base_ch*16, **kwargs)
        self.layer_attention = layer_attention
        self.super_reso = super_reso
        #
        img_size = [s * 2 for s in img_size]
        kwargs["img_size"] = img_size
        self.up6 = Upsample(base_ch*16, base_ch*8, base_ch*8, convblock=PCTConv, radix=radix, drop_prob=drop_prob,
                                dilation=1, padding=1, reduction=reduction, avd=avd, avd_first=avd_first,
                                norm_layer=norm_layer, activation=activation, expansion=expansion, **kwargs)
        img_size = [s * 2 for s in img_size]
        kwargs["img_size"] = img_size
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
        if out.size() != x.size():
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        return out