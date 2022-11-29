# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:unet_blocks
    author: 12718
    time: 2022/1/15 14:52
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv2d
from .splat import SplAtConv2d
from .dropout import DropPath
try:
    from torch.nn import LayerNorm
except:
    from .layernorm import LayerNorm

__all__ = ["DoubleConv3D", "Downsample3D", "Upsample3D", "AttentionBlock", "VGGBlock", "Upsample",
           "DoubleConv", "Downsample", "SplAtBlock", "RRBlock", "ResBlock", "ConvNeXtBlock"]

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1):
        super(DoubleConv3D, self).__init__()
        hidden_ch = out_ch // reduction
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample3D, self).__init__()
        self.conv = DoubleConv3D(in_ch, out_ch, reduction=2)
        self.down = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        net = self.conv(x)
        down = self.down(net)
        return net, down

class Upsample3D(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(Upsample3D, self).__init__()
        self.conv = DoubleConv3D(in_ch1+in_ch2, out_ch)


    def forward(self, x1, x2):
        up = F.interpolate(x2, x1.size()[2:], mode="trilinear", align_corners=True)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net

class AttentionBlock(nn.Module):
    def __init__(self, in_ch_g, in_ch_l, out_ch):
        super(AttentionBlock, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch_l, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_g = Conv2d(in_ch_g, out_ch, ksize=1, stride=1, padding=0)
        self.conv_l = Conv2d(out_ch, out_ch, ksize=1, stride=1, padding=0)
        self.psi = Conv2d(out_ch, 1, activation=nn.Sigmoid())

    def forward(self, x1, x2):
        x2 = self.up_conv(x2)
        x2 = self.conv_l(x2)
        x1 = self.conv_g(x1)
        net = x1+x2
        net = F.relu(net, inplace=True)
        psi = self.psi(net)
        return net * psi


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, **kwargs):
        """
        Implmentation of Unet's double 3x3 conv.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"<https://arxiv.org/pdf/1505.04597.pdf>
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
        """
        super(DoubleConv, self).__init__()
        expansion_out = int(round(out_ch*expansion))
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv2d(in_ch, expansion_out, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)
        self.conv2 = Conv2d(expansion_out, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        return net


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, expansion=1.0,radix=2, drop_prob=0.0,
                 dilation=1, padding=1, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True),avd=False, avd_first=False, num_current=2):
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
                              reduction=reduction, expansion=expansion, num_current=num_current)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feature = self.conv(x)
        downsample = self.down(feature)
        return feature, downsample


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


class  ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=1.0,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2, **kwargs):
        """
        Implementation of ResNet's block.
        References:
            "Deep Residual Learning for Image Recognition"<https://arxiv.org/pdf/1512.03385.pdf>
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(ResBlock, self).__init__()
        expansion_out = int(round(out_ch * expansion))
        self.conv1 = Conv2d(in_ch, expansion_out, ksize=3, stride=1,
                            padding=dilation, dilation=dilation,
                            activation=activation, norm_layer=norm_layer)
        self.conv2 = Conv2d(expansion_out,  out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            activation=None, norm_layer=norm_layer)
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0,
                            activation=None, norm_layer=norm_layer)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcut(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net

class RConv(nn.Module):
    def __init__(self, out_ch, num_recurrent=2, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), **kwargs):
        """
        Recurrent conv.
        References:
            "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>
        Args:
            out_ch (int): number of output channels
            num_recurrent (int): times or recurrent
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
        """
        super(RConv, self).__init__()
        self.conv = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation)
        self.num_recurrent = num_recurrent

    def forward(self, x):
        x1 = x
        for i in range(self.num_recurrent):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class RRBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_current=2, radix=2, drop_prob=0.0, dilation=1,
                 padding=1, expansion=3, reduction=4, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), avd=False, avd_first=False, **kwargs):
        """
        Recurrent block in R2Unet.
        References:
             "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>

        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            num_current(int): times or recurrent
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
        """
        super(RRBlock, self).__init__()
        self.conv1x1 = Conv2d(in_ch, out_ch, norm_layer=None, activation=None)
        self.conv = nn.Sequential(
            RConv(out_ch, num_current, norm_layer=norm_layer),
            RConv(out_ch, num_current, norm_layer=norm_layer)
        )

    def forward(self, x):
        net = self.conv1x1(x)
        identify = net
        net = self.conv(net)
        net = net + identify
        return net


class SplAtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1, expansion=3,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 avd=False, avd_first=False, num_current=2, **kwargs):
        """
        Implementation of block with Split Attention Module.
        References:
            "ResNeSt: Split-Attention Networks",https://hangzhang.org/files/resnest.pdf
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            radix (int): number of groups
            drop_prob (float): dropout rate
            dilation (int): dilation rate for conv
            padding (Union[int,tuple]): the padding size
            expansion (float): expansion rate for channels
            reduction (int): the reduction rate for Split Attention Conv2d
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
            avd (bool): whether use avd layer
            avd_first (bool): whether use avd layer first
            num_current(int): times or recurrent
        """
        super(SplAtBlock, self).__init__()
        expansion_out = int(round(out_ch * expansion))
        self.conv1 = Conv2d(in_ch, expansion_out, norm_layer=norm_layer, activation=activation)
        self.conv2 = SplAtConv2d(expansion_out, expansion_out, ksize=3, stride=1, padding=padding, dilation=dilation,
                                 radix=radix, drop_prob=drop_prob, norm_layer=norm_layer,
                                 nolinear=activation, reduction=reduction)
        self.conv3 = Conv2d(expansion_out, out_ch, norm_layer=norm_layer, activation=None)
        self.avd_layer = nn.Identity()
        if avd:
            self.avd_layer = nn.AvgPool2d(3, stride=1, padding=1)
        self.avd_first = avd_first
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0,
                                   activation=None, norm_layer=norm_layer)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcut(x)
        net = self.conv1(x)
        if self.avd_first:
            net = self.avd_layer(net)
        net = self.conv2(net)
        if not self.avd_first:
            net = self.avd_layer(net)
        net = self.conv3(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, **kwargs):
        super(ConvNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, groups=in_ch, bias=False)
        self.layernorm = LayerNorm(in_ch, eps=1e-6)
        self.conv2 = nn.Conv2d(in_ch, in_ch*4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_ch*4, in_ch, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()
        drouppath_rate = kwargs.get("dropout_rate", 0)
        self.identity = DropPath(drouppath_rate) if drouppath_rate > 0. else nn.Identity()
        self.post_norm = LayerNorm(in_ch, eps=1e-6)

    def forward(self, x):
        net = self.conv1(x)
        net = net.permute(0, 2, 3, 1)
        net = self.layernorm(net)
        net = net.permute(0, 3, 1, 2)
        net = self.conv2(net)
        net = self.activation(net)
        net = self.conv3(net)
        net = self.post_norm(net.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        net = x + self.identity(net)
        return net

VGGBlock = DoubleConv