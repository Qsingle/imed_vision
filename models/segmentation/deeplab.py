# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/16 20:24
    @filename: deeplab.py
    @software: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.classification.resnet import *
from models.classification import create_backbone
from layers.utils import *
from layers.task_fusion import CrossGL, CAGL
from layers.task_fusion import CrossTaskAttention

__all__ = ["DeeplabV3", "DeeplabV3Plus", "ASPP", "ImagePooling"]


class ImagePooling(nn.Module):
    def __init__(self, in_planes, out_ch=256, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        '''
            implementation of ImagePooling in deeplabv3,section 3.3
            paper:https://arxiv.org/abs/1706.05587
            args:
                in_planes (int):input planes for the pooling
                out_ch (int): output channels, in paper is 256
                norm_layer (nn.Module): the batch normalization module
                activation(nn.Module): the activation function module
        '''
        super(ImagePooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = Conv2d(in_planes, out_ch, ksize=1, stride=1, padding=0, norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        net = self.avgpool(x)
        net = self.conv(net)
        return net


class ASPP(nn.Module):
    def __init__(self, in_planes, out_ch, rates, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        '''
            implementation of ASPP(Atrous Spatial Pyramid Pooling) in deeplabv3,section 3.3
            References:
                https://arxiv.org/abs/1706.05587
            args:
                in_planes (int):input planes for the pooling
                out_ch (int): output channels, in paper is 256
                norm_layer (nn.Module): the batch normalization module
                activation(nn.Module): the activation function module
        '''
        super(ASPP, self).__init__()
        self.branch1 = Conv2d(in_planes, 256, 1, stride=1, padding=0, dilation=rates[0], norm_layer=norm_layer, activation=activation)
        self.branch2 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[1], dilation=rates[1], norm_layer=norm_layer,
                              activation=activation)
        self.branch3 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[2], dilation=rates[2], norm_layer=norm_layer,
                              activation=activation)
        self.branch4 = Conv2d(in_planes, 256, 3, stride=1, padding=rates[3], dilation=rates[3], norm_layer=norm_layer,
                              activation=activation)
        self.branch5 = ImagePooling(in_planes, 256)

        self.conv = nn.Sequential(
            Conv2d(1280, out_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)
        branch5 = F.interpolate(branch5, size=branch4.size()[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        conv = self.conv(concat)
        return conv

class DeeplabV3(nn.Module):
    def __init__(self, in_ch, num_classes, backbone="resnet50",  output_stride=16, **kwargs):
        super(DeeplabV3, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))

        multi_grids = [1, 2, 4]
        # self.backbone = backbones[backbone](in_ch=in_ch, strides=strides,
        #                                     dilations=dilations, multi_grids=multi_grids, **kwargs)
        self.backbone = create_backbone(backbone)(in_ch=in_ch, strides=strides,
                                                  dilations=dilations, multi_grids=multi_grids, **kwargs)
        del self.backbone.avg_pool
        del self.backbone.fc
        self.aspp = ASPP(in_planes=2048, out_ch=256, rates=rates)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        # net = self.backbone.conv1(x)
        # net = self.backbone.max_pool(net)
        # net = self.backbone.layer1(net)
        # net = self.backbone.layer2(net)
        # net = self.backbone.layer3(net)
        # net = self.backbone.layer4(net)
        features = self.backbone.forward_features(x)
        net = self.aspp(features[-1])
        net = self.conv5(net)
        net = F.interpolate(net, size=size, mode="bilinear", align_corners=False)
        return net

class RCAB(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4,
                 norm=nn.BatchNorm2d,activation=nn.ReLU(inplace=True)):
        super(RCAB, self).__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            norm(out_ch),
            activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
            norm(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_ch),
        )
        self.se = SEModule(out_ch, reduction=reduction, norm_layer=norm,
                           sigmoid=nn.Sigmoid(), activation=activation)
        self.act = activation

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.se(net)*net + net
        net = self.act(net)
        return net
    

class Decoder(nn.Module):
    def __init__(self, low_ch, high_ch):
        super(Decoder, self).__init__()
        self.conv_low = nn.Conv2d(low_ch, 48, 1, 1)
        fusion_ch = 48 + high_ch
        self.fusion_conv = nn.Sequential(
            Conv2d(fusion_ch, out_ch=256, ksize=3, stride=1, padding=1),
            Conv2d(256, 256, ksize=3, stride=1, padding=1),
        )

    def forward(self, low_fe, high_fe):
        low_fe = self.conv_low(low_fe)
        high_fe = F.interpolate(high_fe, size=low_fe.size()[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([low_fe, high_fe], dim=1)
        fusion_out = self.fusion_conv(concat)
        return fusion_out

class DeeplabV3Plus(nn.Module):
    def __init__(self, in_ch, num_classes, sr_ch=None, backbone="resnet50",  output_stride=16,
                 middle_layer=False, super_reso=False, upscale_rate=2, sr_seg_fusion=False, cross_att=False, both=False,
                 **kwargs):
        super(DeeplabV3Plus,self).__init__()
        sr_ch = in_ch if sr_ch is None else sr_ch
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))

        multi_grids = [1, 2, 4]
        # self.backbone = backbones[backbone](in_ch=in_ch, strides=strides,
        #                                     dilations=dilations, multi_grids=multi_grids, **kwargs)
        self.both = both
        self.backbone = create_backbone(backbone, in_ch=in_ch, strides=strides,
                                            dilations=dilations, multi_grids=multi_grids, **kwargs)
        try:
            norm_layer = kwargs["norm_layer"]
        except KeyError as e:
            norm_layer = nn.BatchNorm2d
        try:
            activation = kwargs["activation"]
        except KeyError as e:
            activation = nn.ReLU(inplace=True)
        try:
            reduction = kwargs["reduction"]
        except KeyError as e:
            reduction = 16

        del self.backbone.avg_pool
        del self.backbone.fc
        self.aspp = ASPP(in_planes=2048, out_ch=256, rates=rates, norm_layer=norm_layer,
                         activation=activation)
        if middle_layer:
            low_ch = 512
        else:
            low_ch = 256
        self.decoder_seg = Decoder(low_ch, 256)
        self.middle_layer = middle_layer
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.sr_seg_fusion = sr_seg_fusion and self.super_reso
        self.out_conv = nn.Conv2d(256, num_classes, 1, 1)
        if super_reso:
            self.decoder_sr = Decoder(low_ch, 256)
            self.sr = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=upscale_rate)
            ) if sr_ch == 3 else nn.Conv2d(256, sr_ch, 1, 1)
            if self.sr_seg_fusion:
                if cross_att:
                    self.sr_seg_fusion_module = CrossTaskAttention(256, 256, patch_size=16)
                else:
                    # self.sr_seg_fusion_module = CrossGL(256, 256)
                    self. sr_seg_fusion_module = CAGL(256, 256)


    def forward(self, x):
        size = x.size()[2:]
        # net = self.backbone.conv1(x)
        # net = self.backbone.max_pool(net)
        # net = self.backbone.layer1(net)
        # h = net
        # net = self.backbone.layer2(net)
        # if self.middle_layer:
        #     h = net
        # net = self.backbone.layer3(net)
        # net = self.backbone.layer4(net)
        features = self.backbone.forward_features(x)
        h = features[0]
        if self.middle_layer:
            h = features[1]
        aspp = self.aspp(features[-1])
        net = self.decoder_seg(h, aspp)
        seg_out = self.out_conv(net)
        seg_out = F.interpolate(seg_out, size=size, mode="bilinear", align_corners=False)
        sr = None
        fusion_sr = None
        fusion_seg = None
        if self.super_reso:
            seg_out = F.interpolate(seg_out, scale_factor=self.upscale_rate, align_corners=False, mode="bilinear")
            if self.training or self.both:
                sr_fe = self.decoder_sr(h, aspp)
                if self.sr_seg_fusion:
                    fusion_sr, fusion_seg = self.sr_seg_fusion_module(sr_fe, net)
                fusion_seg = self.out_conv(net)
                fusion_seg = F.interpolate(fusion_seg, seg_out.size()[2:], mode="bilinear", align_corners=False)
                sr_fe = F.interpolate(sr_fe, size=size, mode="bilinear", align_corners=False)
                sr = self.sr(sr_fe)
                fusion_sr = F.interpolate(fusion_sr, size=size, mode="bilinear", align_corners=False)
                fusion_sr = self.sr(fusion_sr)
                    # fusion_seg = net * fusion + net
        if sr is not None:
            if self.super_reso and self.training:
                if self.sr_seg_fusion:
                    return seg_out, sr, fusion_seg, fusion_sr
                return seg_out, sr
            elif self.super_reso and self.both:
                return seg_out, sr
        return seg_out

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = DeeplabV3Plus(3, 20, layer_attention=True)
    model.eval()
    out = model(x)
    print(out.shape)