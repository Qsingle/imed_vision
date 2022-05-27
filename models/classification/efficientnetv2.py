# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:efficientnetv2
    author: 12718
    time: 2021/9/24 9:38
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .create_model import BACKBONE_REGISTER

def make_division(chs, multiplier=1.0, min_depth=8, divisor=8):
    """Round number of filters based on depth multiplier."""

    chs *= multiplier
    min_depth = min_depth or multiplier
    new_filters = max(min_depth, int(chs + divisor / 2) // divisor * divisor)
    return int(new_filters)

class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=4.0, activation=nn.Sigmoid()):
        super(SEModule, self).__init__()
        if activation is None:
            activation = nn.Sigmoid()
        hidden_ch = int(round(in_ch*(1/reduction)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_ch, in_ch, kernel_size=1, stride=1)
        self.activation = activation

    def forward(self, x):
        identity = x
        net = self.avg_pool(x)
        net = self.fc1(net)
        net = self.act1(net)
        net = self.fc2(net)
        net = self.activation(net) * identity
        return net

if hasattr(nn, "SiLU"):
    SiLU = nn.SiLU
else:
    class SiLU(nn.Module):
        def __init__(self):
            super(SiLU, self).__init__()

        def forward(self, x):
            return x*torch.sigmoid(x)

class MBConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4.0, se=True, se_reduction=4, activation=nn.ReLU(),
                 norm_layer=None, se_activation=nn.Sigmoid()):
        super(MBConv2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU()

        hidden_size = int(round(in_ch * expansion))
        #pw
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_size, kernel_size=1, stride=1, bias=False),
            norm_layer(hidden_size),
            activation
        )
        #dw
        self.conv2  = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, stride=stride,padding=1,
                      groups=hidden_size, bias=False),
            norm_layer(hidden_size),
            activation
        )
        self.se = None
        if se:
            self.se = SEModule(hidden_size, se_reduction, se_activation)

        #pw linear
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, out_ch, 1, 1, bias=False),
            norm_layer(out_ch)
        )
        self.short_cut = True
        if stride > 1 or in_ch != out_ch:
            self.short_cut = False
        self.activation = activation

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        if self.se is not None:
            net = self.se(net)
        net = self.conv3(net)
        if self.short_cut:
            net = net + x
        net = self.activation(net)
        return net

class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4.0, stride=1, se=True, se_reduction=4,
                 activation=nn.ReLU(), se_activation=nn.Sigmoid(), norm_layer=None):
        super(FusedMBConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU()
        hidden_size = int(round(in_ch*expansion))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_size, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(hidden_size),
            activation
        )
        self.se = None
        if se:
            self.se = SEModule(hidden_size, reduction=se_reduction, activation=se_activation)
        self.identity = True
        if in_ch != out_ch or stride > 1:
            self.identity = False
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, out_ch, 1,  bias=False),
            norm_layer(out_ch)
        )
        self.activation = activation

    def forward(self,x):
        net = self.conv1(x)
        if self.se is not None:
            net = self.se(net)
        net = self.conv2(net)
        if self.identity:
            net = net + x
        net = self.activation(net)
        return net

class EfficientNetV2(nn.Module):
    def __init__(self,  cfgs, in_ch=3, num_classes=1000, multiplier=1.0, min_depth=8, divisor=8,
                 norm_layer=nn.BatchNorm2d, activation=nn.PReLU(), se_reduction=4.0,
                 se_activation=nn.Sigmoid()):
        super(EfficientNetV2, self).__init__()
        self.blocks = nn.ModuleList()
        out_ch = make_division(cfgs[0][2], multiplier, min_depth, divisor)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            activation
        )
        block_type = {0: MBConv2d, 1: FusedMBConv}
        in_planes = out_ch
        for e, n, c, se, s, b in cfgs:
            block = block_type[b]
            in_ch = in_planes
            out_ch = make_division(c, multiplier, min_depth=min_depth, divisor=divisor)
            layers = []
            for i in range(n):
                layers.append(block(in_ch, out_ch, stride=s if i == 0 else 1, expansion=e, se=se, norm_layer=norm_layer,
                                    se_reduction=se_reduction, se_activation=se_activation))
                in_ch = out_ch
            self.blocks.append(nn.Sequential(*layers))
            in_planes = out_ch
        self.act_conv = nn.Conv2d(in_planes, 1280, 1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        net = self.conv1(x)
        for i in range(len(self.blocks)):
            net = self.blocks[i](net)
        net = self.act_conv(net)
        net = self.avg_pool(net)
        net = self.fc(net)
        return net

def _effnet_v2(cfgs, **kwargs):
    kwargs["cfgs"] = cfgs
    model = EfficientNetV2(**kwargs)
    return model

base_cfgs = {
    "s1":  [
        #e, n, c, se, s, block
        [4, 2, 24, False, 1, 1],
        [4, 4, 48, False, 2, 1],
        [4, 4, 96, False, 2, 1],
        [4, 6, 128, True, 2, 0],
        [4, 9, 160, True, 1, 0],
        [4, 15, 256, True, 2, 0]
    ]
}

@BACKBONE_REGISTER.register()
def efficientnetv2_s(**kwargs):
    return _effnet_v2(base_cfgs["s1"], **kwargs)

if __name__ == "__main__":
    model = efficientnetv2_s()
    x = torch.randn((2, 3, 224, 224))
    print(model(x).shape)