# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:lnet
    author: 12718
    time: 2022/6/23 9:34
    tool: PyCharm
"""
import torch
import torch.nn as nn

class SplitConv(nn.Module):
    def __init__(self, out_ch, groups=4):
        super(SplitConv, self).__init__()
        assert out_ch % groups == 0, "out_ch must be divided by groups, but got {}/{}={}"\
            .format(out_ch, groups, out_ch % groups)
        g_ch = out_ch // groups
        self.branchs = nn.ModuleList([nn.Conv2d(g_ch, g_ch, 1, 1, 0)])
        for i in range(1, groups):
            self.branchs.append(nn.Conv2d(g_ch, g_ch, 3, 1,
                                          padding=int(pow(2, i-1)),
                                          dilation=int(pow(2, i-1)),
                                          groups=g_ch))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.groups = groups


    def forward(self, x):
        _, ch, _, _ = x.size()
        feas = torch.split(x, ch // self.groups, dim=1)
        features = []
        for i in range(len(self.branchs)):
            features.append(self.branchs[i](feas[i]))
        feature = torch.cat(features, 1)
        net = self.bn(feature)
        net = self.relu(net)
        return net

class SplitBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=4):
        super(SplitBlock, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = SplitConv(out_ch, groups=groups)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, 1, 1, 0),
        # )
        # self.bn_relu = nn.Sequential(
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        self.identity = nn.Identity()
        if in_ch != out_ch:
            self.identity = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        identity = self.identity(x)
        net = self.conv1(x)
        net = self.conv2(net)
        # net = self.conv3(net)
        net = net + identity
        # net = self.bn_relu(net)
        return net

class LightNet(nn.Module):
    def __init__(self, in_ch, num_classes=1000, groups=4, layers=[2, 2, 2, 2]):
        super(LightNet, self).__init__()
        base_ch = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, 2, 3),
            nn.BatchNorm2d(base_ch),
            nn.ReLU()
        )
        self.stage1 = self._make_layer(base_ch, base_ch*2, 1, layers[0], groups)
        self.stage2 = self._make_layer(base_ch*2, base_ch*4, 2, layers[1], groups)
        self.stage3 = self._make_layer(base_ch*4, base_ch*8, 2, layers[2], groups)
        self.state4 = self._make_layer(base_ch*8, base_ch*16, 2, layers[3], groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_ch*16, num_classes)

    def _make_layer(self, in_ch, out_ch, stride, blocks, groups):
        layers = []
        if stride != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 2, 2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
        else:
            layers.append(SplitBlock(in_ch, out_ch, groups=groups))
        for i in range(1, blocks):
            layers.append(SplitBlock(out_ch, out_ch, groups))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        features = [x]
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.state4(x)
        features.append(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        x = self.pool(features[-1])
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out



if __name__ == "__main__":
    from torchstat import stat
    from torchvision import models
    import sys
    import os
    sys.path.append(os.path.abspath("../../"))
    x = torch.randn(1, 3, 224, 224)
    model = LightNet(3, groups=4, layers=[2, 4, 16, 8])
    out = model(x)
    print(out.shape)
    stat(model, (3, 224, 224))
    shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
    stat(shufflenet, (3, 224, 224))