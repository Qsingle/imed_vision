# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gmlp_unet
    author: 12718
    time: 2022/10/12 15:31
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.cross_att import CrossAtt
from layers.mbconv import MBConv

class RCAB(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, ksize=3, reduction=0.25):
        super(RCAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride=1, padding=(ksize-1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, ksize, stride=stride, padding=(ksize-1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, int(out_ch*reduction), 1, 1),
            nn.Conv2d(int(out_ch*reduction), out_ch, 1, 1),
            nn.Sigmoid()
        )
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, 1)
            if stride != 1:
                self.skip = nn.Sequential(
                    nn.AvgPool2d(2, 2),
                    nn.Conv2d(in_ch, out_ch, 1, 1)
                )
        elif stride != 1:
            self.skip = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(in_ch, out_ch, 1, 1)
            )

    def forward(self, x):
        identity = self.skip(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.se(net) * net
        net = net + identity
        return net

class Encoder(nn.Module):
    def __init__(self, ch, out_ch, n_blocks):
        super(Encoder, self).__init__()
        convs = [MBConv(ch, out_ch)]
        # convs = [RCAB(ch, out_ch)]
        for _ in range(1, n_blocks):
            convs.append(MBConv(out_ch, out_ch))
            # convs.append(RCAB(out_ch, out_ch))
        self.conv = nn.Sequential(*convs)
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        features = self.conv(x)
        down = self.down(features)
        return features, down

class CrossBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_head=4, expansion_rate=2, qkv_bias=True, drop_att=0., drop_proj=0., img_size=(512, 512)):
        super(CrossBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1)
        self.cross_att = CrossAtt(out_ch, num_head=num_head,qkv_bias=qkv_bias,
                                  drop_att=drop_att, drop_proj=drop_proj, img_size=img_size)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_ch, out_ch*expansion_rate, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_ch*expansion_rate, out_ch, 1, 1)
        )
        # self.norm = LayerNorm(out_ch)

    def forward(self, x):
        net = self.conv(x)
        net = self.cross_att(net)
        # net = self.mlp(net) + net
        # net = self.norm(net)
        return net

class Decoder(nn.Module):
    def __init__(self, ch1, ch2, out_dim, att=False, num_head=4,
                 qkv_bias=True, drop_att=0., drop_proj=0., img_size=(512, 512)):
        super(Decoder, self).__init__()
        self.fuse = MBConv(ch1+ch2, out_dim)
        self.att = CrossAtt(out_dim, num_head=num_head, qkv_bias=qkv_bias,
                            drop_att=drop_att, drop_proj=drop_proj, img_size=img_size) \
                            if att else MBConv(out_dim, out_dim)
        self.up_conv = nn.Conv2d(ch1, ch1, 1, 1)

    def forward(self, low_level, high_level):
        high_level = F.interpolate(high_level, size=low_level.size()[2:], mode="bilinear", align_corners=True)
        high_level = self.up_conv(high_level)
        feature_fuse = torch.cat([high_level, low_level], dim=1)
        feature_fuse = self.fuse(feature_fuse)
        feature_fuse = self.att(feature_fuse)
        return feature_fuse

class GANet(nn.Module):
    def __init__(self, ch=3, num_classes=12, qkv_bias=True, drop_att=0., drop_proj=0.,
                 num_head=4, s=1., img_size=(512, 512)):
        super(GANet, self).__init__()
        _img_size = list(img_size)
        bash_ch = int(32 * s)
        self.stem = nn.Sequential(
            nn.Conv2d(ch, bash_ch, 3, 2, 1),
            nn.BatchNorm2d(bash_ch),
            nn.ReLU(),
            nn.Conv2d(bash_ch, bash_ch, 3, 1, 1),
            nn.BatchNorm2d(bash_ch),
            nn.ReLU()
        )
        down_size = lambda x: [x[0] // 2, x[1] // 2]
        up_size = lambda x: [x[0]*2, x[1]*2]
        _img_size = down_size(_img_size)
        self.s1 = Encoder(bash_ch, bash_ch, 2)
        _img_size = down_size(_img_size)
        self.s2 = Encoder(bash_ch, bash_ch*2, 4)
        _img_size = down_size(_img_size)
        self.s3 = Encoder(bash_ch*2, bash_ch*4, 4)
        blocks = [CrossBlock(bash_ch*4, bash_ch*8, num_head=num_head, qkv_bias=qkv_bias,
                            drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size)]
        for i in range(1, 4):
            blocks.append(CrossBlock(bash_ch*8, bash_ch*8, num_head=num_head, qkv_bias=qkv_bias,
                                     drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size))
        self.s4 = nn.Sequential(
            *blocks
        )
        # self.s4_down = nn.Conv2d(bash_ch*8, bash_ch*8, 3, 2, 1)
        # _img_size = down_size(_img_size)
        # blocks = [CrossBlock(bash_ch*8, bash_ch*16, num_head=num_head, qkv_bias=qkv_bias, drop_att=drop_att,
        #                      drop_proj=drop_proj, img_size=_img_size)]
        # for i in range(1, 2):
        #     blocks.append(CrossBlock(bash_ch*16, bash_ch*16, num_head=num_head, qkv_bias=qkv_bias, drop_att=drop_att,
        #                      drop_proj=drop_proj, img_size=_img_size))
        # self.s5 = nn.Sequential(
        #     *blocks
        # )
        _img_size = up_size(_img_size)
        self.d1 = Decoder(bash_ch*8, bash_ch*4, bash_ch*4, att=True, num_head=num_head, qkv_bias=qkv_bias,
                          drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size)
        _img_size = up_size(_img_size)
        self.d2 = Decoder(bash_ch*4, bash_ch*2, bash_ch*2, att=True, num_head=num_head, qkv_bias=qkv_bias,
                          drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size)
        self.d3 = Decoder(bash_ch*2, bash_ch, bash_ch, att=False)
        # self.d4 = Decoder(bash_ch*2, bash_ch, bash_ch, att=False)

        # self.align = nn.Sequential(
        #     nn.Conv2d(bash_ch, bash_ch, 1, 1),
        #     nn.ReLU(),
        #     MBConv(bash_ch, bash_ch),
        #     MBConv(bash_ch, bash_ch)
        # )
        self.out_conv = nn.Conv2d(bash_ch, num_classes, 1, 1)


    def forward(self, x):
        stem = self.stem(x)
        f1, down = self.s1(stem)
        f2, down = self.s2(down)
        f3, down = self.s3(down)
        f4 = self.s4(f3)
        # f4_down = self.s4_down(f4)
        # f5 = self.s5(f4_down)
        d1 = self.d1(f3, f4)
        d2 = self.d2(f2, d1)
        d3 = self.d3(f1, d2)
        # d4 = self.d4(f1, d3)
        out_fea = F.interpolate(d3, size=x.size()[2:], mode="bilinear", align_corners=True)
        # out_fea = self.align(out_fea)
        out = self.out_conv(out_fea)
        return out




if __name__ == "__main__":
    x = torch.randn(1, 3, 496, 256).cuda()
    model = GANet(3, 12, img_size=(496, 256)).cuda()
    out = model(x)
    print(out.shape)
