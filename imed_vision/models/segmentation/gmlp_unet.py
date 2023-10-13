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

from imed_vision.layers.mbconv import MBConv
from imed_vision.layers import DropPath

class SSGU(nn.Module):
    def __init__(self, img_size, dim, norm_layer=nn.LayerNorm):
        super(SSGU, self).__init__()
        self.norm = norm_layer(dim // 2)
        self.y_axis_linear = nn.Linear(img_size[0], img_size[0])
        self.x_axis_linear = nn.Linear(img_size[1], img_size[1])
        # self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v_x, v_y = v.chunk(2, dim=-1)
        v_x = self.x_axis_linear(v_x.transpose(2, 3)).transpose(2, 3)
        v_y = self.y_axis_linear(v_y.transpose(1, 3)).transpose(1, 3)
        spatial = torch.cat([v_x, v_y], dim=-1)
        out = spatial * u
        return out

class GateMLP(nn.Module):
    def __init__(self, dim, img_size, ffn_dim=768,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU(),
                 drop_path_rate=0.):
        super(GateMLP, self).__init__()
        self.pre_norm = norm_layer(dim)
        self.act = act_layer
        self.channel_proj_pre = nn.Linear(dim, ffn_dim)
        self.sgu = SSGU(img_size, ffn_dim, norm_layer=norm_layer)
        self.channel_proj_after = nn.Linear(ffn_dim // 2, dim)
        self.drop = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        net = self.pre_norm(x)
        net = self.channel_proj_pre(net)
        net = self.act(net)
        net = self.sgu(net)
        net = self.channel_proj_after(net)
        net = x + self.drop(net)
        return net

        

class GateEncoder(nn.Module):
    def __init__(self, dim, out_dim, img_size,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU(), gate_layer=nn.Sigmoid(),
                 exp_ratio=4., ffn_exp_ratio=2,
                 reduction=0.25, drop_path_rate=0., down=True):
        super(GateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, out_dim, 1, 1)
        self.conv2 = MBConv(out_dim, out_dim, exp_ratio=exp_ratio, reduction=reduction, ksize=3, stride=1,
                           drop_path_rate=drop_path_rate, gate_layer=gate_layer)
        self.gatemlp = GateMLP(out_dim, img_size, ffn_dim=int(out_dim*ffn_exp_ratio),
                               norm_layer=norm_layer, act_layer=act_layer,
                               drop_path_rate=drop_path_rate)
        self.drop = DropPath(drop_path_rate)
        self.down = nn.MaxPool2d(2, stride=2) if down else nn.Identity()

    def forward(self, x):
        identity = self.conv1(x)
        net = self.conv2(identity)
        net = self.gatemlp(net.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        net = self.drop(net) + identity
        down = self.down(net)
        return net, down

class GateDecoder(nn.Module):
    def __init__(self, dim1, dim2, out_dim, img_size,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU(), gate_layer=nn.Sigmoid(),
                 exp_ratio=4., ffn_exp_ratio=2,
                 reduction=0.25, drop_path_rate=0.,
                 ):
        super(GateDecoder, self).__init__()
        self.conv1 = nn.Conv2d(dim1+dim2, out_dim, 1, 1)
        self.conv2 = MBConv(out_dim, out_dim, exp_ratio=exp_ratio, reduction=reduction, ksize=3, stride=1,
                            drop_path_rate=drop_path_rate, gate_layer=gate_layer)
        self.gatemlp = GateMLP(out_dim, img_size, ffn_dim=int(out_dim * ffn_exp_ratio),
                               norm_layer=norm_layer, act_layer=act_layer,
                               drop_path_rate=drop_path_rate)
        self.drop = DropPath(drop_path_rate)

    def forward(self, low, high):
        high = F.interpolate(high, size=low.size()[2:], mode="bilinear", align_corners=True)
        net = torch.cat([low, high], dim=1)
        net = self.conv1(net)
        identity = net
        net = self.conv2(net)
        net = self.gatemlp(net.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        net = self.drop(net) + identity
        return net

class GateMLPUNet(nn.Module):
    def __init__(self, img_size, in_ch=3, num_classes=12, act_layer=nn.GELU(), exp_ratio=4.,
                 ffn_exp_ratio=2, reduction=0.25, drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(GateMLPUNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            act_layer,
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            act_layer,
        )
        self.encoder1 = GateEncoder(32, 32, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s//2 for s in img_size]
        self.encoder2 = GateEncoder(32, 64, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s // 2 for s in img_size]
        self.encoder3 = GateEncoder(64, 128, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s // 2 for s in img_size]
        self.encoder4 = GateEncoder(128, 256, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s // 2 for s in img_size]
        self.encoder5 = GateEncoder(256, 512, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio, down=False)
        img_size = [s * 2 for s in img_size]
        self.decoder6 = GateDecoder(512, 256, 256, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s * 2 for s in img_size]
        self.decoder7 = GateDecoder(256, 128, 128, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s * 2 for s in img_size]
        self.decoder8 = GateDecoder(128, 64, 64, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        img_size = [s * 2 for s in img_size]
        self.decoder9 = GateDecoder(64, 32, 32, img_size, norm_layer=norm_layer, act_layer=act_layer,
                                    ffn_exp_ratio=ffn_exp_ratio, reduction=reduction,
                                    drop_path_rate=drop_path_rate, exp_ratio=exp_ratio)
        self.out_conv = nn.Conv2d(32, num_classes, 1, 1)

    def forward(self, x):
        net = self.stem(x)
        f1, down = self.encoder1(net)
        f2, down = self.encoder2(down)
        f3, down = self.encoder3(down)
        f4, down = self.encoder4(down)
        f5, _ = self.encoder5(down)
        net = self.decoder6(f4, f5)
        net = self.decoder7(f3, net)
        net = self.decoder8(f2, net)
        net = self.decoder9(f1, net)
        net = self.out_conv(net)
        return net

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512).cuda()
    model = GateMLPUNet((512, 512), 3, 12).cuda()
    out = model(x)
    print(out.shape)
