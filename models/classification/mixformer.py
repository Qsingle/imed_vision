# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:mixformer
    author: 12718
    time: 2022/4/29 9:21
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math
import os.path

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from functools import partial
import torch.nn.functional as F
from collections import OrderedDict

from layers.mix_transformer_layers import *

from .create_model import BACKBONE_REGISTER

__all__ = ["MixTransformer", "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]

class MixTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], global_pool='avg',
                 pretrain=False, pretrained_model=None, **kwargs):
        super(MixTransformer, self).__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed, setting the embedding for 4 stage
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        #encoders
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        #stage 1
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                           qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur+i], qk_scale=qk_scale,
                                           reduce_ratio=sr_ratios[0], norm_layer=norm_layer, att_drop=attn_drop_rate)
                                        for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        #stage2
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                           qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i], qk_scale=qk_scale,
                                           reduce_ratio=sr_ratios[1], norm_layer=norm_layer, att_drop=attn_drop_rate)
                                     for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        #stage3
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                           qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i], qk_scale=qk_scale,
                                           reduce_ratio=sr_ratios[2], norm_layer=norm_layer, att_drop=attn_drop_rate)
                                     for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        #stage 4
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                           qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i], qk_scale=qk_scale,
                                           reduce_ratio=sr_ratios[3], norm_layer=norm_layer, att_drop=attn_drop_rate)
                                     for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        cur += depths[3]

        #classification head
        self.head = nn.Linear(embed_dims[3], num_classes)
        self.global_pool = global_pool
        self.apply(self._init_weights)
        #pretrain
        if pretrain:
            assert pretrained_model is not None, "Please give the path of the pretrained model"
            self.load_check_point(pretrained_model)

    def load_check_point(self, path):
        state = torch.load(path, map_location="cpu")
        print("Loading pretrained model from {}".format(path))
        try:
            self.load_state_dict(state)
        except:
            new_state = OrderedDict()
            for k, v in state.items():
                if "head" not in k:
                    new_state[k] = v
            self.state_dict().update(new_state)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.head.in_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]

        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]

        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]

        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        bs = x.shape[0]
        features = []
        #stage 1
        net, H, W = self.patch_embed1(x)
        for block in self.block1:
            net = block(net, H, W)
        net = self.norm1(net)
        net = net.transpose(1, 2).reshape(bs, -1, H, W)
        features.append(net)

        #stage 2
        net, H, W = self.patch_embed2(net)
        for block in self.block2:
            net = block(net, H, W)
        net = self.norm2(net)
        net = net.transpose(1, 2).reshape(bs, -1, H, W)
        features.append(net)

        # stage 3
        net, H, W = self.patch_embed3(net)
        for block in self.block3:
            net = block(net, H, W)
        net = self.norm3(net)
        net = net.transpose(1, 2).reshape(bs, -1, H, W)
        features.append(net)

        # stage 2
        net, H, W = self.patch_embed4(net)
        for block in self.block4:
            net = block(net, H, W)
        net = self.norm4(net)
        net = net.transpose(1, 2).reshape(bs, -1, H, W)
        features.append(net)
        return features

    def forward_head(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        out = self.head(x)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        out = self.forward_head(features[-1])
        return out

@BACKBONE_REGISTER.register()
def mit_b0(**kwargs):
    return MixTransformer(embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@BACKBONE_REGISTER.register()
def mit_b1(**kwargs):
    return MixTransformer(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@BACKBONE_REGISTER.register()
def mit_b2(**kwargs):
    return MixTransformer(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@BACKBONE_REGISTER.register()
def mit_b3(**kwargs):
    return MixTransformer(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@BACKBONE_REGISTER.register()
def mit_b4(**kwargs):
    return MixTransformer(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@BACKBONE_REGISTER.register()
def mit_b5(**kwargs):
    return MixTransformer(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 256, 256)
    model = mit_b2(img_size=256)
    out = model(x)
    print(out.shape)