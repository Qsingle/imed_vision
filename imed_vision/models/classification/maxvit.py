# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:maxvit
    author: 12718
    time: 2022/9/19 15:12
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn

from imed_vision.layers import MaxSA

max_vits = {
    "maxvit_t": {
        "url": "",
        "cfg": {
            "num_heads": [2*(2**i) for i in range(4)],
            "embidding_dims":[64, 64, 128, 256, 512],
            "num_blocks": [2, 2, 5, 2],
            "droppath_rate": 0.2
        }
    },
    "maxvit_s": {
        "url": "",
        "cfg": {
            "num_heads": [3*(2**i) for i in range(4)],
            "embidding_dims": [64, 96, 192, 384, 768],
            "num_blocks": [2, 2, 5, 2],
            "droppath_rate": 0.3
        }
    },
    "maxvit_b": {
        "url": "",
        "cfg": {
            "num_heads": [3*(2**i) for i in range(4)],
            "embidding_dims": [64, 96, 192, 384, 768],
            "num_blocks": [2, 6, 14, 2],
            "droppath_rate": 0.4
        }
    },
    "maxvit_l": {
        "url": "",
        "cfg": {
            "num_heads": [4*(2**i) for i in range(4)],
            "embidding_dims": [128, 128, 256, 512, 1024],
            "num_blocks": [2, 6, 14, 2],
            "droppath_rate": 0.6
        }
    }
}

class MaxViT(nn.Module):
    def __init__(self, in_ch, num_classes=1000, embidding_dims=[64, 64, 128, 256, 512],
                 num_blocks=[2, 2, 2, 2],
                 patch_sizes=[
                     [7, 7],
                     [7, 7],
                     [7, 7],
                     [7, 7]
                 ], grid_sizes=[
                     [7, 7],
                     [7, 7],
                     [7, 7],
                     [7, 7]
                 ],
                 num_heads=[2, 2, 2, 2],
                 expansion=4,
                 ffn_expansion=4,
                 droppath_rate=0.,
                 att_drop=0.,
                 proj_drop=0.,
                 reduction=0.25,
                 img_size=(224, 224)
                 ):
        super(MaxViT, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, embidding_dims[0], 3, 2, 1),
            nn.BatchNorm2d(embidding_dims[0]),
            nn.GELU(),
            nn.Conv2d(embidding_dims[0], embidding_dims[0], 3, 1, 1),
            nn.BatchNorm2d(embidding_dims[0]),
            nn.GELU()
        )
        self.stages = nn.ModuleDict()
        img_size = [s // 2 for s in img_size]
        for i in range(4):
            img_size = [s // 2 for s in img_size]
            blocks = [MaxSA(
                embidding_dims[i],
                embidding_dims[i+1],
                stride=2,
                patch_size=patch_sizes[i],
                grid_size=grid_sizes[i],
                num_head=num_heads[i],
                expansion=expansion,
                ffn_expansion_rate=ffn_expansion,
                droppath_rate=droppath_rate,
                att_drop=att_drop,
                proj_drop=proj_drop,
                reduction=reduction,
                img_size=img_size
            )]
            for k in range(1, num_blocks[i]):
                blocks.append(
                    MaxSA(
                        embidding_dims[i + 1],
                        embidding_dims[i + 1],
                        stride=1,
                        patch_size=patch_sizes[i],
                        grid_size=grid_sizes[i],
                        num_head=num_heads[i],
                        expansion=expansion,
                        ffn_expansion_rate=ffn_expansion,
                        droppath_rate=droppath_rate,
                        att_drop=att_drop,
                        proj_drop=proj_drop,
                        reduction=reduction,
                        img_size=img_size
                    )
                )
            self.stages["stage_{}".format(i+1)] = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embidding_dims[-1], num_classes)

    def forward_features(self, x):
        stem = self.stem(x)
        net = stem
        features = []
        for k, v in self.stages.items():
            net = self.stages[k](net)
            features.append(net)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        out = self.avgpool(features[-1])
        out = self.fc(out.flatten(1))
        return out

def _max_vit(arch_name, pretrained=False, progress=True, **kwargs):
    in_ch = kwargs.get("in_ch", 3)
    model = MaxViT(in_ch=in_ch, **max_vits[arch_name]["cfg"], **kwargs)
    if pretrained:
        if max_vits[arch_name]["url"] != "":
            state = torch.hub.load_state_dict_from_url(max_vits[arch_name]["url"], progress=progress)
            model.load_state_dict(state)
        else:
            print(f"We not provide the pretrained weights for {arch_name} now, skip loading weights")
    return model

def maxvit_t(pretrained=False, progress=True, **kwargs):
    model = _max_vit(arch_name="maxvit_t", pretrained=pretrained,
                     progress=progress, **kwargs)
    return model

def maxvit_s(pretrained=False, progress=True, **kwargs):
    model = _max_vit(arch_name="maxvit_s", pretrained=pretrained,
                     progress=progress, **kwargs)
    return model

def maxvit_b(pretrained=False, progress=True, **kwargs):
    model = _max_vit(arch_name="maxvit_b", pretrained=pretrained,
                     progress=progress, **kwargs)
    return model

def maxvit_l(pretrained=False, progress=True, **kwargs):
    model = _max_vit(arch_name="maxvit_t", pretrained=pretrained,
                     progress=progress, **kwargs)
    return model

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224).cuda()
    model = maxvit_t(num_classes=100).cuda()
    out = model(x)
    print(out.shape)