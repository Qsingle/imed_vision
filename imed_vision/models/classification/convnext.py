#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   convnext.py
    @Time    :   2023/09/07 11:37:11
    @Author  :   12718 
    @Version :   1.0
'''

import torch
import torch.nn as nn

from imed_vision.comm.init import trunc_normal_
from imed_vision.layers.layernorm import LayerNorm
from imed_vision.layers.convnext import Block
from .create_model import BACKBONE_REGISTER

__all__ = ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge"]

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

class ConvNeXt(nn.Module):
    def __init__(self, in_ch=3, num_classes=1000, num_blocks=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.) -> None:
        super(ConvNeXt, self).__init__()
        stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        self.features = [dims[0]]
        #stem and other three time downsampling layer
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], 2, 2)
                )
            )
        
        self.stages = []
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path_rate=dp_rates[cur+j],
                          layer_scale_init_value=layer_scale_init_value) 
                    for j in range(num_blocks[i])
                ]
            )
            self.stages.append(stage)
            self.features.append(dims[i])
            cur += num_blocks[i]
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channel_last")
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        features = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        feature = features[-1]
        x = self.norm(feature.mean([-2, -1]))
        x = self.head(x)
        return x

@BACKBONE_REGISTER.register()
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONE_REGISTER.register()
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONE_REGISTER.register()
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONE_REGISTER.register()
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONE_REGISTER.register()
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = convnext_tiny(pretrained=False, in_22k=False)
    out = model(x)