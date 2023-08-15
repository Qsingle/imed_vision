#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   learnable_sam.py
    @Time    :   2023/08/15 10:41:31
    @Author  :   12718 
    @Version :   1.0
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

from layers.prompt import FFTPrompt
from layers.maf import SSC as MSConv2d
from layers.adapter import PromptGen
from models.segmentation.segment_anything import sam_model_registry
from models.segmentation.segment_anything.modeling.common import LayerNorm2d


class PromptSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, reduction=4, upsample_times=2, groups=4, 
                 prompt_input=False, prompt_type="fft", fft_type="highpass", freq_num=0.25, ms=False) -> None:
        super(PromptSAM, self).__init__()
        #load same from the pretrained model
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint)
        del self.sam.prompt_encoder
        del self.sam.mask_decoder
        out_dim = self.sam.image_encoder.neck[0].out_channels
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        self.img_size = self.sam.image_encoder.img_size
        blocks = []
        for block in self.sam.image_encoder.blocks:
            blocks.append(
                PromptGen(block, reduction=reduction)
            )
        self.sam.image_encoder.blocks = nn.Sequential(
            *blocks
        )
        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    # nn.Conv2d(dim, dim // 2, 1, 1, 0),
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )
        
        if prompt_input:
            if prompt_type == "fft":
                self.prompt_input = FFTPrompt(rate=freq_num, prompt_type=fft_type)
        else:
            self.prompt_input = nn.Identity()

    def upscale(self, x, times=2):
        for i in range(times):
            # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x):
        x = self.prompt_input(x)
        out = self.sam.image_encoder(x)
        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        seg_out = self.decoder(out)
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out