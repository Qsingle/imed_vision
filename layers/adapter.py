#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   adapter.py
    @Time    :   2023/08/15 10:08:12
    @Author  :   12718 
    @Version :   1.0
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torch import Tensor
import torch.nn as nn

from models.segmentation.segment_anything.modeling.common import LayerNorm2d
from models.segmentation.segment_anything.modeling.image_encoder import Block

__all__ = ["PromptGen"]

class PromptGen(nn.Module):
    def __init__(self, blk:Block, reduction=4, cls_token=False, reshape=False, seq_size=None, no_transpose=False, dim=None) -> None:
        """
            One type of adapter introduced in 
            "Learnable Ophthalmology SAM"<https://arxiv.org/abs/2304.13425>
        Args:
            blk (Union[nn.Module, Block]): The Vision Transformer Block
            reduction (int, optional): The reduction rate. Defaults to 4.
            cls_token (bool, optional): Whether use the class token in the block. Defaults to False.
            reshape (bool, optional): Whether needs to reshape. Defaults to False.
            seq_size ([type], optional): The length of token. Defaults to None.
            no_transpose (bool, optional): Whether need to transpose the output. Defaults to False.
            dim ([type], optional): The dimension of the input. Defaults to None.
        """
        super(PromptGen, self).__init__()
        self.block = blk
        dim = dim or blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )
        self.no_transpose = no_transpose
        self.cls_token = cls_token
        if self.cls_token:
            self.prompt_cls = nn.Sequential(
                nn.Linear(dim, prompt_dim),
                nn.LayerNorm(prompt_dim),
                nn.GELU(),
                nn.Linear(prompt_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
        self.reshape = reshape
        self.seq_size = seq_size
    
    def forward(self, x:Tensor) -> Tensor:
        """AI is creating summary for forward

        Args:
            x (Tensor): The input features.

        Returns:
            Tensor: The features extracted by the block
        """
        if self.cls_token:
            tokens = x[:,1:]
            cls_token = x[:, 0].unsqueeze(1)
            # cls_token = self.prompt_cls(cls_token)
            bs, seq_len, dim = tokens.size()
            if self.reshape:
                tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
            prompt = self.prompt_learn(tokens)
            promped = tokens + prompt
            if self.reshape:
                promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
            promped = torch.cat([cls_token, promped], dim=1)
        else:
            if self.no_transpose:
                prompt = self.prompt_learn(x)
            else:
                prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            promped = x + prompt
        net = self.block(promped)
        return net