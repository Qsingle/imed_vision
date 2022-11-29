# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cross_att
    author: 12718
    time: 2022/10/24 19:23
    tool: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn

from layers.attentions import MHA
from layers.layernorm import LayerNorm

def sinusoidal_embeddings(seq_len, dim, base=10000):
    """计算pos位置的dim维sinusoidal编码
    """
    assert dim % 2 == 0
    indices = torch.arange(0, dim // 2, dtype=torch.float)
    indices = torch.pow(torch.ones(dim // 2)*base, -2 * indices / dim)
    pos = torch.arange(0, seq_len).unsqueeze(-1).to(torch.float)
    embeddings = pos @ indices.unsqueeze(0)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.reshape(-1, dim)
    return embeddings

class CrossAtt(nn.Module):
    def __init__(self, dim, num_head=4, qkv_bias=True, drop_proj=0., drop_att=0., img_size=(256, 256)):
        super(CrossAtt, self).__init__()
        self.x_att = MHA(dim, num_head, qkv_bias=qkv_bias, att_drop=drop_att, proj_drop=drop_proj)
        # self.register_buffer("x_pos", sinusoidal_embeddings(img_size[1], dim=dim))
        self.y_att = MHA(dim, num_head, qkv_bias=qkv_bias, att_drop=drop_att, proj_drop=drop_proj)
        # self.register_buffer("y_pos", sinusoidal_embeddings(img_size[0], dim=dim))
        # self.attention_fusio = nn.Conv2d()
        self.norm_pre = LayerNorm(dim)
        self.norm_aft = LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)


    def forward(self, x):
        """

        Args:
            x (Tensor): Input tensor, [bs, ch, h, w]

        Returns:

        """
        x = self.norm_pre(x)
        # y_pos = self.get_buffer("y_pos")
        # x_pos = self.get_buffer("x_pos")
        out_y = torch.mean(x, dim=3).transpose(-2, -1)
        out_y = self.y_att(out_y)
        out_y = out_y.transpose(-2, -1).unsqueeze(-1)
        out_x = torch.mean(x, dim=2).transpose(-2, -1)
        out_x = self.x_att(out_x).transpose(-2, -1).unsqueeze(-2) + out_y
        out_x = self.conv(out_x)
        out_x = out_x + x
        out_x = self.norm_aft(out_x)
        return out_x


if __name__ == "__main__":
    embd = sinusoidal_embeddings(32, 64, 100)
    print(embd.shape)
    x = torch.randn(2, 64, 64, 32).cuda()
    m = CrossAtt(64, 8, img_size=(64, 32)).cuda()
    out = m(x)
    print(out.shape)