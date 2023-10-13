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

from .attentions import MHA
from .layernorm import LayerNorm
from imed_vision.comm.activation import HSwish
from .mbconv import MBConv

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

class RelPosEmbedding(nn.Module):
    def __init__(self, dim, num_head, seq_len):
        super(RelPosEmbedding, self).__init__()
        # self.pos = nn.Parameter(torch.randn(num_head, dim // num_head, seq_len, seq_len), requires_grad=True)
        indexes = sinusoidal_embeddings(seq_len ** 2, dim).transpose(0, 1).reshape(num_head, dim // num_head,
                                                                         seq_len, seq_len)
        self.register_buffer("rel_index", indexes)

    def forward(self, x):
        pos = self.rel_index
        x = x + pos
        return x

class AxisAttention(nn.Module):
    def __init__(self, dim, num_head=4, axis="x", seq_len=256, pos_embed=False,
                 qkv_bias=True, att_drop=0., proj_drop=0.):
        super(AxisAttention, self).__init__()
        self.head_dim = dim // num_head
        self.num_head = num_head
        self.gamma = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim*3, 1, 1, bias=qkv_bias)
        self.pos_embd = nn.Identity()
        if pos_embed:
            self.pos_embd = RelPosEmbedding(dim, num_head, seq_len)
        self.axis = axis
        self.att_drop = nn.Dropout(att_drop)
        # self.proj

    def forward(self, x):
        bs, ch, h, w = x.size()
        qkv = self.qkv(x)
        if self.axis == "x":
            qkv = qkv.reshape(bs, 3, self.num_head, -1, h, w).permute(1, 0, 2, 3, 4, 5) #3, bs, num_head, head_dim, h, w
        elif self.axis == "y":
            qkv = qkv.reshape(bs, 3, self.num_head, -1, h, w).permute(1, 0, 2, 3, 5, 4) #3, bs, num_head, head_dim, w, h
        else:
            raise ValueError("Unknown mode axis")
        q, k, v = qkv.unbind(0)
        att = q.transpose(-2, -1) @ k #bs, num_head, head_dim, w, w
        att = self.pos_embd(att)
        att = torch.softmax(att*self.gamma, dim=-1)
        att = self.att_drop(att)
        net = v @ att #bs, num_head, head_dim, w, h
        if self.axis == "y":
            net = net.transpose(-2, -1)
        net = net.reshape(bs, ch, h, w)

        return net


class CrossAtt(nn.Module):
    def __init__(self, dim, num_head=4, qkv_bias=True, drop_proj=0., drop_att=0., img_size=(256, 256), ksize=1):
        super(CrossAtt, self).__init__()

        self.norm_pre = LayerNorm(dim)
        self.att_x = AxisAttention(dim // 2, num_head, axis="x", seq_len=img_size[1],
                                   pos_embed=False, qkv_bias=qkv_bias)
        self.att_y = AxisAttention(dim // 2, num_head, axis="y", seq_len=img_size[0],
                                   pos_embed=False, qkv_bias=qkv_bias)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        #     LayerNorm(dim),
        #     HSwish()
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, ksize, 1, padding=(ksize-1)//2),
            LayerNorm(dim),
            HSwish()
        )
        # self.norm_aft = LayerNorm(dim)

    def forward(self, x):
        """

        Args:
            x (Tensor): Input tensor, [bs, ch, h, w]

        Returns:

        """
        x = self.norm_pre(x)
        x_x, x_y = x.chunk(2, dim=1)
        x_x = self.att_x(x_x)
        x_y = self.att_y(x_y)
        x_out = torch.cat([x_x, x_y], dim=1)
        # out_x = self.conv(x) + x
        out_x = self.conv(x_out) + x
        # out_x = self.norm_aft(x)
        return out_x


if __name__ == "__main__":
    embd = sinusoidal_embeddings(32, 64, 100)
    print(embd.shape)
    x = torch.randn(2, 64, 64, 32).cuda()
    m = CrossAtt(64, 8, img_size=(64, 32)).cuda()
    out = m(x)
    print(out.shape)