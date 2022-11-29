# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:seq_trans
    author: 12718
    time: 2022/8/14 15:29
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from layers import MHA

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, activation=nn.GELU()):
        """

        Args:
            dim:
            hidden_dim:
            out_dim:
            activation:
        """
        super(MLP, self).__init__()
        out_dim = out_dim or dim
        self.l1 = nn.Linear(dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.activate = activation

    def forward(self, x):
        net = self.l1(x)
        net = self.activate(net)
        net = self.l2(net)
        return net

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, att_drop=0., proj_drop=0., mlp_exp_rate=4.0,
                 activation=nn.GELU(), norm_layer=nn.LayerNorm):
        """
        Implementation of the transformer layer
        Args:
            dim:
            num_heads:
            qkv_bias:
            att_drop:
            proj_drop:
            mlp_exp_rate:
            activation:
            norm_layer:
        """
        super(Block, self).__init__()
        self.attention = MHA(dim, num_heads, qkv_bias=qkv_bias,
                             att_drop=att_drop, proj_drop=proj_drop)
        self.att_norm = norm_layer(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_exp_rate), activation=activation)
        self.mlp_norm = norm_layer(dim)

    def forward(self, x):
        net = self.attention(x)
        atten = self.att_norm(x+net)
        net = self.mlp(atten)
        net = self.mlp_norm(net + atten)
        return net

class PatchEmbed(nn.Module):
    def __init__(self, in_dim, seq_len, patch_size=32):
        super(PatchEmbed, self).__init__()


class SeqTrans(nn.Module):
    def __init__(self, in_dim, seq_len, num_heads=8, num_block=6, qkv_bias=False,
                 att_drop=0., proj_drop=0., mlp_exp_rate=4.0,
                 activation=nn.GELU(), norm_layer=nn.LayerNorm, patch_size=32):
        super(SeqTrans, self).__init__()
