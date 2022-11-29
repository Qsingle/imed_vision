# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pct
    author: 12718
    time: 2022/10/25 9:59
    tool: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Parameter

from comm.helper import _pair

class Channels(nn.Module):
    def forward(self, x):
        y_mean = torch.mean(x, dim=2)
        y_mean = torch.unsqueeze(y_mean, dim=2)
        y_std = torch.std(x, unbiased=False, dim=2)
        y_std = torch.unsqueeze(y_std, dim=2)
        x = (x - y_mean) / y_std
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class PCTLayer(nn.Module):
    def __init__(self, fea_size,gate_channel, groups=4):
        super(PCTLayer, self).__init__()
        size = lambda x, y:x*y
        fea_size = _pair(fea_size)
        num_fea = size(fea_size[0], fea_size[1])

        self.channels = gate_channel
        self.groups = groups
        self.compress = ChannelPool()
        self.chan = Channels()

        self.weight = Parameter(torch.Tensor(1, 7))
        self.weight.data.fill_(0)
        self.bias = Parameter(torch.Tensor(num_fea))
        self.bias.data.fill_(0)

        self.sigmoid = nn.Sigmoid()

    def _style_integration(self, t):
        z = t * self.weight[None, :, :]  # B x C x 3

        z = torch.sum(z, dim=2)[:, :, None, None] + self.bias[None, :, None, None]  # B x C x 1 x 1
        # z_hat = self.bn(z)

        return z

    def forward(self, x):
        b, c, height, width = x.size()

        x_0 = self.compress(x).view(b, 1, height * width)

        # two scales
        x_1 = self.compress(x[:, :self.channels // 2, :, :]).view(b, 1, height * width)
        x_2 = self.compress(x[:, self.channels // 2:, :, :]).view(b, 1, height * width)

        # four scales
        pool_0 = self.compress(x[:, :self.channels // 4, :, :]).view(b, 1, height * width)
        pool_1 = self.compress(x[:, self.channels // 4:self.channels // 2, :, :]).view(b, 1, height * width)
        pool_2 = self.compress(x[:, self.channels // 2:self.channels * 3 // 4, :, :]).view(b, 1, height * width)
        pool_3 = self.compress(x[:, self.channels * 3 // 4:, :, :]).view(b, 1, height * width)
        # print(x_2.size())
        # y3 = self.avg_pool4(x).view(b, 16 * c)
        y_t = torch.cat((x_0, x_1, x_2, pool_0, pool_1, pool_2, pool_3), 1)
        y_t = self.chan(y_t)
        y = y_t.transpose(1, 2)

        out = self._style_integration(y)

        out = out.transpose(1, 2)
        out = self.sigmoid(out.view(b, 1, height, width))
        # broadcasting
        return x * out

if __name__ == "__main__":
    x = torch.randn(1, 64, 64, 64).cuda()
    m = PCTLayer((64, 64), 64, 4).cuda()
    m(x)