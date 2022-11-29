# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dedcgcnee
    author: 12718
    time: 2022/7/27 15:19
    tool: PyCharm
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

def edge_conv2d64(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 64, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 64, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(im.device)
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 64, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 64, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).to(im.device)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 64, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 64, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).to(im.device)
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 64, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 64, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).to(im.device)
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def edge_conv2d128(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 128, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 128, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(im.device)
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 128, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 128, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).to(im.device)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 128, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 128, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).to(im.device)
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 128, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 128, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).to(im.device)
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def edge_conv2d256(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 256, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 256, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(im.device)
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 256, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 256, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).to(im.device)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 256, axis=1)
    #sobel_kernel2 = np.repeat(sobel_kernel2, 256, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).to(im.device)
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 256, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 256, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).to(im.device)
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class GCN(nn.Module):
    def __init__(self, channel, img_size):
        super(GCN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.para = torch.nn.Parameter(torch.ones((1, channel) + img_size, dtype=torch.float32))
        self.adj = torch.nn.Parameter(torch.ones((channel, channel), dtype=torch.float32))

    def forward(self, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()
        fea_matrix = x.view(b, c, H * W)
        c_adj = self.avg_pool(x).view(b, c)

        m = torch.ones((b, c, H, W), dtype=torch.float32)

        for i in range(0, b):
            t1 = c_adj[i].unsqueeze(0)
            t2 = t1.t()
            c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1 - t2) - 0.5) - 0.5) * 2
            c_adj_s = (c_adj_s.t() + c_adj_s) / 2

            output0 = torch.mul(torch.mm(self.adj * c_adj_s, fea_matrix[i]).view(1, c, H, W), self.para)

            m[i] = output0

        output = torch.nn.functional.relu(m.to(x.device))

        return output


class EEblock(nn.Module):
    def __init__(self, channel):
        super(EEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1))
        self.sconv31 = nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, y, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b, c, 1, 1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b, c, 1, 1))

        k = x1 * map_y31 + x2 * map_y13 + x

        return k

def edge_conv2d(im):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(im.device)
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).to(im.device)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).to(im.device)
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).to(im.device)
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect + edge_detect1 + edge_detect2 + edge_detect3

    return sobel_out


class DEDCGCNEE(nn.Module):
    def __init__(self, in_c, n_classes, img_size=(512, 512)):
        super(DEDCGCNEE, self).__init__()
        self.n_classes = n_classes
        self.down = downsample()

        self.Conv1 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Conv5 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv6 = conv_block(ch_in=64, ch_out=128)
        self.Conv7 = conv_block(ch_in=128, ch_out=256)
        self.Conv8 = conv_block(ch_in=256, ch_out=512)

        self.GCN_layer = GCN(channel=512, img_size=(img_size[0]//8, img_size[1] // 8))

        self.EEblock1 = EEblock(channel=256)
        self.EEblock2 = EEblock(channel=128)
        self.EEblock3 = EEblock(channel=64)

        self.Up4 = up_conv(512, 256)
        self.Up_conv4 = Decoder(512, 256)

        self.Up3 = up_conv(256, 128)
        self.Up_conv3 = Decoder(256, 128)

        self.Up2 = up_conv(128, 64)
        self.Up_conv2 = Decoder(128, 64)

        self.fconv = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)



    def forward(self, x):
        y = edge_conv2d(x)
        x1 = self.Conv1(x)

        x2 = self.down(x1)
        x2 = self.Conv2(x2)

        x3 = self.down(x2)
        x3 = self.Conv3(x3)

        x4 = self.down(x3)
        x4 = self.Conv4(x4)

        e1 = edge_conv2d64(x1)
        e2 = edge_conv2d128(x2)
        e3 = edge_conv2d256(x3)

        y1 = self.Conv5(y) + e1

        y2 = self.down(y1)
        y2 = self.Conv6(y2) + e2

        y3 = self.down(y2)
        y3 = self.Conv7(y3) + e3

        y4 = self.down(y3)
        y4 = self.Conv8(y4)
        GCN_output = self.GCN_layer(x4 + y4)

        d4 = self.Up4(GCN_output)
        m3 = self.EEblock1(y3, x3)
        l4 = torch.cat((m3, d4), dim=1)
        d4 = self.Up_conv4(l4)

        d3 = self.Up3(d4)
        m2 = self.EEblock2(y2, x2)
        l3 = torch.cat((m2, d3), dim=1)
        d3 = self.Up_conv3(l3)

        d2 = self.Up2(d3)
        m1 = self.EEblock3(y1, x1)
        l2 = torch.cat((m1, d2), dim=1)
        d2 = self.Up_conv2(l2)

        out = self.fconv(d2)

        return out