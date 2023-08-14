# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gmlp_unet
    author: 12718
    time: 2022/10/12 15:31
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, groups=4):
        super().__init__()
        assert in_ch % groups == 0, "Except in_ch % groups = 0, but {} % {} = {}".format(in_ch, groups, in_ch % groups)
        group_out_ch = out_ch // groups
        self.branchs = nn.ModuleList()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        for i in range(groups):
            if i == 0:
                self.branchs.append(nn.Conv2d(group_out_ch, group_out_ch, 3, 1, 1, bias=False))
            else:
                self.branchs.append(
                        nn.Conv2d(group_out_ch, group_out_ch, 3, 1, padding=i, dilation=i, bias=False, groups=group_out_ch)
                )
        self.groups = groups
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, 1),
            nn.BatchNorm2d(out_ch)
        )


    def forward(self, x):
        x = self.conv1(x)
        xs = x.chunk(self.groups, dim=1)
        fes = []
        fe = self.branchs[0](xs[0])
        fes.append(fe)
        for i, branch in enumerate(self.branchs[1:], start=1):
            fe = branch(xs[i]) + fe
            fes.append(fe)
        fe = self.conv2(torch.cat(fes, dim=1)) + x
        return fe
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, depth=1, groups=4):
        super(Encoder, self).__init__()
        convs = [MSConv2d(in_ch, out_ch, groups=groups)]
        for i in range(1, depth):
            convs.append(MSConv2d(out_ch, out_ch, groups=groups))
        self.block = nn.Sequential(*convs)
        self.down = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        net = self.block(x)
        down = self.down(net)
        return net, down


class RelPos(nn.Module):
    def __init__(self, num_head, seq_len):
        super().__init__()
        self.rel_pos = nn.Parameter(torch.randn(num_head, seq_len**2), requires_grad=True)
        query_index = torch.arange(seq_len).unsqueeze(0)
        key_index = torch.arange(seq_len).unsqueeze(1)
        relative_index = key_index - query_index + seq_len - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        self.num_head = num_head
        self.seq_len = seq_len
    @torch.no_grad()
    def get_rel_pos(self):
        return torch.index_select(self.rel_pos, 1, self.flatten_index).reshape(self.num_head, self.seq_len, self.seq_len)
    def forward(self, x):
        net = x + self.get_rel_pos().unsqueeze(0)
        return net

class MHA(nn.Module):
    def __init__(self, dim, head_dim, num_head=4, qkv_bias=True, proj_bias=True, rel_pos=None):
        super(MHA, self).__init__()
        self.head_dim = head_dim
        self.gamma = self.head_dim ** -0.5
        self.num_head = num_head
        self.d = head_dim * num_head
        self.q = nn.Conv2d(dim, head_dim * num_head, 1, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, head_dim * num_head, 1, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, head_dim * num_head, 1, 1, bias=qkv_bias)
        # self.talking1 = nn.Conv2d(num_head, num_head, 1, 1)
        # self.talking2 = nn.Conv2d(num_head, num_head, 1, 1)
        self.proj = nn.Conv2d(head_dim * num_head, dim, 1, 1, bias=proj_bias)
        self.rel_pos = rel_pos if rel_pos else nn.Identity()
    def forward(self, x):
        B, CH, H, W = x.size()
        q = self.q(x).permute(0, 2, 1, 3).reshape(-1, self.num_head, self.head_dim, W).transpose(2, 3)
        k = self.k(x).permute(0, 2, 1, 3).reshape(-1, self.num_head, self.head_dim, W)
        # v = self.v(x).permute(0, 2, 1, 3).reshape(-1, self.num_head, self.head_dim, W).transpose(2, 3)
        v = self.v(x)
        # v_local = self.v_local(v).permute(0, 2, 1, 3).reshape(-1, self.num_head, self.dh, W).transpose(2, 3)
        v = v.permute(0, 2, 1, 3).reshape(-1, self.num_head, self.head_dim, W).transpose(2, 3)
        att = q @ k
        att = att * self.gamma
        att = self.rel_pos(att)
        # att = self.talking1(att)
        att = torch.softmax(att, dim=3)
        # att = self.talking2(att)
        net = att @ v
        net = net.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        net = self.proj(net) + x
        return net

class CroA(nn.Module):
    def __init__(self, dim, head_dim, num_head=4, qkv_bias=True, proj_bias=True, rel_pos_x=None, rel_pos_y=None):
        super(CroA, self).__init__()
        self.att_x = MHA(dim // 2, head_dim, num_head, qkv_bias=qkv_bias, proj_bias=proj_bias, rel_pos=rel_pos_x)
        self.att_y = MHA(dim // 2, head_dim, num_head, qkv_bias=qkv_bias, proj_bias=proj_bias, rel_pos=rel_pos_y)
        self.proj = nn.Conv2d(dim, dim, 1, 1)

    def forward(self, x):
        axis_x, axis_y = x.chunk(2, dim=1)
        axis_x = self.att_x(axis_x)
        axis_y = axis_y.transpose(2, 3)
        axis_y = self.att_y(axis_y).transpose(2, 3)
        net = self.proj(torch.cat([axis_x, axis_y], dim=1))
        return net

class CrossBlock(nn.Module):
    def __init__(self, dim, out_dim, num_head=4, qkv_bias=True, proj_bias=True, use_rel_pos=False, img_size=(7, 7), drop_path_rate=0.):
        super(CrossBlock, self).__init__()
        rel_pos_x = None
        rel_pos_y = None
        if use_rel_pos:
            rel_pos_x = RelPos(num_head, img_size[1])
            rel_pos_y = RelPos(num_head, img_size[0])
        self.att = CroA(dim, head_dim=dim//2, num_head=num_head, qkv_bias=qkv_bias, proj_bias=proj_bias,
                        rel_pos_x=rel_pos_x, rel_pos_y=rel_pos_y)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim*2, out_dim, 1, 1, 0)
        )
        self.layer_scale1 = nn.Parameter(torch.ones((dim, 1, 1)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(torch.ones((out_dim, 1, 1)), requires_grad=True)
        self.skip = nn.Identity()
        if dim != out_dim:
            self.skip = nn.Conv2d(dim, out_dim, 1, 1)

    def forward(self, x):
        net = self.att(x)*self.layer_scale1
        net = net + x
        net = self.mlp(net)*self.layer_scale2 + self.skip(net)
        return net

class Decoder(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, att=True, num_head=4, qkv_bias=True,
                          drop_att=0., drop_proj=0., img_size=(7, 7), depth=2, use_rel_pos=False,
                 groups=4):
        super(Decoder, self).__init__()
        self.att = att
        block = [
                MSConv2d(in_ch1+in_ch2, out_ch, groups=groups)
            ]

        if att:
            for i in range(1, depth):
                block.append(
                    CrossBlock(out_ch, out_ch, num_head, qkv_bias, use_rel_pos=use_rel_pos,
                               img_size=img_size, drop_path_rate=drop_att)
                )
        else:
            for i in range(1, depth):
                block.append(MSConv2d(out_ch, out_ch, groups=groups))
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch2, 1, 1),
            nn.BatchNorm2d(in_ch2)
        )
        self.block = nn.Sequential(*block)

    def forward(self, low, high):
        high = F.interpolate(high, size=low.size()[2:], mode="bilinear", align_corners=True)
        high = self.up_conv(high)
        net = torch.cat([low, high], dim=1)
        net = self.block(net)
        return net

class GANet(nn.Module):
    def __init__(self, ch=3, num_classes=12, qkv_bias=True, drop_att=0., drop_proj=0.,
                 num_head=8, img_size=(512, 512), depths=[2, 2, 6, 2], chs=[32, 64, 96, 128],
                 ):
        super(GANet, self).__init__()
        _img_size = list(img_size)
        self.stem = nn.Sequential(
            nn.Conv2d(ch, chs[0], 3, 2, 1),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(),
            nn.Conv2d(chs[0], chs[0], 3, 1, 1),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU()
        )
        down_size = lambda x: [x[0] // 2, x[1] // 2]
        up_size = lambda x: [x[0]*2, x[1]*2]
        # _img_size = down_size(_img_size)
        self.s1 = Encoder(chs[0], chs[0], depths[0])
        _img_size = down_size(_img_size)
        self.s2 = Encoder(chs[0], chs[1], depths[1])
        _img_size = down_size(_img_size)
        self.s3 = Encoder(chs[1], chs[2], depths[2])
        _img_size = down_size(_img_size)
        blocks = [CrossBlock(chs[2], chs[3], num_head=num_head, qkv_bias=qkv_bias, drop_path_rate=drop_att, use_rel_pos=False, img_size=_img_size)]
        for i in range(1, depths[3]):
            blocks.append(CrossBlock(chs[3], chs[3], num_head=num_head, qkv_bias=qkv_bias, drop_path_rate=drop_att, use_rel_pos=False, img_size=_img_size))
        self.s4 = nn.Sequential(*blocks)
        # self.s4_down = nn.Conv2d(bash_ch*8, bash_ch*8, 3, 2, 1)
        # _img_size = down_size(_img_size)
        # blocks = [CrossBlock(bash_ch*8, bash_ch*16, num_head=num_head, qkv_bias=qkv_bias, drop_att=drop_att,
        #                      drop_proj=drop_proj, img_size=_img_size)]
        # for i in range(1, 2):
        #     blocks.append(CrossBlock(bash_ch*16, bash_ch*16, num_head=num_head, qkv_bias=qkv_bias, drop_att=drop_att,
        #                      drop_proj=drop_proj, img_size=_img_size))
        # self.s5 = nn.Sequential(
        #     *blocks
        # )
        _img_size = up_size(_img_size)
        self.d1 = Decoder(chs[2], chs[3], chs[2], att=False, num_head=num_head, qkv_bias=qkv_bias,
                          drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size, depth=depths[2], use_rel_pos=False)
        _img_size = up_size(_img_size)
        self.d2 = Decoder(chs[1], chs[2], chs[1], att=False, num_head=num_head, qkv_bias=qkv_bias,
                          drop_att=drop_att, drop_proj=drop_proj, img_size=_img_size, depth=depths[1])
        self.d3 = Decoder(chs[0], chs[1], chs[0], att=False, depth=depths[0])
        # self.d4 = Decoder(bash_ch*2, bash_ch, bash_ch, att=False)

        # self.align = FuseMBConv(chs[0], chs[0])
        self.out_conv = nn.Conv2d(chs[0], num_classes, 1, 1)



    def forward(self, x):
        stem = self.stem(x)
        f1, down = self.s1(stem)
        f2, down = self.s2(down)
        f3, down = self.s3(down)
        f4 = self.s4(down)
        # f4_down = self.s4_down(f4)
        # f5 = self.s5(f4_down)
        d1 = self.d1(f3, f4)
        d2 = self.d2(f2, d1)
        d3 = self.d3(f1, d2)
        # d4 = self.d4(f1, d3)
        out_fea = F.interpolate(d3, size=x.size()[2:], mode="bilinear", align_corners=True)
        # out_fea = self.align(out_fea)
        out = self.out_conv(out_fea)
        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 1024, 512).cuda()
    model = GANet(3, 12, img_size=(512, 512)).cuda()
    from pytorch_benchmark.benchmark import measure_flops
    from pytorch_benchmark.benchmark import format_num
    out = model(x)
    flops = measure_flops(model, x, print_details=False)
    print(format_num(flops))
    # print(out.shape)