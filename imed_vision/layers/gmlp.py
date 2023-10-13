# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gmlp
    author: 12718
    time: 2022/8/26 14:30
    tool: PyCharm
    From:https://arxiv.org/abs/2105.08050
"""
import torch
import torch.nn as nn

from .dropout import DropPath

__all__ = ["SGU", "GateMLP", "CGB", "MAB"]

class SGU(nn.Module):
    """
        Implementation of the Spatial Gating Unit in gMLP.
        "Pay Attention to MLPs"<https://arxiv.org/abs/2105.08050>
    """
    def __init__(self, dim, seq_len):
        super(SGU, self).__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)
        nn.init.ones_(self.spatial_proj.bias)

    def forward(self, x):
        """

        Args:
            x: Tensor([bs, seq_len, dim])

        Returns:
            Gated tensor
        """
        assert x.size()[-2] % 2 == 0, "The dim must divided by two, " \
                                   "expect {}%2 = 0, but got".format(x.size()[-2], x.size()[-2] % 2)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v.transpose(-1, -2)).transpose(-1, -2)
        out = u*v
        return out


class GateMLP(nn.Module):
    def __init__(self,  seq_len, dim=128, dim_ffn=768, drop_path_rate=0.):
        """
            Implementation of the gateMLP layer in gMLP.
            "Pay Attention to MLPs"<https://arxiv.org/abs/2105.08050>
        """
        super(GateMLP, self).__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.channel_proj_pre = nn.Linear(dim, dim_ffn)
        self.act = nn.GELU()
        self.sgu = SGU(dim_ffn, seq_len)
        self.channel_proj_after = nn.Linear(dim_ffn // 2, dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        net = self.pre_norm(x) #bs, h*w, ch
        net = self.channel_proj_pre(net)
        net = self.act(net)
        net = self.sgu(net)
        net = self.channel_proj_after(net)
        net = self.drop_path(net) + x
        return net

def rerange(x, patch_size, channel_format="channel_last"):
    """
    Rerange the tensor based on patch size.
    Args:
        x (tensor): input tensor
        patch_size (tuple): size of the patch

    Returns:
        tensor: after rerange
    """

    if channel_format == "channel_last":
        bs, h, w, c = x.size()
        patch_h, patch_w = patch_size
        assert h % patch_h == 0 and w % patch_w, "Feature size {} must divide by patch size {}".format((h, w),
                                                                                                       (patch_h,
                                                                                                        patch_w))
        out = x.reshape(bs, patch_size[0], h // patch_size[0],  patch_size[1], w // patch_size[1], c)
        out = out.permute(0, 1, 3, 2, 4, 5) #bs, psize[0], psize[1], num_ph, num_pw, ch
        out = out.reshape(bs, patch_size[0]*patch_size[1], -1, c)
    elif channel_format == "channel_first":
        bs, c, h, w = x.size()
        patch_h, patch_w = patch_size
        assert h % patch_h == 0 and w % patch_w == 0, "Feature size {} must divide by patch size {}".format((h,w),
                                                                                        (patch_h, patch_w))
        out = x.reshape(bs, c, patch_size[0], h // patch_size[0], patch_size[1], w // patch_size[1])
        out = out.permute(0, 1, 2, 4, 3, 5)  # bs,ch, psize[0], psize[1], num_ph, num_pw
        out = out.reshape(bs, c, patch_size[0] * patch_size[1], -1)
    else:
        raise ValueError("Unsupported channel format:{}".format(channel_format))
    return out

def unbind(x, grid_size, patch_size, channel_format="channel_last"):
    """
    Unbind the tensor from patch.
    Args:
        x (tensor): input tensor
        grid_size (tuple): size of the grid
        patch_size (tuple): size of the patch
        channel_format (str): format of the channel

    Returns:
        tensor: tensor after unbind
    """
    bs = x.size(0)
    if channel_format == "channel_last":
        ch = x.size(-1)
        out = x.reshape(bs, grid_size[0], grid_size[1], patch_size[0], patch_size[1], ch)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(bs, grid_size[0]*patch_size[0], grid_size[1]*patch_size[1], ch)
    elif channel_format == "channel_first":
        ch = x.size(1)
        out = x.reshape(bs, ch, grid_size[0], grid_size[1], patch_size[0], patch_size[1])
        out = out.permute(0, 1, 2, 4, 3, 5) #bs, ch, g_size[0], p_size[0], g_size[1], p_size[1]
        out = out.reshape(bs, ch, grid_size[0] * patch_size[0], grid_size[1] * patch_size[1])
    else:
        raise ValueError("Unsupported channel format:{}".format(channel_format))
    return out

class BlockMLP(nn.Module):
    def __init__(self, dim, img_size, patch_size=(16, 16), expansion_rate=4, drop_rate=0.):
        super(BlockMLP, self).__init__()
        size = lambda x,y:x*y
        seq_len = int(size(img_size[0], img_size[1]) / size(patch_size[0], patch_size[1]))
        self.mlp = GateMLP(seq_len, dim, dim_ffn=int(dim*expansion_rate), drop_path_rate=drop_rate)
    def forward(self, x):
        out = self.mlp(x)
        return out


class MAB(nn.Module):
    def __init__(self, dim, img_size, patch_size=(16, 16), grid_size=(16, 16),
                 expansion_rate=2, gmlp_expansion_rate=2, drop_rate=0.):
        """
         Multi-Axis Gated MLP.
        "MAXIM: Multi-Axis MLP for Image Processing"<https://arxiv.org/abs/2201.02973v1>
        Args:
            dim (int): dimensions
            img_size (tuple): size of the image
            patch_size (tuple): patch size
            grid_size (tuple): size of the grid
            expansion_rate (float): expansion rate, default:2.
            gmlp_expansion_rate (float): expansion rate for the gmlp block, default:2
            drop_rate (float): dropout rate, default:0.
        """
        super(MAB, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dense_pre = nn.Linear(dim, int(dim*expansion_rate))
        self.patch_size = patch_size
        self.block_mlp = BlockMLP(dim, img_size, patch_size, expansion_rate=gmlp_expansion_rate,
                                  drop_rate=drop_rate)
        self.grid_size = grid_size
        self.grid_mlp = BlockMLP(dim, img_size,
                                 patch_size=[img_size[0] // grid_size[0], img_size[1] // grid_size[1]],
                                 expansion_rate=expansion_rate)
        self.dense_fuse = nn.Linear(dim*expansion_rate, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.dense_pre(x)
        x = self.act(x)
        block_branch, grid_branch = x.chunk(2, dim=-1)
        bs, h, w, ch = grid_branch.size()
        grid_size = (h // self.patch_size[0], w // self.patch_size[1])
        patch_size = self.patch_size
        block_branch = rerange(block_branch, patch_size=patch_size)
        block_branch = self.block_mlp(block_branch)
        block_branch = unbind(block_branch, grid_size, patch_size)
        patch_size = (h // self.grid_size[0], w // self.grid_size[1])
        grid_size = self.grid_size
        grid_branch = rerange(grid_branch, patch_size)
        grid_branch = self.grid_mlp(grid_branch)
        grid_branch = unbind(grid_branch, grid_size, patch_size)
        x = torch.cat([block_branch, grid_branch], dim=-1)
        x = self.dense_fuse(x).permute(0, 3, 1, 2)
        x = self.drop_path(x) + identity
        return x

class CGB(nn.Module):
    def __init__(self, dim_x, dim_y, dim, img_size, upsample_y=True, patch_size=(16, 16),
                 grid_size=(16, 16), expansion_rate=2, gmlp_expansion_rate=2, drop_rate=0.):
        """
        Cross Gate MLP Block.
        ""<>
        Args:
            dim_x (int): dimension of the input x
            dim_y (int): dimension of the input y
            dim (int): dimension of output
            img_size (tuple): size of the image
            upsample_y (bool): whether upsampling the y
            patch_size (tuple): size of the patch
            grid_size (tuple): size of the grid
            expansion_rate (float): expansion rate
            gmlp_expansion_rate (float): expansion rate for the gate mlp block
            drop_rate (float): dropout rate
        """
        super(CGB, self).__init__()
        self.dense_x = nn.Conv2d(dim_x, dim, 1, 1, 0)
        self.norm_x1 = nn.LayerNorm(dim_x)
        self.dense_x1 = nn.Linear(dim, dim)
        self.cross_x = MAB(dim_x, img_size, patch_size=patch_size, grid_size=grid_size,
                            expansion_rate=expansion_rate, gmlp_expansion_rate=gmlp_expansion_rate,
                            drop_rate=0.)
        self.act = nn.GELU()

        self.upsample_y = nn.ConvTranspose2d(dim_y, dim, 2, 2) if upsample_y else nn.Identity()
        self.dense_y = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm_y1 = nn.LayerNorm(dim_x)
        self.dense_y1 = nn.Linear(dim_x, dim_x)
        self.cross_y = MAB(dim_x, img_size, patch_size=patch_size, grid_size=grid_size,
                            expansion_rate=expansion_rate, gmlp_expansion_rate=gmlp_expansion_rate,
                            drop_rate=0.)

        self.dense_gate_x = nn.Linear(dim_x, dim_x)
        self.dense_gate_y = nn.Linear(dim_x, dim_x)
        self.droppath_x = DropPath(drop_rate)
        self.droppath_y = DropPath(drop_rate)

    def forward(self, x, y):
        """

        Args:
            x (Tensor): (bs, ch, h, w)
            y (Tensor): (bs, ch, h, w)

        Returns:
            (Tensor, Tensor)
        """
        shortcut_x = x

        x = self.dense_x(x).permute(0, 2, 3, 1)
        x = self.norm_x1(x)
        x = self.dense_x1(x)
        x = self.act(x).permute(0, 3, 1, 2)
        gx = self.cross_x(x)
        y = self.upsample_y(y)
        shortcut_y = y
        y = y
        y = self.dense_y(y).permute(0, 2, 3, 1)
        y = self.norm_y1(y)
        y = self.dense_y1(y)
        y = self.act(y).permute(0, 3, 1, 2)
        gy = self.cross_y(y)

        y = gx * y
        y = self.droppath_x(self.dense_gate_y(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) + shortcut_y

        x = x * gy
        x = self.droppath_x(self.dense_gate_x(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) + y + shortcut_x
        return x, y

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(1, 3, 512, 512).to(device)
    # hidden = nn.Conv2d(3, 3, 16, 16).to(device)
    # embd = hidden(x)
    # embd = nn.Conv2d(3, 128, 1, 1).to(device)(embd).reshape(1, 128, -1).transpose(-2, -1)
    # model = GateMLP(embd.shape[1]).to(device)
    # out = model(embd)
    # print(out.shape)
    x = torch.randn(1, 32, 256, 256).to(device)
    y = torch.randn(1, 16, 128, 128).to(device)
    # model = MAB(32, (512, 512)).to(device)
    model = CGB(32, 16, 32, (256, 256)).to(device)
    out = model(x, y)
    print(out[0].shape, out[1].shape)