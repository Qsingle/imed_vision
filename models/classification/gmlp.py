# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gmlp
    author: 12718
    time: 2022/8/28 11:22
    tool: PyCharm
"""

import torch.nn as nn

from layers import GateMLP, PatchEmbedding

from .create_model import BACKBONE_REGISTER

model_url = {
    "gmlp_ti_s224_b16": "",
    "gmlp_s_s224_b16": "",
    "gmlp_b_s224_b16": ""
}

class gMLP(nn.Module):
    def __init__(self, in_dim=3, dim=128, num_blocks=30,
                 expansion_rate=6, num_classes=1000,
                 global_pool="avg"):
        super(gMLP, self).__init__()
        self.patch_embd = PatchEmbedding(in_dim, dim)
        blocks = []
        for i in range(num_blocks):
            blocks.append(GateMLP(self.patch_embd.num_patchs, dim, dim_ffn=int(dim*expansion_rate)))
        self.block = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)
        self.global_pool = global_pool

    def forward_features(self, x):
        net = self.patch_embd(x)
        net = net.flatten(2).transpose(1, 2)
        net = self.block(net)
        net = self.norm(net)
        return net

    def forward_head(self, x):
        if self.global_pool == "avg":
            net = x.mean(1)
        elif self.global_pool == "max":
            net = x.max(1)[0]
        else:
            raise ValueError("Unknown pool way:{}".format(self.global_pool))
        net = self.fc(net)
        return net

    def forward(self, x):
        net = self.forward_features(x)
        net = self.forward_head(net)
        return net

def _create_gmlp(arch, pretrained=False, progress=True, **kwargs):
    model = gMLP(**kwargs)
    if pretrained:
        if model_url != "":
            state = torch.hub.load_state_dict_from_url(model_url[arch],
                                                       model_dir="./weights",
                                                       progress=progress)
            model.load_state_dict(state)
        else:
            print("The pretrained weights for {} is not provided now, skip load pretrained weight")
    return model

@BACKBONE_REGISTER.register()
def gmlp_ti_s224_b16(pretrained=False, progress=True, **kwargs):
    kwargs["dim"] = 128
    return _create_gmlp(arch="gmlp_ti_s224_b16", pretrained=pretrained, progress=progress, **kwargs)

@BACKBONE_REGISTER.register()
def gmlp_s_s224_b16(pretrained=False, progress=True, **kwargs):
    kwargs["dim"] = 256
    return _create_gmlp(arch="gmlp_s_s224_b16", pretrained=pretrained, progress=progress, **kwargs)

@BACKBONE_REGISTER.register()
def gmlp_b_s224_b16(pretrained=False, progress=True, **kwargs):
    kwargs["dim"] = 512
    return _create_gmlp(arch="gmlp_b_s224_b16", pretrained=pretrained, progress=progress, **kwargs)

if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 224, 224).to(device)
    model = gMLP(3).to(device)
    out = model(x)
    print(out.shape)