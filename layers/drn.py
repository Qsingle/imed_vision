#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   drn.py
    @Time    :   2023/09/13 09:52:34
    @Author  :   12718 
    @Version :   1.0
    @Desc    : Implementation of the domain representation normal module
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRN(nn.Module):
    def __init__(self, dim, eps=1e-5) -> None:
        """Implementation of the Domain Representation Normalization module
         "Source-Free Domain Adaptation for Real-World Image Dehazing"<https://dl.acm.org/doi/abs/10.1145/3503161.3548410>

        Args:
            dim (int): The number of dimension for the feature
            eps (float, optional): Minimal non-zero value to avoid zero. Defaults to 1e-5.
        """
        super(DRN, self).__init__()
        self.eps = eps
        self.di = nn.InstanceNorm2d(dim)
        # self.dv_conv = nn.Sequential(
        #     nn.Conv2d(dim*3, dim, 1, 1),
        #     nn.Conv2d(dim, dim, 1, 1)
        # )
        self.dv_conv = nn.Sequential(
            nn.Conv2d(dim*3, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        di = self.di(x)
        res = x - di
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.mean((x-mean)**2, dim=[2,3], keepdim=True)+self.eps)
        mean = mean.expand_as(x)
        std = std.expand_as(x)
        dv = self.dv_conv(torch.cat([res, mean, std], dim=1))
        out = di + dv
        return out
    

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32).cuda()
    m = DRN(64).cuda()(x)
    print(m.shape)