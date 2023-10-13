#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   fpn.py
    @Time    :   2023/07/28 13:34:22
    @Author  :   12718 
    @Version :   1.0
'''

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, chs=[512, 1024, 2048], hidden_state=256, use_last_stage=False) -> None:
        """
            Feature Pyramid Network introduced in 
            "Feature Pyramid Networks for Object Detection"<https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html>
        Args:
            chs (Union[List,tuple]): The number of channels for the input features. Defaults to [512, 1024, 2048].
            hidden_state (int, optional): The dimension of the hidden state. Defaults to 256.
            use_last_stage (bool, optional): Whether use the last input feature to do the downsampling. Defaults to False.
        """
        super(FPN, self).__init__()
        self.proj_convs = nn.ModuleList()
        for ch in chs:
            self.proj_convs.append(nn.Conv2d(ch, hidden_state, 1, 1, 0))
        
        self.p_convs = nn.ModuleList()
        self.use_last_state = use_last_stage
        for _ in range(len(chs)):
            self.p_convs.append(
                nn.Conv2d(hidden_state, hidden_state, 3, 1, 1)
            )
        if use_last_stage:
            self.p_convs.append(
                nn.Conv2d(chs[-1], hidden_state, 3, 2, 1)
            )
        else:
            self.p_convs.append(
                nn.Conv2d(hidden_state, hidden_state, 3, 2, 1)
            )
        self.p_convs.append(nn.Conv2d(hidden_state, hidden_state, 3, 2, 1))
    
    def forward(self, fes):
        pfs = []
        for fe, conv in zip(fes, self.proj_convs):
            pfs.append(conv(fe))
        
        for i in range(len(pfs) - 2, -1, -1):
            pfs[i] = pfs[i] + F.interpolate(pfs[i+1], size=pfs[i].size()[2:], mode="nearest")
        
        for i in range(len(self.p_convs)-2):
            pfs[i] = self.p_convs[i](pfs[i])
        if self.use_last_state:
            fe = fes[-1]
        else:
            fe = pfs[i]
        for i in range(len(self.p_convs)-2, len(self.p_convs)):
            fe = self.p_convs[i](fe)
            pfs.append(fe)
        return pfs

if __name__ == "__main__":
    import torch
    fes = [
        torch.randn(1, 512, 56, 56),
        torch.randn(1, 1024, 28, 28),
        torch.randn(1, 2048, 14, 14)
    ]
    m = FPN(use_last_stage=False)
    outs = m(fes)
    for out in outs:
        print(out.shape)