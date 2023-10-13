#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   msbdn.py
    @Time    :   2023/09/13 11:14:42
    @Author  :   12718 
    @Version :   1.0
'''

import torch
import torch.nn as nn


class FDL(nn.Module):
    def __init__(self, lambda_p=1., lambda_a=1.) -> None:
        """Implementation of the Frequency domain Losses
         "Source-Free Domain Adaptation for Real-World Image Dehazing"<https://dl.acm.org/doi/abs/10.1145/3503161.3548410>

        Args:
            lambda_p (float, optional): Trade-off parameters. Defaults to 1.
            lambda_a (float, optional): Trade-off parameters. Defaults to 1.
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_a = lambda_a
    
    def forward(self, o_t, o_s, o_clahe):
        f_t = torch.fft.fft(o_t, norm="forward")
        f_s = torch.fft.fft(o_s, norm="forward")
        f_clahe = torch.fft.fft(o_clahe, norm="backward")
        p_t = torch.arctan(f_t.imag/(f_t.real+1e-5))
        p_s = torch.arctan(f_s.imag/(f_s.real+1e-5))
        a_s = torch.sqrt(f_s.real**2+f_s.imag**2)
        bs, ch, h, w = p_s.shape
        a_clahe = torch.sqrt(f_clahe.real**2+f_clahe.imag**2)
        l_pha = (2*torch.norm(torch.abs(p_t[:, :, h//2, :])-torch.abs(p_s[:, :, h//2, :]), p=1)).mean()
        l_amp = (2*torch.norm(torch.abs(a_s[:, :, h//2, :])-torch.abs(a_clahe[:, :, h//2, :]), p=1)).mean()
        loss = self.lambda_p*l_pha + self.lambda_a*l_amp
        return loss

class DCP(nn.Module):
    def __init__(self, win_size=3) -> None:
        """
            Implementation of the dark channel prior loss
        Args:
            win_size (int, optional): The size of window to do the dehaze. Defaults to 3.
        """
        super().__init__()
        self.win_size = win_size
    
    def forward(self, x):
        x = nn.functional.pad(x, pad=[self.win_size//2, self.win_size//2, self.win_size//2, self.win_size//2])
        x = x.unfold(2, self.win_size, 1).unfold(3, self.win_size, 1)
        x = torch.min(x, dim=1)[0]
        x = torch.norm(x, p=1)
        return x

class PPL(nn.Module):
    def __init__(self, lambda_d=1e-3, lambda_c=1e-3, win_size=3, eps=1e-8) -> None:
        """Implementation of the Physical Prior Losses
         "Source-Free Domain Adaptation for Real-World Image Dehazing"<https://dl.acm.org/doi/abs/10.1145/3503161.3548410>

        Args:
            lambda_d (float, optional): [description]. Defaults to 1e-3.
            lambda_c (float, optional): Trade. Defaults to 1e-3.
            win_size (int, optional): window size for dcp. Defaults to 3.
            eps (float, optional): Non-zero value. Defaults to 1e-8.
        """
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c
        self.eps = eps
        self.dcp = DCP(win_size=win_size)
    
    def forward(self, o_s):
        dcp = self.dcp(o_s)
        max_c = torch.max(o_s, dim=1)[0]
        min_c = torch.min(o_s, dim=1)[0]
        s = (max_c-min_c) / (max_c + self.eps)
        s[max_c==0] = 0
        v = max_c
        cap = torch.norm(v-s, p=1)
        loss = self.lambda_c*cap + self.lambda_d*dcp
        return loss

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    z = torch.randn(1, 3, 224, 224)
    fdl = FDL()(x, y, z)
    print("fdl:", fdl)
    print(224*224)
    ppl = PPL()(x)
    print(ppl)
