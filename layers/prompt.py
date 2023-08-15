#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   prompt.py
    @Time    :   2023/08/15 10:54:29
    @Author  :   12718 
    @Version :   1.0
'''

import torch
import torch.nn as nn

__all__ = ["FFTPrompt"]

class FFTPrompt(nn.Module):
    def __init__(self, rate=0.25, prompt_type="highpass") -> None:
        super(FFTPrompt, self).__init__()
        assert prompt_type in ["highpass", "lowpass"], "The prompt type must in " \
        "['highpass', 'lowpass'], but got {}".format(prompt_type)
        self.rate = rate
        self.prompt_type = prompt_type
    
    def forward(self, x):
        fft = torch.fft.fft2(x, norm="forward")
        fft = torch.fft.fftshift(fft)
        h, w = x.shape[2:]
        radio = int((h*w*self.rate)**.5 // 2)
        mask = torch.zeros_like(x)
        c_h, c_w = h // 2, w // 2
        mask[:, :, c_h-radio:c_h+radio, c_w-radio:c_w+radio] = 0
        if self.prompt_type == "highpass":
            fft = fft*(1-mask)
        else:
            fft = fft * mask
        real, imag = fft.real, fft.imag
        shift = torch.fft.fftshift(torch.complex(real, imag))
        inv = torch.fft.ifft2(shift, norm="forward")
        inv = inv.real
        return torch.abs(inv)