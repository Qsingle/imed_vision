# -*- coding:utf-8 -*-
"""
    FileName: ema
    Author: 12718
    Create Time: 2023-02-22 11:22
"""
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel, DataParallel
from copy import deepcopy
from typing import Any

def is_distributed(model):
    """
    Whether the model is distributed
    Args:
        model (nn.Module): the model

    Returns:
        bool(is distributed or not)
    """
    return type(model) in (DistributedDataParallel, DataParallel)

class ModelEMA(nn.Module):
    """
        Model Exponential Moving Average

        References:
            timm:https://github.com/huggingface/pytorch-image-models/blob/a32c4eff69a3b0c117708003579e9a1c14abc0d6/timm/utils/model_ema.py#L82

    """
    def __init__(self, model, decay=0.999, device=None):
        super(ModelEMA, self).__init__()
        self.model = deepcopy(model.module if is_distributed(model) else model)
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.model.to(device)

    @torch.no_grad()
    def _update(self, model, update_fn):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            if self.device is not None:
                model_p.to(self.device)
            updated_para = update_fn(ema_p, model_p)
            ema_p.copy_(updated_para)

    def update(self, model):
        self._update(model, lambda e, m: self.decay*e + (1-self.decay)*m)
        
    def set(self, model):
        self._update(model, lambda e, m: m)

    def state_dict(self, *args: Any,
               destination: Any = None,
               prefix: Any = '',
               keep_vars: Any = False):
        state = {
            "model": self.model.state_dict(),
            "decay": self.decay,
            "device": self.device
        }
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        tmp_state = {}
        keys = ["model", "decay", "device"]
        for key in keys:
            if strict:
                if key not in state_dict:
                    raise KeyError(
                        "key '{}'' is not specified in "
                        "state_dict when loading state dict".format(key)
                    )
            tmp_state[key] = state_dict[key]
        self.model.load_state_dict(tmp_state["model"])
        tmp_state.pop("model")
        self.__dict__.update(tmp_state)