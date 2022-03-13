# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:scheduler
    author: 12718
    time: 2022/1/15 16:44
    tool: PyCharm
"""
from torch.optim import Optimizer
from abc import ABCMeta


class LRScheduler(metaclass=ABCMeta):
    def __init__(self, optimizer: Optimizer, start=-1):

        self.optimizer = optimizer
        self.current_step = start

        if start == -1:
            for group in self.optimizer.param_groups:
                if hasattr(group, "initial_lr"):
                    continue
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in "
                        "param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], self.optimizer.param_groups)
        )

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.current_step += 1
        values = self.get_lr()
        for groups, lr in zip(self.optimizer.param_groups, values):
            groups["lr"] = lr