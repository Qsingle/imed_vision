# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:hybird
    author: 12718
    time: 2022/12/12 9:08
    tool: PyCharm
"""

from .scheduler import LRScheduler


class CyclicLR(LRScheduler):
    def __init__(self, optimizer, T=5, warm_up_interval=1, gamma=0.1,
                 lr_decay_epochs=[50, 100, 130, 160, 190, 220, 250, 280],
                 start=-1, stepping=True
                 ):
        super(CyclicLR, self).__init__(optimizer, start)
        self.T = T
        self.count_step = 1
        self.lr_decay_epochs = lr_decay_epochs
        self.gamma = gamma
        self.cycle_step = 0
        self.warm_up_interval = warm_up_interval
        self.stepping = stepping


    def state_dict(self):
        state_dict = {
            key:value
            for key, value in self.__dict__.items()
            if key in ["current_step", "count_step", "lr_decay_epochs", "T", "gamma", "cycle_step"]
        }
        return state_dict

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["current_step", "count_step",
                "lr_decay_epochs", "T", "gamma",
                "cycle_step", "warm_up_interval"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)

    def get_lr(self):
        def calc_lr(group):
            if self.count_step >= len(self.lr_decay_epochs):
                self.stepping = False
                self.count_step -= 1
            if self.current_step % self.lr_decay_epochs[self.count_step] == 0 and self.stepping:
                group["initial_lr"] = group["initial_lr"] * self.gamma
                self.count_step += 1
                self.cycle_step = 0
            lr = group["initial_lr"]
            if self.cycle_step >= self.warm_up_interval:
                if self.cycle_step < self.T:
                    current_t = self.cycle_step
                    lr = (group["initial_lr"]*self.T) - (current_t % self.T) * group["initial_lr"]
                    self.cycle_step += 1
                    if self.cycle_step >= self.T:
                        self.cycle_step = 0
                else:
                    self.cycle_step = 0
            else:
                self.cycle_step += 1
                if self.cycle_step == self.warm_up_interval:
                    self.warm_up_interval = 0
            return lr
        return [calc_lr(group) for group in self.optimizer.param_groups]

class LinearLR(LRScheduler):
    def __init__(self, optimizer, max_epochs, start=-1):
        super(LinearLR, self).__init__(optimizer, start)
        self.max_epochs = max_epochs

    def get_lr(self):
        def calc_lr(group):
            base_lr = group['initial_lr']
            lr = base_lr - (base_lr * (self.current_step / (self.max_epochs)))
            return lr
        return [calc_lr(group) for group in self.optimizer.param_groups]

    def state_dict(self):
        state_dict = {
            key:value
            for key, value in self.__dict__.items()
            if key in ["current_step", "max_epochs"]
        }
        return state_dict

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["current_step", "max_epochs"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)


class HybridLR(LRScheduler):
    def __init__(self, optimizer, clr_max, max_epochs, cycle_lent=5,
                 warm_up_interval=1, gamma=0.1,
                 lr_decay_epochs=[50, 100, 130, 160, 190, 220, 250, 280],
                 start=-1, stepping=True):
        super(HybridLR, self).__init__(optimizer, start)
        self.linear_epochs = max_epochs - clr_max + 1
        self.clr_epochs = clr_max
        self.linear = LinearLR(optimizer, max_epochs=self.linear_epochs)
        self.clr = CyclicLR(optimizer, T=cycle_lent,
                            gamma=gamma, warm_up_interval=warm_up_interval,
                            lr_decay_epochs=lr_decay_epochs, stepping=stepping)

    def step(self):
        self.current_step += 1
        if self.current_step < self.clr_epochs:
            self.clr.step()
        else:
            self.linear.step()

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key in ["current_step", "base_lrs", 'linear_epochs', 'clr_epochs', 'linear', 'clr']
        }
        return state_dict

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["current_step", "base_lrs", 'linear_epochs', 'clr_epochs', 'linear', 'clr']
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

if __name__ == "__main__":
    import torch
    model = torch.nn.Linear(1024, 20)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sche = HybridLR(opt, gamma=0.5, max_epochs=100, clr_max=50)
    for i in range(100):
        print(i, sche.get_lr())
        sche.step()