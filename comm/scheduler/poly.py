# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:poly
    author: 12718
    time: 2022/1/15 16:44
    tool: PyCharm
"""

from .scheduler import LRScheduler

class PolyLRScheduler(LRScheduler):
    def __init__(self, optimizer,  num_images, batch_size, epochs, gamma=0.9, start=-1, drop_last=False):
        super(PolyLRScheduler, self).__init__(optimizer, start)
        if num_images % batch_size == 0 or drop_last:
            total_iterations = num_images // batch_size * epochs
        else:
            total_iterations = (num_images // batch_size + 1) * epochs

        self.total_iterations = total_iterations
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_images = num_images
        self.epochs = epochs
        print("Initial learning rate set to:{}".format([group["initial_lr"] for group
                                                        in self.optimizer.param_groups]))

    def get_lr(self):
        def calc_lr(group):
            lr = group["initial_lr"] * (1-self.current_step/self.total_iterations)**self.gamma
            return lr
        return [calc_lr(group) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            key:value
            for key, value in self.__dict__.items()
            if key in ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        }

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)