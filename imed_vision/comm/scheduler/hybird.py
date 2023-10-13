# -*- coding:utf-8 -*-
"""
    FileName: hybird
    Author: 12718
    Create Time: 2023-01-09 09:59
"""
from .scheduler import LRScheduler


class WarmupPoly(LRScheduler):
    def __init__(self, optimizer,  num_images, batch_size, epochs, gamma=0.9, start=-1, drop_last=False, warmup_iter=1000):
        super(WarmupPoly, self).__init__(optimizer, start)
        if num_images % batch_size == 0 or drop_last:
            total_iterations = num_images // batch_size * epochs
        else:
            total_iterations = (num_images // batch_size + 1) * epochs
        self.warmup_iter = warmup_iter
        self.total_iterations = total_iterations
        self.poly_total = total_iterations - warmup_iter + 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_images = num_images
        self.epochs = epochs
    def get_lr(self):
        def calc_lr(group):
            if self.current_step < self.warmup_iter:
                lr = group["initial_lr"] * (self.current_step / self.warmup_iter)
            else:
                lr = group["initial_lr"] * (1-(self.current_step - self.warmup_iter)/self.poly_total)**self.gamma
            return lr
        return [calc_lr(group) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            key:value
            for key, value in self.__dict__.items()
            if key in ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs", "warmup_iter"]
        }

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs", "warmup_iter"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)

if __name__ == "__main__":
    import torch
    model = torch.nn.Linear(1024, 20)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sche = WarmupPoly(opt, gamma=0.9, warmup_iter=10, batch_size=1, epochs=100, num_images=1)
    for i in range(100):
        print(i, sche.get_lr())
        sche.step()