# -*- coding:utf-8 -*-
"""
    FileName: depth
    Author: 12718
    Create Time: 2023-05-22 19:13
"""
import torch

from collections import OrderedDict

def abs_rel(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    return torch.abs(gt - pred).mean()

def rmse(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    return torch.sqrt(((gt-pred)**2).mean())

def mse(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    return torch.mean((gt-pred)**2)

def depth_a123(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).to(torch.float32).mean()
    a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
    a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()
    return a1, a2, a3
def log_10(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    return torch.abs(torch.log10(pred) - torch.log10(gt)).mean()

def silog(pred, gt):
    assert pred.shape == gt.shape, "The shape of pred must equal gt, got {} and {}".format(pred.shape, gt.shape)
    err = torch.log(pred) - torch.log(gt)
    _silog = torch.sqrt(torch.mean(err**2) - torch.mean(err)**2) * 100
    return _silog
class DepthMetrics:
    def __init__(self):
        self.results = OrderedDict()
        self.reset()

    def reset(self):
        metrics = [
            'mse', 'rmse', 'a1', 'a2', 'a3', 'silog', 'abs_rel']
        for metric in metrics:
            self.results[metric] = []

    def update(self, pred, gt):
        for i in range(pred.shape[0]):
            p = pred[i]
            g = gt[i]
            _mse = mse(p, g)
            _rmse = rmse(p, g)
            _a1, _a2, _a3 = depth_a123(p, g)
            # _log_10 = log_10(p, g)
            _silog = silog(p, g)
            _abs_rel = abs_rel(p, g)
            self.results['mse'].append(_mse.item())
            self.results['rmse'].append(_rmse.item())
            self.results['a1'].append(_a1.item())
            self.results['a2'].append(_a2.item())
            self.results['a3'].append(_a3.item())
            # self.results['log10'].append(_log_10.item())
            self.results['silog'].append(_silog.item())
            self.results['abs_rel'].append(_abs_rel.item())
    def evaluate(self):
        return self.results

if __name__ == "__main__":
    metric = DepthMetrics()
    metric.update(torch.randn(1, 224, 224), torch.randn(1, 224, 224))
    print(metric.evaluate())