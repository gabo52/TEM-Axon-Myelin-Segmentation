import torch.nn.functional as F
from torch import nn

class GRatioLoss(nn.Module):

    def __init__(self, base, weight=1.0):
        super().__init__()
        self.base, self.weight = base, weight

    def forward(self, x, y):
        p, y0 = x.softmax(1), y[:, 0]
        g = lambda a, m: (a / (a + m + 1)).sqrt()
        gp = g(p[:, 1].sum((1, 2)), p[:, 2].sum((1, 2)))
        gt = g((y0 == 1).float().sum((1, 2)), (y0 == 2).float().sum((1, 2)))
        return self.base(x, y) + self.weight * (gp - gt).abs().mean()

class EdgeLoss(nn.Module):

    def __init__(self, base, ignore, weight=1.0, kernel=7):
        super().__init__()
        self.base, self.ignore, self.weight, self.kernel = base, ignore, weight, kernel

    def forward(self, x, y):
        y2 = y[:, 0].long()
        my = (y2 == 2).float()[:, None]
        pad = self.kernel // 2
        band = (F.max_pool2d(my, self.kernel, 1, pad) + F.max_pool2d(-my, self.kernel, 1, pad))[:, 0]
        ce = F.cross_entropy(x, y2, ignore_index=self.ignore, reduction="none")
        return self.base(x, y2[:, None]) + self.weight * (ce * band).sum() / band.sum().clamp(min=1)
