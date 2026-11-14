import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .accept import MIN_AXON, RING_PAD, accept_map
except ImportError:
    from accept import MIN_AXON, RING_PAD, accept_map

class AcceptLoss(nn.Module):

    def __init__(self, base, head, features, weight, every, min_axon=MIN_AXON, pad=RING_PAD,
                 target=None):
        super().__init__()
        self.base, self.head, self.features = base, head, features
        self.weight, self.every = weight, every
        self.min_axon, self.pad, self.steps = min_axon, pad, 0
        self.target = target or accept_map

    def forward(self, out, target):
        total = self.base(out, target)
        self.steps += 1
        if not torch.is_grad_enabled() or self.steps % self.every or "x" not in self.features:
            return total
        p = out[0].float().softmax(1)
        feat = self.head(torch.cat([self.features["x"], p[:, 1:3]], dim=1))
        pred = out[0].detach().argmax(1).cpu().numpy().astype(np.uint8)
        sem = target[0][:, 0].cpu().numpy()
        terms = [t for t in (self._bce(feat, b, pred, sem) for b in range(len(pred))) if t is not None]
        add = self.weight * torch.stack(terms).mean() if terms else None
        return total if add is None or not torch.isfinite(add) else total + add

    def _bce(self, feat, b, pred, sem):
        y, w = self.target(pred[b], pred[b], sem[b], self.min_axon, self.pad)
        if y is None or w.sum() == 0:
            return None
        pair = torch.from_numpy(np.stack([y * w, w])).to(feat.device)[None]
        tm, wm = F.interpolate(pair, size=feat.shape[-2:], mode="nearest")[0]
        pos, neg = (tm > 0.5) & (wm > 0), (tm <= 0.5) & (wm > 0)
        if not pos.any() or not neg.any():
            return None
        h = feat[b, 0]
        bce = 0.5 * (
            F.binary_cross_entropy_with_logits(h, torch.ones_like(h), weight=pos.float(), reduction="sum") / pos.sum().clamp(min=1.0)
            + F.binary_cross_entropy_with_logits(h, torch.zeros_like(h), weight=neg.float(), reduction="sum") / neg.sum().clamp(min=1.0)
        )
        return bce.clamp(max=20.0)
