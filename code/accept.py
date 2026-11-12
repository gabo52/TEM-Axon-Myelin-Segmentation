import numpy as np
import torch.nn as nn
from scipy import ndimage as ndi

#we can edit this 
MIN_AXON = 1500
RING_GAPS = 4
RING_PAD = 4
REF_RES = 1504.0

def ring_gaps(axon, myelin, n=24, pad=RING_PAD):
    if not axon.any():
        return 99
    cy, cx = ndi.center_of_mass(axon)
    ys, xs = np.mgrid[0:axon.shape[0], 0:axon.shape[1]]
    sec = (((np.arctan2(ys - cy, xs - cx) + np.pi) % (2 * np.pi)) / (2 * np.pi) * n).astype(int)
    ring = ndi.binary_dilation(axon, iterations=pad) & ~axon
    idx = sec[ring]
    tot = np.bincount(idx, minlength=n).astype(np.float64)
    yes = np.bincount(idx, weights=myelin[ring], minlength=n).astype(np.float64)
    cov = np.where(tot > 0, yes / np.maximum(tot, 1), 0.0)
    return int((cov < 0.3).sum())

class AcceptHead(nn.Module):

    def __init__(self, width=16):
        super().__init__()
        self.width, self.body = width, None

    def forward(self, feat):
        if self.body is None:
            c = feat.shape[1]
            self.body = nn.Sequential(
                nn.AvgPool2d(4),
                nn.Conv2d(c, self.width, 3, 1, 1), nn.GroupNorm(8, self.width), nn.ReLU(inplace=True),
                nn.Conv2d(self.width, self.width, 3, 1, 1), nn.GroupNorm(8, self.width), nn.ReLU(inplace=True),
                nn.Conv2d(self.width, 1, 1),
            ).to(feat.device)
        return self.body(feat)

def accept_map(pred, myelin, gt, min_axon=MIN_AXON, pad=RING_PAD):
    comp, n = ndi.label(pred == 1)
    if n == 0:
        return None, None
    valid = (gt != 3).astype(np.int64)
    g, n_g = ndi.label((gt == 1) & (valid > 0))
    g, c = g * valid, comp * valid
    area_c = np.bincount(c.ravel(), minlength=n + 1)[1:]
    tgt, wgt = np.zeros(n + 1, np.float32), np.zeros(n + 1, np.float32)
    wgt[1:] = area_c >= min_axon
    if n_g:
        inter = np.bincount((g * (n + 1) + c).ravel(),
                            minlength=(n_g + 1) * (n + 1)).reshape(n_g + 1, n + 1)[1:, 1:]
        area_g = np.bincount(g.ravel(), minlength=n_g + 1)[1:]
        matched = (inter / (area_g[:, None] + area_c[None] - inter + 1e-9)).max(0) >= 0.3
        tgt[1:] = matched
        my = myelin == 2
        boxes = ndi.find_objects(comp)
        for i in range(1, n + 1):
            if not matched[i - 1] and wgt[i] > 0:
                box = boxes[i - 1]
                crop = tuple(slice(max(b.start - pad - 2, 0), b.stop + pad + 2) for b in box) \
                    if box is not None else tuple(slice(None) for _ in range(2))
                if ring_gaps(comp[crop] == i, my[crop], pad=pad) >= RING_GAPS:
                    tgt[i] = 0.0
                else:
                    wgt[i] = 0.0
    return tgt[comp], wgt[comp]


def oracle_map(pred, myelin, gt, min_axon=MIN_AXON, pad=RING_PAD):
    comp, n = ndi.label(pred == 1)
    if n == 0:
        return None, None
    valid = (gt != 3).astype(np.int64)
    g, n_g = ndi.label((gt == 1) & (valid > 0))
    g, c = g * valid, comp * valid
    area = np.bincount(comp.ravel(), minlength=n + 1)[1:].astype(np.float64)
    area_c = np.bincount(c.ravel(), minlength=n + 1)[1:]
    tgt, wgt = np.zeros(n + 1, np.float32), np.zeros(n + 1, np.float32)
    wgt[1:] = area_c >= min_axon
    if n_g:
        inter = np.bincount((g * (n + 1) + c).ravel(),
                            minlength=(n_g + 1) * (n + 1)).reshape(n_g + 1, n + 1)[1:, 1:]
        area_g = np.bincount(g.ravel(), minlength=n_g + 1)[1:]
        tgt[1:] = (inter / (area_g[:, None] + area_c[None] - inter + 1e-9)).max(0) >= 0.3
    art = np.bincount(comp[gt == 3].ravel(), minlength=n + 1)[1:].astype(np.float64)
    wgt[1:] *= (~(art > 0.5 * np.maximum(area, 1))) | (tgt[1:] > 0.5)
    return tgt[comp], wgt[comp]
