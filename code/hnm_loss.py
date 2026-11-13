"""Mineria de negativos duros dirigida por el oraculo.

El diagnostico en fold 0 (DistGPEM) da el techo:

    quitar solo los FP de fondo .............. 316.6 nm
    quitar TODAS las fibras no emparejadas .. 163.2 nm   <- el objetivo son 150

O sea: **la segmentacion que conservamos ya vale 163 nm**; todo el error esta en
las fibras que predecimos y no deberian existir. No es un problema de bordes,
es de deteccion.

Los filtros post-hoc fallan porque rechazar una fibra verdadera abre un hueco.
La perdida de distancia fallo por otra razon: castiga `p_fg` en TODAS las
regiones lejanas, donde ya vale ~0, asi que no hay gradiente. Los falsos
positivos reales son pocos pixeles con `p_fg` alta.

Aqui se senalan exactamente esos: en cada paso se identifican las fibras
predichas que NO emparejan con un axon anotado (IoU >= 0.3) y se penaliza su
probabilidad de primer plano. La red aprende a no dibujarlas.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn

IOU_MIN = 0.3


def fp_mask(pred, gt, min_px=200):
    """Mascara de las fibras predichas que no emparejan con ningun axon del GT."""
    if pred.max() == 0:
        return None
    comp, n = ndi.label(pred == 1)
    if n == 0:
        return None
    valid = (gt != 3).astype(np.int64)
    g, n_g = ndi.label((gt == 1) & (valid > 0))
    g, c = g * valid, comp * valid
    a_c = np.bincount(c.ravel(), minlength=n + 1)[1:]
    keep = a_c >= min_px
    matched = np.zeros(n, bool)
    if n_g:
        inter = np.bincount((g * (n + 1) + c).ravel(),
                            minlength=(n_g + 1) * (n + 1)).reshape(n_g + 1, n + 1)[1:, 1:]
        a_g = np.bincount(g.ravel(), minlength=n_g + 1)[1:]
        matched = (inter / (a_g[:, None] + a_c[None] - inter + 1e-9)).max(0) >= IOU_MIN
    bad = keep & ~matched
    if not bad.any():
        return None
    return bad[c - 1] if c.max() > 0 else None


class HNMLoss(nn.Module):
    """Penaliza p_fg en las regiones que la red predijo como fibra y no lo son."""

    def __init__(self, inner, weight=0.2, every=2, min_px=2000, warmup=2500):
        super().__init__()
        self.inner, self.weight, self.every, self.min_px = inner, weight, every, min_px
        self.warmup, self.steps = warmup, 0

    def forward(self, out, target):
        total = self.inner(out, target)
        self.steps += 1
        # warm-up: minar negativos mientras el modelo todavia es malo amplifica el
        # ruido (produce muchisimos falsos positivos y el castigo domina la CE+Dice).
        if not torch.is_grad_enabled() or self.steps <= self.warmup or self.steps % self.every:
            return total

        logits = out[0] if isinstance(out, (list, tuple)) else out
        fg = logits.float().softmax(1)[:, 1:3].sum(1)
        sem = target[0][:, 0].cpu().numpy()
        pred = logits.detach().argmax(1).cpu().numpy().astype(np.uint8)
        dev = fg.device

        terms = []
        for b in range(len(pred)):
            m = fp_mask(pred[b], sem[b], self.min_px)
            if m is None or not m.any():
                continue
            m = torch.from_numpy(m).to(dev)
            terms.append(fg[b][m].mean())
        add = self.weight * torch.stack(terms).mean() if terms else None
        return total if add is None or not torch.isfinite(add) else total + add
