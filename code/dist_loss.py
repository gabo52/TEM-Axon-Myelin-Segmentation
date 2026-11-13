"""Perdida de distancia: ataca la cola del HD95 directamente.

El diagnostico sobre el modelo de 150 ep: el 30.2 % de los pixeles de borde
estan a mas de 150 nm del borde GT y aportan el **99.7 %** de la suma de
distancias al cuadrado. Ninguna perdida que usamos (CE, Dice, BCE) es sensible
a eso: miden solapamiento, no distancia. Y la direccion GT->prediccion
(fibras perdidas) es solo el 0.8 %, o sea que el problema son los falsos
positivos lejanos.

    L = media ponderada de p_primer_plano   con peso  relu(DT_gt - tau)^2

Es decir: **donde estas lejos de cualquier fibra anotada, tu probabilidad de
primer plano debe ser cero.** Si L = 0 entonces no hay prediccion a mas de tau
del GT, luego P95 <= tau y HD95 <= tau. Con tau = 70 px = 150 nm, el cero de la
perdida es el objetivo del challenge.

La bisagra deja gradiente CERO dentro de tau, asi que no pelea con Dice en el
70 % que ya esta bien segmentado.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn

TAU_PX = 70.0      # 150 nm con pixel de 2.152 nm
CAP_PX = 500.0     # recorte de la distancia para estabilidad


def far_weight(sem, tau=TAU_PX, cap=CAP_PX):
    """Peso por pixel: 0 dentro de tau del borde GT, sube al cuadrado fuera."""
    valid = sem != 3                       # 3 = artefacto, no lo mide el challenge
    fg = (sem == 1) | (sem == 2)
    if not fg.any() or not valid[fg].any():
        return None
    b = fg & ~ndi.binary_erosion(fg)
    dt = ndi.distance_transform_edt(~b)
    w = np.clip((dt - tau) / cap, 0.0, 1.0).astype(np.float32)
    w[~valid] = 0.0
    w = w * w
    return w if w.sum() > 0 else None


class DistLoss(nn.Module):
    """Envuelve la cadena del benchmark y anade el termino de distancia."""

    def __init__(self, inner, weight=0.2, every=2, tau=TAU_PX, cap=CAP_PX):
        super().__init__()
        self.inner, self.weight, self.every = inner, weight, every
        self.tau, self.cap, self.steps = tau, cap, 0

    def forward(self, out, target):
        total = self.inner(out, target)
        self.steps += 1
        if not torch.is_grad_enabled() or self.steps % self.every:
            return total

        logits = out[0] if isinstance(out, (list, tuple)) else out
        fg = logits.float().softmax(1)[:, 1:3].sum(1)
        sem = target[0][:, 0].cpu().numpy()
        dev = fg.device

        terms = []
        for b in range(len(sem)):
            w = far_weight(sem[b], self.tau, self.cap)
            if w is None:
                continue
            w = torch.from_numpy(w).to(dev)
            terms.append((w * fg[b]).sum() / w.sum().clamp(min=1e-6))
        add = self.weight * torch.stack(terms).mean() if terms else None
        return total if add is None or not torch.isfinite(add) else total + add
