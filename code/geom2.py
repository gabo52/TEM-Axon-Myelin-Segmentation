"""Entrada enriquecida: misma imagen con mas preprocesados (fusion temprana).

Sobre los 7 mapas de geom.py se anaden bordes y orientacion local, que son
informacion que la red no tiene y que distingue un axon de un vaso:

  - contraste local                          1
  - core-ring y sur-ring a 4 escalas         8   (mas radios que los 3 de geom.py)
  - modulo del gradiente (imagen suavizada)  1   borde de mielina
  - orientacion local  cos(2t), sin(2t)      2   tensor de estructura

Todo con medias de caja por imagen integral: O(1) por pixel sea cual sea el
radio, y se calcula del parche YA aumentado (por eso no se guarda en disco).
"""
import torch
import torch.nn.functional as F

try:
    from .geom import RADII as R_REF
    from .geom import _annulus, _box, _integral
except ImportError:
    from geom import RADII as R_REF
    from geom import _annulus, _box, _integral

RADII = (20, 30, 45, 68)


def maps(x):
    img = x[:, :1].float()
    I = _integral(img)
    out = [img - _box(I, 3 * max(R_REF))[0]]

    for r in RADII:
        core = _box(I, 0.6 * r)[0]
        ring = _annulus(I, 0.9 * r, 1.4 * r)
        sur = _annulus(I, 1.6 * r, 2.2 * r)
        out += [core - ring, sur - ring]

    sm = _box(I, 3)[0]                                  # suavizar antes de derivar
    gx = F.pad(sm[..., :, 1:] - sm[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(sm[..., 1:, :] - sm[..., :-1, :], (0, 0, 0, 1))
    out.append(torch.sqrt(gx * gx + gy * gy + 1e-9))

    jxx = _box(_integral(gx * gx), 6)[0]                # tensor de estructura
    jyy = _box(_integral(gy * gy), 6)[0]
    jxy = _box(_integral(gx * gy), 6)[0]
    num, den = 2 * jxy, jxx - jyy
    nrm = torch.sqrt(num * num + den * den + 1e-9)
    out += [den / nrm, num / nrm]

    return torch.cat(out, 1)


N_MAPS = 1 + 2 * len(RADII) + 3
