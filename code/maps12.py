"""Enriched input

On top of the 7 maps of geom.py we add edges and local orientation, information
the network does not have and that tells an axon from a vessel:

  - local contrast                            1
  - core-ring and sur-ring at 4 scales        8   (more radii than the 3 of geom.py)
  - gradient magnitude (smoothed image)       1   myelin border
  - local orientation  cos(2t), sin(2t)       2   structure tensor

"""
import torch
import torch.nn.functional as F

try:
    from .maps7 import RADII as R_REF
    from .maps7 import _annulus, _box, _integral
except ImportError:
    from maps7 import RADII as R_REF
    from maps7 import _annulus, _box, _integral

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

    sm = _box(I, 3)[0]                                 
    gx = F.pad(sm[..., :, 1:] - sm[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(sm[..., 1:, :] - sm[..., :-1, :], (0, 0, 0, 1))
    out.append(torch.sqrt(gx * gx + gy * gy + 1e-9))

    jxx = _box(_integral(gx * gx), 6)[0]             
    jyy = _box(_integral(gy * gy), 6)[0]
    jxy = _box(_integral(gx * gy), 6)[0]
    num, den = 2 * jxy, jxx - jyy
    nrm = torch.sqrt(num * num + den * den + 1e-9)
    out += [den / nrm, num / nrm]

    return torch.cat(out, 1)


N_MAPS = 1 + 2 * len(RADII) + 3