import torch
import torch.nn.functional as F

RADII = (28, 48, 80)        


def _integral(x):
    return F.pad(x, (1, 0, 1, 0)).cumsum(-1).cumsum(-2)


def _box(I, r):
    H, W = I.shape[-2] - 1, I.shape[-1] - 1
    r = max(int(round(r)), 1)
    yy = torch.arange(H, device=I.device)
    xx = torch.arange(W, device=I.device)
    y0, y1 = (yy - r).clamp(min=0), (yy + r + 1).clamp(max=H)
    x0, x1 = (xx - r).clamp(min=0), (xx + r + 1).clamp(max=W)
    s = (I[..., y1[:, None], x1[None, :]] - I[..., y0[:, None], x1[None, :]]
         - I[..., y1[:, None], x0[None, :]] + I[..., y0[:, None], x0[None, :]])
    a = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).to(s.dtype)
    return s / a, a


def _annulus(I, r_in, r_out):
    m_in, a_in = _box(I, r_in)
    m_out, a_out = _box(I, r_out)
    d = a_out - a_in
    return torch.where(d > 0, (m_out * a_out - m_in * a_in) / d.clamp(min=1.0), m_out)


def maps(x):
    img = x[:, :1].float()
    I = _integral(img)
    out = [img - _box(I, 3 * max(RADII))[0]]
    for r in RADII:
        core = _box(I, 0.6 * r)[0]
        ring = _annulus(I, 0.9 * r, 1.4 * r)
        sur = _annulus(I, 1.6 * r, 2.2 * r)
        out += [core - ring, sur - ring]
    return torch.cat(out, 1)


N_MAPS = 1 + 2 * len(RADII)
