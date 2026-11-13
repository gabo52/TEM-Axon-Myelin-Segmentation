"""Close open myelin rings and fill the axon they enclose.

For every fibre (axon 2k, myelin 2k+1) of an instance map:

1. cast N_RAYS rays from the axon centroid;
2. on each ray, r_in / r_out = where the fibre's own myelin starts / ends
   (a ray is invalid if it meets another fibre first or never meets the myelin);
3. if the fraction of valid rays >= MIN_COV, interpolate r_in / r_out (circularly)
   on the invalid rays;
4. background inside the inner contour -> axon 2k;
   background between inner and outer contour on the missing sectors -> myelin 2k+1.

Only background pixels change; other fibres and artefacts are never touched.
Fibres cut by the image border, tiny axons and non star-shaped axons are skipped.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon

N_RAYS = 180
MIN_COV = 0.7
MAX_GAP = 2          # background px tolerated inside the myelin along a ray

def profile(sub, cy, cx, k, rmax):
    th = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    r = np.arange(rmax)
    ys = np.clip(np.round(cy + np.sin(th)[:, None] * r).astype(int), 0, sub.shape[0] - 1)
    xs = np.clip(np.round(cx + np.cos(th)[:, None] * r).astype(int), 0, sub.shape[1] - 1)
    v = sub[ys, xs]
    r_in, r_out = np.full(N_RAYS, np.nan), np.full(N_RAYS, np.nan)
    for i, row in enumerate(v):
        my = row == 2 * k + 1
        if not my.any():
            continue
        a = int(np.argmax(my))
        if ((row[:a] >= 2) & (row[:a] // 2 != k)).any():
            continue
        b, gap = a, 0
        while b + 1 < rmax and gap <= MAX_GAP:
            b += 1
            gap = 0 if row[b] == 2 * k + 1 else gap + 1
        r_in[i], r_out[i] = a, b - gap
    return th, r_in, r_out

def fill_polygon(th, r, cy, cx, shape):
    m = np.zeros(shape, bool)
    m[polygon(cy + np.sin(th) * r, cx + np.cos(th) * r, shape)] = True
    return m

def close_rings(inst, min_cov=MIN_COV):
    out = inst.copy()
    H, W = inst.shape
    for k, sl in enumerate(ndi.find_objects(np.where(inst >= 2, inst // 2, 0)), 1):
        if sl is None or sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue
        if (inst[sl] == 2 * k).sum() < 50:
            continue
        hy, hx = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        pad = max(hy, hx) // 2 + 10
        y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        sub = out[y0:min(sl[0].stop + pad, H), x0:min(sl[1].stop + pad, W)]
        cy, cx = ndi.center_of_mass(sub == 2 * k)
        if sub[int(round(cy)), int(round(cx))] != 2 * k:
            continue
        th, r_in, r_out = profile(sub, cy, cx, k, int(np.hypot(hy, hx)) + 10)
        valid = ~np.isnan(r_in)
        if valid.mean() < min_cov:
            continue
        ri = np.interp(th, th[valid], r_in[valid], period=2 * np.pi)
        inner = fill_polygon(th, ri, cy, cx, sub.shape)
        add_my = np.zeros(sub.shape, bool)
        if (~valid).any():
            ro = np.interp(th, th[valid], r_out[valid], period=2 * np.pi)
            outer = fill_polygon(th, ro, cy, cx, sub.shape)
            yy, xx = np.mgrid[:sub.shape[0], :sub.shape[1]]
            idx = np.round(np.arctan2(yy - cy, xx - cx) % (2 * np.pi) / (2 * np.pi) * N_RAYS).astype(int) % N_RAYS
            miss = ~valid
            miss = miss | np.roll(miss, 1) | np.roll(miss, -1)
            add_my = outer & ~inner & miss[idx] & (sub == 0)
        add_ax = inner & (sub == 0)
        sub[add_ax] = 2 * k
        sub[add_my] = 2 * k + 1
    return out
