import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

PX_NM = 2.152  
COMPARTMENTS = (("axon", 0), ("myelin", 1))

def dice(p, g):
    s = p.sum() + g.sum()
    return 2 * (p & g).sum() / s if s else np.nan

def get_border(m):
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool))

def hd95(p, g):
    if not p.any() or not g.any():
        return np.nan
    bp, bg = get_border(p), get_border(g)
    if not bp.any() or not bg.any():
        return np.nan
    d_pg = ndi.distance_transform_edt(~bg)[bp]
    d_gp = ndi.distance_transform_edt(~bp)[bg]
    return max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)) * PX_NM

def fibers(y):
    return np.where(y >= 2, y // 2, 0).astype(np.int64)

def diameter(area):
    return 2 * np.sqrt(area / np.pi)

def match(p, g):
    fp, fg = fibers(p), fibers(g)
    n_p, n_g = fp.max() + 1, fg.max() + 1
    inter = np.bincount((fg * n_p + fp).ravel(), minlength=n_g * n_p).reshape(n_g, n_p)
    inter[0] = inter[:, 0] = 0
    area_p = np.bincount(fp.ravel(), minlength=n_p)
    area_g = np.bincount(fg.ravel(), minlength=n_g)
    iou = inter / (area_g[:, None] + area_p[None] - inter + 1e-9)
    return [(i, iou[i].argmax()) for i in range(1, n_g) if area_g[i] and iou[i].max() > 0]

def g_ratio_errors(p, g, pairs):
    area_p, area_g = np.bincount(fibers(p).ravel()), np.bincount(fibers(g).ravel())
    ax_p = np.bincount(p[(p >= 2) & (p % 2 == 0)] // 2, minlength=len(area_p))
    ax_g = np.bincount(g[(g >= 2) & (g % 2 == 0)] // 2, minlength=len(area_g))
    
    return [
        abs((diameter(ax_p[j]) / diameter(area_p[j])) - (diameter(ax_g[i]) / diameter(area_g[i])))
        for i, j in pairs if area_p[j] > 0 and area_g[i] > 0
    ]

def evaluate(pred, gt):
    valid = (gt != 1) & (pred != 1)
    p, g = np.where(valid, pred, 0), np.where(valid, gt, 0)
    scores = {}
    for name, odd in COMPARTMENTS:
        pm, gm = (p >= 2) & (p % 2 == odd), (g >= 2) & (g % 2 == odd)
        scores[f"dice_{name}"] = dice(pm, gm)
        scores[f"hd95_{name}"] = hd95(pm, gm)
    return scores, g_ratio_errors(p, g, match(p, g))

def aggregate(results):
    rows, errors = [r[0] for r in results], [e for r in results for e in r[1]]
    m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
    m["dice_mean"] = (m["dice_axon"] + m["dice_myelin"]) / 2
    m["hd95_mean"] = (m["hd95_axon"] + m["hd95_myelin"]) / 2
    m["g_ratio_mae"] = np.mean(errors) if errors else np.nan
    return m

def evaluate_dir(pred_dir, gt_dir):
    return aggregate(
        [evaluate(tifffile.imread(f), tifffile.imread(Path(gt_dir) / f.name)) for f in sorted(Path(pred_dir).glob("*.tif"))]
    )

if __name__ == "__main__":
    for k, v in evaluate_dir(sys.argv[1], sys.argv[2]).items():
        print(f"{k}: {v:.4f}")
