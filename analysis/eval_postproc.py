"""Compare post-processing variants on saved semantic predictions, with analysis/metrics.py.

Input : <sem_dir>/<case>_sem.tif (0/1/2, full resolution) and <case>_score.npy (accept score,
        1/4 of network resolution), as written by `code/predict.py --save-sem`.
Output: <out>.csv, one row per (case, variant) + one GLOBAL row per variant (repo aggregation:
        nanmean per image for Dice/HD95, g-ratio MAE over all matched fibres).

    python analysis/eval_postproc.py <sem_dir> <gt_dir> <out> [--tau 0.2]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from metrics import aggregate, evaluate
from otros.postproc import postprocess

VARIANTS = {"main": dict(close=False), "close_rings": dict(close=True)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sem_dir", type=Path)
    ap.add_argument("gt_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tau", type=float, default=0.2)
    a = ap.parse_args()

    results = {v: [] for v in VARIANTS}
    rows = []
    for f in sorted(a.sem_dir.glob("*_sem.tif")):
        case = f.name[:-len("_sem.tif")]
        sem = tifffile.imread(f)
        gt = tifffile.imread(a.gt_dir / f"{case}.tif").astype(np.int64)
        score = torch.from_numpy(np.load(a.sem_dir / f"{case}_score.npy").astype(np.float32))[None, None]
        score = F.interpolate(score, size=sem.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        for v, kw in VARIANTS.items():
            r = evaluate(postprocess(sem, score, a.tau, **kw).astype(np.int64), gt)
            results[v].append(r)
            s = r[0]
            rows.append(dict(case=case, variant=v, **s,
                             dice_mean=np.nanmean([s["dice_axon"], s["dice_myelin"]]),
                             hd95_mean=np.nanmean([s["hd95_axon"], s["hd95_myelin"]]),
                             g_ratio_mae=np.mean(r[1]) if r[1] else np.nan))
        print(case, flush=True)

    for v, res in results.items():
        rows.append(dict(case="GLOBAL", variant=v, **aggregate(res)))
        print(v, {k: round(float(x), 4) for k, x in aggregate(res).items()})
    with open(a.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
