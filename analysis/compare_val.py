"""Cellpose (out of fold) vs nnU-Net on the same validation images, official metrics.

    python analysis/compare_val.py <nnunet_pred_dir> [<cellpose_variant>]

Images = the .tif files in <nnunet_pred_dir>. Cellpose predictions are read from
code/checkpoints/cellpose_kfold/eval/<variant>/fold*/ (default cp-1_multiscale).
Prints DSC/HD95 per compartment, MAE_g, % reference fibers detected (IoU >= 0.5)
and % open fibers (myelin does not enclose the axon; border fibers excluded).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from fiber_check import fiber_quality, matched  # noqa: E402
from metrics import aggregate, evaluate          # noqa: E402

GT = ROOT / "data" / "train_instance_labels"


def one(args):
    pred_path, name = args
    p = tifffile.imread(pred_path).astype(np.int64)
    g = tifffile.imread(GT / name).astype(np.int64)
    ids, ok = matched(g, p)
    q = [v for v in fiber_quality(p).values() if not v[2]]
    return evaluate(p, g), len(ids), len(ok), len(q), sum(not c for c, _, _ in q)


def report(label, paths):
    with ProcessPoolExecutor(6) as ex:
        out = list(ex.map(one, [(p, p.name) for p in paths]))
    m = aggregate([o[0] for o in out])
    ngt, nok, nq, nopen = (sum(o[i] for o in out) for i in range(1, 5))
    print(f"{label:>22}: DSC ax {m['dice_axon']:.3f} my {m['dice_myelin']:.3f} mean {m['dice_mean']:.3f} | "
          f"HD95 ax {m['hd95_axon']:.0f} my {m['hd95_myelin']:.0f} mean {m['hd95_mean']:.0f} | "
          f"MAE_g {m['g_ratio_mae']:.4f} | detected {100 * nok / ngt:.1f}% ({nok}/{ngt}) | "
          f"open {100 * nopen / nq:.1f}% ({nopen}/{nq}) | n={len(paths)}", flush=True)


if __name__ == "__main__":
    nn = sorted(Path(sys.argv[1]).glob("*.tif"))
    variant = sys.argv[2] if len(sys.argv) > 2 else "cp-1_multiscale"
    ev = ROOT / "code" / "checkpoints" / "cellpose_kfold" / "eval" / variant
    cp = [next(ev.glob(f"fold*/{p.name}")) for p in nn]
    report("nnU-Net GPEM15", nn)
    report(f"Cellpose {variant}", cp)
