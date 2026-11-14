"""Control de calidad por fibra: ¿se pierden fibras? ¿quedan abiertas?

Una fibra (axón 2k + mielina 2k+1) está CERRADA si su mielina rodea por completo al
axón: al rellenar los huecos de la mielina, el relleno cubre el axón (>= 99 %).
`anillo` = fracción del contorno exterior del axón que toca su propia mielina.
Las fibras cortadas por el borde de la imagen se cuentan aparte (no pueden cerrar).

    python analysis/fiber_check.py <carpeta_pred> [<carpeta_pred2> ...] [--gt <carpeta_gt>]

Con --gt: además, fibras reales detectadas (IoU >= 0.5 sobre la fibra completa).
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

IOU = 0.5


def fiber_quality(inst):
    """Por fibra: (cerrada, anillo, toca_borde)."""
    out = {}
    fib = np.where(inst >= 2, inst // 2, 0)
    h, w = inst.shape
    for k, box in enumerate(ndi.find_objects(fib), 1):
        if box is None:
            continue
        border = box[0].start == 0 or box[1].start == 0 or box[0].stop == h or box[1].stop == w
        box = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in box)
        axon, myelin = inst[box] == 2 * k, inst[box] == 2 * k + 1
        if not axon.any() or not myelin.any():
            out[k] = (False, 0.0, border)
            continue
        enclosed = (ndi.binary_fill_holes(myelin) & axon).sum() / axon.sum()
        ring = ndi.binary_dilation(axon) & ~axon
        out[k] = (enclosed >= 0.99, float(myelin[ring].mean()), border)
    return out


def matched(a, b):
    """Fibras de `a` que tienen pareja en `b` con IoU >= 0.5."""
    fa, fb = np.where(a >= 2, a // 2, 0), np.where(b >= 2, b // 2, 0)
    na, nb = fa.max() + 1, fb.max() + 1
    inter = np.bincount((fa * nb + fb).ravel(), minlength=na * nb).reshape(na, nb)
    inter[0] = inter[:, 0] = 0
    sa, sb = np.bincount(fa.ravel(), minlength=na), np.bincount(fb.ravel(), minlength=nb)
    iou = inter / (sa[:, None] + sb[None] - inter + 1e-9)
    ids = [k for k in range(1, na) if sa[k]]
    return ids, {k for k in ids if iou[k].max() >= IOU}


def summary(name, q):
    inner = [v for v in q if not v[2]]
    n_open = sum(not c for c, _, _ in inner)
    ring = np.array([r for _, r, _ in inner])
    print(f"{name:>22}: {len(q)} fibras ({len(q) - len(inner)} en el borde) | "
          f"abiertas {n_open}/{len(inner)} ({100 * n_open / max(len(inner), 1):.1f} %) | "
          f"anillo < 90 %: {(ring < 0.9).sum()} | anillo mediano {np.median(ring):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--gt")
    ap.add_argument("--names", nargs="+")
    a = ap.parse_args()
    names = a.names or [Path(p).name for p in a.preds]
    files = sorted(f.name for f in Path(a.preds[0]).glob("*.tif"))

    if a.gt:                                   # el GT como referencia de "cerrada"
        summary("GT", [v for f in files
                       for v in fiber_quality(tifffile.imread(Path(a.gt) / f)).values()])
    for name, d in zip(names, a.preds):
        q, n_gt, n_found = [], 0, 0
        for f in files:
            p = tifffile.imread(Path(d) / f)
            q += fiber_quality(p).values()
            if a.gt:
                ids, ok = matched(tifffile.imread(Path(a.gt) / f), p)
                n_gt, n_found = n_gt + len(ids), n_found + len(ok)
        summary(name, q)
        if a.gt:
            print(f"{'':>22}  fibras reales detectadas: {n_found}/{n_gt} "
                  f"({100 * n_found / n_gt:.1f} %), perdidas {n_gt - n_found}")

    if len(a.preds) == 2 and not a.gt:         # sin GT: lo que encuentra solo uno de los dos
        only = [0, 0]
        for f in files:
            x, y = (tifffile.imread(Path(d) / f) for d in a.preds)
            for i, (u, v) in enumerate(((x, y), (y, x))):
                ids, ok = matched(u, v)
                only[i] += len(ids) - len(ok)
        print(f"fibras sin pareja (IoU >= {IOU}) en el otro modelo: "
              f"solo {names[0]} {only[0]}, solo {names[1]} {only[1]}")


if __name__ == "__main__":
    main()
