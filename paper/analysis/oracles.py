"""Tabla I del paper: análisis de oráculos en validación (inferencia Cellpose por defecto).

Uso:  python paper/analysis/oracles.py [carpeta_salida]

1. Predice las imágenes de validación (code/checkpoints/cellpose/split.json) con la
   inferencia por defecto de Cellpose (cellprob 0, una escala) y las guarda.
2. Clasifica las fibras (IoU sobre X ∪ X+1, umbral 0.5):
     TP      fibra predicha con IoU >= 0.5 con alguna fibra GT
     FP_bg   fibra predicha sin pareja y con <= 50 % de su área en artefacto
     FP_art  fibra predicha sin pareja y con > 50 % en artefacto (no puntúa)
     FN      fibra GT sin ninguna predicha con IoU >= 0.5
3. Evalúa (analysis/metrics.py) cada oráculo, que corrige la predicción con el GT:
     default            predicción tal cual
     no_fp              sin FP_bg
     add_fn             + fibras FN del GT (sin tocar los FP)
     no_fp_add_fn       sin FP_bg + fibras FN
     perfect_tp         cada TP sustituida por su fibra GT
     gt_at_0.25         GT reducido x0.25 y vuelto a ampliar (techo de la escala)
Escribe oracles.json con métricas y conteos.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "cellpose"))
sys.path.insert(0, str(ROOT / "analysis"))
import metrics  # noqa: E402
from common import (DEFAULT_CKPT, load_models, model_path, pair_fibers, predict_image,  # noqa: E402
                    read_image, rescale_lbl, to_axon, to_fiber, train_scale)

GT_DIR = ROOT / "data" / "train_instance_labels"
IMG_DIR = ROOT / "data" / "train_samples"
ORACLES = ["default", "no_fp", "add_fn", "no_fp_add_fn", "perfect_tp", "gt_at_0.25"]


def fibers(y):
    return np.where(y >= 2, y // 2, 0).astype(np.int64)


def classify(g, p):
    fg, fp = fibers(g), fibers(p)
    ng, n_p = fg.max() + 1, fp.max() + 1
    inter = np.bincount((fg * n_p + fp).ravel(), minlength=ng * n_p).reshape(ng, n_p)
    ag, ap = np.bincount(fg.ravel(), minlength=ng), np.bincount(fp.ravel(), minlength=n_p)
    iou = inter / (ag[:, None] + ap[None] - inter + 1e-9)
    iou[0], iou[:, 0] = 0, 0
    art = np.bincount(fp[g == 1], minlength=n_p)
    pids = [k for k in range(1, n_p) if ap[k]]
    gids = [k for k in range(1, ng) if ag[k]]
    tp = [k for k in pids if iou[:, k].max() >= 0.5]
    fp_all = [k for k in pids if iou[:, k].max() < 0.5]
    fp_art = [k for k in fp_all if art[k] / ap[k] > 0.5]
    fp_bg = [k for k in fp_all if k not in fp_art]
    fn = [k for k in gids if iou[k].max() < 0.5]
    best_gt = {k: int(iou[:, k].argmax()) for k in tp}
    return dict(fg=fg, fp=fp, tp=tp, fp_bg=fp_bg, fp_art=fp_art, fn=fn, gids=gids,
                best_gt=best_gt, fn_area=[int(ag[k]) for k in fn], tp_area=[int(ap[k]) for k in tp])


def add_gt_fibers(pred, g, fg, ids, offset):
    """Pinta las fibras GT `ids` donde la predicción está vacía (ids desplazados para no chocar)."""
    out = pred.copy()
    m = np.isin(fg, ids) & (out == 0) & (g >= 2)
    out[m] = g[m] + offset
    return out


def analyze(name, pred_dir):
    g = tifffile.imread(GT_DIR / f"{name}.tif").astype(np.int64)
    p = tifffile.imread(pred_dir / f"{name}.tif").astype(np.int64)
    c = classify(g, p)
    offset = 2 * (int(p.max()) // 2 + 1)  # par: mantiene axón par / mielina impar

    no_fp = np.where(np.isin(c["fp"], c["fp_bg"]), 0, p)
    perfect = np.where(np.isin(c["fp"], c["tp"]), 0, p)
    perfect = add_gt_fibers(perfect, g, c["fg"], list(c["best_gt"].values()), offset)
    H, W = g.shape
    small = (round(H * 0.25), round(W * 0.25))
    gt_low = pair_fibers(rescale_lbl(rescale_lbl(to_fiber(g), small), (H, W)),
                         rescale_lbl(rescale_lbl(to_axon(g), small), (H, W)))
    variants = {
        "default": p,
        "no_fp": no_fp,
        "add_fn": add_gt_fibers(p, g, c["fg"], c["fn"], offset),
        "no_fp_add_fn": add_gt_fibers(no_fp, g, c["fg"], c["fn"], offset),
        "perfect_tp": perfect,
        "gt_at_0.25": gt_low,
    }
    res = {k: metrics.evaluate(v, g) for k, v in variants.items()}
    counts = {k: len(c[k]) for k in ("tp", "fp_bg", "fp_art", "fn", "gids")}
    return name, res, counts, c["fn_area"], c["tp_area"]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "analysis" / "out"
    pred_dir = out / "pred_default"
    pred_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads((DEFAULT_CKPT / "split.json").read_text())["val"]

    todo = [n for n in names if not (pred_dir / f"{n}.tif").exists()]
    if todo:
        mf, ma = load_models(model_path(DEFAULT_CKPT, "fiber"), model_path(DEFAULT_CKPT, "axon"))
        scale = train_scale(DEFAULT_CKPT)
        for i, n in enumerate(todo, 1):
            pred = predict_image(read_image(IMG_DIR / f"{n}.tif"), mf, ma, scale,
                                 cellprob=0.0, multiscale=False)
            tifffile.imwrite(pred_dir / f"{n}.tif", pred, compression="zlib")
            print(f"pred {i}/{len(todo)} {n}", flush=True)

    with Pool(8) as pool:
        rows = pool.starmap(analyze, [(n, pred_dir) for n in names])

    report = {"oracles": {}, "counts": {}, "n_images": len(rows)}
    for k in ORACLES:
        a = metrics.aggregate([r[1][k] for r in rows])
        report["oracles"][k] = {m: float(v) for m, v in a.items()}
    for key in ("tp", "fp_bg", "fp_art", "fn", "gids"):
        report["counts"][key] = int(sum(r[2][key] for r in rows))
    report["fn_area_median"] = float(np.median([a for r in rows for a in r[3]]))
    report["tp_area_median"] = float(np.median([a for r in rows for a in r[4]]))
    (out / "oracles.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'oráculo':>14} | DSC_mean | HD95_mean (nm) | MAE_g")
    for k, a in report["oracles"].items():
        print(f"{k:>14} | {a['dice_mean']:.4f}   | {a['hd95_mean']:9.2f}      | {a['g_ratio_mae']:.4f}")
    print("conteos:", report["counts"], "| área mediana FN/TP:",
          report["fn_area_median"], report["tp_area_median"])


if __name__ == "__main__":
    main()
