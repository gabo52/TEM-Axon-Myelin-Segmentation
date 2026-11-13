"""Evaluación del pipeline Cellpose en validación, con las métricas OFICIALES.

1. Lee el split del entrenamiento (<ckpt>/split.json) y predice las imágenes de
   validación EXACTAMENTE como predict.py (misma función, misma escala).
2. Calcula Dice, HD95 (nm) y error de g-ratio con analysis/metrics.py, que
   implementa las fórmulas del challenge.
3. Guarda en --out las predicciones (.tif) y metrics.csv (una fila por imagen +
   MEAN). Con --wandb sube la tabla, las medias y ejemplos con GT y predicción.

Uso:
    python code/cellpose/evaluate.py
    python code/cellpose/evaluate.py --wandb --wandb-group cellpose-1111-1750
    python code/cellpose/evaluate.py --limit 3        # prueba rápida

Notas:
  * Un modelo entrenado con --all no tiene validación: no se puede evaluar así.
  * n_hd95_nan cuenta imágenes donde HD95 no está definido (predicción o GT vacíos);
    la media de HD95 las ignora, así que si es > 0 hay que mirarlas.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402
from common import (CELLPROB, DEFAULT_CKPT, DEFAULT_IMAGES, DEFAULT_LABELS, ROOT, fiber_stats,  # noqa: E402
                    list_pairs, load_models, model_path, predict_image, read_image,
                    resolve_scale)

sys.path.insert(0, str(ROOT / "analysis"))
import metrics  # noqa: E402  (analysis/metrics.py: fórmulas oficiales)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="carpeta de train.py")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--split", default=None, help="def. <ckpt>/split.json")
    ap.add_argument("--scale", type=float, default=None,
                    help="por defecto, la de entrenamiento (train_fiber.json)")
    ap.add_argument("--out", default=str(ROOT / "eval_cellpose"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cellprob", type=float, default=CELLPROB,
                    help=f"umbral de probabilidad de Cellpose (def. {CELLPROB}; Cellpose usa 0)")
    ap.add_argument("--no-multiscale", dest="multiscale", action="store_false",
                    help="sin la 2ª pasada a escala/2")
    ap.add_argument("--limit", type=int, default=None, help="evaluar solo N imágenes")
    ap.add_argument("--wandb", action="store_true", help="registrar la evaluación en W&B")
    ap.add_argument("--wandb-name", default=None, help="nombre del run (def. eval-<out>)")
    ap.add_argument("--wandb-group", default=None, help="p. ej. el grupo del entrenamiento")
    ap.add_argument("--wandb-images", type=int, default=4, help="nº de imágenes de ejemplo")
    args = ap.parse_args()

    split = json.loads(Path(args.split or Path(args.ckpt) / "split.json").read_text())
    if split.get("all") or not split["val"]:
        sys.exit("este modelo se entrenó sin validación (--all): no hay imágenes que evaluar")
    names = split["val"][:args.limit]
    pairs = {img.stem: (img, lbl) for img, lbl in list_pairs(args.images, args.labels)}

    scale = resolve_scale(args.ckpt, args.scale)
    m_fiber, m_axon = load_models(model_path(args.ckpt, "fiber"), model_path(args.ckpt, "axon"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"evaluando {len(names)} imágenes de validación | scale {scale} | "
          f"cellprob {args.cellprob} | multiscale {args.multiscale}")

    run = None
    if args.wandb:
        run = wandb_log.start(args.wandb_name or f"eval-{out.name}", args.wandb_group,
                              job_type="eval",
                              config={**vars(args), "scale": scale, "val_images": names})

    rows, results, examples = [], [], []
    for i, name in enumerate(names, start=1):
        t0 = time.time()
        img_path, lbl_path = pairs[name]
        img = read_image(img_path)
        gt = tifffile.imread(lbl_path)
        pred = predict_image(img, m_fiber, m_axon, scale, args.batch_size,
                             args.cellprob, args.multiscale)
        tifffile.imwrite(out / f"{name}.tif", pred, compression="zlib")

        scores, g_errors = metrics.evaluate(pred, gt)
        results.append((scores, g_errors))
        row = {"image": name, **scores,
               "g_ratio_mae": float(np.mean(g_errors)) if g_errors else float("nan"),
               "n_matched": len(g_errors),
               "n_gt_fibers": len(np.unique(gt[gt >= 2] // 2)),
               "n_pred_fibers": fiber_stats(pred)["n_fibers"]}
        rows.append(row)
        print(f"[{i}/{len(names)}] {name}: Dice ax={row['dice_axon']:.3f} my={row['dice_myelin']:.3f} | "
              f"HD95 ax={row['hd95_axon']:.0f} my={row['hd95_myelin']:.0f} nm | "
              f"Δg={row['g_ratio_mae']:.3f} | fibras {row['n_pred_fibers']}/{row['n_gt_fibers']} "
              f"({time.time() - t0:.0f}s)")
        if run is not None and len(examples) < args.wandb_images:
            caption = f"{name}  Dice ax={row['dice_axon']:.3f} my={row['dice_myelin']:.3f}"
            examples.append(wandb_log.mask_image(img, pred, gt, caption))

    # medias oficiales: Dice/HD95 = media por imagen, g-ratio = MAE sobre todas las fibras
    summary = metrics.aggregate(results)
    summary["n_hd95_nan"] = sum(np.isnan(r["hd95_axon"]) or np.isnan(r["hd95_myelin"]) for r in rows)

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + [k for k in summary if k not in cols])
        w.writeheader()
        w.writerows(rows)
        w.writerow({"image": "MEAN", **summary})

    print("\n=== MÉTRICAS OFICIALES (validación) ===")
    for k, v in summary.items():
        print(f"{k:>14}: {v:.4f}")
    print(f"\npredicciones y metrics.csv en {out}")

    if run is not None:
        try:  # metrics.csv ya está escrito: un fallo de W&B no invalida la evaluación
            log = {"per_image": wandb_log.table(rows)}
            if examples:
                log["examples"] = examples
            run.log(log)
            run.summary.update(summary)
            run.finish()
        except Exception as e:
            print(f"[wandb] error al subir la evaluación ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
