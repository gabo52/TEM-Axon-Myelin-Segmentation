"""Inferencia Cellpose-SAM -> predicciones en el formato de entrega del challenge.

Para cada imagen de --images escribe en --out un TIFF uint16 con el MISMO nombre
y el MISMO tamaño (3008×4096): axón = 2k, su mielina = 2k+1, 0 = fondo.
Antes de guardar se comprueba el formato (tamaño, dtype y pares X / X+1).

Uso:
    python code/cellpose/predict.py --images /ruta/test --out predictions
    python code/cellpose/predict.py --images /ruta/test --out predictions --wandb
    python code/cellpose/predict.py --images /ruta/test --out predictions \\
                                    --ckpt code/checkpoints/cellpose_all   # modelo final

--ckpt es la carpeta que crea train.py (models/cpsam_tem_{fiber,axon} y
train_fiber.json, de donde se lee la escala de entrenamiento).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402
from common import (CELLPROB, DEFAULT_CKPT, check_prediction, fiber_stats, list_images,  # noqa: E402
                    load_models, model_path, predict_image, read_image, resolve_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="carpeta con las TIFF de test")
    ap.add_argument("--out", required=True, help="carpeta de salida (predictions/)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="carpeta de train.py")
    ap.add_argument("--scale", type=float, default=None,
                    help="por defecto, la de entrenamiento (train_fiber.json)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cellprob", type=float, default=CELLPROB,
                    help=f"umbral de probabilidad de Cellpose (def. {CELLPROB}; Cellpose usa 0)")
    ap.add_argument("--no-multiscale", dest="multiscale", action="store_false",
                    help="sin la 2ª pasada a escala/2")
    ap.add_argument("--wandb", action="store_true", help="registrar la inferencia en W&B")
    ap.add_argument("--wandb-name", default=None, help="nombre del run (def. predict-<out>)")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-images", type=int, default=4, help="nº de imágenes de ejemplo")
    args = ap.parse_args()

    scale = resolve_scale(args.ckpt, args.scale)
    files = list_images(args.images)
    if not files:
        sys.exit(f"no hay .tif/.tiff en {args.images}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m_fiber, m_axon = load_models(model_path(args.ckpt, "fiber"), model_path(args.ckpt, "axon"))
    print(f"{len(files)} imágenes | ckpt {args.ckpt} | scale {scale} | "
          f"cellprob {args.cellprob} | multiscale {args.multiscale}")

    run = None
    if args.wandb:
        run = wandb_log.start(args.wandb_name or f"predict-{out.name}", args.wandb_group,
                              job_type="predict",
                              config={**vars(args), "scale": scale, "n_images": len(files)})

    rows, examples = [], []
    for i, path in enumerate(files, start=1):
        t0 = time.time()
        img = read_image(path)
        pred = predict_image(img, m_fiber, m_axon, scale, args.batch_size,
                             args.cellprob, args.multiscale)
        check_prediction(pred, img.shape)
        tifffile.imwrite(out / path.name, pred, compression="zlib")

        row = {"image": path.name, **fiber_stats(pred)}
        rows.append(row)
        print(f"[{i}/{len(files)}] {path.name}: {row['n_fibers']} fibras, "
              f"g medio {row['g_ratio_mean']:.3f} ({time.time() - t0:.0f}s)")
        if run is not None and len(examples) < args.wandb_images:
            examples.append(wandb_log.mask_image(img, pred, caption=path.name))

    n_total = sum(r["n_fibers"] for r in rows)
    print(f"\n{len(rows)} predicciones en {out} | {n_total} fibras en total")

    if run is not None:
        log = {"per_image": wandb_log.table(rows)}
        if examples:
            log["examples"] = examples
        run.log(log)
        run.summary.update({"n_images": len(rows), "n_fibers_total": n_total,
                            "g_ratio_mean": float(np.nanmean([r["g_ratio_mean"] for r in rows]))})
        run.finish()


if __name__ == "__main__":
    main()
