"""Inferencia: segmentacion + rechazo -> TIFF uint16 con las instancias.

    # un solo modelo
    python predict.py --checkpoint C.pth --images DIR --plans P.json \
        --dataset-json D.json --out predictions/ [--tau 0]

    # ensemble (los 5 folds de la CV se promedian)
    python predict.py --ensemble f0.pth f1.pth f2.pth f3.pth f4.pth \
        --images DIR --plans P.json --dataset-json D.json --out predictions/

Salida: uint16 a la resolucion nativa, 0 fondo / 2k axon / 2k+1 mielina.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tifffile

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

# nnU-Net lee estas variables al importarse; se apuntan a un temporal.
_tmp = tempfile.mkdtemp(prefix="nnunet_predict_")
for _v in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
    os.environ.setdefault(_v, _tmp)

from infer import load_model, predict      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="un solo modelo")
    ap.add_argument("--ensemble", nargs="+", help="varios checkpoints; se promedian")
    ap.add_argument("--images", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="umbral del rechazo; 0 = solo el gate de borde (lo medido)")
    ap.add_argument("--no-close", action="store_true",
                    help="no cerrar los anillos abiertos (pipeline antiguo, medido peor)")
    ap.add_argument("--save-sem", action="store_true",
                    help="guardar <out>/sem/<caso>_sem.tif y _score.npy para reevaluar")
    a = ap.parse_args()

    paths = a.ensemble or ([a.checkpoint] if a.checkpoint else [])
    if not paths:
        ap.error("hace falta --checkpoint o --ensemble")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_json, plans = json.load(open(a.dataset_json)), json.load(open(a.plans))
    models = [load_model(p, a.plans, dataset_json, device) for p in paths]
    print(f"{len(models)} modelo(s) cargado(s)")

    sem_dir = out / "sem"
    if a.save_sem:
        sem_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.images).glob("*.tif"))
    for i, f in enumerate(files, 1):
        inst, sem, score = predict(f, models, plans, a.tau, device, close=not a.no_close)
        tifffile.imwrite(out / f.name, inst.astype("uint16"))
        if a.save_sem:
            tifffile.imwrite(sem_dir / f"{f.stem}_sem.tif", sem)
            np.save(sem_dir / f"{f.stem}_score.npy", score.astype(np.float32))
        print(f"  [{i}/{len(files)}] {f.name}", flush=True)
    print(f"escritas {len(list(out.glob('*.tif')))} predicciones en {out}")


if __name__ == "__main__":
    main()
