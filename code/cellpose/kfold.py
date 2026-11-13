"""Validación cruzada K-fold de Cellpose + modelo final con todas las imágenes.

Reanudable: cada paso se salta si ya está hecho, así que tras un corte basta con
relanzar el mismo comando. Un paso que falla se anota y se sigue con el resto.

    # lanzar en segundo plano (sigue aunque se cierre la terminal)
    nohup python code/cellpose/kfold.py --wandb > code/checkpoints/kfold.log 2>&1 &
    tail -f code/checkpoints/kfold.log

Pasos:
  1. entrena fiber y axon en cada fold          -> <root>/cellpose_kfold/fold<i>/
  2. entrena el modelo final con las 266 (--all) -> <root>/cellpose_all/
  3. evalúa cada fold en su validación con cada variante de inferencia
                                                -> <root>/cellpose_kfold/eval/<variante>/fold<i>/
  4. resume las métricas fuera de fold (todas las imágenes) -> <root>/cellpose_kfold/summary.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import DEFAULT_CKPT, DEFAULT_IMAGES, DEFAULT_LABELS  # noqa: E402

# variantes de inferencia evaluadas fuera de fold (argumentos de evaluate.py)
VARIANTS = {
    "cellpose_default": ["--cellprob", "0", "--no-multiscale"],
    "cp-1_multiscale": ["--cellprob", "-1"],
    "cp-2_multiscale": ["--cellprob", "-2"],
}
TARGETS = ("fiber", "axon")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    """Lanza un paso; devuelve True si terminó bien."""
    log("$ " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")   # el log no pierde líneas si el proceso muere
    ok = subprocess.run([str(c) for c in cmd], env=env).returncode == 0
    log(f"{'OK' if ok else 'FALLO'} ({(time.time() - t0) / 60:.1f} min)")
    return ok


def trained(marker, **expected):
    """¿Ya entrenado CON ESTA configuración? Si el marcador es de otra configuración
    (p. ej. una prueba con otro k o pocas épocas) se aborta en vez de reutilizarlo."""
    if not marker.exists():
        return False
    args = json.loads(marker.read_text())["args"]
    diff = {k: (args.get(k), v) for k, v in expected.items() if args.get(k) != v}
    if diff:
        sys.exit(f"{marker} es de otra configuración {diff} (guardado, esperado). "
                 f"Bórralo o usa otro --root.")
    return True


def final_plan(kroot, k, target):
    """Cómo entrenar el modelo final para que siga la MISMA trayectoria que un fold:
    mismo calendario de --epochs (la bajada del learning rate de Cellpose depende de él),
    parada en la mejor época mediana de los folds y tantas imágenes por época como el
    train de un fold (con 266 imágenes cada época daría ~25 % más pasos)."""
    infos = [json.loads((kroot / f"fold{i}" / f"train_{target}.json").read_text())
             for i in range(k)]
    best = sorted(i["best_epoch"] for i in infos)
    stop = best[len(best) // 2]                 # mediana (un valor real: época con control)
    nimg = int(np.median([i["n_train"] for i in infos]))
    return stop, nimg, best


# ---------------------------------------------------------------- resumen
def summarize(eval_root, k):
    """Métricas fuera de fold: Dice/HD95 = media por imagen; MAE_g = sobre todas las fibras."""
    summary = {}
    for variant in VARIANTS:
        rows, per_fold = [], []
        for i in range(k):
            f = eval_root / variant / f"fold{i}" / "metrics.csv"
            if not f.exists():
                continue
            fold_rows = [r for r in csv.DictReader(open(f)) if r["image"] != "MEAN"]
            rows += fold_rows
            per_fold.append(_aggregate(fold_rows))
        if not rows:
            continue
        s = _aggregate(rows)
        s["n_folds"] = len(per_fold)
        if len(per_fold) < k:
            log(f"AVISO {variant}: solo {len(per_fold)}/{k} folds evaluados "
                f"({len(rows)} imágenes): no comparable con las demás variantes")
        for key in ("dice_mean", "hd95_mean", "g_ratio_mae"):   # desviación muestral entre folds
            vals = [p[key] for p in per_fold]
            s[f"{key}_fold_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[variant] = s
    return summary


def _aggregate(rows):
    val = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan  # noqa: E731
    out = {k: float(np.nanmean([val(r, k) for r in rows]))
           for k in ("dice_axon", "dice_myelin", "hd95_axon", "hd95_myelin")}
    out["dice_mean"] = (out["dice_axon"] + out["dice_myelin"]) / 2
    out["hd95_mean"] = (out["hd95_axon"] + out["hd95_myelin"]) / 2
    n = np.array([int(r["n_matched"]) for r in rows])
    g = np.array([val(r, "g_ratio_mae") for r in rows])
    ok = n > 0
    out["g_ratio_mae"] = float((g[ok] * n[ok]).sum() / n[ok].sum())
    out["n_images"] = len(rows)
    out["n_hd95_nan"] = int(sum(np.isnan(val(r, "hd95_axon")) or np.isnan(val(r, "hd95_myelin"))
                                for r in rows))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200, help="épocas máximas de cada fold")
    ap.add_argument("--patience", type=int, default=60,
                    help="early stopping por Dice de validación en los folds (0 = no)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--root", default=str(DEFAULT_CKPT.parent), help="def. code/checkpoints")
    ap.add_argument("--skip-final", action="store_true", help="no entrenar el modelo final")
    ap.add_argument("--skip-eval", action="store_true", help="no evaluar los folds")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="cellpose-kfold")
    args = ap.parse_args()

    root = Path(args.root)
    kroot, final = root / "cellpose_kfold", root / "cellpose_all"
    common = ["--images", args.images, "--labels", args.labels, "--epochs", args.epochs,
              "--seed", args.seed]
    wb = ["--wandb", "--wandb-group", args.wandb_group] if args.wandb else []
    failed = []
    log(f"k={args.k} epochs={args.epochs} root={root} wandb={args.wandb}")

    # 1. folds
    fold_cfg = dict(kfold=args.k, seed=args.seed, epochs=args.epochs, patience=args.patience)
    fold_done = lambda i, t: trained(kroot / f"fold{i}" / f"train_{t}.json",  # noqa: E731
                                     fold=i, **fold_cfg)
    for i in range(args.k):
        out = kroot / f"fold{i}"
        for t in TARGETS:
            if fold_done(i, t):
                log(f"fold{i}/{t}: ya entrenado")
                continue
            if not run([sys.executable, CODE / "train.py", "--kfold", args.k, "--fold", i,
                        "--target", t, "--out", out, "--patience", args.patience,
                        *common, *wb]):
                failed.append(f"train fold{i}/{t}")

    # 2. modelo final con todas las imágenes. No tiene validación (no hay early stopping):
    #    sigue la trayectoria de un fold y para en la mejor época mediana (ver final_plan),
    #    así que solo se lanza cuando los k folds de ese objetivo están entrenados.
    if not args.skip_final:
        for t in TARGETS:
            if not all(fold_done(i, t) for i in range(args.k)):
                log(f"final/{t}: faltan folds de {t}; no se entrena (relanza cuando estén)")
                failed.append(f"train final/{t} (faltan folds)")
                continue
            stop, nimg, best = final_plan(kroot, args.k, t)
            stop = None if stop >= args.epochs - 1 else stop   # mejor = última: sin parada
            if trained(final / f"train_{t}.json", all=True, seed=args.seed, epochs=args.epochs,
                       stop_epoch=stop, nimg_per_epoch=nimg):
                log(f"final/{t}: ya entrenado")
                continue
            log(f"final/{t}: calendario de {args.epochs} épocas, parada tras la época {stop}, "
                f"{nimg} imágenes por época (mejor época por fold: {best})")
            cmd = [sys.executable, CODE / "train.py", "--all", "--target", t, "--out", final,
                   "--nimg-per-epoch", nimg, *common, *wb]
            if stop is not None:
                cmd += ["--stop-epoch", stop]
            if not run(cmd):
                failed.append(f"train final/{t}")

    # 3. evaluación fuera de fold
    if not args.skip_eval:
        for i in range(args.k):
            ckpt = kroot / f"fold{i}"
            if not all(fold_done(i, t) for t in TARGETS):
                log(f"fold{i}: faltan modelos, no se evalúa")
                continue
            newest_model = max((ckpt / f"train_{t}.json").stat().st_mtime for t in TARGETS)
            for name, flags in VARIANTS.items():
                out = kroot / "eval" / name / f"fold{i}"
                # una evaluación anterior a los modelos actuales (fold reentrenado) se repite
                if (out / "metrics.csv").exists() and \
                        (out / "metrics.csv").stat().st_mtime > newest_model:
                    log(f"eval {name}/fold{i}: ya hecha")
                    continue
                wb_eval = (["--wandb", "--wandb-group", args.wandb_group,
                            "--wandb-name", f"eval_fold{i}_{name}"] if args.wandb else [])
                if not run([sys.executable, CODE / "evaluate.py", "--ckpt", ckpt,
                            "--images", args.images, "--labels", args.labels,
                            "--out", out, *flags, *wb_eval]):
                    failed.append(f"eval {name}/fold{i}")

        # 4. resumen
        summary = summarize(kroot / "eval", args.k)
        (kroot / "summary.json").write_text(json.dumps(summary, indent=2))
        log("RESUMEN fuera de fold (Dice/HD95 media por imagen, MAE_g sobre fibras):")
        for name, s in summary.items():
            log(f"  {name:>17}: Dice {s['dice_mean']:.4f} ±{s['dice_mean_fold_std']:.4f} | "
                f"HD95 {s['hd95_mean']:.1f} ±{s['hd95_mean_fold_std']:.1f} nm | "
                f"MAE_g {s['g_ratio_mae']:.4f} | {s['n_images']} imgs, {s['n_folds']} folds, "
                f"{s['n_hd95_nan']} HD95 NaN")

    log("TERMINADO" + (f" con fallos: {failed} (relanza el mismo comando)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
