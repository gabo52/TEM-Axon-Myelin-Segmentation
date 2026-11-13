"""Fine-tuning de Cellpose-SAM para el TEM Axon & Myelin Challenge (SIPAIM 2026).

Normalmente se lanza desde code/train.py (--model cellpose), pero también
funciona directamente.

Entrena uno o los dos modelos del pipeline (ver common.py):
    fiber : fibra completa (axón ∪ mielina)
    axon  : solo axón
Los dos usan EXACTAMENTE el mismo split train/val (<out>/split.json), para que
la validación del pipeline completo (mielina = fibra − axón) sea limpia.

Uso (RTX 3080 12 GB):
    python code/cellpose/train.py --target both
    python code/cellpose/train.py --target fiber --epochs 200 --freeze 8
    python code/cellpose/train.py --target axon --split code/checkpoints/cellpose/split.json
    python code/cellpose/train.py --target both --wandb
    python code/cellpose/train.py --target both --all       # modelo final: todas las imágenes

Salida (en --out; por defecto code/checkpoints/cellpose, o .../cellpose_all con --all):
    models/cpsam_tem_fiber    pesos del modelo de fibra
    models/cpsam_tem_axon     pesos del modelo de axón
    split.json                imágenes de train / val
    train_<target>.json       argumentos, versiones y curvas de pérdida
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402
from early_stop import EarlyStopping  # noqa: E402
from common import (DEFAULT_CKPT, DEFAULT_IMAGES, DEFAULT_LABELS, TARGETS, list_pairs,  # noqa: E402
                    load_split, make_kfold, make_split, read_image, rescale_img, rescale_lbl,
                    save_split)

MODEL_NAME = "cpsam_tem_{target}"  # mismos nombres que los modelos ya entrenados


# ---------------------------------------------------------------- datos
def load_dataset(pairs, names, target, scale):
    """Lee y reescala imagen a imagen, para no tener las 266 a 3008×4096 en RAM.

    Devuelve imágenes, etiquetas del objetivo y máscaras de píxeles válidos (no artefacto).
    """
    by_name = {img.stem: (img, lbl) for img, lbl in pairs}
    to_target = TARGETS[target]
    images, labels, valid = [], [], []
    for name in names:
        img_path, lbl_path = by_name[name]
        img = rescale_img(read_image(img_path), scale)
        y = tifffile.imread(lbl_path)
        images.append(img)
        labels.append(rescale_lbl(to_target(y), img.shape))
        valid.append(rescale_lbl((y != 1).astype(np.int32), img.shape) > 0)
    return images, labels, valid


# ---------------------------------------------------------------- encoder
def freeze_encoder(net, n_blocks):
    """Congela patch/pos embedding y los primeros n_blocks del ViT de SAM.

    Congelar los bloques INICIALES ahorra memoria de verdad: autograd no guarda
    sus activaciones porque nada antes de ellos requiere gradiente.
    """
    import torch

    enc = next((m for m in net.modules()
                if isinstance(getattr(m, "blocks", None), torch.nn.ModuleList)), None)
    if enc is None:
        raise AttributeError("No encontré el encoder ViT (.blocks). Revisa print(model.net).")

    n_blocks = min(n_blocks, len(enc.blocks))
    frozen = list(enc.patch_embed.parameters())
    if enc.pos_embed is not None:
        frozen.append(enc.pos_embed)
    for block in enc.blocks[:n_blocks]:
        frozen += list(block.parameters())
    for p in frozen:
        p.requires_grad = False

    n_all = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"congelados {n_blocks}/{len(enc.blocks)} bloques ViT + embeddings | "
          f"entrenables {n_train / 1e6:.0f}M de {n_all / 1e6:.0f}M")


# ---------------------------------------------------------------- train
def train_one(target, args, pairs, train_names, val_names, out):
    import cellpose
    import torch
    from cellpose import models, train

    print(f"\n===== modelo '{target}' =====")
    t0 = time.time()
    train_x, train_y, _ = load_dataset(pairs, train_names, target, args.scale)
    val_x, val_y, val_valid = load_dataset(pairs, val_names, target, args.scale)
    print(f"datos cargados en {time.time() - t0:.0f}s | tamaño reescalado {train_x[0].shape}")

    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained)
    if args.freeze > 0:
        freeze_encoder(model.net, args.freeze)

    info = {
        "target": target,
        "args": vars(args),
        "versions": {"cellpose": getattr(cellpose, "version", "?"), "torch": torch.__version__},
        "n_train": len(train_names),
        "n_val": len(val_names),
    }
    run = None
    if args.wandb:
        run_name = MODEL_NAME.format(target=target)
        if args.kfold:
            run_name = f"fold{args.fold}_{run_name}"
        run = wandb_log.start_train(run_name, args.wandb_group,
                                    config={**info, "val_images": val_names})

    model_path = out / "models" / MODEL_NAME.format(target=target)
    best_path = Path(f"{model_path}_best")
    # early stopping por Dice de validación (sin validación, --all, no hay early stopping)
    result = None
    with EarlyStopping(model, val_x, val_y, val_valid, best_path,
                       patience=args.patience if val_x else 0,
                       every=args.check_every, start=args.check_start,
                       wandb_run=run.run if run is not None else None,
                       stop_epoch=args.stop_epoch) as es:
        result = train.train_seg(
            model.net,
            train_data=train_x, train_labels=train_y,
            test_data=val_x or None, test_labels=val_y or None,   # None con --all
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=0.1,
            nimg_per_epoch=args.nimg_per_epoch,
            min_train_masks=1,
            save_path=str(out),
            model_name=MODEL_NAME.format(target=target),
        )
    if es.stopped:
        # train_seg cortado: no guardó el modelo final. Se guardan los pesos actuales (fp32);
        # con validación se sustituyen abajo por el mejor checkpoint.
        model.net.save_model(str(model_path))
    # el modelo que queda es el de MEJOR Dice de validación: si el entrenamiento llegó
    # al final, el modelo de la última época también compite. Al terminar, Cellpose pasa la
    # red a bf16; se vuelve a fp32 con los pesos finales (guardados en fp32) para medir el
    # Dice en la misma precisión que los controles durante el entrenamiento.
    final_dice = None
    if val_x and not es.stopped:
        try:
            model.net.dtype = torch.float32
            model.net.load_model(str(model_path), device=model.device)
            final_dice = es.check(args.epochs - 1)
        except Exception as e:
            print(f"[early stop] control final fallido, se conserva el mejor anterior ({e})")
    if es.best_epoch is not None and es.best_epoch != args.epochs - 1 and best_path.exists():
        os.replace(best_path, model_path)
    best_path.unlink(missing_ok=True)
    print(f"modelo '{target}' guardado en {model_path} ({(time.time() - t0) / 60:.1f} min) | "
          f"mejor época {es.best_epoch} (Dice {es.best_dice:.4f}) | parada temprana {es.stopped}")

    info["model"] = str(model_path)
    info["best_epoch"] = es.best_epoch
    info["best_val_dice"] = None if es.best_epoch is None else es.best_dice
    info["final_val_dice"] = final_dice
    info["stopped_early"] = es.stopped
    info["last_epoch"] = es.history[-1][0] if es.stopped and es.history else args.epochs - 1
    info["history"] = es.history   # (época, train_loss, val_loss, val_dice) en cada validación
    train_loss, val_loss = (result[1], result[2]) if result is not None else ([], [])
    info["train_loss"] = [float(v) for v in train_loss]
    info["val_loss"] = [float(v) for v in val_loss]
    (out / f"train_{target}.json").write_text(json.dumps(info, indent=2))
    if run is not None:
        run.finish(train_loss, val_loss, model_path,
                   summary={k: info[k] for k in ("best_epoch", "best_val_dice",
                                                 "stopped_early", "last_epoch")})

    del model, train_x, train_y, val_x, val_y
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["fiber", "axon", "both"], default="both",
                    help="qué modelo Cellpose entrenar (both = los dos con el mismo split)")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES), help="carpeta con las TIFF de entrada")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS), help="máscaras uint16 con el mismo nombre")
    ap.add_argument("--out", default=None,
                    help="carpeta de salida (def. code/checkpoints/cellpose, "
                         "o code/checkpoints/cellpose_all con --all)")
    ap.add_argument("--pretrained", default="cpsam", help="pesos base: cpsam | cpsam_v2 | ruta")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="reducción de la imagen; usar el MISMO valor en predict")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=12,
                    help="bloques ViT iniciales congelados (0 = entrenar todo)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None,
                    help="reutilizar un split.json existente (si no, se crea uno)")
    ap.add_argument("--all", action="store_true",
                    help="modelo final: entrena con TODAS las imágenes, sin validación")
    ap.add_argument("--patience", type=int, default=60,
                    help="early stopping: parar si el Dice de validación no mejora en N épocas "
                         "(0 = desactivado)")
    ap.add_argument("--check-every", type=int, default=10,
                    help="cada cuántas épocas se mide el Dice de validación (múltiplo de 10)")
    ap.add_argument("--check-start", type=int, default=40,
                    help="primera época en la que se mide el Dice (múltiplo de 10)")
    ap.add_argument("--stop-epoch", type=int, default=None,
                    help="parar tras esta época sin cambiar el calendario de --epochs "
                         "(modelo final: misma trayectoria de learning rate que un fold)")
    ap.add_argument("--nimg-per-epoch", type=int, default=None,
                    help="imágenes por época (def. todas las de train); el modelo final usa "
                         "el tamaño de train de un fold para dar los mismos pasos por época")
    ap.add_argument("--kfold", type=int, default=0,
                    help="validación cruzada con K folds (usar con --fold); 0 = split simple")
    ap.add_argument("--fold", type=int, default=0, help="fold a entrenar (0..K-1)")
    ap.add_argument("--wandb", action="store_true", help="registrar el entrenamiento en W&B")
    ap.add_argument("--wandb-group", default=f"cellpose-{time.strftime('%m%d-%H%M')}",
                    help="grupo de W&B (fiber y axon de una misma ejecución van juntos)")
    args = ap.parse_args()

    import torch
    from cellpose import io
    io.logger_setup()  # pérdidas en consola y en ~/.cellpose/run.log (wandb_log las lee de aquí)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Cellpose solo informa en la época 5 y en múltiplos de 10: otros valores se saltarían
    if args.check_every % 10 or args.check_start % 10:
        sys.exit("--check-every y --check-start deben ser múltiplos de 10")
    if args.stop_epoch is not None and (args.stop_epoch % 10 or args.stop_epoch >= args.epochs):
        sys.exit("--stop-epoch debe ser múltiplo de 10 y menor que --epochs")
    if args.kfold and (args.all or args.split):
        sys.exit("--kfold no se combina con --all ni con --split")
    if args.kfold and not 0 <= args.fold < args.kfold:
        sys.exit(f"--fold debe estar entre 0 y {args.kfold - 1}")

    # cada modo en su carpeta, para no pisar modelos ni split.json:
    #   cellpose/ (split simple), cellpose_all/ (--all), cellpose_kfold/fold<i>/ (--kfold)
    if args.out:
        out = Path(args.out)
    elif args.kfold:
        out = DEFAULT_CKPT.with_name("cellpose_kfold") / f"fold{args.fold}"
    else:
        out = DEFAULT_CKPT.with_name("cellpose_all" if args.all else "cellpose")
    out.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(args.images, args.labels)
    names = [img.stem for img, _ in pairs]
    print(f"{len(names)} pares imagen/máscara en {args.images}")

    # split: con seed=0 el simple es el mismo que usó cellpose_tem.py (eval_fiber/)
    if args.kfold:
        train_names, val_names = make_kfold(names, args.kfold, args.seed)[args.fold]
    elif args.split:
        train_names, val_names = load_split(args.split)
    else:
        train_names, val_names = make_split(names, args.val_frac, args.seed)
    if args.all:
        train_names, val_names = sorted(train_names + val_names), []
    missing = set(train_names + val_names) - set(names)
    if missing:
        sys.exit(f"el split nombra imágenes que no existen: {sorted(missing)[:5]}")
    save_split(out / "split.json", train_names, val_names, seed=args.seed, all=args.all,
               kfold=args.kfold, fold=args.fold if args.kfold else None)
    print(f"train={len(train_names)}  val={len(val_names)}  -> {out / 'split.json'}")

    targets = ["fiber", "axon"] if args.target == "both" else [args.target]
    for target in targets:
        train_one(target, args, pairs, train_names, val_names, out)


if __name__ == "__main__":
    main()
