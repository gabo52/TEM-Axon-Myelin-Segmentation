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

Salida (en --out):
    models/cpsam_tem_fiber    pesos del modelo de fibra
    models/cpsam_tem_axon     pesos del modelo de axón
    split.json                imágenes de train / val
    train_<target>.json       argumentos, versiones y curvas de pérdida
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402
from common import (DEFAULT_IMAGES, DEFAULT_LABELS, ROOT, TARGETS, list_pairs,  # noqa: E402
                    load_split, make_split, read_image, rescale_img, rescale_lbl, save_split)

MODEL_NAME = "cpsam_tem_{target}"  # mismos nombres que los modelos ya entrenados


# ---------------------------------------------------------------- datos
def load_dataset(pairs, names, target, scale):
    """Lee y reescala imagen a imagen, para no tener las 266 a 3008×4096 en RAM."""
    by_name = {img.stem: (img, lbl) for img, lbl in pairs}
    to_target = TARGETS[target]
    images, labels = [], []
    for name in names:
        img_path, lbl_path = by_name[name]
        img = rescale_img(read_image(img_path), scale)
        lbl = rescale_lbl(to_target(tifffile.imread(lbl_path)), img.shape)
        images.append(img)
        labels.append(lbl)
    return images, labels


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
    train_x, train_y = load_dataset(pairs, train_names, target, args.scale)
    val_x, val_y = load_dataset(pairs, val_names, target, args.scale)
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
        run = wandb_log.start(MODEL_NAME.format(target=target), args.wandb_group,
                              config={**info, "val_images": val_names})

    model_path, train_loss, val_loss = train.train_seg(
        model.net,
        train_data=train_x, train_labels=train_y,
        test_data=val_x or None, test_labels=val_y or None,   # None con --all
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.1,
        min_train_masks=1,
        save_path=str(out),
        model_name=MODEL_NAME.format(target=target),
    )
    print(f"modelo '{target}' guardado en {model_path} ({(time.time() - t0) / 60:.1f} min)")

    info["model"] = str(model_path)
    info["train_loss"] = [float(v) for v in train_loss]
    info["val_loss"] = [float(v) for v in val_loss]
    (out / f"train_{target}.json").write_text(json.dumps(info, indent=2))
    if run is not None:
        run.finish(train_loss, val_loss, model_path)

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
    ap.add_argument("--out", default=str(ROOT / "code" / "checkpoints" / "cellpose"))
    ap.add_argument("--pretrained", default="cpsam", help="pesos base: cpsam | cpsam_v2 | ruta")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="reducción de la imagen; usar el MISMO valor en predict")
    ap.add_argument("--epochs", type=int, default=150)
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
    ap.add_argument("--wandb", action="store_true", help="registrar el entrenamiento en W&B")
    ap.add_argument("--wandb-group", default=f"cellpose-{time.strftime('%m%d-%H%M')}",
                    help="grupo de W&B (fiber y axon de una misma ejecución van juntos)")
    args = ap.parse_args()

    import torch
    from cellpose import io
    io.logger_setup()  # pérdidas en consola y en ~/.cellpose/run.log (wandb_log las lee de aquí)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(args.images, args.labels)
    names = [img.stem for img, _ in pairs]
    print(f"{len(names)} pares imagen/máscara en {args.images}")

    # split: con seed=0 es el mismo que usó cellpose_tem.py (eval_fiber/)
    if args.split:
        train_names, val_names = load_split(args.split)
    else:
        train_names, val_names = make_split(names, args.val_frac, args.seed)
    if args.all:
        train_names, val_names = sorted(train_names + val_names), []
    missing = set(train_names + val_names) - set(names)
    if missing:
        sys.exit(f"el split nombra imágenes que no existen: {sorted(missing)[:5]}")
    save_split(out / "split.json", train_names, val_names, seed=args.seed, all=args.all)
    print(f"train={len(train_names)}  val={len(val_names)}  -> {out / 'split.json'}")

    targets = ["fiber", "axon"] if args.target == "both" else [args.target]
    for target in targets:
        train_one(target, args, pairs, train_names, val_names, out)


if __name__ == "__main__":
    main()
