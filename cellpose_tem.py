"""
Cellpose-SAM para el TEM Instance Segmentation Challenge (axón + mielina emparejados).

Estrategia: dos modelos Cellpose-SAM fine-tuneados
  - modelo "fiber": instancias de fibra completa (axón ∪ mielina)
  - modelo "axon" : instancias de axón
  Luego: mielina = fibra − axón, emparejados por construcción.

Formato de entrada (tus TIFF):
  X/  -> imágenes TEM (grises, 8 o 16 bit), ej. img_001.tif
  Y/  -> máscaras con el MISMO nombre, ej. img_001.tif
         0 = fondo, 1 = artefacto, pares >=2 = axón, impar (par+1) = su mielina

Instalación:
  pip install cellpose tifffile scikit-image scipy

Uso (RTX 3080 12 GB):
  python cellpose_tem.py stats  --x X --y Y
  python cellpose_tem.py train  --x X --y Y --target fiber --batch-size 4 --freeze 12 --out models
  python cellpose_tem.py train  --x X --y Y --target axon  --batch-size 4 --freeze 12 --out models
  python cellpose_tem.py predict --x X_test --fiber-model models/models/cpsam_tem_fiber \
                                 --axon-model models/models/cpsam_tem_axon --scale 0.25 --out preds
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.transform import resize


# ----------------------------------------------------------------------------
# Carga y conversión de etiquetas
# ----------------------------------------------------------------------------
def load_pairs(x_dir, y_dir):
    x_dir, y_dir = Path(x_dir), Path(y_dir)
    exts = {".tif", ".tiff"}
    xs = sorted(p for p in x_dir.iterdir() if p.suffix.lower() in exts)
    imgs, labs, names = [], [], []
    for xp in xs:
        yp = (y_dir / xp.name) if (y_dir / xp.name).exists() else None
        if yp is None:
            print(f"[skip] sin máscara: {xp.name}")
            continue
        img = tifffile.imread(xp)
        if img.ndim == 3 and img.shape[-1] in (3, 4):  # RGB guardado por error -> gris
            img = img[..., :3].mean(-1)
        imgs.append(img)
        labs.append(tifffile.imread(yp))
        names.append(xp.stem)
    print(f"{len(imgs)} pares cargados")
    return imgs, labs, names


def to_fiber(y):
    """Axón 2k y mielina 2k+1 -> misma fibra k. Fondo (0) y artefacto (1) -> 0."""
    return (y // 2).astype(np.int32)


def to_axon(y):
    """Solo axones (IDs pares >= 2) -> instancia k."""
    out = np.where((y >= 2) & (y % 2 == 0), y // 2, 0)
    return out.astype(np.int32)


def equiv_diameters(lbl):
    ids, areas = np.unique(lbl[lbl > 0], return_counts=True)
    return 2 * np.sqrt(areas / np.pi)


# ----------------------------------------------------------------------------
# Reescalado (Cellpose-SAM se entrenó con objetos de ~7.5–120 px de diámetro)
# ----------------------------------------------------------------------------
def rescale_img(img, s):
    if s == 1:
        return img.astype(np.float32)
    shape = (round(img.shape[0] * s), round(img.shape[1] * s))
    return resize(img, shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def rescale_lbl(lbl, shape):
    return resize(lbl, shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)


# ----------------------------------------------------------------------------
# Congelar bloques del encoder ViT (ahorra memoria GPU y acelera)
# ----------------------------------------------------------------------------
def freeze_encoder(net, n_blocks):
    """Congela patch embedding, pos embedding y los primeros n_blocks del ViT.

    Al congelar los bloques INICIALES, autograd no guarda sus activaciones
    (la entrada no requiere gradiente), así que se ahorra memoria de verdad.
    """
    import torch

    enc = None
    for _, m in net.named_modules():
        if isinstance(getattr(m, "blocks", None), torch.nn.ModuleList):
            enc = m
            break
    if enc is None:
        raise AttributeError("No encontré el encoder ViT (.blocks). Revisa print(model.net).")

    n_total = len(enc.blocks)
    n_blocks = min(n_blocks, n_total)
    frozen = []
    if hasattr(enc, "patch_embed"):
        frozen += list(enc.patch_embed.parameters())
    if isinstance(getattr(enc, "pos_embed", None), torch.nn.Parameter):
        frozen.append(enc.pos_embed)
    for blk in enc.blocks[:n_blocks]:
        frozen += list(blk.parameters())
    for p in frozen:
        p.requires_grad = False

    n_all = sum(p.numel() for p in net.parameters())
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"congelados {n_blocks}/{n_total} bloques ViT + embeddings | "
          f"params entrenables: {n_tr/1e6:.0f}M de {n_all/1e6:.0f}M")


# ----------------------------------------------------------------------------
# Comandos
# ----------------------------------------------------------------------------
def cmd_stats(a):
    imgs, labs, _ = load_pairs(a.x, a.y)
    shapes = {im.shape for im in imgs}
    print("tamaños de imagen:", shapes)
    print("dtype/rango ej.:", imgs[0].min(), imgs[0].max())
    fd = np.concatenate([equiv_diameters(to_fiber(y)) for y in labs])
    ad = np.concatenate([equiv_diameters(to_axon(y)) for y in labs])
    for name, d in [("fibra", fd), ("axón", ad)]:
        print(f"diámetro {name} (px): mediana={np.median(d):.0f}  p5={np.percentile(d,5):.0f}  "
              f"p95={np.percentile(d,95):.0f}  n={len(d)}")
    for target in (40, 60, 80):
        print(f"  -> scale para fibra mediana≈{target}px: {target/np.median(fd):.3f}")


def cmd_train(a):
    from cellpose import models, train, io
    io.logger_setup()  # muestra pérdida y tiempo por época (y guarda ~/.cellpose/run.log)

    imgs, labs, names = load_pairs(a.x, a.y)
    conv = to_fiber if a.target == "fiber" else to_axon

    data, labels = [], []
    for im, y in zip(imgs, labs):
        im_s = rescale_img(im, a.scale)
        data.append(im_s)
        labels.append(rescale_lbl(conv(y), im_s.shape))
    del imgs, labs  # liberar RAM de las imágenes a resolución completa

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(data))
    n_val = max(1, int(len(data) * a.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"train={len(tr_idx)}  val={len(val_idx)}  (val: {[names[i] for i in val_idx]})")

    Path(a.out).mkdir(parents=True, exist_ok=True)

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    if a.freeze > 0:
        freeze_encoder(model.net, a.freeze)

    path, tr_loss, te_loss = train.train_seg(
        model.net,
        train_data=[data[i] for i in tr_idx],
        train_labels=[labels[i] for i in tr_idx],
        test_data=[data[i] for i in val_idx],
        test_labels=[labels[i] for i in val_idx],
        batch_size=a.batch_size,
        n_epochs=a.epochs,
        learning_rate=a.lr,
        weight_decay=0.1,
        min_train_masks=1,
        save_path=a.out,
        model_name=f"cpsam_tem_{a.target}",
    )
    print("modelo guardado en:", path)


def combine(fiber, axon, min_px=20):
    """Construye la máscara de entrega: axón = 2k, su mielina = 2k+1."""
    out = np.zeros(fiber.shape, np.int32)
    k = 1
    for f_id, sl in enumerate(ndi.find_objects(fiber), start=1):
        if sl is None:
            continue
        fmask = fiber[sl] == f_id
        ax = (axon[sl] > 0) & fmask
        if ax.sum() < min_px:
            continue  # fibra sin axón -> descartar
        # quedarse con el axón principal y rellenar huecos
        lab, n = ndi.label(ax)
        if n > 1:
            sizes = ndi.sum(ax, lab, range(1, n + 1))
            ax = lab == (np.argmax(sizes) + 1)
        ax = ndi.binary_fill_holes(ax) & fmask
        my = fmask & ~ax
        if my.sum() < min_px:
            continue  # sin anillo de mielina -> descartar
        o = out[sl]
        o[ax] = 2 * k
        o[my] = 2 * k + 1
        k += 1
    return out


def cmd_predict(a):
    from cellpose import models
    from skimage.io import imsave

    exts = {".tif", ".tiff"}
    files = sorted(p for p in Path(a.x).iterdir() if p.suffix.lower() in exts)
    m_fiber = models.CellposeModel(gpu=True, pretrained_model=a.fiber_model)
    m_axon = models.CellposeModel(gpu=True, pretrained_model=a.axon_model)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        img = tifffile.imread(p).astype(np.float32)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(-1)
        im_s = rescale_img(img, a.scale)
        fib = m_fiber.eval(im_s, batch_size=a.batch_size)[0]
        axn = m_axon.eval(im_s, batch_size=a.batch_size)[0]
        fib = rescale_lbl(fib, img.shape)
        axn = rescale_lbl(axn, img.shape)
        pred = combine(fib, axn)
        # PNG de un canal; uint16 por si hay >127 fibras
        imsave(out_dir / f"{p.stem}.png", pred.astype(np.uint16), check_contrast=False)
        print(f"{p.name}: {pred.max() // 2} fibras")


# ----------------------------------------------------------------------------
# Evaluación sobre el split de validación (mismo split que en train)
# ----------------------------------------------------------------------------
NM_PER_PX = 2.15


def val_files(x_dir, y_dir, val_frac):
    """Reproduce exactamente el split de cmd_train (mismo orden, misma semilla)."""
    exts = {".tif", ".tiff"}
    xs = sorted(p for p in Path(x_dir).iterdir()
                if p.suffix.lower() in exts and (Path(y_dir) / p.name).exists())
    idx = np.random.default_rng(0).permutation(len(xs))
    n_val = max(1, int(len(xs) * val_frac))
    return [xs[i] for i in idx[:n_val]]


def dice(a, b, valid):
    a, b = a & valid, b & valid
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def hd95(a, b):
    """HD95 simétrico entre bordes (en px), estilo medpy."""
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return np.nan
    sa = a ^ ndi.binary_erosion(a)
    sb = b ^ ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sa)
    db = ndi.distance_transform_edt(~sb)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def match_instances(gt, pr, thr=0.5):
    """Empareja instancias por IoU > thr (con thr>=0.5 el emparejamiento es único)."""
    ag = np.bincount(gt.ravel())
    ap_ = np.bincount(pr.ravel())
    m = (gt > 0) & (pr > 0)
    key = gt[m].astype(np.int64) * (pr.max() + 1) + pr[m]
    keys, inter = np.unique(key, return_counts=True)
    g_ids, p_ids = keys // (pr.max() + 1), keys % (pr.max() + 1)
    iou = inter / (ag[g_ids] + ap_[p_ids] - inter)
    ok = iou > thr
    n_gt = int((ag[1:] > 0).sum())
    n_pr = int((ap_[1:] > 0).sum())
    return g_ids[ok], p_ids[ok], iou[ok], n_gt, n_pr


def evaluate_image(y, pred, has_axon=True):
    """y: GT original (0,1,pares/impares). pred: 2k/2k+1 (o fibras k si has_axon=False)."""
    valid = y != 1
    gt_fib = to_fiber(y)
    r = {}
    if has_axon:
        pr_fib = (pred // 2).astype(np.int32)
        gt_ax, gt_my = (y >= 2) & (y % 2 == 0), (y >= 3) & (y % 2 == 1)
        pr_ax, pr_my = (pred >= 2) & (pred % 2 == 0), (pred >= 3) & (pred % 2 == 1)
        r["dice_axon"] = dice(gt_ax, pr_ax, valid)
        r["dice_myelin"] = dice(gt_my, pr_my, valid)
        r["hd95_axon_nm"] = hd95(gt_ax & valid, pr_ax & valid) * NM_PER_PX
        r["hd95_myelin_nm"] = hd95(gt_my & valid, pr_my & valid) * NM_PER_PX
    else:
        pr_fib = pred.astype(np.int32)
    r["dice_fiber"] = dice(gt_fib > 0, pr_fib > 0, valid)

    g, p, iou, n_gt, n_pr = match_instances(gt_fib, pr_fib)
    tp = len(g)
    r["n_gt"], r["n_pred"], r["tp"] = n_gt, n_pr, tp
    r["precision"] = tp / n_pr if n_pr else 0.0
    r["recall"] = tp / n_gt if n_gt else 0.0
    r["f1"] = 2 * tp / (n_gt + n_pr) if (n_gt + n_pr) else 1.0
    r["sq"] = float(iou.mean()) if tp else 0.0          # calidad de segmentación
    r["pq"] = r["sq"] * r["f1"]                          # panoptic quality

    if has_axon and tp:
        # g-ratio por fibra emparejada: sqrt(área axón / área fibra) (diámetros equivalentes)
        a_gt_ax = np.bincount(y[gt_ax] // 2, minlength=gt_fib.max() + 1)
        a_gt_f = np.bincount(gt_fib.ravel())
        a_pr_ax = np.bincount(pred[pr_ax] // 2, minlength=pr_fib.max() + 1)
        a_pr_f = np.bincount(pr_fib.ravel())
        g_gt = np.sqrt(a_gt_ax[g] / a_gt_f[g])
        g_pr = np.sqrt(a_pr_ax[p] / a_pr_f[p])
        r["g_gt_mean"] = float(g_gt.mean())
        r["g_pred_mean"] = float(g_pr.mean())
        r["delta_g_mae"] = float(np.abs(g_pr - g_gt).mean())
    return r


def predict_image(img, m_fiber, m_axon, scale, bs):
    im_s = rescale_img(img, scale)
    fib = rescale_lbl(m_fiber.eval(im_s, batch_size=bs)[0], img.shape)
    if m_axon is None:
        return fib
    axn = rescale_lbl(m_axon.eval(im_s, batch_size=bs)[0], img.shape)
    return combine(fib, axn)


def cmd_eval(a):
    import csv
    from cellpose import models
    from skimage.io import imsave

    # métricas con la fórmula oficial del challenge (comparables con nnU-Net)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
    from metrics import aggregate as official_aggregate
    from metrics import evaluate as official_evaluate

    files = val_files(a.x, a.y, a.val_frac)
    print(f"evaluando {len(files)} imágenes de validación")
    m_fiber = models.CellposeModel(gpu=True, pretrained_model=a.fiber_model)
    m_axon = models.CellposeModel(gpu=True, pretrained_model=a.axon_model) if a.axon_model else None
    has_axon = m_axon is not None
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = wandb_start(a, len(files)) if a.wandb else None

    rows, official, examples = [], [], []
    for p in files:
        img = tifffile.imread(p)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(-1)
        y = tifffile.imread(Path(a.y) / p.name).astype(np.int32)
        pred = predict_image(img, m_fiber, m_axon, a.scale, a.batch_size)
        imsave(out_dir / f"{p.stem}.png", pred.astype(np.uint16), check_contrast=False)
        r = {"image": p.stem, **evaluate_image(y, pred, has_axon)}
        rows.append(r)
        msg = f"{p.stem}: F1={r['f1']:.3f} PQ={r['pq']:.3f} DiceFib={r['dice_fiber']:.3f}"
        if has_axon:
            msg += (f" DiceAx={r['dice_axon']:.3f} DiceMy={r['dice_myelin']:.3f}"
                    f" HD95ax={r['hd95_axon_nm']:.0f}nm HD95my={r['hd95_myelin_nm']:.0f}nm"
                    f" Δg={r.get('delta_g_mae', np.nan):.3f}")
            official.append(official_evaluate(pred, y))
            if run is not None and len(examples) < a.wandb_images:
                caption = f"{p.stem}  DiceAx={r['dice_axon']:.3f}  DiceMy={r['dice_myelin']:.3f}"
                examples.append(wandb_example(img, y, pred, caption))
        print(msg)

    # unión de columnas: delta_g_* solo existe en imágenes con alguna fibra emparejada
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "image"))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image"] + keys)
        w.writeheader()
        w.writerows(rows)
        mean = {"image": "MEAN"}
        for k in keys:
            mean[k] = float(np.nanmean([r.get(k, np.nan) for r in rows]))
        w.writerow(mean)

    print("\n=== MEDIA validación ===")
    for k in keys:
        if k in ("n_gt", "n_pred", "tp"):
            print(f"{k:>16}: {int(sum(r[k] for r in rows))} (total)")
        else:
            print(f"{k:>16}: {mean[k]:.4f}")

    official_mean = official_aggregate(official) if official else {}
    if official_mean:
        print("\n=== MÉTRICAS OFICIALES (analysis/metrics.py, HD95 en nm) ===")
        for k, v in official_mean.items():
            print(f"{k:>16}: {v:.4f}")
    print(f"\nCSV y predicciones en: {out_dir}")

    if run is not None:
        wandb_finish(run, rows, mean, official_mean, examples)


# ----------------------------------------------------------------------------
# W&B para eval (opcional, --wandb). Proyecto: WANDB_PROJECT (def. "tem-challenge")
# ----------------------------------------------------------------------------
CLASS_LABELS = {0: "fondo", 1: "axón", 2: "mielina", 3: "artefacto"}


def to_classes(lbl):
    """Instancias (0, 1, 2k, 2k+1) -> clase por píxel, para pintar las máscaras."""
    cls = np.zeros(lbl.shape, np.uint8)
    cls[(lbl >= 2) & (lbl % 2 == 0)] = 1   # axón
    cls[(lbl >= 3) & (lbl % 2 == 1)] = 2   # mielina
    cls[lbl == 1] = 3                      # artefacto (solo en GT)
    return cls


def wandb_start(a, n_images):
    """Abre un run de tipo 'eval'. Si wandb falla (no instalado / sin login) devuelve None."""
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "tem-challenge"),
                          entity=os.environ.get("WANDB_ENTITY"),
                          name=a.wandb_name or f"eval-{Path(a.out).name}",
                          group=a.wandb_group, job_type="eval",
                          config={"fiber_model": a.fiber_model, "axon_model": a.axon_model,
                                  "scale": a.scale, "val_frac": a.val_frac,
                                  "n_images": n_images})
    except Exception as e:
        print(f"[wandb] desactivado ({type(e).__name__}: {e})")
        return None


def wandb_example(img, y, pred, caption, step=4):
    """Imagen reducida 1/step con GT y predicción como máscaras (se activan en la UI)."""
    import wandb
    small = (slice(None, None, step), slice(None, None, step))
    return wandb.Image(img[small].astype(np.uint8), caption=caption, masks={
        "prediction": {"mask_data": to_classes(pred[small]), "class_labels": CLASS_LABELS},
        "ground_truth": {"mask_data": to_classes(y[small]), "class_labels": CLASS_LABELS},
    })


def wandb_finish(run, rows, mean, official_mean, examples):
    """Tabla por imagen + ejemplos + medias (resumen del run)."""
    import wandb
    cols = list(dict.fromkeys(k for r in rows for k in r))
    table = wandb.Table(columns=cols, data=[[r.get(k) for k in cols] for r in rows])
    log = {"per_image": table}
    if examples:
        log["examples"] = examples
    run.log(log)
    run.summary.update({f"mean/{k}": v for k, v in mean.items() if k != "image"})
    run.summary.update({f"official/{k}": v for k, v in official_mean.items()})
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats")
    s.add_argument("--x", required=True)
    s.add_argument("--y", required=True)

    t = sub.add_parser("train")
    t.add_argument("--x", required=True)
    t.add_argument("--y", required=True)
    t.add_argument("--target", choices=["fiber", "axon"], required=True)
    t.add_argument("--scale", type=float, default=0.25, help="factor de reducción (usar 'stats')")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--freeze", type=int, default=12,
                   help="nº de bloques ViT iniciales a congelar (0 = entrenar todo)")
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--out", default="models")

    p = sub.add_parser("predict")
    p.add_argument("--x", required=True)
    p.add_argument("--fiber-model", required=True)
    p.add_argument("--axon-model", required=True)
    p.add_argument("--scale", type=float, default=0.25, help="el MISMO que en train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="preds")

    e = sub.add_parser("eval")
    e.add_argument("--x", required=True)
    e.add_argument("--y", required=True)
    e.add_argument("--fiber-model", required=True)
    e.add_argument("--axon-model", default=None, help="opcional: sin él solo se evalúan fibras")
    e.add_argument("--scale", type=float, default=0.25)
    e.add_argument("--val-frac", type=float, default=0.15, help="el MISMO que en train")
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--out", default="eval_val")
    e.add_argument("--wandb", action="store_true", help="registrar el eval en W&B")
    e.add_argument("--wandb-name", default=None, help="nombre del run (def. eval-<out>)")
    e.add_argument("--wandb-group", default=None, help="p. ej. el mismo grupo que el entrenamiento")
    e.add_argument("--wandb-images", type=int, default=4, help="nº de imágenes de ejemplo a subir")

    a = ap.parse_args()
    {"stats": cmd_stats, "train": cmd_train, "predict": cmd_predict, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()