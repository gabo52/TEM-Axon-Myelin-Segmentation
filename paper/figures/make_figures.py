"""Figuras del paper (se regeneran con: python paper/figures/make_figures.py [n_imagenes]).

  pipeline.pdf     esquema del método
  qualitative.png  imagen de validación: GT, Cellpose por defecto y nuestra inferencia,
                   con las fibras recuperadas por la inferencia final marcadas en amarillo

Necesita los modelos de code/checkpoints/cellpose (GPU recomendada).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "cellpose"))
from common import (DEFAULT_CKPT, load_models, model_path, predict_image,  # noqa: E402
                    read_image, train_scale)

AXON, MYELIN, ARTIFACT = (0.10, 0.85, 0.20), (0.90, 0.15, 0.85), (0.55, 0.55, 0.55)


# ------------------------------------------------------------------ esquema
def pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color, ec="0.25", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color="0.25"))

    box(0.5, 9, 12, 8, "TEM image\n3008×4096\n2.15 nm/px", "0.92")
    box(17, 15, 13, 8, "downscale\n×0.25", "#dbe9f6")
    box(17, 3, 13, 8, "downscale\n×0.125", "#dbe9f6")
    box(35, 15, 17, 8, "Cellpose-SAM\nfiber model", "#fde2c8")
    box(35, 3, 17, 8, "Cellpose-SAM\naxon model", "#fde2c8")
    box(57, 7.5, 16, 11, "upsample to\nfull resolution;\nmerge scales\n(new if overlap ≤ 20%)", "#e4f2dc")
    box(78, 7.5, 21.5, 11, "pairing\nmyelin = fiber − axon;\nmerged fibers split\nby nearest axon\n→ labels (2k, 2k+1)", "#efe0f5")
    arrow(13.2, 13, 16.6, 19)
    arrow(13.2, 13, 16.6, 7)
    arrow(30.6, 19, 34.6, 19)
    arrow(30.6, 7, 34.6, 7)
    arrow(30.6, 19, 34.6, 8.5)
    arrow(30.6, 7, 34.6, 17.5)
    arrow(52.6, 19, 56.6, 14.5)
    arrow(52.6, 7, 56.6, 11.5)
    arrow(73.6, 13, 77.6, 13)
    ax.text(43.5, 0.3, "both models run at both scales, cellprob threshold −1", ha="center",
            fontsize=6, style="italic", color="0.3")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ cualitativa
def overlay(img, y):
    rgb = np.repeat((img / 255.0)[..., None], 3, axis=-1)
    for mask, color in (((y >= 2) & (y % 2 == 0), AXON), ((y >= 3) & (y % 2 == 1), MYELIN),
                        (y == 1, ARTIFACT)):
        rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array(color)
    return rgb


def matched_gt(gt, pred, thr=0.5):
    """Ids de fibra GT emparejados (IoU >= thr) con alguna fibra predicha."""
    fg = np.where(gt >= 2, gt // 2, 0).astype(np.int64)
    fp = np.where(pred >= 2, pred // 2, 0).astype(np.int64)
    ng, n_p = fg.max() + 1, fp.max() + 1
    inter = np.bincount((fg * n_p + fp).ravel(), minlength=ng * n_p).reshape(ng, n_p)
    ag, ap = np.bincount(fg.ravel(), minlength=ng), np.bincount(fp.ravel(), minlength=n_p)
    iou = inter / (ag[:, None] + ap[None] - inter + 1e-9)
    iou[:, 0] = 0
    return {k for k in range(1, ng) if ag[k] and iou[k].max() >= thr}, fg


def qualitative_figure(path, n_images):
    split = json.loads((DEFAULT_CKPT / "split.json").read_text())
    scale = train_scale(DEFAULT_CKPT)
    mf, ma = load_models(model_path(DEFAULT_CKPT, "fiber"), model_path(DEFAULT_CKPT, "axon"))
    best = None
    for name in split["val"][:n_images]:
        img = read_image(ROOT / "data" / "train_samples" / f"{name}.tif")
        gt = tifffile.imread(ROOT / "data" / "train_instance_labels" / f"{name}.tif")
        base = predict_image(img, mf, ma, scale, cellprob=0.0, multiscale=False)
        ours = predict_image(img, mf, ma, scale)
        m_base, fg = matched_gt(gt, base)
        m_ours, _ = matched_gt(gt, ours)
        recovered = sorted(m_ours - m_base)
        print(f"{name}: recuperadas {len(recovered)}", flush=True)
        if best is None or len(recovered) > len(best[5]):
            best = (name, img, gt, base, ours, recovered, fg)
    name, img, gt, base, ours, recovered, fg = best

    # recorte centrado en las fibras recuperadas
    H, W = gt.shape
    ch, cw = 1500, 2000
    if recovered:
        ys, xs = np.nonzero(np.isin(fg, recovered))
        cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy, cx = H // 2, W // 2
    y0, x0 = int(np.clip(cy - ch // 2, 0, H - ch)), int(np.clip(cx - cw // 2, 0, W - cw))
    crop = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    rec_edge = binary_dilation(find_boundaries(np.isin(fg, recovered)[crop], mode="outer"),
                               iterations=5)

    panels = [("TEM image", np.repeat((img[crop] / 255.0)[..., None], 3, axis=-1)),
              ("Ground truth", overlay(img[crop], gt[crop])),
              ("Cellpose defaults", overlay(img[crop], base[crop])),
              ("Ours (cellprob −1 + multi-scale)", overlay(img[crop], ours[crop]))]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.55))
    for i, (ax, (title, im)) in enumerate(zip(axes, panels)):
        if i >= 2:
            im = im.copy()
            im[rec_edge] = (1.0, 0.85, 0.0)
        ax.imshow(im[::2, ::2], interpolation="nearest")
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03, left=0, right=1, top=0.86, bottom=0)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"figura: {name}, recorte y={y0} x={x0}, {len(recovered)} fibras recuperadas en la imagen")


if __name__ == "__main__":
    pipeline_figure(OUT / "pipeline.pdf")
    qualitative_figure(OUT / "qualitative.png", int(sys.argv[1]) if len(sys.argv) > 1 else 12)
