"""Utilidades compartidas por train.py / predict.py del pipeline Cellpose-SAM.

Codificación del challenge (GT, uint16):
    0 = fondo, 1 = artefacto (ignorado), X par >= 2 = axón, X+1 = su mielina.

Cellpose trabaja con instancias "planas", así que entrenamos dos modelos:
    fiber : fibra completa (axón ∪ mielina)  -> id k = X // 2
    axon  : solo el axón                     -> id k = X // 2
y en inferencia mielina = fibra − axón (pareja X / X+1 por construcción).
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from skimage.transform import resize

PX_NM = 2.152           # nm/px (calibración oficial)
EXTS = {".tif", ".tiff"}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = ROOT / "data" / "train_samples"
DEFAULT_LABELS = ROOT / "data" / "train_instance_labels"


# ---------------------------------------------------------------- lectura
def read_image(path):
    """Imagen TEM -> float32 2D (si viene en RGB por error, se pasa a gris)."""
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(-1)
    return img.astype(np.float32)


def list_pairs(images_dir, labels_dir):
    """Imágenes con máscara del mismo nombre, en orden alfabético."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    pairs = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        if (labels_dir / p.name).exists():
            pairs.append((p, labels_dir / p.name))
        else:
            print(f"[skip] sin máscara: {p.name}")
    return pairs


# ---------------------------------------------------------------- etiquetas
def to_fiber(y):
    """Axón X y mielina X+1 -> misma fibra X//2. Fondo y artefacto -> 0."""
    return np.where(y >= 2, y // 2, 0).astype(np.int32)


def to_axon(y):
    """Solo axones (pares >= 2) -> instancia X//2."""
    return np.where((y >= 2) & (y % 2 == 0), y // 2, 0).astype(np.int32)


TARGETS = {"fiber": to_fiber, "axon": to_axon}


# ---------------------------------------------------------------- escala
def rescale_img(img, s):
    if s == 1:
        return img.astype(np.float32)
    shape = (round(img.shape[0] * s), round(img.shape[1] * s))
    return resize(img, shape, order=1, preserve_range=True,
                  anti_aliasing=s < 1).astype(np.float32)


def rescale_lbl(lbl, shape):
    if tuple(lbl.shape) == tuple(shape):
        return lbl.astype(np.int32)
    return resize(lbl, shape, order=0, preserve_range=True,
                  anti_aliasing=False).astype(np.int32)


# ---------------------------------------------------------------- split
def make_split(names, val_frac, seed=0):
    """Split determinista. Con seed=0 reproduce el de cellpose_tem.py."""
    idx = np.random.default_rng(seed).permutation(len(names))
    n_val = max(1, int(len(names) * val_frac))
    return ([names[i] for i in sorted(idx[n_val:])],
            [names[i] for i in sorted(idx[:n_val])])


def save_split(path, train, val, **meta):
    Path(path).write_text(json.dumps({"train": train, "val": val, **meta}, indent=2))


def load_split(path):
    d = json.loads(Path(path).read_text())
    return d["train"], d["val"]
