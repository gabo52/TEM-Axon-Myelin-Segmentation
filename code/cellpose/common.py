"""Utilidades compartidas por train.py, predict.py y evaluate.py (pipeline Cellpose-SAM).

Codificación del challenge (uint16):
    0 = fondo, 1 = artefacto (ignorado), X par >= 2 = axón, X+1 = su mielina.

Cellpose trabaja con instancias "planas", así que hay dos modelos:
    fiber : fibra completa (axón ∪ mielina)  -> id k = X // 2
    axon  : solo el axón                     -> id k = X // 2
y en inferencia mielina = fibra − axón (pareja X / X+1 por construcción).
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.transform import resize

PX_NM = 2.152           # nm/px (calibración oficial)
EXTS = {".tif", ".tiff"}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = ROOT / "data" / "train_samples"
DEFAULT_LABELS = ROOT / "data" / "train_instance_labels"
DEFAULT_CKPT = ROOT / "code" / "checkpoints" / "cellpose"   # salida de train.py


# ---------------------------------------------------------------- lectura
def read_image(path):
    """Imagen TEM -> float32 2D (si viene en RGB por error, se pasa a gris)."""
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(-1)
    return img.astype(np.float32)


def list_images(folder):
    """TIFF de una carpeta (.tif/.tiff, sin importar mayúsculas), en orden alfabético."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def list_pairs(images_dir, labels_dir):
    """Imágenes con máscara del mismo nombre, en orden alfabético."""
    pairs = []
    for p in list_images(images_dir):
        if (Path(labels_dir) / p.name).exists():
            pairs.append((p, Path(labels_dir) / p.name))
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


def to_classes(y):
    """Máscara del challenge -> clase por píxel: 0 fondo, 1 axón, 2 mielina, 3 artefacto."""
    cls = np.zeros(y.shape, np.uint8)
    cls[(y >= 2) & (y % 2 == 0)] = 1
    cls[(y >= 3) & (y % 2 == 1)] = 2
    cls[y == 1] = 3
    return cls


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


def make_kfold(names, k=5, seed=0):
    """k folds deterministas: lista de (train, val), cada imagen en exactamente un val."""
    idx = np.random.default_rng(seed).permutation(len(names))
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = set(folds[i].tolist())
        out.append(([names[j] for j in range(len(names)) if j not in val],
                    [names[j] for j in sorted(val)]))
    return out


def save_split(path, train, val, **meta):
    Path(path).write_text(json.dumps({"train": train, "val": val, **meta}, indent=2))


def load_split(path):
    d = json.loads(Path(path).read_text())
    return d["train"], d["val"]


# ---------------------------------------------------------------- modelos
def model_path(ckpt_dir, target):
    """Ruta de los pesos que guarda train.py: <ckpt>/models/cpsam_tem_<target>."""
    return Path(ckpt_dir) / "models" / f"cpsam_tem_{target}"


def train_scale(ckpt_dir):
    """Escala con la que se entrenó (leída de train_fiber.json), o None si no está."""
    info = Path(ckpt_dir) / "train_fiber.json"
    if not info.exists():
        return None
    return float(json.loads(info.read_text())["args"]["scale"])


def resolve_scale(ckpt_dir, scale=None):
    """La escala de inferencia DEBE ser la del entrenamiento: por defecto se lee de ahí."""
    trained = train_scale(ckpt_dir)
    if scale is None:
        if trained is None:
            raise SystemExit(f"no encuentro {Path(ckpt_dir) / 'train_fiber.json'}: pasa --scale")
        return trained
    if trained is not None and abs(trained - scale) > 1e-9:
        print(f"[aviso] --scale {scale} distinto del de entrenamiento ({trained})")
    return scale


def load_models(fiber_model, axon_model, gpu=True):
    """Modelos en fp32: misma precisión con la que se entrenaron y se eligió el checkpoint
    (Cellpose carga en bf16 por defecto; se midió hasta ~0.01 de Dice de diferencia)."""
    from cellpose import models
    return (models.CellposeModel(gpu=gpu, pretrained_model=str(fiber_model), use_bfloat16=False),
            models.CellposeModel(gpu=gpu, pretrained_model=str(axon_model), use_bfloat16=False))


# ---------------------------------------------------------------- inferencia
def pair_fibers(fiber, axon, min_px=20):
    """Fibras + axones (instancias) -> máscara del challenge: axón 2k, mielina 2k+1.

    Para cada fibra predicha:
      * sin axón dentro            -> se descarta (no se puede medir el g-ratio)
      * con un axón                -> axón (huecos rellenos) y mielina = fibra − axón
      * con varios axones          -> la fibra son varias fusionadas: cada píxel se
                                      asigna al axón más cercano y sale un par por axón
    Pares sin mielina (< min_px) se descartan.
    """
    out = np.zeros(fiber.shape, np.uint16)
    k = 1
    for f_id, box in enumerate(ndi.find_objects(fiber), start=1):
        if box is None:
            continue
        fmask = fiber[box] == f_id
        ax_ids, sizes = np.unique(axon[box][fmask], return_counts=True)
        ax_ids = ax_ids[(ax_ids > 0) & (sizes >= min_px)]
        if len(ax_ids) == 0:
            continue
        ax_lbl = np.where(fmask & np.isin(axon[box], ax_ids), axon[box], 0)

        if len(ax_ids) == 1:
            owner = np.where(fmask, ax_ids[0], 0)
        else:  # dueño de cada píxel = axón más cercano
            _, (iy, ix) = ndi.distance_transform_edt(ax_lbl == 0, return_indices=True)
            owner = np.where(fmask, ax_lbl[iy, ix], 0)

        o = out[box]  # vista: escribir en o escribe en out
        for a_id in ax_ids:
            region = owner == a_id
            ax = ndi.binary_fill_holes(ax_lbl == a_id) & region
            my = region & ~ax
            if ax.sum() < min_px or my.sum() < min_px:
                continue
            o[ax] = 2 * k
            o[my] = 2 * k + 1
            k += 1
    return out


# Parámetros de inferencia elegidos en validación (ver code/cellpose/README.md):
#  * CELLPROB = -1: Cellpose usa 0; las fibras perdidas tienen cellprob mediano -0.7
#  * MULTISCALE: 2ª pasada a escala/2 que recupera fibras grandes no detectadas
CELLPROB = -1.0
MULTISCALE = True


def add_missing(base, extra, max_overlap=0.2):
    """Añade a `base` las instancias de `extra` que apenas solapan (<= 20 %) con lo ya detectado."""
    out = base.copy()
    next_id = int(base.max()) + 1
    for i, box in enumerate(ndi.find_objects(extra), start=1):
        if box is None:
            continue
        m = extra[box] == i
        if (base[box][m] > 0).mean() <= max_overlap:
            o = out[box]  # vista
            o[m & (o == 0)] = next_id
            next_id += 1
    return out


def segment(model, img, scale, batch_size=8, cellprob=CELLPROB, multiscale=MULTISCALE):
    """Instancias de un modelo Cellpose, devueltas a la resolución de `img`."""
    def run(s):
        masks = model.eval(rescale_img(img, s), batch_size=batch_size, cellprob_threshold=cellprob)[0]
        return rescale_lbl(masks, img.shape)

    masks = run(scale)
    if multiscale:
        masks = add_missing(masks, run(scale / 2))
    return masks


def predict_image(img, m_fiber, m_axon, scale, batch_size=8, cellprob=CELLPROB, multiscale=MULTISCALE):
    """Imagen a resolución completa -> máscara del challenge al MISMO tamaño (uint16)."""
    fiber = segment(m_fiber, img, scale, batch_size, cellprob, multiscale)
    axon = segment(m_axon, img, scale, batch_size, cellprob, multiscale)
    return pair_fibers(fiber, axon)


def check_prediction(pred, shape):
    """Comprueba el formato de entrega: tamaño, uint16 y pares X / X+1 válidos."""
    assert pred.shape == tuple(shape), f"tamaño {pred.shape} != imagen {shape}"
    assert pred.dtype == np.uint16, f"dtype {pred.dtype} != uint16"
    ids = np.unique(pred)
    ids = ids[ids > 0]
    assert 1 not in ids, "la predicción no debe contener la etiqueta 1 (artefacto)"
    axons, myelins = set(ids[ids % 2 == 0]), set(ids[ids % 2 == 1])
    assert myelins == {a + 1 for a in axons}, "hay axones sin mielina o mielinas sin axón"


def fiber_stats(pred):
    """Nº de fibras y g-ratio medio (D_axón / D_fibra = sqrt(A_axón / A_fibra))."""
    fib = np.bincount((pred // 2).ravel())[1:]
    ax = np.bincount((pred[(pred >= 2) & (pred % 2 == 0)] // 2), minlength=len(fib) + 1)[1:]
    ok = fib > 0
    g = np.sqrt(ax[ok] / fib[ok])
    return {"n_fibers": int(ok.sum()), "g_ratio_mean": float(g.mean()) if ok.any() else float("nan")}
