"""Comparación visual Cellpose vs nnU-Net en el test: 4 paneles por imagen.

python analysis/compare_test.py data/test_samples predictions_cellpose <pred_nnunet> comparison_test
"""
import html
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

IMG, CP, NN, OUT = map(Path, sys.argv[1:5])
F = 4                                    # factor de reducción (3008x4096 -> 752x1024)
AXON, MYELIN = (46, 155, 255), (255, 176, 0)
CP_ONLY, NN_ONLY, SWAP = (230, 0, 126), (60, 210, 90), (255, 255, 255)


def classes(inst):
    """0 fondo, 1 axón, 2 mielina."""
    c = np.zeros(inst.shape, np.uint8)
    c[(inst >= 2) & (inst % 2 == 0)] = 1
    c[(inst >= 3) & (inst % 2 == 1)] = 2
    return c


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def overlay(gray, layers, alpha=0.5):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for mask, color in layers:
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, np.float32)
    return rgb


def save(rgb, path):
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, quality=82)


def process(name):
    img = tifffile.imread(IMG / name).astype(np.float32)
    h, w = img.shape
    gray = img[:h // F * F, :w // F * F].reshape(h // F, F, w // F, F).mean((1, 3))
    cp_full, nn_full = tifffile.imread(CP / name), tifffile.imread(NN / name)
    ccp, cnn = classes(cp_full), classes(nn_full)

    stats = {
        "name": name,
        "fibers_cp": int(len(np.unique(cp_full[cp_full >= 2] // 2))),
        "fibers_nn": int(len(np.unique(nn_full[nn_full >= 2] // 2))),
        "dice_axon": dice(ccp == 1, cnn == 1),
        "dice_myelin": dice(ccp == 2, cnn == 2),
        "disagree_pct": float(100 * ((ccp != cnn) & ((ccp > 0) | (cnn > 0))).mean()),
    }

    cp, nn = cp_full[::F, ::F], nn_full[::F, ::F]
    a, b = ccp[::F, ::F], cnn[::F, ::F]
    stem = Path(name).stem
    for tag, inst, c in (("cp", cp, a), ("nn", nn, b)):
        rgb = overlay(gray, [(c == 1, AXON), (c == 2, MYELIN)])
        rgb[find_boundaries(inst, mode="inner")] = (0, 0, 0)
        save(rgb, OUT / "img" / f"{stem}_{tag}.jpg")
    same = (a == b) & (a > 0)
    save(overlay(gray, [(same & (a == 1), AXON), (same & (a == 2), MYELIN)]),
         OUT / "img" / f"{stem}_common.jpg")
    save(overlay(gray, [((a > 0) & (b == 0), CP_ONLY), ((b > 0) & (a == 0), NN_ONLY),
                        ((a > 0) & (b > 0) & (a != b), SWAP)], alpha=0.65),
         OUT / "img" / f"{stem}_diff.jpg")
    return stats


PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cellpose vs nnU-Net</title>
<style>
:root {{ --bg:#ffffff; --fg:#1d1d1f; --muted:#6b6b70; --card:#f4f4f6; --line:#dcdce0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#141416; --fg:#ececef; --muted:#9a9aa2; --card:#1e1e22; --line:#33333a; }} }}
:root[data-theme="dark"] {{ --bg:#141416; --fg:#ececef; --muted:#9a9aa2; --card:#1e1e22; --line:#33333a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:0 16px 48px; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
header {{ max-width:1600px; margin:0 auto; padding:24px 0 8px; }}
h1 {{ font-size:22px; margin:0 0 6px; }}
p {{ margin:4px 0; color:var(--muted); }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:12px 0; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.sw {{ width:14px; height:14px; border-radius:3px; display:inline-block; border:1px solid var(--line); }}
.cols, .row {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; }}
.cols {{ position:sticky; top:0; z-index:2; background:var(--bg); padding:10px 0;
  border-bottom:1px solid var(--line); font-weight:600; max-width:1600px; margin:0 auto; }}
section {{ max-width:1600px; margin:18px auto 0; background:var(--card); border-radius:10px; padding:10px; }}
section h2 {{ font-size:15px; margin:0 0 8px; display:flex; flex-wrap:wrap; gap:4px 16px; align-items:baseline; }}
section h2 small {{ font-weight:400; color:var(--muted); }}
.row img {{ width:100%; height:auto; display:block; border-radius:6px; cursor:zoom-in; }}
table {{ border-collapse:collapse; margin:10px 0; font-variant-numeric:tabular-nums; }}
td, th {{ padding:3px 10px; border-bottom:1px solid var(--line); text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }}
a {{ color:inherit; }}
@media (max-width:700px) {{ .cols, .row {{ grid-template-columns:repeat(2, 1fr); }} }}
</style></head><body>
<header>
<h1>Test SIPAIM 2026 — Cellpose vs nnU-Net</h1>
<p>{n} imágenes de test. Cellpose: <code>code/checkpoints/cellpose_all</code> (cellprob −1 + multiescala).
nnU-Net: <code>predictions/</code> de la rama <code>danitza2</code> (GPEM15).</p>
<p>Concordancia media entre ambos: Dice axón {da:.3f}, Dice mielina {dm:.3f}.
Fibras totales: Cellpose {fc}, nnU-Net {fn}. Es acuerdo entre modelos, no precisión (el test no tiene ground truth).</p>
<div class="legend">
<span><i class="sw" style="background:rgb{axon}"></i>axón</span>
<span><i class="sw" style="background:rgb{myelin}"></i>mielina</span>
<span><i class="sw" style="background:#000"></i>borde de instancia</span>
<span><i class="sw" style="background:rgb{cp_only}"></i>solo Cellpose</span>
<span><i class="sw" style="background:rgb{nn_only}"></i>solo nnU-Net</span>
<span><i class="sw" style="background:rgb{swap}"></i>clase distinta (axón ↔ mielina)</span>
</div>
<details><summary>Imágenes ordenadas por desacuerdo (mayor primero)</summary>
<table><tr><th>imagen</th><th>% píxeles en desacuerdo</th><th>Dice axón</th><th>Dice mielina</th><th>fibras CP</th><th>fibras nnU</th></tr>
{table}</table></details>
</header>
<div class="cols"><div>Cellpose</div><div>nnU-Net</div><div>Zona común</div><div>Zona no común</div></div>
{rows}
</body></html>
"""

ROW = """<section id="{stem}"><h2>{name}
<small>Dice axón {dice_axon:.3f} · Dice mielina {dice_myelin:.3f} · desacuerdo {disagree_pct:.1f} % ·
fibras Cellpose {fibers_cp} / nnU-Net {fibers_nn}</small></h2>
<div class="row">
<a href="img/{stem}_cp.jpg" target="_blank"><img loading="lazy" src="img/{stem}_cp.jpg" alt="{name} Cellpose"></a>
<a href="img/{stem}_nn.jpg" target="_blank"><img loading="lazy" src="img/{stem}_nn.jpg" alt="{name} nnU-Net"></a>
<a href="img/{stem}_common.jpg" target="_blank"><img loading="lazy" src="img/{stem}_common.jpg" alt="{name} zona común"></a>
<a href="img/{stem}_diff.jpg" target="_blank"><img loading="lazy" src="img/{stem}_diff.jpg" alt="{name} zona no común"></a>
</div></section>"""


if __name__ == "__main__":
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in IMG.glob("*.tif"))
    with ProcessPoolExecutor(6) as ex:
        stats = list(ex.map(process, names))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    rows = "\n".join(ROW.format(stem=Path(s["name"]).stem, **s) for s in stats)
    table = "\n".join(
        f'<tr><td><a href="#{Path(s["name"]).stem}">{html.escape(s["name"])}</a></td>'
        f'<td>{s["disagree_pct"]:.1f}</td><td>{s["dice_axon"]:.3f}</td>'
        f'<td>{s["dice_myelin"]:.3f}</td><td>{s["fibers_cp"]}</td><td>{s["fibers_nn"]}</td></tr>'
        for s in sorted(stats, key=lambda s: -s["disagree_pct"]))
    (OUT / "index.html").write_text(PAGE.format(
        n=len(stats), rows=rows, table=table,
        da=np.mean([s["dice_axon"] for s in stats]), dm=np.mean([s["dice_myelin"] for s in stats]),
        fc=sum(s["fibers_cp"] for s in stats), fn=sum(s["fibers_nn"] for s in stats),
        axon=AXON, myelin=MYELIN, cp_only=CP_ONLY, nn_only=NN_ONLY, swap=SWAP))
    print(f"{len(stats)} imágenes -> {OUT / 'index.html'}")
