Our solution for the *TEM Axon & Myelin Segmentation Challenge* (SIPAIM 2026):

> **Method:** a 2D nnU-Net whose decoder feeds a small
> **learned fiber-rejection head** — trained with rule-derived targets — so the
> model decides by itself which predicted fibers to keep, instead of relying on
> an external critic or a hand-tuned rule.

## Results (validation fold 0)

| model | HD95 global | Dice |
|---|---|---|
| nnU-Net baseline (+ g-ratio loss) (our first proposal) | 750.0 nm | 0.862 |
| **benchmark: + learned rejection** | **565.7 nm** | **0.890** |
| Ours |  nm | — |

Rejection quality: **AUC 0.876** (kept vs discarded fibers), equal to an
external appearance critic — but with a **single model** and a single forward
pass. Matched per-fiber HD95 is 31.8 nm; boundary error on detected fibers is
42.9 nm.

## Repository layout

```
.
├── README.md                 ← this file
├── benchmark_description.md  ← full method: architecture, losses, pipeline
├── code/                     ← deliverable
│   ├── benchmark.py          ← the trained architecture (nnUNetTrainerAccept)
│   ├── predict.py            ← inference → uint16 instance TIFFs
│   ├── requirements.txt
│   ├── nnU-Net baseline
│   ├── checkpoints/          ← trained weights
│   └── predictions/          ← predictions
├── otros/                    ← # algunos scripts de uso comun
│   ├── instances.py          ← paired instances (watershed)
│   └── accept_filter.py      ← score-based fiber filtering
│   └── visualizar.ipynb      ← view data
├── analysis/
│   └── metrics.py            ← #revisar
```

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# 2. evaluate 
python 
```

Training and inference commands are in [`code/README.md`](code/README.md).

## Metrics

Implemented by us, ask to commite (`analysis/metrics.py`):

* **Dice** per compartment over the global masks.
* **HD95** — `max(P95 d(∂P→∂G), P95 d(∂G→∂P)) × 2.152 nm/px`, per compartment,
  pooled boundary pixels.
* **g-ratio error** — per paired fiber, `g = D_axon / D_fiber`, `D = 2√(A/π)`.
* Artifact regions (label `1`) are excluded from every metric.
