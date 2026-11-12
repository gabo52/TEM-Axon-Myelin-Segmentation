Our solution for the *TEM Axon & Myelin Segmentation Challenge* (SIPAIM 2026):

> **Method:** a 2D nnU-Net whose encoder fuses a **frozen geometric prior**
> (the radial material profile measured on the image) through a **learned gate**
> (GPEM), and whose decoder feeds a **learned fiber-rejection head**. A single
> model and a single forward pass: no external critic, no hand-tuned rule at
> inference.

## Results (validation fold 0, 54 held-out cases, 150 epochs)

| model | HD95 raw | HD95 + border gate | Dice |
|---|---|---|---|
| nnU-Net baseline (20 ep) | 750.0 nm | — | 0.862 |
| **GPEM** | 744.7 nm | **572.5 nm** | **0.883** |
| GPEM + learned rejection (τ = 0.2) | — | **566.5 nm** | **0.885** |

The **border gate** removes fibers cut by the image edge, whose diameter and
g-ratio are not measurable; τ is calibrated on validation. Matched per-fiber
HD95 is 31.8 nm and the boundary error on correctly detected fibers is 42.9 nm.

## Repository layout

```
.
├── README.md                 ← this file
├── benchmark_description.md  ← full method: architecture, losses, pipeline
├── code/                     ← deliverable
│   ├── benchmark.py          ← the trainer (nnUNetTrainerAccept)
│   ├── gpem.py               ← + geometric prior fused in the encoder
│   ├── geom.py               ← the frozen geometric maps
│   ├── predict.py            ← inference → uint16 instance TIFFs
│   ├── requirements.txt
│   ├── nnU-Net_baseline/     ← how to reproduce the baseline
│   ├── checkpoints/          ← trained weights
│   └── predictions/          ← predictions
├── otros/                    ← some scripts of common use
│   ├── instances.py          ← paired instances (watershed)
│   ├── accept_filter.py      ← score-based fiber filtering
│   ├── visualizar.ipynb      ← view the data
│   ├── dataset_summary.md    ← what the data looks like
│   ├── experimentos.md       ← every experiment we ran
│   └── ablations.md          ← what worked and what did not
└── analysis/
    └── metrics.py            ← Dice / HD95 / g-ratio
```

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# 2. train (GPEM = the benchmark)
cp code/*.py "$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')"
nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerGPEM

# 3. predict
python code/predict.py --checkpoint code/checkpoints/checkpoint_final.pth \
    --images <test_dir> --plans code/checkpoints/plans.json \
    --dataset-json code/checkpoints/dataset.json --out code/predictions
```

Full instructions in [`code/README.md`](code/README.md).

## Metrics

Implemented exactly as the challenge specifies ([`analysis/metrics.py`](analysis/metrics.py)):

* **Dice** per compartment over the global masks.
* **HD95** — `max(P95 d(∂P→∂G), P95 d(∂G→∂P)) × 2.152 nm/px`, per compartment,
  pooled boundary pixels.
* **g-ratio error** — per paired fiber, `g = D_axon / D_fiber`, `D = 2√(A/π)`.
* Artifact regions (label `1`) are excluded from every metric.
