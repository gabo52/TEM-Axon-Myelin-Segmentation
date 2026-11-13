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

Cellpose-SAM pipeline (second method, see [`code/cellpose/README.md`](code/cellpose/README.md)):

```
code/
├── train.py                  ← single training entry point: --model cellpose | nnunet
├── cellpose/
│   ├── train.py              ← fine-tunes the fiber and axon Cellpose-SAM models
│   ├── kfold.py              ← 5-fold CV + final model on all 266 images (resumable)
│   ├── early_stop.py         ← early stopping on validation Dice
│   ├── evaluate.py           ← validation with the official metrics
│   ├── predict.py            ← test images → uint16 paired TIFFs
│   ├── common.py             ← data, scaling, fiber/axon pairing, format check
│   ├── wandb_log.py          ← optional W&B logging (--wandb)
│   └── requirements.txt
└── checkpoints/
    ├── cellpose/             ← single-split models (227 train / 39 val)
    ├── cellpose_kfold/       ← fold<i>/ models, eval/<variant>/fold<i>/, summary.json
    └── cellpose_all/         ← final models trained on all 266 images
paper/                        ← SIPAIM method description (LaTeX) + figure scripts
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

## Execution pipeline — Cellpose-SAM

All commands run from the repository root, in the environment of
[`code/cellpose/requirements.txt`](code/cellpose/requirements.txt)
(Python 3.10, cellpose 4.2.1.1, torch 2.14). Data is expected in
`data/train_samples/` (images) and `data/train_instance_labels/` (masks, same
file names).

```bash
# 0. environment
pip install -r code/cellpose/requirements.txt

# 1. 5-fold cross-validation + final model (≈ 19 h on one RTX 3080; resumable)
setsid nohup python code/cellpose/kfold.py --wandb \
      > code/checkpoints/kfold.log 2>&1 < /dev/null &
tail -f code/checkpoints/kfold.log | grep -vE "it/s\]"

# 2. out-of-fold results (official metrics over the 266 images) → pick the inference variant
cat code/checkpoints/cellpose_kfold/summary.json

# 3. test predictions with the final model (uint16 TIFF, same names, 3008×4096)
python code/cellpose/predict.py --images /path/to/test --out predictions \
       --ckpt code/checkpoints/cellpose_all
```

Step-by-step details (what each stage writes, how to resume, how to choose the
inference variant, submission layout) are in
[`code/cellpose/README.md`](code/cellpose/README.md#execution-pipeline-step-by-step).

Submission archive expected by the challenge:

```
TeamOrLastName_Institution_SIPAIM2026.zip
├── method_description.pdf      ← paper/main.pdf (compiled, authors filled in)
├── code/                       ← README.md, requirements, train.py, predict.py, checkpoints/ (or link)
└── predictions/                ← one uint16 .tif per test image, same file name
```

## Metrics

Implemented by us, ask to commite (`analysis/metrics.py`):

* **Dice** per compartment over the global masks.
* **HD95** — `max(P95 d(∂P→∂G), P95 d(∂G→∂P)) × 2.152 nm/px`, per compartment,
  pooled boundary pixels.
* **g-ratio error** — per paired fiber, `g = D_axon / D_fiber`, `D = 2√(A/π)`.
* Artifact regions (label `1`) are excluded from every metric.
