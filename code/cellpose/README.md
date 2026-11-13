# Cellpose-SAM — paired axon + myelin

Two fine-tuned Cellpose-SAM (`cpsam`) models:

| model | what it segments | training label |
|---|---|---|
| `fiber` | whole fiber (axon ∪ myelin) | `X // 2` for `X >= 2` |
| `axon` | axon only | `X // 2` for even `X >= 2` |

At inference, **myelin = fiber − axon**, so every axon `2k` is paired with its
myelin `2k+1` by construction. If the fiber model merges two fibers (two axons
inside), each pixel is assigned to the nearest axon and two pairs are produced.
Images are downscaled ×0.25 (fibers fall within Cellpose's size range) and the
masks are returned at 3008×4096.

## Files

| file | what it does |
|---|---|
| `train.py` | trains `fiber`, `axon` or both (same train/val split) |
| `kfold.py` | 5-fold cross-validation + final model on all images |
| `early_stop.py` | early stopping on validation Dice (Cellpose has no callbacks) |
| `predict.py` | test → uint16 TIFF in the submission format |
| `evaluate.py` | validation with the official metrics (`analysis/metrics.py`) |
| `common.py` | data, scaling, fiber/axon pairing, format check |
| `wandb_log.py` | optional W&B logging (`--wandb`) |

## Installation

```bash
pip install -r code/cellpose/requirements.txt
```

## Usage (from the repository root)

```bash
# 1. train (validation = 15 % of the images, seed 0)
python code/train.py --model cellpose --target both --epochs 200 --wandb

# 2. evaluate on validation (Dice, HD95 in nm, g-ratio error)
python code/cellpose/evaluate.py --wandb

# 3. final model on ALL images (goes to code/checkpoints/cellpose_all)
python code/train.py --model cellpose --target both --epochs 200 --all

# 4. test predictions
python code/cellpose/predict.py --images /path/test --out predictions \
       --ckpt code/checkpoints/cellpose_all
```

`predict.py` and `evaluate.py` read the training scale from
`<ckpt>/train_fiber.json`, so it does not need to be repeated. W&B uses the
project `WANDB_PROJECT` (default `tem-challenge`); without `--wandb` nothing is
sent.

> Step 3 above trains the final model for a fixed number of epochs. The k-fold
> pipeline below (`kfold.py`) builds `cellpose_all` differently: it reproduces
> the trajectory of a fold and stops at the median best epoch. That is the
> recommended way.

## Inference: threshold and multi-scale

The dominant error on validation was **whole fibers never detected** (156 of
1094, mostly large ones): filling them in would bring HD95 from 669 to 24 nm,
whereas removing false positives or perfecting boundaries barely moves it. The
fiber model "almost" sees them (median cellprob −0.7 vs 3.4 for detected
fibers), so by default inference:

* uses `cellprob_threshold = -1` (Cellpose uses 0) — `--cellprob`
* runs a second pass at scale/2 and adds the instances that overlap by at most
  20 % with what was already detected — `--no-multiscale` to disable it

Validation (39 images, official metrics, same weights):

| inference | mean Dice | mean HD95 | g-ratio MAE |
|---|---|---|---|
| Cellpose defaults | 0.828 | 669 nm | 0.0167 |
| multi-scale | 0.859 | 528 nm | 0.0163 |
| **cellprob −1 + multi-scale (default)** | **0.865** | **535 nm** | **0.0151** |

The parameters were chosen among 5 variants by looking at these same 39 images,
so the figure may be somewhat optimistic. Inference takes ~2× longer (2 passes).

> These numbers were measured with Cellpose's default bf16 inference. Models are
> now loaded in fp32 (`common.load_models`), the precision in which they were
> trained and in which early stopping selects the checkpoint (up to ~0.01 Dice
> difference was measured between the two). The k-fold results are computed in
> fp32.

## Cross-validation (k-fold) and final model

```bash
nohup python code/cellpose/kfold.py --wandb > code/checkpoints/kfold.log 2>&1 &
tail -f code/checkpoints/kfold.log
```

`kfold.py` is resumable (if it stops, relaunch the same command) and does:

1. 5 folds × {fiber, axon}, 200 epochs max. → `code/checkpoints/cellpose_kfold/fold<i>/`
2. final model on the 266 images → `code/checkpoints/cellpose_all/`. It has no
   validation, so it reproduces the trajectory of a fold: same 200-epoch
   schedule (Cellpose's learning-rate decay depends on it), stop at the median
   best epoch of the folds and as many images per epoch as a fold's training set
3. evaluation of each fold on its validation images with 3 inference variants
   (Cellpose defaults, cellprob −1 + multi-scale, cellprob −2 + multi-scale)
4. `cellpose_kfold/summary.json`: official out-of-fold metrics over the 266
   images (mean ± std across folds)

**Early stopping** (`early_stop.py`, only with validation): every 10 epochs from
epoch 40 the validation images are segmented with the current network and the
Dice of the target (fiber or axon, artifacts excluded, mean per image) is
measured. The checkpoint with the best Dice is kept (the last epoch also
competes) and training stops if it does not improve for `--patience` = 60
epochs. A short patience would stop too early: validation oscillates and the
best points come in the final learning-rate decay phase (epochs 150–199).

## Execution pipeline (step by step)

All commands run from the repository root. Expected data:
`data/train_samples/*.tif` (images) and `data/train_instance_labels/*.tif`
(masks with the same file names).

### 0. Environment

```bash
conda activate segmentation_env      # or: pip install -r code/cellpose/requirements.txt
```

Tested with Python 3.10, cellpose 4.2.1.1, torch 2.14.1 (CUDA 13), one RTX 3080
(12 GB). The base weights (`cpsam`) are downloaded by Cellpose to
`~/.cellpose/models/` on first use.

### 1. Cross-validation + final model (unattended, ~19 h)

```bash
setsid nohup python code/cellpose/kfold.py --wandb \
      > code/checkpoints/kfold.log 2>&1 < /dev/null &
```

`setsid nohup` keeps it running after the terminal (or the editor) is closed.

| stage | output | approx. time (RTX 3080) |
|---|---|---|
| 10 fold models (5 folds × fiber/axon) | `cellpose_kfold/fold<i>/models/`, `train_<target>.json` | ~90 min each (incl. ~16 Dice checks of ~70 s) |
| final fiber + axon models | `cellpose_all/models/` | ≤ 70 min each |
| out-of-fold evaluation (3 variants × 266 images) | `cellpose_kfold/eval/<variant>/fold<i>/metrics.csv` + predictions | ~2–3 h |
| summary | `cellpose_kfold/summary.json` | seconds |

Early stopping may shorten the fold trainings.

**Monitoring**

```bash
tail -f code/checkpoints/kfold.log | grep -vE "it/s\]"   # progress
grep "early stop" code/checkpoints/kfold.log              # validation Dice at each check
grep -E "OK|FALLO" code/checkpoints/kfold.log             # finished / failed steps
nvidia-smi                                                # GPU in use
```

With `--wandb` everything appears in W&B under the group `cellpose-kfold`
(training loss, validation Dice, per-image evaluation tables, example images).

**If it stops** (reboot, error): relaunch the same command. Every step is
skipped if it is already done *with the same configuration*; a step that fails
is recorded and the rest continue. Use `>>` instead of `>` to keep the previous
log. If a marker belongs to another configuration (e.g. a test run with fewer
epochs in the same folder), `kfold.py` aborts with a message instead of reusing
it: delete that folder or use `--root`.

### 2. Read the out-of-fold results and pick the inference variant

```bash
python -m json.tool code/checkpoints/cellpose_kfold/summary.json
```

For each variant: `dice_mean`, `hd95_mean` (nm) and `g_ratio_mae` over the 266
images, their std across folds and `n_hd95_nan` (images where HD95 is
undefined, ignored in the mean — check it is 0). Pick the variant with the best
balance of the three metrics. The flags of each variant:

| variant in `summary.json` | `predict.py` flags |
|---|---|
| `cellpose_default` | `--cellprob 0 --no-multiscale` |
| `cp-1_multiscale` | *(none, it is the default)* |
| `cp-2_multiscale` | `--cellprob -2` |

Known biases, to be reported: each fold's validation images are used both to
select the checkpoint (early stopping) and to report the metrics, so the CV is
slightly optimistic (estimated ≤ 0.005 Dice); fold 0 contains the 39 images
on which cellprob −1 and multi-scale were originally chosen; `sample_0118` and
`sample_0178` partially overlap (same tissue) and fall in different folds.

### 3. Test predictions with the final model

```bash
python code/cellpose/predict.py --images /path/to/test --out predictions \
       --ckpt code/checkpoints/cellpose_all [variant flags] [--wandb]
```

One uint16 TIFF per test image, **same file name**, 3008×4096, axon `2k` /
myelin `2k+1`. Before saving, `predict.py` checks size, dtype and that every
axon has its myelin (and vice versa); it stops with an error otherwise.

### 4. Submission

```
TeamOrLastName_Institution_SIPAIM2026.zip
├── method_description.pdf      ← paper/main.pdf (compiled, authors filled in)
├── code/                       ← README.md, requirements, train.py, predict.py,
│                                 checkpoints/ (or a download link: 2 × 1.2 GB)
└── predictions/                ← output of step 3
```

### Output layout

```
code/checkpoints/
├── cellpose/                  single-split models (227 train / 39 val)
├── cellpose_kfold/
│   ├── fold<i>/               models/cpsam_tem_{fiber,axon}, split.json, train_{fiber,axon}.json
│   ├── eval/<variant>/fold<i>/  predictions (.tif) + metrics.csv
│   └── summary.json
├── cellpose_all/              final models (all 266 images) + train_{fiber,axon}.json
└── kfold.log
```

`train_<target>.json` records the arguments, versions, best epoch and its Dice,
whether early stopping fired, and the loss/Dice history. Model weights
(`models/`) and `.tif` predictions are not in git.

## Weights

`<ckpt>/models/cpsam_tem_fiber` and `cpsam_tem_axon` (~1.2 GB each, not in
git).
