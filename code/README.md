# code/ — SIPAIM 2026 TEM: axon + myelin instance segmentation

We propose **GPEM15**: a 2D nnU-Net whose encoder fuses a frozen 12-channel
*geometric prior* (radial material profile at 4 scales + gradient magnitude +
local orientation) through a learned gate, plus a per-fiber rejection head.

Fold-0 validation: **HD95 570.5 nm, Dice 0.8834** (global, unmatched).

## Structure

```
code/
├── train.py                training entry point
├── predict.py              inference entry point → uint16 instance TIFFs
├── gpem12.py               nnUNetTrainerGPEM15 — THE METHOD (12-channel prior)
├── gpem7.py                nnUNetTrainerGPEM — ablation (7-channel prior)
├── accept_trainer.py       nnUNetTrainerAccept — the method's base
├── maps7.py / maps12.py    frozen geometric maps (7 / 12 channels)
├── accept.py               rejection head + rule-derived targets
├── accept_loss.py          head supervision
├── losses.py               g-ratio and edge losses
├── step.py                 train step that skips non-finite batches (bf16)
├── instances.py            instance assembly (watershed + ring filter)
├── accept_filter.py        score filter + border gate
├── checkpoints/            weights + plans.json + dataset.json
└── requirements.txt        environment
```

`gpem12.py` registers the classifier `nnUNetTrainerGPEM15` (the `--trainer gpem15`
key). `gpem7.py` is its 7-channel ablation; `accept_trainer.py` is their shared
base (nnU-Net + g-ratio + edge + rejection, no prior).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Predict

```bash
python code/predict.py \
    --model-folder code/checkpoints \
    --images <test dir> \
    --out predictions \
    --folds 0 1 2 3 4          # 5 folds = final ensemble
```

`--folds 0` runs a single model (demo); `--folds 0 1 2 3 4` averages the softmax
of the five folds inside nnU-Net.

Output: one `uint16` TIFF per image at native resolution (3008×4096), with
`0` = background and each fiber a consecutive pair `(X, X+1)`: **axon = `X`
(even ≥ 2), myelin = `X+1` (odd ≥ 3)** — the challenge encoding (e.g. `10/11`,
`12/13`, …).

## Train

```bash
export nnUNet_raw=<raw> nnUNet_preprocessed=<prep> nnUNet_results=<results>
python code/train.py --dataset 505 --fold 0 --trainer gpem15
```

`train.py` copies the method's modules into nnU-Net's package and runs
`run_training_entry`. The resulting weights live in `code/checkpoints/`
(e.g. `fold_0/checkpoint_final.pth`).

## Weights

`checkpoints/` holds `plans.json`, `dataset.json` and one `fold_X` subfolder per
fold, each with `checkpoint_final.pth`. They are updated when the 5-fold
cross-validation finishes, for the final ensemble.