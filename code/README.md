# code/ — deliverable

Two trainers, same data and schedule, so any team member can compare them:

| file | role |
|---|---|
| `baseline.py` | **nnU-Net baseline** — stock 2D nnU-Net, default loss |
| `benchmark.py` | **the benchmark** — adds g-ratio, edge and the rejection head |
| `losses.py` | g-ratio and edge terms |
| `accept.py` | rejection head, rule-derived targets, ring helpers |
| `accept_loss.py` | balanced BCE that supervises the head |
| `step.py` | train step that skips non-finite batches (keeps long runs alive) |
| `tracking.py` | optional W&B logging (no-op if wandb is absent) |
| `tiling.py` | sliding-window forward pass |
| `predict.py` | inference → uint16 instance TIFFs |
| `requirements.txt` | pinned environment |
| `nnU-Net_baseline/README.md` | baseline result and how it compares |
| `checkpoints/` | trained weights (not in git) |
| `predictions/` | output folder for the test set |

Every file stays under ~90 lines on purpose: this is the reference the rest of
the team will extend.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export nnUNet_raw=$PWD/nnunet_raw
export nnUNet_preprocessed=$PWD/nnunet_preprocessed
export nnUNet_results=$PWD/nnunet_results
```

## Train

Both trainers are nnU-Net trainers, so they are used through the nnU-Net CLI.
Copy the modules together (nnU-Net finds the class by name in its own folder):

```bash
cp baseline.py benchmark.py losses.py accept.py accept_loss.py step.py tracking.py \
   "$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')"

EPOCHS=250 nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerBaseline   # baseline
EPOCHS=250 nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerAccept     # benchmark
```

Use `all` instead of `0` for the final model. Dataset format is nnU-Net's with
the challenge encoding: `0` background, `1` artifact → ignore, `X` axon,
`X+1` paired myelin.

## Predict

```bash
python predict.py \
    --checkpoint checkpoints/checkpoint_final.pth \
    --images     /path/to/test/images \
    --plans      /path/to/nnUNetPlans.json \
    --dataset-json /path/to/dataset.json \
    --out        predictions/ --tau 0.2
```

`--tau` is the rejection threshold calibrated on validation (0.2 in our runs);
fibers touching the image border are dropped as well. Output: uint16 TIFFs.

## Checkpoints

Weights are not stored in git (~470 MB each). Place
`checkpoints/checkpoint_final.pth` before predicting; `predict.py` reads the
nnU-Net checkpoint dict (`["network_weights"]`).
