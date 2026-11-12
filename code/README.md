# code/ — deliverable

| file | role |
|---|---|
| `benchmark.py` | the trainer `nnUNetTrainerAccept`: g-ratio, edge and the rejection head |
| **`gpem.py`** | **`nnUNetTrainerGPEM`: the benchmark + the geometric prior fused in the encoder** |
| **`geom.py`** | the frozen geometric maps (radial profile, integral-image box means) |
| `baseline.py` | stock 2D nnU-Net, for the reference numbers |
| `accept.py` | rejection head, rule-derived targets, ring helpers |
| `accept_loss.py` | balanced BCE that supervises the head |
| `losses.py` | g-ratio and edge terms |
| `step.py` | train step that skips non-finite batches (bf16 by default) |
| `tracking.py` | optional W&B logging (no-op if wandb is absent) |
| `tiling.py` | sliding-window forward pass |
| `predict.py` | inference → uint16 instance TIFFs |
| `requirements.txt` | pinned environment |
| `nnU-Net_baseline/README.md` | baseline result and how it compares |
| `checkpoints/` | trained weights + `plans.json` + `dataset.json` |
| `predictions/` | output folder for the test set |

Every file stays under ~100 lines on purpose: this is the reference the rest of
the team extends.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export nnUNet_raw=$PWD/nnunet_raw
export nnUNet_preprocessed=$PWD/nnunet_preprocessed
export nnUNet_results=$PWD/nnunet_results
```

## Train

Copy the modules next to nnU-Net (it finds the trainer by name in its own
folder), then use the usual `nnUNetv2_train`:

```bash
cp benchmark.py gpem.py geom.py accept.py accept_loss.py losses.py step.py tracking.py \
   "$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')"
```

| what | command |
|---|---|
| **the benchmark (GPEM)**, one fold | `EPOCHS=150 nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerGPEM` |
| cross-validation, 5 folds | `nnUNetv2_train <DATASET_ID> 2d <0..4> -tr nnUNetTrainerGPEM` |
| delivery model (all cases) | `EPOCHS=150 nnUNetv2_train <DATASET_ID> 2d all -tr nnUNetTrainerGPEM` |
| without the geometric prior | `EPOCHS=150 nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerAccept` |
| the baseline | `EPOCHS=150 nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerBaseline` |

Environment variables: `EPOCHS` (default 250), `ACCEPT_W` (0.1), `ACCEPT_EVERY`
(2), `AMP` (1), `AMP_DTYPE` (`bf16`; `fp16` overflows on large patches).

Dataset format is nnU-Net's, with the challenge encoding remapped to
`0` background, `1` axon, `2` myelin, `3` artifact → ignore.

## Predict

```bash
python predict.py \
    --checkpoint checkpoints/checkpoint_final.pth \
    --images     /path/to/test/images \
    --plans      checkpoints/plans.json \
    --dataset-json checkpoints/dataset.json \
    --out        predictions/ --tau 0.2
```

`--tau` is the rejection threshold calibrated on validation (0.2 in our runs);
fibers touching the image border are dropped as well. Output: uint16 TIFFs.

To use the 5-fold ensemble, average the probability maps of `fold_0 … fold_4`
before instance assembly (`nnUNetv2_predict -f 0 1 2 3 4` does it natively).

## Checkpoints

Weights are not stored in git (~475 MB each); they are published as a GitHub
**Release** asset. Download `checkpoint_final.pth` into `checkpoints/` before
predicting; `predict.py` reads the nnU-Net checkpoint dict
(`["network_weights"]`).
