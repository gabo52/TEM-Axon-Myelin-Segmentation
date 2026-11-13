# Branch `postproc` — ring closure post-processing + `predict.py` fixes

This branch changes only inference and post-processing. Nothing is retrained: it uses the
same GPEM fold-0 weights as `main`.

## Result (GPEM fold 0, 54 validation cases, τ = 0.2 + border gate)

| pipeline | Dice | HD95 (nm) | g-ratio MAE |
|---|---|---|---|
| `main` | 0.8873 | 553.9 | 0.0150 |
| this branch (ring closure, ring filter moved after it) | **0.8917** | **538.1** | **0.0149** |

Fibres whose myelin was only partly predicted are repaired before the ring test instead of
being dropped by it. Full ablation (15 configurations) in `otros/postproc_ablation.md`.

## Files changed from `main`

### `code/predict.py`

| change | before (`main`) | now |
|---|---|---|
| Trainer | Always built `nnUNetTrainerAccept`, so GPEM weights could not be loaded. | `--trainer gpem\|accept` (default `gpem`). |
| Output resolution | Instances stayed at network resolution (1504×2048). | Vote fractions are upsampled bilinearly to the original 3008×4096 before the argmax (`to_full_res`). |
| Post-processing | `to_instances(sem)` → `drop(inst, score, tau)` inline. | One call to `otros.postproc.postprocess`. |
| `--no-close` | — | Restores the `main` post-processing (no ring closure). |
| `--save-sem` | — | Also writes `<out>/sem/<case>_sem.tif` and `<case>_score.npy`, so post-processing can be re-evaluated without running the network. |

Existing arguments (`--checkpoint`, `--images`, `--plans`, `--dataset-json`, `--out`, `--tau`)
are unchanged.

### `code/tiling.py`

- `forward()` has a new optional argument `return_votes=False`. With `True` it also returns
  the per-class vote counts (`int16 [H, W, 3]`), which `predict.py` needs for the upsampling.
- The default call returns the same `(sem, score)` as before, so existing callers are not
  affected.

## New files

| file | what it does |
|---|---|
| `otros/ring_closure.py` | `close_rings(inst)`: closes open myelin rings and fills the axon they enclose. Casts 180 rays from the axon centroid, finds where the fibre's own myelin starts and ends on each ray, and interpolates the contour on the missing sectors when at least 70 % of the rays are valid. Only background pixels change. Fibres cut by the image border, axons under 50 px and non star-shaped axons are skipped. |
| `otros/postproc.py` | `postprocess(sem, score, tau, close, border)`: the whole semantic → instances step. With `close=True`: watershed without ring filter → `close_rings` → `drop_unwrapped(0.5)` → `accept_filter.drop`. With `close=False` it is exactly the `main` pipeline. |
| `analysis/eval_postproc.py` | Compares `main` vs `close_rings` on the maps saved by `--save-sem`, using `analysis/metrics.py`. Writes a CSV with one row per (case, variant) plus a GLOBAL row per variant. |
| `otros/postproc_ablation.md` | Ablation write-up: steps tested, results table, takeaways, reproduce commands. |
| `otros/postproc_ablation.xlsx` | Per-image numbers behind the ablation. |

Files in `main` that are used but not modified: `otros/instances.py`, `otros/accept_filter.py`,
`analysis/metrics.py`, `code/gpem.py`, `code/benchmark.py`.

## Usage

```bash
# predict (ring closure on by default)
python code/predict.py --checkpoint code/checkpoints/checkpoint_final.pth \
    --plans code/checkpoints/plans.json --dataset-json code/checkpoints/dataset.json \
    --images <images_dir> --out <out_dir> --save-sem

# compare main vs ring closure on the saved semantic maps
python analysis/eval_postproc.py <out_dir>/sem <gt_dir> <out_dir>/postproc_eval
```

## Caveats

- **Split not confirmed.** The fold-0 validation split was reconstructed with
  `KFold(5, shuffle=True, random_state=12345)` over the sorted case names and has not been
  checked against the team's `splits_final.json`.
- **The gain is concentrated in a few images.** 21 images improve HD95, 11 get worse, and the
  median change is 0. Trimmed mean (dropping the 3 best and 3 worst): −10.5 nm. Worst case:
  `sample_0165` (+305 nm).
- **The gain depends on the τ filter.** Without τ, Dice changes by +0.0001 and HD95 by −9.7 nm.
- **`predict.py` on this branch has not been run end to end.** The ablation numbers come from
  local scripts that implement the same logic.
