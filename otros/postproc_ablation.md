# Post-processing ablation — ring closure

Model: GPEM fold 0 (release `v0.1-gpem-fold0`). Fold-0 validation split: 54 cases, reconstructed
with `KFold(5, shuffle=True, random_state=12345)` over the sorted case names. This split still
needs to be confirmed against `splits_final.json`.

Inference:
- Sliding window with votes, as in `code/tiling.py`.
- Vote fractions are upsampled bilinearly to the full 3008×4096 resolution before the argmax
  (`predict.py` now writes full-resolution instances).

Evaluation: `analysis/metrics.py` against the full-resolution `train_instance_labels`, with the
same aggregation as in the repo.

All variants run on the same semantic maps and use the same accept filter (τ = 0.2 plus the
border gate).

## Steps tested

| code | step |
|---|---|
| — (`repo`) | `to_instances` (watershed + `drop_unwrapped` 0.5) → `accept_filter.drop` (main pipeline) |
| C | `otros/ring_closure.py`: radial closure of open myelin rings plus filling of the enclosed axon |
| R | `drop_unwrapped` moved **after** the other steps (watershed runs with `min_ring = 0`) |
| H | fill background holes enclosed by a single fibre |
| T | trim myelin thicker than 1.6 × the fibre's median thickness |
| S | trim protrusions on fibres with solidity < 0.85 |
| M | minimum axon area (6000 px at full resolution): small axon blobs become myelin or background |

## Results (54 validation images, global)

| config | Dice ax | Dice my | **Dice** | HD95 ax | HD95 my | **HD95 (nm)** | **g MAE** |
|---|---|---|---|---|---|---|---|
| repo (main) | 0.8981 | 0.8764 | 0.8873 | 580.8 | 527.0 | 553.9 | 0.0150 |
| + C | 0.8998 | 0.8772 | 0.8885 | 558.9 | 520.6 | 539.7 | 0.0150 |
| + H | 0.8982 | 0.8766 | 0.8874 | 578.6 | 527.4 | 553.0 | 0.0150 |
| + T | 0.8981 | 0.8758 | 0.8869 | 580.8 | 524.5 | 552.6 | 0.0152 |
| + S | 0.8981 | 0.8765 | 0.8873 | 580.8 | 526.7 | 553.7 | 0.0150 |
| + R | 0.8981 | 0.8764 | 0.8873 | 580.8 | 527.0 | 553.9 | 0.0150 |
| + M | 0.8981 | 0.8759 | 0.8870 | 580.9 | 531.6 | 556.2 | 0.0152 |
| **+ C R** | **0.9045** | **0.8790** | **0.8917** | 558.9 | **517.2** | **538.1** | **0.0149** |
| + C H | 0.8997 | 0.8773 | 0.8885 | 558.6 | 520.8 | 539.7 | 0.0150 |
| + C T | 0.8998 | 0.8769 | 0.8883 | 558.9 | 518.0 | 538.5 | 0.0150 |
| + C H T | 0.8997 | 0.8773 | 0.8885 | 558.6 | 518.0 | 538.3 | 0.0149 |
| + C H R | 0.9041 | 0.8772 | 0.8907 | 577.3 | 533.9 | 555.6 | 0.0149 |
| + C H T S R | 0.9041 | 0.8791 | 0.8916 | 577.3 | 531.8 | 554.6 | 0.0149 |
| + M C T S R | 0.9044 | 0.8788 | 0.8916 | 556.6 | 522.1 | 539.3 | 0.0150 |
| + M C H T S R (all) | 0.9032 | 0.8769 | 0.8901 | 570.1 | 539.5 | 554.8 | 0.0149 |

Reference without filters (watershed only): Dice 0.8636, HD95 801.8 nm.

## Takeaways

- **C + R is the selected flow: Dice +0.0045, HD95 −15.8 nm, g MAE −0.0001 vs main.**
  - C closes open rings and fills the axon.
  - R alone changes nothing: its result is identical to main.
  - Together they work because fibres whose myelin was only partly predicted are now repaired
    before the ring test, instead of being dropped by it.
- H, T, S and M bring almost nothing.
  - H combined with R makes HD95 worse (+17 nm). Almost all of that comes from one image,
    `sample_0220`, where one far false-positive fibre survives (HD95 1017 → 1937 nm).
  - M raises fibre-level F1 (+0.005) but is not an official metric, and it raises HD95.
- The gains are concentrated in the images with open rings.
  - With C + R, 21 images improve HD95 and 11 get worse; the median change is 0.
  - Trimmed mean (dropping the 3 best and 3 worst images): HD95 −10.5 nm.
  - Biggest wins: `sample_0176` (−498 nm), `0069`, `0113`, `0158`.
  - Biggest losses: `sample_0165` (+305 nm), `0240`.

## Reproduce

```bash
python code/predict.py --checkpoint code/checkpoints/checkpoint_final.pth --plans code/checkpoints/plans.json \
    --dataset-json code/checkpoints/dataset.json --images <val_images> --out <out> --save-sem
python analysis/eval_postproc.py <out>/sem <gt_dir> <out>/postproc_eval    # main vs close_rings
```

`--no-close` restores the main pipeline in `predict.py`.
