# Ablations — what worked and what did not

All runs: Dataset 505 (half resolution, 212 train / 54 validation, fold 0),
same code (bf16), same evaluation pipeline. HD95 is the global pooled metric in
nm; Dice is instance-level after the border gate.

## Epoch budget (benchmark, `nnUNetTrainerAccept`)

| epochs | HD95 + border | Dice |
|---|---|---|
| 20 | 653.9 | 0.8715 |
| 60 (older code, fp16) | 574.6 | 0.888 |
| 250 | 634.6 | 0.8803 |

More epochs help vs 20; the 250 result is only comparable with the 20 result
because both ran the same code.

## The four proposed extensions (20 epochs each)

| idea | what it does | AUC (honest) | HD95 + border | Dice |
|---|---|---|---|---|
| — benchmark, own score τ=0.2 | reference | 0.829 | **604.0** | **0.8692** |
| (1) ranking loss | fibre-level pairwise order | 0.797 | — | — |
| (2) learned arbiter | MLP over head score + 5 ring/shape descriptors | **0.922** | 598.2 | 0.8629 |
| (3) GNN over fibre graph | 2 SAGE layers, k-NN = 6 | 0.904 | 820.8 | 0.8395 |
| (4) heteroscedastic head | mean + variance | 0.842 | — | — |

Notes that matter:

* **AUC is not the metric.** (2) raises AUC by +0.09 and the deliverable HD95 by
  5.8 nm only, while losing 0.6 Dice. (3) raises AUC by +0.075 and *breaks* the
  filtering (−217 nm worse than the plain border gate).
* **Why:** HD95 is dominated by a few very large false positives, which the
  border gate already removes; AUC counts thousands of small fibres whose
  rejection barely moves the metric.
* (3) has a real implementation flaw: it is trained on patch graphs (~50 nodes)
  and applied to full-image graphs (~1500 nodes), so the score distribution does
  not transfer. Its validation Dice (0.8984, the best of all runs) never reached
  the instance level (0.8658 after the gate).

## Verdict

The benchmark (learned rejection, one model, one forward pass) remains the best
deliverable. The four extensions are documented negative results: they improve a
proxy, not the measured metric.
