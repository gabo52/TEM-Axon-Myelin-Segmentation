# Method description — Paired axon/myelin instance segmentation in TEM

## 1. Problem formulation

Each myelinated fiber is a **nested** structure: a roughly circular, light
axoplasm surrounded by a concentric, electron-dense ring of myelin. The ground
truth encodes instances as consecutive label pairs:

| value | meaning |
|---|---|
| `0` | background / extra-axonal space |
| `1` | artifact — **excluded from every metric** |
| `X` (even ≥ 2) | axon instance |
| `X+1` (odd ≥ 3) | myelin sheath paired with axon `X` |

We therefore **segment three classes** (background / axon / myelin) and recover
*instances* in a post-processing step. Artifact is remapped to the *ignore*
class so it never contributes a gradient. The instance pairing (`X`, `X+1`) is a
structural constraint of the reconstruction.

## 2. Key observation driving the design

Error analysis on the validation split (fold 0) decomposed the global HD95 into two very different terms:

| source | contribution | nature |
|---|---|---|
| false-positive fibers ("ghosts") | **~325 nm** | confident predictions on *discarded* regions: mostly myelin rings that are broken/interrupted |
| false negatives + boundary error | ~195 nm | missed fibers, and 42.9 nm of boundary error on the fibers that *are* detected |

Two properties of the ghosts are decisive:

1. A **broken myelin ring is a real, measurable signal** (24 angular sectors
   around the axon; the number of empty sectors separates kept from discarded
   fibers with AUC ≈ 0.83). Texture, intensity histograms, bubble counts and
   noise statistics do **not** separate them (AUC 0.51–0.61).
2. **≈5 % of annotated fibers also have interrupted rings.** Applying the rule
   as a hard gate therefore destroys true positives: the rejection decision has
   to be **learned and soft**, integrated in the model, not thresholded at
   inference.

This is the reason the method is *not* "segment then filter with a rule": the
rule only provides **training targets**, and the network learns to apply it.

## 3. Architecture

```
                      ┌──────────────────────────────┐
  TEM image (uint8)   │  nnU-Net 2D PlainConvUNet    │
  ─────────────────►  │  encoder ── decoder          │
                      └──────────────┬───────────────┘
                                     │ decoder features (last stage)
             ┌───────────────────────┼────────────────────────┐
             ▼                       ▼                        ▼
      SegHead (3 classes)     AcceptHead (rejection)     (deep supervision)
      bg / axon / myelin      p(keep) per fiber
             │                       │
             ▼                       ▼
        semantic mask           fiber score map
             │                       │
             └──────────► instances ◄┘
                    (watershed + pairing)
                             │
                             ▼
              post-process: drop if score<τ or fiber touches image border
                             │
                             ▼
                 uint16 instance map (X, X+1)
```

### 3.1 Segmentation backbone

Stock **nnU-Net v2.8.1** 2D configuration (`PlainConvUNet`, patch
1024×1536, InstanceNorm, LeakyReLU, deep supervision). The 2D configuration was
chosen because fibers are elongated structures whose context fits in-plane; a
half-resolution variant (1504×2048) is the primary model, with full-resolution
(3008×4096) explored as a variant.

### 3.2 AcceptHead — learned fiber rejection

A three-layer convolutional head (width 16) applied to the last decoder stage:

* input: `[decoder_features, p(axon), p(myelin)]` (concatenated; channel count
  is read lazily so the head adapts to any decoder width),
* `AvgPool2d(4) → conv3×3 → GroupNorm → ReLU → conv3×3 → GroupNorm → ReLU → conv1×1`,
* one logit per 4×4 px block, i.e. a score map at 1/4 resolution.

A fiber's score is the mean of this map over its axon pixels. The head is part
of the trained network, so **a single forward pass returns both the
segmentation and the rejection score**.

### 3.3 GPEM — geometric prior fused into the encoder

The visual appearance of a TEM section is ambiguous: large vessels and somas
look like oversized axons. The **geometry** is not: a real fiber is a bright
core (axoplasm) inside a thin dark ring (myelin), and that ring is surrounded by
normal tissue. A false positive is a dark blob whose "ring" is weak and which
stays dark **outside** the ring.

We therefore measure, on the **image** and with no learned parameters:

```
core  = mean over the disc 0.6r          ring = mean over the annulus 0.9r–1.4r
sur   = mean over the annulus 1.6r–2.2r
maps  = { img − box(3R_max),  core − ring,  sur − ring }   for r in {28, 48, 80}
```

All means are box means computed with an **integral image**, so the cost is O(1)
per pixel regardless of the radius (with `avg_pool2d` a 80 px radius would take
minutes per patch). Seven channels, ≈90 ms per 1024×1536 patch.

The maps are fused into the last three encoder stages with a **learned gated
residual**:

```
g' = proj(g)                         1×1 projection to the stage width
a  = sigmoid(gate([h, g']))          per-pixel gate
h  = h + a · g'                      fused feature
```

`proj` is initialised to **zero**, so at step 0 the network is *identical* to the
benchmark without the prior, and it learns to use the geometry only if it helps.

## 4. Training objective

```
L = L_seg + L_gratio + L_edge + λ_accept · L_accept
```

**L_seg** — nnU-Net default (cross-entropy + soft Dice, batch Dice), with the
artifact class mapped to *ignore*.

**L_gratio** — per-image consistency with the challenge's main biomarker.
With `A` = area, `D = 2√(A/π)`, `g = D_axon / D_fiber`:

```
L_gratio = | sqrt(A_ax / (A_ax + A_my + 1)) − sqrt(Â_ax / (Â_ax + Â_my + 1)) |
```
computed from the soft class probabilities (prediction) and the ground truth.

**L_edge** — cross-entropy concentrated on a band around the axon/myelin
interface, to sharpen the thin myelin border:

```
band = maxpool(myelin, k=7) + maxpool(−myelin, k=7)      (myelin boundary, width 7 px)
L_edge = Σ (CE · band) / Σ band
```

**L_accept** — balanced BCE on the rejection head. Targets are **derived from
rules on the training data only**:

| predicted fiber | target | weight |
|---|---|---|
| matches a GT axon (IoU ≥ 0.3) | 1 (keep) | 1 |
| no match **and** broken ring (≥ 4 of 24 empty sectors) | 0 (discard) | 1 |
| no match, intact ring (ambiguous) | — | **0** |

Ignoring the ambiguous group was essential: those fibers are mostly *unlabeled*
(discarded) regions, and forcing them to 0 injects label noise.

Two robustness details: the BCE is averaged over positive and negative pixels
separately (class balance, since ghosts are a minority), and terms are clamped
plus checked for NaN, because NaNs in this term silently poisoned long runs.

### 4.1 Resolution-aware constants

Constants expressed in pixels are scaled with the dataset resolution
(`res = median_height / 1504`): minimum axon area `1500·res²` and ring dilation
`4·res`. Without this scaling, full-resolution training supervises a completely
different physical area and degrades.

## 5. Inference pipeline

1. **Normalisation** — nnU-Net z-score statistics from the plans file.
2. **Tiled forward pass** with the training patch size and 50 % overlap; class
   probabilities are accumulated as votes (argmax after averaging) and the
   accept score map is averaged.
3. **Instance assembly** (`otros/instances.py`):
   * axons = connected components of the axon class;
   * myelin is split among axons with a **watershed on the distance transform**
     seeded by the axons, so touching fibers separate;
   * labels are written as `2k` (axon) / `2k+1` (myelin) — the pair is
     guaranteed by construction.
4. **Fiber filtering** (the only post-process):
   ```
   discard fiber  ⟺  mean accept-score < τ   OR   fiber bbox touches the image border
   ```
   The border rule removes partial fibers whose measurement is not defined.
5. **Output** — uint16 TIFF with the challenge's paired encoding.

No external critic, no per-image threshold tuning: the *rejection* decision is
the network's, the two thresholds are fixed and calibrated on validation folds.

## 6. Experimental results (fold 0, validated on the cluster)

54 held-out cases (fold 0 = 212 train / 54 validation), 150 epochs, HD95 in nm
pooled over boundary pixels. The **border gate** removes fibers cut by the image
edge: their diameter and g-ratio are not measurable.

| model | HD95 raw | HD95 + border gate | Dice |
|---|---|---|---|
| nnU-Net baseline (20 ep) | 750.0 | — | 0.8623 |
| AcceptHead, rule targets | 759.0 | 593.0 | 0.8788 |
| AcceptHead, oracle targets | 751.9 | 594.0 | 0.8785 |
| **GPEM** | **744.7** | **572.5** | **0.8833** |
| GPEM + learned rejection (τ=0.2) | — | **566.5** | **0.8851** |

Reference points: removing every false positive would give 234.9 nm. The trained
rejection head activates on ≈12 % of fibers at τ=0.2 (the score map is compared
per fiber).

### 6.1 What was tested and rejected (all measured, same split)

| variant | result | why it failed |
|---|---|---|
| radial "nested" head (star-convex r_inner/r_outer) | 765.9 nm | no gain over baseline |
| myelin ring-closure loss | 885.7 nm | over-segments myelin |
| adversarial patch discriminator | ≈ accept-head | no benefit |
| discard as a 4th class | — | artifact/unlabeled are not separable classes |
| committee (ring judge + border judge) | AUC 0.825 | double head interferes; border judge inert |
| ResEncM backbone (residual encoder) | 838.9 nm | smaller patch (448×576) loses the fiber context |
| fiber-level ranking loss | AUC 0.797 | optimises the order, not the operating point |
| learned arbiter over ring descriptors | AUC 0.922 / 598 nm | better AUC, no HD95 gain |
| GNN over the fiber graph | AUC 0.904 / 649 nm | trained on patch graphs, applied to whole images |
| uncertain head (mean + variance) | AUC 0.842 | no gain |
| post-hoc filters on image features | 590–870 nm | never beat the plain border gate (580 nm) |

The pattern is consistent: **context and a single learned rejection head help;
extra heads, extra backbones and extra gates do not.** A high AUC on fiber
classification is not a guide here — the delivery metric depends on the
operating point, not on the global ranking.

## 7. Training configuration

| item | value |
|---|---|
| framework | nnU-Net v2.8.1, custom trainer `nnUNetTrainerAccept` |
| configuration | 2D, PlainConvUNet, patch 1024×1536, batch 2 |
| optimiser | SGD + Nesterov (nnU-Net default), 250 epochs |
| augmentation | nnU-Net default (flips, rotations, elastic, gamma, brightness) |
| λ_accept | 0.1 (applied every 2 steps) |
| hardware | single GPU (validated on RTX 5090 / 32 GB; fits in 8 GB at half resolution) |
| reproducibility | decisions taken on the cluster (torch 2.12), not on the dev box (torch 2.3): the same 20-epoch run differs by ~38 nm |
