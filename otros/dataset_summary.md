# Dataset notes (read this before training)

Numbers measured on the annotated training set. They are what the benchmark
architecture is built on, and they are a good starting point for other ideas.

## 1. Format

| | value |
|---|---|
| images | 266 annotated, `uint8`, 1 channel, **3008 × 4096 px** |
| labels | `uint16`, paired instances |
| pixel size | **2.152 nm/px** → field of view 8.81 × 6.47 µm |

Label encoding:

| value | meaning |
|---|---|
| `0` | background / extra-axonal space |
| `1` | **artifact — excluded from every metric** |
| `X` (even ≥ 2) | axon instance |
| `X+1` (odd ≥ 3) | myelin sheath paired with axon `X` |

So there are 4 roles in the ground truth but only **3 classes to segment**
(background / axon / myelin): artifact is remapped to *ignore*, and the
`X` / `X+1` pairing is a property of the instance reconstruction.

## 2. Fiber statistics

| | value |
|---|---|
| median axon area | ~55 700 px (Ø ≈ 0.55 µm at full resolution) |
| largest axon | ~935 000 px — check these with the organizers, they look like somas/vessels |
| annotated fibers | ~8 000 |

## 3. Where the HD95 error comes from (the key table)

Of the large predicted fibers, 247 land on background (they *hurt* the metric)
and 766 land on artifact (ignored). Removing subsets of the 247:

| variant | fibers removed | Dice | HD95 (nm) |
|---|---|---|---|
| base | 0 | 0.866 | 784.1 |
| no broken ring (`gaps ≥ 4`) | 156 | 0.907 | **514.0** |
| no intact ring (`gaps < 4`) | 91 | 0.890 | 738.9 |
| none of the 247 | 247 | 0.936 | 384.2 |

**67 % of the excess HD95 comes from 156 fibers with a broken myelin ring.**
That is why rejection is worth learning.

## 4. Which features actually separate kept from discarded fibers

Means per class over 2 516 large predicted fibers (`defectos_por_fibra.csv`,
copied next to this file — one row per fiber):

| feature | annotated | background | artifact | AUC |
|---|---|---|---|---|
| empty ring sectors (`gaps`) | 0.77 | **8.47** | 6.52 | **0.81** |
| mean ring coverage (`cov_mean`) | **0.96** | 0.62 | 0.72 | **0.82** |
| thickness irregularity (`irreg`) | 0.25 | 0.53 | 0.67 | 0.72 |
| circularity | 0.54 | 0.42 | 0.47 | 0.70 |
| bubble fraction | 0.171 | 0.179 | 0.218 | **0.52 ✗** |
| number of bubbles | 367 | 356 | 396 | **0.52 ✗** |

Read it as: **the ring is the signal, texture/bubbles are noise.** Note that
background and artifact have the *same* defects — they are the same kind of
object (a discarded region), which is why the "artifact class" cannot be
predicted as a 4th class.

## 5. Consequence for the method

Because ~5 % of **annotated** fibers also have a broken ring, the rule cannot
be applied as a hard gate: that would delete true positives. Instead the rule
is used only to build **training targets**, and the network learns a soft
decision (`code/accept.py`, `code/accept_loss.py`).

## 6. Ideas worth trying (open)

* multi-channel input (gradient / EDT) — not yet evaluated;
* full resolution 3008 × 4096 — evaluated, did not beat half-resolution;
* better instance separation where axons touch (watershed is the current step);
* the artifact class is ignored by the metric: predicting it well has no reward.
