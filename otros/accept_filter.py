"""Drop predicted fibers that the model does not trust.

Two independent, cheap criteria (both calibrated on validation folds):

* low accept score  -- the learned rejection head is unsure about this fiber;
* touches the border -- the fiber is cut by the image edge, so its anatomy
  (diameter, myelin thickness, g-ratio) is not measurable.
"""
import numpy as np

def fibers(inst, k):
    return (inst == 2 * k) | (inst == 2 * k + 1)

def drop(inst, score, tau=0.2, border=True):
    out = inst.copy()
    h, w = inst.shape
    axon = (inst >= 2) & (inst % 2 == 0)
    ids = np.unique(np.where(axon, inst // 2, 0))
    for k in ids[ids > 0]:
        m = axon & (inst // 2 == k)
        reject = score[m].mean() < tau
        if border and not reject:
            ys, xs = np.nonzero(m)
            reject = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        if reject:
            out[fibers(inst, k)] = 0
    return out
