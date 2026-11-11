import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.io import imsave
from skimage.measure import label
from skimage.segmentation import watershed

MIN_RING = 0.5

def drop_unwrapped(inst, min_ring):
    out = inst.copy()
    for k, box in enumerate(ndi.find_objects(inst // 2), 1):
        if box is None:
            continue
        box = tuple(slice(max(s.start - 5, 0), s.stop + 5) for s in box)
        axon, myelin = inst[box] == 2 * k, inst[box] == 2 * k + 1
        ring = ndi.binary_dilation(axon, iterations=4) & ~axon
        if myelin[ring].mean() < min_ring:
            out[box][axon | myelin] = 0
    return out

def to_instances(sem, min_ring=MIN_RING):
    axons = label(sem == 1, connectivity=1)
    dist = ndi.distance_transform_edt(axons == 0)
    ws = watershed(dist, axons, mask=sem > 0)
    out = np.zeros(sem.shape, np.uint16)
    out[ws > 0] = ws[ws > 0] * 2 + 1
    out[axons > 0] = axons[axons > 0] * 2
    out[(ws > 0) & (sem == 0)] = 0
    return drop_unwrapped(out, min_ring) if min_ring else out

if __name__ == "__main__":
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(sys.argv[1]).glob("*.tif")):
        inst = to_instances(tifffile.imread(f))
        tifffile.imwrite(dst / f.name, inst)
        imsave(dst / f"{f.stem}.png", inst, check_contrast=False)
