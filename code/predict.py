"""Delivery: native nnU-Net inference -> uint16 instances at native resolution.

Fold-0 pipeline (GPEM15, 570.5 nm / Dice 0.8834):

    1. nnUNetPredictor  (nnU-Net inference: gaussian + mirroring)
    2. nearest x2       (the network works at half resolution)
    3. to_instances     (watershed + ring filter)
    4. border gate      (drop fibers cut by the image edge)

    python code/predict.py --model-folder <trainer dir> --folds 0 1 2 3 4 \\
        --images <dir> --out <dir>

`--model-folder` must point to the `nnUNetTrainerGPEM15__nnUNetPlans__2d` folder
(the one holding `dataset.json`, `plans.json` and the `fold_X` subfolders).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tifffile

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

for _v in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
    os.environ.setdefault(_v, tempfile.mkdtemp(prefix="nnunet_"))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  
from accept_filter import drop                                      
from instances import to_instances                                 


def predict_folder(model_folder, images_dir, out_dir, folds=("0",)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=True, device=device, verbose=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds,
                                                   checkpoint_name="checkpoint_final.pth")

    # nnU-Net expects <case>_0000.tif; the test images have no suffix.
    tmp = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
    for f in sorted(Path(images_dir).glob("*.tif")):
        os.symlink(f.resolve(), tmp / f"{f.stem}_0000.tif")

    sem_dir = tmp.parent / f"{tmp.name}_sem"
    predictor.predict_from_files(str(tmp), str(sem_dir), save_probabilities=False,
                                 overwrite=True, num_processes_preprocessing=1,
                                 num_processes_segmentation_export=1)

    for f in sorted(sem_dir.glob("*.tif")):
        sem = tifffile.imread(f)
        if sem.shape[0] <= 1600:
            sem = sem.repeat(2, 0).repeat(2, 1)          
        inst = to_instances(sem)
        ones = np.ones(inst.shape, np.float32)
        tifffile.imwrite(out / f.name, drop(inst, ones, 0.0).astype("uint16"))
    return len(list(out.glob("*.tif")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True,
                    help="trainer folder (dataset.json + plans.json + fold_X)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", nargs="+", default=("0",),
                    help="folds to ensemble: '0' for demo, '0 1 2 3 4' for final")
    a = ap.parse_args()
    n = predict_folder(a.model_folder, a.images, a.out, tuple(a.folds))
    print(f"wrote {n} predictions to {a.out}")


if __name__ == "__main__":
    main()