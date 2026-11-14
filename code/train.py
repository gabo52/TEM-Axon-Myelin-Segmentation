"""Training of the GPEM15 method — single entry point (the challenge asks for code/train.py).

GPEM15 is the delivered method: a 2D nnU-Net whose encoder fuses a 12-channel
geometric prior (radial profile at 4 scales + gradient + local orientation)
through a learned gate, plus a per-fiber rejection head.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

# gpem15
NNUNET_TRAINERS = {
    "gpem15": "nnUNetTrainerGPEM15",     # method: 12-channel geometric prior    (gpem12.py)
    "gpem":   "nnUNetTrainerGPEM",       # ablation: 7-channel geometric prior   (gpem7.py)
    "accept": "nnUNetTrainerAccept",     # base: nnU-Net + g-ratio + edge + reject (accept_trainer.py)
}


NNUNET_FILES = ["accept_trainer.py", "gpem7.py", "gpem12.py",
                "maps7.py", "maps12.py",
                "accept.py", "accept_loss.py", "losses.py",
                "step.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="nnU-Net dataset ID, e.g. 505")
    ap.add_argument("--fold", default="0", help="0-4, or 'all' for the final model")
    ap.add_argument("--trainer", choices=NNUNET_TRAINERS, default="gpem15")
    ap.add_argument("--config", default="2d")
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()

    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not os.environ.get(var):
            sys.exit(f"missing env var {var} (see code/README.md)")

    import nnunetv2.training.nnUNetTrainer as trainers_pkg
    dst = Path(trainers_pkg.__path__[0])
    for f in NNUNET_FILES:
        shutil.copy(CODE / f, dst / f)
    print(f"trainers copied to {dst}")

    env = dict(os.environ, EPOCHS=str(a.epochs))

    cmd = [sys.executable, "-c",
           "from nnunetv2.run.run_training import run_training_entry; run_training_entry()",
           a.dataset, a.config, a.fold, "-tr", NNUNET_TRAINERS[a.trainer], "-p", a.plans]
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()