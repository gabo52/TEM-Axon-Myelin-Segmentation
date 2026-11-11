"""Entrenamiento — punto de entrada único (el challenge pide code/train.py).

Elige el modelo con --model; el resto de opciones van al pipeline elegido:

  Cellpose-SAM  (code/cellpose/train.py)
    python code/train.py --model cellpose --target both --wandb
    python code/train.py --model cellpose --target fiber --epochs 200 --freeze 8
    python code/train.py --model cellpose --help          # todas sus opciones

  nnU-Net       (trainers de code/baseline.py y code/benchmark.py)
    python code/train.py --model nnunet --dataset 501 --fold 0 --trainer accept --wandb
    python code/train.py --model nnunet --dataset 501 --fold all --trainer accept   # modelo final
    python code/train.py --model nnunet --help

W&B: con --wandb se registra el entrenamiento (proyecto WANDB_PROJECT, por
defecto "tem-challenge"). Sin --wandb no se envía nada.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- Cellpose
def train_cellpose(args):
    """Delega en code/cellpose/train.py con las mismas opciones."""
    cmd = [sys.executable, str(CODE / "cellpose" / "train.py"), *args]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- nnU-Net
NNUNET_TRAINERS = {
    "baseline": "nnUNetTrainerBaseline",  # nnU-Net estándar        (baseline.py)
    "accept": "nnUNetTrainerAccept",      # + g-ratio, borde, rechazo (benchmark.py)
}
# nnU-Net busca el trainer por nombre dentro de su propio paquete,
# así que estos módulos se copian allí antes de entrenar.
NNUNET_FILES = ["baseline.py", "benchmark.py", "losses.py", "accept.py",
                "accept_loss.py", "step.py", "tracking.py"]


def train_nnunet(args):
    ap = argparse.ArgumentParser(prog="train.py --model nnunet")
    ap.add_argument("--dataset", required=True, help="ID del dataset nnU-Net, p. ej. 501")
    ap.add_argument("--fold", default="0", help="0-4, o 'all' para el modelo final")
    ap.add_argument("--trainer", choices=NNUNET_TRAINERS, default="accept")
    ap.add_argument("--config", default="2d")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--wandb", action="store_true", help="registrar el entrenamiento en W&B")
    a = ap.parse_args(args)

    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not os.environ.get(var):
            sys.exit(f"falta la variable de entorno {var} (ver code/README.md)")

    import nnunetv2.training.nnUNetTrainer as trainers_pkg
    dst = Path(trainers_pkg.__path__[0])
    for f in NNUNET_FILES:
        shutil.copy(CODE / f, dst / f)
    print(f"trainers copiados a {dst}")

    env = dict(os.environ, EPOCHS=str(a.epochs))
    if not a.wandb:
        env["WANDB_MODE"] = "disabled"  # code/tracking.py pasa a ser un no-op

    cmd = ["nnUNetv2_train", a.dataset, a.config, a.fold, "-tr", NNUNET_TRAINERS[a.trainer]]
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


# ---------------------------------------------------------------- main
PIPELINES = {"cellpose": train_cellpose, "nnunet": train_nnunet}


def main():
    # Solo se lee --model; el resto de argumentos pasan tal cual al pipeline elegido
    # (por eso add_help=False: "--model cellpose --help" muestra la ayuda de Cellpose).
    ap = argparse.ArgumentParser(description=__doc__, add_help=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=PIPELINES, help="pipeline a entrenar")
    args, rest = ap.parse_known_args()
    if args.model is None:
        ap.print_help()
        sys.exit(1)
    PIPELINES[args.model](rest)


if __name__ == "__main__":
    main()
