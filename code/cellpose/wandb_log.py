"""Registro en Weights & Biases para el pipeline Cellpose (train / predict / evaluate).

Todo es opcional: si wandb no está instalado o no hay login, start() devuelve
None, se avisa y el script sigue sin registrar.

Proyecto / entidad: variables WANDB_PROJECT (por defecto "tem-challenge") y
WANDB_ENTITY, igual que el pipeline de nnU-Net (code/tracking.py).

Entrenamiento: cellpose.train.train_seg() no acepta callbacks, pero escribe su
progreso en el logger "cellpose.train":

    10, train_loss=0.8123, test_loss=0.9541, LR=0.000010, time 52.31s

Un logging.Handler lee esas líneas y las manda a W&B (Cellpose las escribe en
la época 5 y cada 10). Al terminar se sube la curva completa de train.
"""
import logging
import os
import re

from common import to_classes

CLASS_LABELS = {0: "fondo", 1: "axón", 2: "mielina", 3: "artefacto"}


def start(name, group=None, job_type=None, config=None):
    """Abre un run de W&B, o devuelve None si no se puede."""
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "tem-challenge"),
                          entity=os.environ.get("WANDB_ENTITY"),
                          name=name, group=group, job_type=job_type, config=config)
    except Exception as e:  # sin wandb o sin login: se sigue sin registrar
        print(f"[wandb] desactivado ({type(e).__name__}: {e})")
        return None


def mask_image(img, pred, gt=None, caption="", step=4):
    """Imagen reducida 1/step con la predicción (y el GT) como máscaras conmutables."""
    import numpy as np
    import wandb

    small = (slice(None, None, step), slice(None, None, step))
    masks = {"prediction": {"mask_data": to_classes(pred[small]), "class_labels": CLASS_LABELS}}
    if gt is not None:
        masks["ground_truth"] = {"mask_data": to_classes(gt[small]), "class_labels": CLASS_LABELS}
    return wandb.Image(np.clip(img[small], 0, 255).astype(np.uint8), caption=caption, masks=masks)


def table(rows):
    """Lista de dicts -> wandb.Table (columnas = unión de claves; faltantes = None)."""
    import wandb
    cols = list(dict.fromkeys(k for r in rows for k in r))
    return wandb.Table(columns=cols, data=[[r.get(k) for k in cols] for r in rows])


# ---------------------------------------------------------------- entrenamiento
NUM = r"([-+\d.eE]+|nan|inf)"
EPOCH_LINE = re.compile(rf"^(\d+), train_loss={NUM}, test_loss={NUM}, LR={NUM}")


class _CellposeLogHandler(logging.Handler):
    """Convierte cada línea de progreso de Cellpose en un run.log()."""

    def __init__(self, run):
        super().__init__(level=logging.INFO)
        self.run = run

    def emit(self, record):
        m = EPOCH_LINE.match(record.getMessage())
        if m is None:
            return
        epoch, train_loss, val_loss, lr = m.groups()
        data = {"epoch": int(epoch), "train_loss": float(train_loss), "lr": float(lr)}
        if float(val_loss) > 0:  # Cellpose escribe 0.0000 cuando no hay validación (--all)
            data["val_loss"] = float(val_loss)
        # emit() se ejecuta DENTRO de train_seg: un fallo de W&B no debe tumbar el entrenamiento
        try:
            self.run.log(data, step=int(epoch))
        except Exception as e:
            print(f"[wandb] no se pudo registrar la época {epoch} ({type(e).__name__}: {e})")


class TrainRun:
    """Un run de W&B por modelo entrenado (fiber / axon)."""

    def __init__(self, run):
        self.run = run
        self.handler = _CellposeLogHandler(run)
        logging.getLogger("cellpose.train").addHandler(self.handler)

    def finish(self, train_loss, val_loss, model_path, summary=None):
        """train_loss/val_loss: curvas completas de train_seg (vacías si hubo early stopping)."""
        import wandb
        logging.getLogger("cellpose.train").removeHandler(self.handler)
        try:
            if len(train_loss):
                # curva completa época -> pérdida. val_loss = 0 donde Cellpose no la calcula.
                rows = [[e, float(t), float(v) if v else None]
                        for e, (t, v) in enumerate(zip(train_loss, val_loss))]
                curve = wandb.Table(columns=["epoch", "train_loss", "val_loss"], data=rows)
                self.run.log({"full_curve/train_loss": wandb.plot.line(
                    curve, "epoch", "train_loss", title="train loss (todas las épocas)")})
                self.run.summary["final_train_loss"] = float(train_loss[-1])
            self.run.summary["model_path"] = str(model_path)
            self.run.summary.update(summary or {})
            self.run.finish()
        except Exception as e:  # un fallo de W&B no debe parar el entrenamiento del otro modelo
            print(f"[wandb] error al cerrar el run ({type(e).__name__}: {e})")


def start_train(name, group, config):
    run = start(name, group=group, job_type="train", config=config)
    return TrainRun(run) if run is not None else None
