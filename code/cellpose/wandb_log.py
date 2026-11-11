"""Registro en Weights & Biases para el entrenamiento de Cellpose.

Problema: cellpose.train.train_seg() hace todo el bucle por dentro y no acepta
callbacks. Pero sí escribe su progreso en el logger "cellpose.train":

    10, train_loss=0.8123, test_loss=0.9541, LR=0.000010, time 52.31s

Solución:
  1. Durante el entrenamiento, un logging.Handler lee esas líneas y las manda
     a W&B (Cellpose las escribe en la época 5 y cada 10 épocas).
  2. Al terminar, se suben las curvas completas que devuelve train_seg()
     (pérdida de train en TODAS las épocas).

Proyecto / entidad: variables WANDB_PROJECT (por defecto "tem-challenge") y
WANDB_ENTITY, igual que el pipeline de nnU-Net (code/tracking.py).
"""
import logging
import os
import re

# "10, train_loss=0.8123, test_loss=0.9541, LR=0.000010, ..."
NUM = r"([-+\d.eE]+|nan|inf)"
EPOCH_LINE = re.compile(rf"^(\d+), train_loss={NUM}, test_loss={NUM}, LR={NUM}")


class _CellposeLogHandler(logging.Handler):
    """Convierte cada línea de progreso de Cellpose en un wandb.log()."""

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
        self.run.log(data, step=int(epoch))


class WandbRun:
    """Un run de W&B por modelo entrenado (fiber / axon)."""

    def __init__(self, name, group, config):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "tem-challenge"),
                              entity=os.environ.get("WANDB_ENTITY"),
                              name=name, group=group, config=config)
        self.handler = _CellposeLogHandler(self.run)
        logging.getLogger("cellpose.train").addHandler(self.handler)

    def finish(self, train_loss, val_loss, model_path):
        logging.getLogger("cellpose.train").removeHandler(self.handler)

        # curvas completas: tabla época -> pérdida. Cellpose deja val_loss = 0 en las
        # épocas en que no la calcula (y en todas con --all): ahí se guarda None.
        rows = [[e, float(t), float(v) if v else None]
                for e, (t, v) in enumerate(zip(train_loss, val_loss))]
        table = self.wandb.Table(columns=["epoch", "train_loss", "val_loss"], data=rows)
        self.run.log({"full_curve/train_loss": self.wandb.plot.line(
            table, "epoch", "train_loss", title="train loss (todas las épocas)")})

        self.run.summary["model_path"] = str(model_path)
        self.run.summary["final_train_loss"] = float(train_loss[-1])
        self.run.finish()


def start(name, group, config):
    """Devuelve un WandbRun, o None si wandb no está instalado / falla al iniciar."""
    try:
        return WandbRun(name, group, config)
    except Exception as e:  # sin wandb o sin login: se entrena igual
        print(f"[wandb] desactivado ({type(e).__name__}: {e})")
        return None
