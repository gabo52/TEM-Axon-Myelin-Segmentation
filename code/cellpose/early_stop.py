"""Early stopping por Dice de validación para cellpose.train.train_seg (sin callbacks).

Cellpose valida (pérdida) en la época 5 y cada 10 épocas y lo escribe en el logger
"cellpose.train":

    120, train_loss=0.7012, test_loss=0.7461, LR=0.000010, time 2710.3s

Este handler lee esas líneas y, cada `every` épocas a partir de `start`:
  * segmenta las imágenes de validación con la red en su estado actual y calcula
    el Dice del objetivo del modelo (fibra o axón) frente al GT, sin los píxeles de
    artefacto, como media por imagen (igual que la métrica del challenge);
  * si el Dice mejora, guarda el modelo (<modelo>_best);
  * si no mejora en `patience` épocas, activa una bandera: el siguiente lote de
    entrenamiento lanza StopTraining (se intercepta cellpose.train._get_batch,
    que train_seg llama en cada lote) y train_seg termina.

Uso:
    with EarlyStopping(model, val_x, val_y, val_valid, best_path, patience=60) as es:
        train.train_seg(model.net, ...)
    es.score()                     # Dice del estado actual (p. ej. el modelo final)
    es.best_epoch, es.best_dice, es.stopped, es.history
"""
import logging
import os
import time
import traceback

import numpy as np
from cellpose import train as cp_train

from wandb_log import EPOCH_LINE


class StopTraining(Exception):
    """Señal interna para salir de train_seg."""


def dice(pred, gt, valid):
    p, g = pred & valid, gt & valid
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


class EarlyStopping(logging.Handler):
    def __init__(self, model, val_x, val_y, val_valid, best_path, patience=60,
                 every=20, start=40, batch_size=8, wandb_run=None, stop_epoch=None):
        super().__init__(level=logging.INFO)
        self.model, self.best_path = model, str(best_path)
        self.val = list(zip(val_x, val_y, val_valid))
        self.patience, self.every, self.start = patience, every, start
        self.stop_epoch = stop_epoch  # parar tras esta época (modelo final: misma trayectoria que un fold)
        self.batch_size, self.wandb_run = batch_size, wandb_run
        self.best_dice, self.best_epoch = -1.0, None
        self.history = []   # (época, train_loss, val_loss, val_dice o None)
        self.flag = False   # pedir parada
        self.stopped = False

    def score(self):
        """Dice medio en validación con la red en su estado actual."""
        scores = []
        for img, gt, valid in self.val:
            masks = self.model.eval(img, batch_size=self.batch_size)[0]
            scores.append(dice(masks > 0, gt > 0, valid))
        return float(np.mean(scores))

    def check(self, epoch):
        """Evalúa el Dice; guarda si mejora; devuelve el Dice."""
        t0 = time.time()
        d = self.score()
        secs = time.time() - t0
        if d > self.best_dice:
            # guardado atómico; el "mejor" solo se actualiza si el archivo se escribió entero
            tmp = self.best_path + ".tmp"
            self.model.net.save_model(tmp)
            os.replace(tmp, self.best_path)
            self.best_dice, self.best_epoch = d, epoch
        if self.wandb_run is not None:
            try:  # un fallo de W&B no debe tumbar el entrenamiento
                self.wandb_run.log({"val_dice": d, "best_val_dice": self.best_dice}, step=epoch)
            except Exception as e:
                print(f"[wandb] no se pudo registrar el Dice ({type(e).__name__}: {e})")
        print(f"[early stop] época {epoch}: Dice val {d:.4f} "
              f"(mejor {self.best_dice:.4f} en la época {self.best_epoch}) [{secs:.0f}s]", flush=True)
        return d

    # ---- lectura de las líneas de progreso de Cellpose
    def emit(self, record):
        m = EPOCH_LINE.match(record.getMessage())
        if m is None:
            return
        epoch = int(m.group(1))
        d = None
        if self.val and epoch >= self.start and epoch % self.every == 0:
            # emit() corre DENTRO de train_seg: un fallo aquí (p. ej. falta de memoria en la
            # GPU o de disco) no debe tirar horas de entrenamiento; se salta este control
            try:
                d = self.check(epoch)
            except Exception:
                print(f"[early stop] control de la época {epoch} fallido, se ignora:", flush=True)
                traceback.print_exc()
            if self.patience and self.best_epoch is not None and \
                    epoch - self.best_epoch >= self.patience:
                self.flag = True
        if self.stop_epoch is not None and epoch >= self.stop_epoch:
            self.flag = True
        self.history.append((epoch, float(m.group(2)), float(m.group(3)), d))

    # ---- activar / desactivar alrededor de train_seg
    def __enter__(self):
        logging.getLogger("cellpose.train").addHandler(self)
        self._get_batch = cp_train._get_batch

        def guarded_get_batch(*args, **kwargs):
            if self.flag:
                raise StopTraining()
            return self._get_batch(*args, **kwargs)

        cp_train._get_batch = guarded_get_batch
        return self

    def __exit__(self, exc_type, exc, tb):
        logging.getLogger("cellpose.train").removeHandler(self)
        cp_train._get_batch = self._get_batch
        if exc_type is StopTraining:
            self.stopped = True
            last = self.history[-1][0]
            why = (f"época de parada fijada ({self.stop_epoch})"
                   if self.stop_epoch is not None and last >= self.stop_epoch
                   else f"sin mejora de Dice en {self.patience} épocas")
            print(f"[early stop] {why}: parado tras la época {last}; mejor época "
                  f"{self.best_epoch} (Dice {self.best_dice:.4f})", flush=True)
            return True  # la excepción se consume aquí
        return False
