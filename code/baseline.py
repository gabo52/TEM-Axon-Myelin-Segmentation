import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from . import tracking
except ImportError:
    import tracking

class nnUNetTrainerBaseline(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("EPOCHS", 250))
        self._wb = None

    def _do_i_compile(self):
        return False

    # W&B (no-op si wandb no está o WANDB_MODE=disabled), igual que benchmark.py
    def on_train_start(self):
        super().on_train_start()
        self._wb = tracking.start(self)

    def on_epoch_end(self):
        super().on_epoch_end()
        tracking.epoch(self._wb, self)

    def on_train_end(self):
        super().on_train_end()
        tracking.finish(self._wb)
