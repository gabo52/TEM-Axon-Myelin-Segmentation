"""Train step that survives non-finite values, without freezing the model.

fp16 can overflow on large patches; a single non-finite gradient NaN-poisoned a
whole run twice. Here such batches are skipped. Skipping is a *safety net*, not
a strategy: if it triggers often the model stops learning, so `skipped` is
counted and reported. bf16 is the default because it does not overflow.
"""
import os

import torch
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context

DTYPE = torch.bfloat16 if os.environ.get("AMP_DTYPE", "bf16") == "bf16" else torch.float16


class SafeStep:
    """Mixin: nnU-Net train step + non-finite guard."""

    skipped = 0

    def _ctx(self):
        amp = getattr(self, "use_amp", True)
        if self.device.type == "cuda" and amp:
            return autocast(self.device.type, dtype=DTYPE)
        return dummy_context()

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with self._ctx():
            loss = self.loss(self.network(data), target)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            return {"loss": 0.0}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        if not all(torch.isfinite(p.grad).all() for p in self.network.parameters() if p.grad is not None):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            return {"loss": 0.0}

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.skipped:
            n = self.skipped
            self.skipped = 0
            print(f"  [warn] {n} batch(es) skipped (non-finite); "
                  f"if this is not rare the run is not learning")
