"""nnUNetTrainerAccept: nnU-Net + g-ratio + edge + learned fiber rejection.

    L = CE+Dice + w_g·|g_pred-g_true| + w_e·CE(band) + w_a·BCE(accept)

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerAccept
Env:   EPOCHS=150  ACCEPT_W=0.1  ACCEPT_EVERY=2  AMP=1
"""
import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:                                   
    from .accept import MIN_AXON, REF_RES, RING_PAD, AcceptHead
    from .accept_loss import AcceptLoss
    from .losses import EdgeLoss, GRatioLoss
    from .step import DTYPE, SafeStep
except ImportError:                  
    from accept import MIN_AXON, REF_RES, RING_PAD, AcceptHead
    from accept_loss import AcceptLoss
    from losses import EdgeLoss, GRatioLoss
    from step import DTYPE, SafeStep

def materialize(net, n_in, hw):
    feat = {}
    h = net.decoder.stages[-1].register_forward_hook(lambda m, i, o: feat.__setitem__("x", o))
    with torch.no_grad():
        out = net(torch.zeros(2, n_in, hw[0], hw[1]))
    h.remove()
    x = feat["x"]
    x = x[0] if isinstance(x, (list, tuple)) else x
    seg = out[0] if isinstance(out, (list, tuple)) else out
    net.accept_head(torch.cat([x, seg.float().softmax(1)[:, 1:3]], dim=1))

class nnUNetTrainerAccept(SafeStep, nnUNetTrainer):
    accept_target = None        

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("EPOCHS", 250))
        self.use_amp = os.environ.get("AMP", "1").lower() not in ("0", "false", "no")
        if not self.use_amp or DTYPE == torch.bfloat16:  
            self.grad_scaler = None

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):

        net = nnUNetTrainer.build_network_architecture(plans_manager, configuration_manager,
                                                       num_input_channels, num_output_channels,
                                                       enable_deep_supervision)
        net.accept_head = AcceptHead()
        materialize(net, num_input_channels,
                    tuple(max(p // 2, 64) for p in configuration_manager.patch_size))
        return net

    def _build_loss(self):
        self.features = {}
        self.network.decoder.stages[-1].register_forward_hook(
            lambda m, i, o: self.features.__setitem__("x", o))

        ignore = self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        base = super()._build_loss()
        inner = base.loss if hasattr(base, "loss") else base    
        chain = GRatioLoss(EdgeLoss(inner, ignore))
        if hasattr(base, "loss"):
            base.loss = chain
        else:
            base = chain

        res = float(self.plans_manager.plans["configurations"]["2d"]["median_image_size_in_voxels"][0]) / REF_RES
        return AcceptLoss(base, self.network.accept_head, self.features,
                          float(os.environ.get("ACCEPT_W", 0.1)),
                          int(os.environ.get("ACCEPT_EVERY", 2)),
                          min_axon=int(MIN_AXON * res * res), pad=max(2, int(RING_PAD * res)),
                          target=self.accept_target)

    def on_train_start(self):
        super().on_train_start()
        print(f"AMP={self.use_amp} dtype={DTYPE}")

    def on_train_end(self):
        super().on_train_end()
