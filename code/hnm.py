"""Propuesta: GPEM + mineria de negativos duros dirigida por el oraculo.

El oraculo dice que la segmentacion conservada vale 163 nm y que el error son las
fibras no emparejadas. Esta perdida las senala una a una durante el entrenamiento
(ver hnm_loss.py).

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerHNM
Env:   EPOCHS=150  HNM_W=0.2  HNM_MIN_PX=2000  HNM_WARMUP=2500
"""
import os

try:
    from .gpem import nnUNetTrainerGPEM
    from .hnm_loss import HNMLoss
except ImportError:
    from gpem import nnUNetTrainerGPEM
    from hnm_loss import HNMLoss


class _HNM:
    """Envuelve la cadena del padre con la mineria de negativos duros."""

    def _build_loss(self):
        inner = super()._build_loss()
        return HNMLoss(inner,
                       weight=float(os.environ.get("HNM_W", 0.2)),
                       every=int(os.environ.get("HNM_EVERY", 2)),
                       min_px=int(os.environ.get("HNM_MIN_PX", 2000)),
                       warmup=int(os.environ.get("HNM_WARMUP", 2500)))


class nnUNetTrainerHNM(_HNM, nnUNetTrainerGPEM):
    """GPEM + mineria de negativos duros sobre los falsos positivos."""
