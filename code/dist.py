"""Experimentos con la perdida de distancia (ver dist_loss.py).

  nnUNetTrainerDist      -> benchmark + distancia   (aisla la distancia)
  nnUNetTrainerDistGPEM  -> GPEM      + distancia   (combina con el mejor actual)

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerDist
Env:   EPOCHS=150  DIST_W=0.2  DIST_EVERY=2  DIST_TAU=70  DIST_CAP=500
"""
import os

try:
    from .benchmark import nnUNetTrainerAccept
    from .dist_loss import DistLoss
    from .gpem import nnUNetTrainerGPEM
except ImportError:
    from benchmark import nnUNetTrainerAccept
    from dist_loss import DistLoss
    from gpem import nnUNetTrainerGPEM


class _Dist:
    """Envuelve la cadena del padre con el termino de distancia."""

    def _build_loss(self):
        inner = super()._build_loss()
        return DistLoss(inner,
                        weight=float(os.environ.get("DIST_W", 0.2)),
                        every=int(os.environ.get("DIST_EVERY", 2)),
                        tau=float(os.environ.get("DIST_TAU", 70.0)),
                        cap=float(os.environ.get("DIST_CAP", 500.0)))


class nnUNetTrainerDist(_Dist, nnUNetTrainerAccept):
    """Benchmark + perdida de distancia."""


class nnUNetTrainerDistGPEM(_Dist, nnUNetTrainerGPEM):
    """GPEM + perdida de distancia."""
