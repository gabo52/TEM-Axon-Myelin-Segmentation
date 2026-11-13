"""Cuarta propuesta: GPEM con la entrada enriquecida (12 canales).

Misma imagen, mas preprocesados como entrada (ver geom2.py): mas escalas del
perfil radial, gradiente y orientacion local. Fusion temprana en la misma
pasada, sin circularidad y sin coste en disco.

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerGPEM15
"""
try:
    from .geom2 import maps as maps2
    from .gpem import nnUNetTrainerGPEM
except ImportError:
    from geom2 import maps as maps2
    from gpem import nnUNetTrainerGPEM


class nnUNetTrainerGPEM15(nnUNetTrainerGPEM):
    """GPEM con 12 canales de prior geometrico en vez de 7."""

    geom_maps = staticmethod(maps2)
