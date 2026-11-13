"""Tercera propuesta: supervisar el artefacto como FONDO en vez de ignorarlo.

El challenge excluye el artefacto (label 3) de la metrica, asi que en el
entrenamiento se puso como *ignore*: la red no recibe gradiente ahi y es libre
de predecir fibras dentro de esas regiones.

El diagnostico dice que eso cuesta caro: de las 1128 fibras que el oraculo borra,
**872 (77 %) caen en region de artefacto**, y sus bordes se salen a la region
valida, que es lo que domina el HD95 global.

Aqui se remapea label 3 -> 0 (fondo) antes de calcular la perdida, de modo que
la red recibe la senal de "aqui no hay fibra". La metrica sigue excluyendo el
artefacto: solo cambia el entrenamiento.

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerArtBG
"""
import torch
from torch import nn

try:
    from .gpem import nnUNetTrainerGPEM
except ImportError:
    from gpem import nnUNetTrainerGPEM

ARTIFACT = 3


class ArtBGLoss(nn.Module):
    """Remapea el artefacto a fondo y delega en la perdida de dentro."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @staticmethod
    def _remap(x):
        if not torch.is_tensor(x):
            return x
        return torch.where(x == ARTIFACT, torch.zeros_like(x), x)

    def forward(self, out, target):
        if isinstance(target, list):
            target = [self._remap(t) for t in target]
        else:
            target = self._remap(target)
        return self.inner(out, target)


class nnUNetTrainerArtBG(nnUNetTrainerGPEM):
    """GPEM + el artefacto supervisado como fondo."""

    def _build_loss(self):
        return ArtBGLoss(super()._build_loss())
