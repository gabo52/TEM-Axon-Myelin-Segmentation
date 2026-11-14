import torch
import torch.nn.functional as F
from torch import nn

try:
    from .accept_trainer import nnUNetTrainerAccept
    from .maps7 import N_MAPS, maps
except ImportError:
    from accept_trainer import nnUNetTrainerAccept
    from maps7 import N_MAPS, maps

LAST_STAGES = 3


class GeoFusion(nn.Module):
    def __init__(self, channels, n_maps=N_MAPS, width=16):
        super().__init__()
        self.proj = nn.Conv2d(n_maps, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * channels, width, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, channels, 1))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, g):
        gp = self.proj(g)
        a = torch.sigmoid(self.gate(torch.cat([h, gp], dim=1)))
        return h + a * gp


def _hook(net, n):
    def hook(module, inp, out):
        g = getattr(net, "_geom", None)
        if g is None or not torch.is_tensor(out):
            return out
        gp = F.interpolate(g, size=out.shape[-2:], mode="bilinear", align_corners=False)
        return net.geo_fusion[n](out, gp)
    return hook


def install(net, hw, maps_fn=maps):
    stages = list(net.encoder.stages)
    idx = list(range(len(stages)))[-LAST_STAGES:]
    info = {}
    hs = [stages[i].register_forward_hook(
        (lambda i: lambda m, a, o: info.__setitem__(i, o.shape[1]))(i)) for i in idx]
    with torch.no_grad():
        net(torch.zeros(1, 1, hw[0], hw[1]))
    for h in hs:
        h.remove()

    n_maps = maps_fn(torch.zeros(1, 1, 64, 64)).shape[1]   # actual channels
    net.geo_fusion = nn.ModuleList([GeoFusion(info[i], n_maps) for i in idx])
    for n, i in enumerate(idx):
        stages[i].register_forward_hook(_hook(net, n))

    orig = net.forward

    def forward(x, *a, **k):
        net._geom = maps_fn(x[:, :1]).to(x.dtype)
        return orig(x, *a, **k)

    net.forward = forward


class nnUNetTrainerGPEM(nnUNetTrainerAccept):
    #nnUNetTrainerGPEM (7-channel prior): the base trainer + prior fused in the encoder.

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        net = nnUNetTrainerAccept.build_network_architecture(plans_manager, configuration_manager,
                                                             num_input_channels, num_output_channels,
                                                             enable_deep_supervision)
        install(net, tuple(max(p // 2, 64) for p in configuration_manager.patch_size), maps)
        return net
