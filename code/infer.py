"""Carga de modelos y prediccion de una imagen, con soporte de ensemble.

El ensemble de nnU-Net NO fusiona los pesos en un unico .pth: los 5 folds se
entrenan de forma independiente, asi que sus pesos viven en zonas distintas del
espacio de soluciones y promediarlos daria basura. Lo que se promedia son las
**probabilidades de salida**: los 5 modelos opinan sobre la misma imagen y se
promedian antes de decidir. Los votos se suben a la resolucion NATIVA
antes del argmax, asi el watershed afina la frontera de las fibras.
"""
import numpy as np
import torch
import tifffile
from skimage.transform import resize
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from benchmark import AcceptHead, materialize
from geom import maps as maps7
from geom2 import maps as maps12
from gpem import install
from otros.postproc import postprocess
from tiling import forward


def load_model(ckpt, plans_path, dataset_json, device):
    """Construye la red desde plans.json y carga los pesos.

    GPEM y DistGPEM comparten red (la perdida de distancia solo actua en
    entrenamiento), asi que basta mirar si el checkpoint trae `geo_fusion` y
    cuantos canales tiene la proyeccion para elegir los mapas de entrada.
    """
    sd = torch.load(ckpt, map_location="cpu")["network_weights"]
    geo = [k for k in sd if k.endswith("proj.weight")]
    n_geom = sd[geo[0]].shape[1] if geo else 0
    cm = PlansManager(plans_path).get_configuration("2d")
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import, input_channels=1,
                                 output_channels=len([k for k in dataset_json["labels"] if k != "ignore"]),
                                 allow_init=True, deep_supervision=True)
    hw = tuple(max(int(p) // 2, 64) for p in cm.patch_size)
    net.accept_head = AcceptHead()
    materialize(net, 1, hw)
    if n_geom:
        install(net, hw, maps12 if n_geom > 7 else maps7)
    net = net.to(device).eval()
    feat = {}
    net.decoder.stages[-1].register_forward_hook(lambda m, i, o: feat.__setitem__("x", o))
    net.load_state_dict(sd)
    return net, feat, (int(cm.patch_size[0]), int(cm.patch_size[1]))


def predict(path, models, plans, tau, device, close=True):
    """Instancias de una imagen, promediando los modelos que se le pasen."""
    raw = tifffile.imread(path)
    out_hw = raw.shape[:2]
    st = plans["foreground_intensity_properties_per_channel"]["0"]
    hw = [int(v) for v in plans["configurations"]["2d"]["median_image_size_in_voxels"]]
    img = resize(raw.astype(np.float32), tuple(hw), order=1, preserve_range=True)
    img = (img - st["mean"]) / st["std"]

    acc = score = None
    for net, feat, tile_hw in models:
        pr, sc = forward(net, img, tile_hw, feat, 1)
        acc = pr if acc is None else acc + pr
        score = sc if score is None else score + sc
    probs, score = acc / len(models), score / len(models)

    if probs.shape[1:] != tuple(out_hw):         # votos a resolucion NATIVA
        probs = np.stack([resize(p, out_hw, order=1, preserve_range=True) for p in probs])
        score = resize(score, out_hw, order=1, preserve_range=True)
    sem = probs.argmax(0).astype(np.uint8)
    return postprocess(sem, score, tau, close), sem, score
