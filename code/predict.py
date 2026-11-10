
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tifffile
from skimage.transform import resize

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

from benchmark import nnUNetTrainerAccept   
from tiling import forward                   
from otros.accept_filter import drop        
from otros.instances import to_instances      

def load(ckpt, plans_path, dataset_json, device):
    plans = dict(json.load(open(plans_path)), continue_training=False)
    trainer = nnUNetTrainerAccept(plans, "2d", 0, dataset_json, device=device)
    trainer.initialize()
    net = trainer.network.eval()
    th, tw = [int(v) for v in trainer.configuration_manager.patch_size]

    feat = {}
    net.decoder.stages[-1].register_forward_hook(lambda m, i, o: feat.__setitem__("x", o))
    with torch.no_grad():   
        out = net(torch.zeros(2, 1, th, tw).to(next(net.parameters()).device))
        logits = out[0] if isinstance(out, (list, tuple)) else out
        net.accept_head(torch.cat([feat["x"], logits.float().softmax(1)[:, 1:3]], dim=1))

    net.load_state_dict(torch.load(ckpt, map_location=device)["network_weights"])
    return net.eval(), (th, tw), feat, plans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, tile_hw, feat, plans = load(args.checkpoint, args.plans,
                                     json.load(open(args.dataset_json)), device)

    stats = plans["foreground_intensity_properties_per_channel"]["0"]
    target_hw = [int(v) for v in plans["configurations"]["2d"]["median_image_size_in_voxels"]]
    n_in = getattr(net, "num_input_channels", 1)

    for f in sorted(Path(args.images).glob("*.tif")):
        img = tifffile.imread(f).astype(np.float32)
        img = (resize(img, tuple(target_hw), order=1, preserve_range=True) - stats["mean"]) / stats["std"]

        sem, score = forward(net, img, tile_hw, feat, n_in)
        if sem.shape[0] < target_hw[0]:
            sem = sem.repeat(2, 0).repeat(2, 1)
        inst = to_instances(sem)

        if score.shape != inst.shape:
            score = torch.nn.functional.interpolate(
                torch.from_numpy(score)[None, None], size=inst.shape,
                mode="bilinear", align_corners=False)[0, 0].numpy()

        tifffile.imwrite(out / f.name, drop(inst, score, args.tau).astype(np.uint16))

    print(f"wrote {len(list(out.glob('*.tif')))} predictions to {out}")

if __name__ == "__main__":
    main()
