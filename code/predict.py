
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from skimage.transform import resize

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

from benchmark import nnUNetTrainerAccept
from gpem import nnUNetTrainerGPEM
from tiling import forward
from otros.postproc import postprocess

TRAINERS = {"gpem": nnUNetTrainerGPEM, "accept": nnUNetTrainerAccept}

def load(ckpt, plans_path, dataset_json, device, trainer_name="gpem"):
    plans = dict(json.load(open(plans_path)), continue_training=False)
    trainer = TRAINERS[trainer_name](plans, "2d", 0, dataset_json, device=device)
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

def to_full_res(votes, shape):
    """Vote fractions (network resolution) -> class map at the original resolution."""
    v = torch.from_numpy(votes).float().permute(2, 0, 1)[None]
    v = F.interpolate(v / v.sum(1, keepdim=True).clamp(min=1), size=shape, mode="bilinear", align_corners=False)
    return v[0].argmax(0).numpy().astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer", choices=list(TRAINERS), default="gpem")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--no-close", action="store_true", help="main pipeline (no ring closure)")
    ap.add_argument("--save-sem", action="store_true", help="also write <case>_sem.tif / _score.npy")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, tile_hw, feat, plans = load(args.checkpoint, args.plans,
                                     json.load(open(args.dataset_json)), device, args.trainer)

    stats = plans["foreground_intensity_properties_per_channel"]["0"]
    target_hw = [int(v) for v in plans["configurations"]["2d"]["median_image_size_in_voxels"]]
    n_in = getattr(net, "num_input_channels", 1)

    for f in sorted(Path(args.images).glob("*.tif")):
        raw = tifffile.imread(f).astype(np.float32)
        img = (resize(raw, tuple(target_hw), order=1, preserve_range=True) - stats["mean"]) / stats["std"]

        _, score_q, votes = forward(net, img.astype(np.float32), tile_hw, feat, n_in, return_votes=True)
        sem = to_full_res(votes, raw.shape)
        score = F.interpolate(torch.from_numpy(score_q)[None, None], size=sem.shape,
                              mode="bilinear", align_corners=False)[0, 0].numpy()
        if args.save_sem:   # lets analysis/eval_postproc.py re-run post-processing without the network
            (out / "sem").mkdir(exist_ok=True)
            tifffile.imwrite(out / "sem" / f"{f.stem}_sem.tif", sem, compression="zlib")
            np.save(out / "sem" / f"{f.stem}_score.npy", score_q.astype(np.float16))

        inst = postprocess(sem, score, args.tau, close=not args.no_close)
        tifffile.imwrite(out / f.name, inst.astype(np.uint16))

    print(f"wrote {len(list(out.glob('*.tif')))} predictions to {out}")

if __name__ == "__main__":
    main()
