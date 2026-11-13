"""Sliding-window inference: returns the class probabilities and the accept score."""
import numpy as np
import torch


def starts(size, tile, step):
    """Top-left coordinates covering `size` with `tile`, stride `step`."""
    if size <= tile:
        return [0]
    out = list(range(0, size - tile + 1, step))
    if out[-1] != size - tile:
        out.append(size - tile)
    return out


def forward(net, img, tile_hw, feat, n_in=1):
    """Run the network over the image.

    Returns
    -------
    probs : float32 [3, H, W]    mean class probability (for the ensemble)
    score : float32 [H/4, W/4]   mean accept score
    """
    h, w = img.shape
    th, tw = tile_hw
    acc = np.zeros((3, h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    score = np.zeros((h // 4, w // 4), np.float32)
    scnt = np.zeros((h // 4, w // 4), np.float32)

    for y0 in starts(h, th, th // 2):
        for x0 in starts(w, tw, tw // 2):
            crop = img[y0:y0 + th, x0:x0 + tw]
            x = torch.from_numpy(crop)[None, None].to(next(net.parameters()).device)
            if n_in > 1:
                x = x.repeat(1, n_in, 1, 1)
            with torch.no_grad():
                out = net(x)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                p = logits.float().softmax(1)
                sc = torch.sigmoid(net.accept_head(torch.cat([feat["x"], p[:, 1:3]], dim=1)))
            acc[:, y0:y0 + th, x0:x0 + tw] += p[0].cpu().numpy()
            cnt[y0:y0 + th, x0:x0 + tw] += 1
            ay, ax = y0 // 4, x0 // 4
            ah, aw = sc.shape[-2:]
            score[ay:ay + ah, ax:ax + aw] += sc[0, 0].cpu().numpy()
            scnt[ay:ay + ah, ax:ax + aw] += 1

    return acc / np.maximum(cnt, 1)[None], score / np.maximum(scnt, 1)
