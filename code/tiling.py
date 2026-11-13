import numpy as np
import torch

def starts(size, tile, step):
    if size <= tile:
        return [0]
    out = list(range(0, size - tile + 1, step))
    if out[-1] != size - tile:
        out.append(size - tile)
    return out

def forward(net, img, tile_hw, feat, n_in=1, return_votes=False):
    """Run the network over the image.

    Returns
    -------
    sem   : uint8 [H, W] class map (argmax of accumulated votes)
    score : float32 [H/4, W/4] mean accept score
    votes : int16 [H, W, 3] per-class vote counts (only if return_votes)
    """
    h, w = img.shape
    th, tw = tile_hw
    votes = np.zeros((h, w, 3), np.int16)
    score = np.zeros((h // 4, w // 4), np.float32)
    count = np.zeros((h // 4, w // 4), np.float32)

    for y0 in starts(h, th, th // 2):
        for x0 in starts(w, tw, tw // 2):
            crop = img[y0:y0 + th, x0:x0 + tw]
            x = torch.from_numpy(crop)[None, None]
            x = x.to(next(net.parameters()).device)
            if n_in > 1:
                x = x.repeat(1, n_in, 1, 1)
            with torch.no_grad():
                out = net(x)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                probs = logits.float().softmax(1)
                pred = logits[0].argmax(0).cpu().numpy().astype(np.uint8)
                sc = torch.sigmoid(net.accept_head(torch.cat([feat["x"], probs[:, 1:3]], dim=1)))
            votes[y0:y0 + th, x0:x0 + tw] += np.eye(3, dtype=np.int16)[pred]
            ay, ax = y0 // 4, x0 // 4
            ah, aw = sc.shape[-2:]
            score[ay:ay + ah, ax:ax + aw] += sc[0, 0].cpu().numpy()
            count[ay:ay + ah, ax:ax + aw] += 1

    score = score / np.maximum(count, 1)
    if return_votes:
        return votes.argmax(-1).astype(np.uint8), score, votes
    return votes.argmax(-1).astype(np.uint8), score
