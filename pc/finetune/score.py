"""score.py <segment dir> <out.json> name=map.npy [name=map.npy ...]
Protocol metrics (docs/finetune-protocol-2026-11-17.md) for each arm's map against the segment's labels, inside its supervision mask
(and only where the map is finite): pixel AUC, top-1% precision (share of the top 1% scores labelled ink), F0.5 at 0.5.
Paired bootstrap over 512 px tiles of the mask (1,000 resamples, seed 0) for the AUC difference of every pair of arms: mean, 95% CI.
CPU only.
"""
import json, sys
import numpy as np
from PIL import Image
SEG, OUT = sys.argv[1:3]; arms = dict(a.split('=', 1) for a in sys.argv[3:])
lab = np.asarray(Image.open(f'{SEG}/labels/inklabels.png')) > 127
mask = np.asarray(Image.open(f'{SEG}/labels/mask.png')) > 0
maps = {k: np.load(v).astype(np.float32) for k, v in arms.items()}
valid = mask.copy()
for m in maps.values(): valid &= np.isfinite(m)
S = 512; H, W = valid.shape
tiles = [(y, x) for y in range(0, H, S) for x in range(0, W, S) if valid[y:y + S, x:x + S].sum() > 0.1 * S * S]


def auc(s, y):
    """rank AUC (Mann-Whitney) with tied scores given their average rank (vectorised)"""
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0: return float('nan')
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]]); avg = start + (cnt + 1) / 2.0
    return float((avg[inv][y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(m, sel):
    s = m[sel]; y = lab[sel]; k = max(1, int(0.01 * s.size)); top = np.argpartition(-s, k - 1)[:k]
    pr = y[s >= 0.5].mean() if (s >= 0.5).any() else 0.0; rc = (y & (s >= 0.5)).sum() / max(1, y.sum()); b2 = 0.25
    f05 = (1 + b2) * pr * rc / max(1e-9, b2 * pr + rc)
    return dict(auc=auc(s, y), top1_precision=float(y[top].mean()), f05_at_0p5=float(f05), px=int(s.size), ink_share=float(y.mean()))


# per-tile pixel lists for the bootstrap (subsampled to at most 4,000 px per tile; fixed seed); point metrics use every valid px
rng = np.random.default_rng(0); idx = []
for y, x in tiles:
    t = np.flatnonzero(valid[y:y + S, x:x + S].ravel())
    if t.size > 4000: t = np.sort(rng.choice(t, 4000, replace=False))
    yy, xx = np.unravel_index(t, (min(S, H - y), min(S, W - x))); idx.append((yy + y, xx + x))
res = dict(segment=SEG, tiles=len(tiles), tile_px=S, arms={k: metrics(m, valid) for k, m in maps.items()}, pairs={})
names = list(maps)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a_, b_ = names[i], names[j]; diffs = []
        for _ in range(1000):
            pick = rng.integers(0, len(tiles), len(tiles)); Y = np.concatenate([idx[t][0] for t in pick]); X = np.concatenate([idx[t][1] for t in pick])
            yl = lab[Y, X]; diffs.append(auc(maps[a_][Y, X], yl) - auc(maps[b_][Y, X], yl))
        d = np.array(diffs); res['pairs'][f'{a_}-{b_}'] = dict(auc_diff=res['arms'][a_]['auc'] - res['arms'][b_]['auc'],
                                                             ci95=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(res['arms'], indent=1))
