"""eval_map.py <segment dir> <ckpt|base> <out.npy> [--reverse] [--roi-tiles tiles.json]
A-hn map of a Paris4 segment prepared by fetch_paris4.py (layers/00..23.tif, labels/mask.png), with the winner's inference code
(infer.load_model + infer.predict_array: TTA 1, nsig 3, stride 21, min_cov 0.25), the protocol's recipe
(docs/finetune-protocol-2026-11-17.md). Only tiles overlapping the supervision mask (or the listed 512 px eval tiles) are run.
--reverse flips the stored 00..23 order before the model (the same switch infer.py has); which order is right is decided once,
on segment A, by the zero-shot layer-order check, and then used for every arm.
Runs in fl-claim-a7:final with the GPU; writes <out.npy> and <out>.json (recipe, timing, shapes).
"""
import argparse, json, os, sys, time
import numpy as np, tifffile
from PIL import Image
B = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'
sys.path.insert(0, B); os.chdir(B)
import torch, infer
ap = argparse.ArgumentParser(); ap.add_argument('seg'); ap.add_argument('ckpt'); ap.add_argument('out')
ap.add_argument('--reverse', action='store_true'); ap.add_argument('--roi-tiles'); ap.add_argument('--vram', type=float, default=9.0)
a = ap.parse_args(); T0 = time.time()
if os.path.exists(a.out) and os.path.exists(a.out[:-4] + '.json'): print('done already', a.out); sys.exit(0)
vol = np.stack([tifffile.imread(f'{a.seg}/layers/{i:02d}.tif') for i in range(24)]).astype(np.uint8)
mask = np.asarray(Image.open(f'{a.seg}/labels/mask.png')) > 0
roi = mask
if a.roi_tiles:
    t = json.load(open(a.roi_tiles)); S = t['tile']; roi = np.zeros_like(mask)
    for y, x in t['tiles']: roi[y:y + S, x:x + S] = True
    roi &= mask
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
infer.vram_cap(a.vram); model, info = infer.load_model(a.ckpt, 1.0, 'cuda')
# Row blocks: the whole crop doesn't fit on a 12 GB card at once. Each block's core rows get a TILE (64 px) margin above and below,
# so every tile that touches a core row lies inside the block, and the grid stays the global stride-21 grid (origin shifted by the
# block start). Tested on a 640 x 594 crop, blocks of 150 vs one pass: same finite px, 99.78% bit-equal, max diff 2.4e-4 (one
# float16 step; fp16 autocast batches differ).
H = vol.shape[1]; BLK, M = int(os.environ.get('EVAL_BLK', 1536)), infer.TILE; p = np.full(vol.shape[1:], np.nan, np.float32); cov = vol[12] > 0
for c0 in range(0, H, BLK):
    c1 = min(H, c0 + BLK); r0, r1 = max(0, c0 - M), min(H, c1 + M)
    if not roi[c0:c1].any(): continue
    rm = roi[r0:r1]   # mask over the whole block: a tile touching the core with mask only in the margin is run, as in a full pass
    q = infer.predict_array(model, np.ascontiguousarray(vol[:, r0:r1]), cov[r0:r1], reverse=a.reverse, tta=1, nsig=3, min_cov=0.25,
                            stride=21, origin=((-r0) % 21, 0), roi_mask=rm, batch=8)
    p[c0:c1] = q[c0 - r0:c1 - r0]; torch.cuda.empty_cache()
    print('block', c0, c1, 'finite', int(np.isfinite(q[c0 - r0:c1 - r0]).sum()), round(time.time() - T0), 's', flush=True)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); np.save(a.out, p.astype(np.float16))
json.dump(dict(seg=a.seg, ckpt=a.ckpt, reverse=a.reverse, roi_tiles=a.roi_tiles, shape=list(p.shape), finite=int(np.isfinite(p).sum()),
               roi_px=int(roi.sum()), seconds=round(time.time() - T0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               recipe='infer.predict_array tta 1 nsig 3 stride 21 min_cov 0.25, roi = supervision mask (or eval tiles)'),
          open(a.out[:-4] + '.json', 'w'), indent=1)
print('EVAL_MAP_DONE', a.out, p.shape, round(time.time() - T0), 's')
