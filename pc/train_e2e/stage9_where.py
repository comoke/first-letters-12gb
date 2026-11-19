"""Where do the stage-9 claim-image differences sit? For a variant PNG (ours vs the package's): histogram of the per-pixel max channel
difference, the count of pixels above 8 / 64 / 128, their bounding box, and how many of them form connected blobs (cv2) with the
largest blob sizes; also the same counts restricted to pixels where the package image is not the page background.
Usage (in fl-claim-a7:final): python stage9_where.py <ref.png> <ours.png> <out.json>
"""
import json, sys
import numpy as np, cv2
a = cv2.imread(sys.argv[1]).astype(np.int16); b = cv2.imread(sys.argv[2]).astype(np.int16); OUT = sys.argv[3]
d = np.abs(a - b).max(-1)
res = dict(shape=list(d.shape), hist={str(t): int((d > t).sum()) for t in (0, 1, 2, 4, 8, 16, 32, 64, 128, 200)})
for t in (8, 64):
    m = (d > t).astype(np.uint8); ys, xs = np.nonzero(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    sizes = sorted(st[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    res[f'gt{t}'] = dict(px=int(m.sum()), bbox=[int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None,
                         blobs=int(n - 1), largest_blobs=sizes[:10], blobs_of_1_px=int(sum(s == 1 for s in sizes)))
json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(res))
