"""Reference pairs for the top-1% Jaccard of the A-hn members (a yardstick for ensA's 0.703): how much do top-pixel sets move between maps that
differ only by training variation? In the claim box (render px 1639:2745, 2431:6829), finite px of both maps:
  seeds      the winner's shipped s0 / s1 / s2 member maps against each other (seed-to-seed spread)
  retrain    our first single-member retrain from the shipped packs (the same-recipe bar, r 0.99958) against the shipped s0
  e2e        our end-to-end members against their shipped counterparts
Each pair: Pearson r and top-1% / top-5% Jaccard. ROI-sized maps are placed into the render raster with their sidecar JSON.
Usage (in fl-claim-a7:final, data at /data): python topk_ref.py <out.json>
"""
import json, sys
import numpy as np
B = (1639, 2745, 2431, 6829); H, W = 4900, 7780
SH = '/data/winner/hf-data/results/member_maps/fwd_hn4_{}.npy'; E2E = '/data/train_e2e/maps/fwd_hn4_{}.npy'
RT = '/data/retrain/eval/fwd_hn4_s0_pc.npy'


def load(p):
    a = np.load(p, mmap_mode='r')
    if a.shape != (H, W):
        j = json.load(open(p[:-4] + '.json'))
        box = next(b for b in (j.get('roi'), j.get('frame')) if b and a.shape == (b[1] - b[0], b[3] - b[2]))
        full = np.full((H, W), np.nan, np.float32); full[box[0]:box[1], box[2]:box[3]] = a; a = full
    return np.asarray(a[B[0]:B[1], B[2]:B[3]], np.float32)


def pair(pa, pb):
    a, b = load(pa), load(pb); m = np.isfinite(a) & np.isfinite(b); r = dict(px=int(m.sum()), r=round(float(np.corrcoef(a[m], b[m])[0, 1]), 5))
    for q in (1, 5):
        ta = (a >= np.percentile(a[m], 100 - q)) & m; tb = (b >= np.percentile(b[m], 100 - q)) & m
        r[f'top{q}_jaccard'] = round(float((ta & tb).sum() / (ta | tb).sum()), 4)
    return r


out = dict(box_render_px=list(B), seeds={}, retrain={}, e2e={})
for x, y in (('s0', 's1'), ('s0', 's2'), ('s1', 's2')): out['seeds'][f'{x}~{y}'] = pair(SH.format(x), SH.format(y))
out['retrain']['s0_pc~s0'] = pair(RT, SH.format('s0'))
for s in ('s0', 's1', 's2', 'decoy'): out['e2e'][f'{s}~{s}'] = pair(E2E.format(s), SH.format(s))
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out))
