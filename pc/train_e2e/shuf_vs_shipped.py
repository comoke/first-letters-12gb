"""Our depth-shuffle maps vs the winner's shipped ones (hf-data results/member_maps/shuf_*, results/fused/shuf_*).
The shipped shuffle maps come from the SHIPPED checkpoints on the winner's shuffled render (their shuffle_control.py prep: layers 8..31,
seed 1, box + 256 px), member recipe roi = the claim box, clip_to_roi false (shuf_hn4_s0.json). We rebuilt that render with the same
code and ran the same checkpoints and recipe (<ours dir>), so the maps should agree to float16 precision if the inference reproduces.
Inside the claim box (render px 1639-2745 x 2431-6829), pixels finite in both: Pearson r, max and mean abs diff, share within 1e-3.
Maps may be full-raster (4900 x 7780) or box-sized; both are handled.
Usage (in fl-claim-a7:final): python shuf_vs_shipped.py <ours dir> <out.json>
"""
import json, sys
import numpy as np
OURS, OUT = sys.argv[1:3]; y0, y1, x0, x1 = 1639, 2745, 2431, 6829; R = '/data/winner/hf-data/results'
def box(p):
    a = np.load(p).astype(np.float64)
    if a.shape == (4900, 7780): return a[y0:y1, x0:x1]
    if a.shape == (y1 - y0, x1 - x0): return a
    raise SystemExit(f'{p}: unexpected shape {a.shape}')
pairs = {'trace': (f'{OURS}/fwd_tr.npy', f'{R}/member_maps/shuf_trace_v3B.npy'),
         's0': (f'{OURS}/fwd_hn4_s0.npy', f'{R}/member_maps/shuf_hn4_s0.npy'),
         's1': (f'{OURS}/fwd_hn4_s1.npy', f'{R}/member_maps/shuf_hn4_s1.npy'),
         's2': (f'{OURS}/fwd_hn4_s2.npy', f'{R}/member_maps/shuf_hn4_s2.npy'),
         'ensA': (f'{OURS}/fused/fwd_ensA.npy', f'{R}/fused/shuf_ensA.npy'),
         'prod': (f'{OURS}/fused/fwd_prod.npy', f'{R}/fused/shuf_prod.npy')}
res = dict(box_px=[y0, y1, x0, x1], maps={})
for k, (o, s) in pairs.items():
    a, b = box(o), box(s); m = np.isfinite(a) & np.isfinite(b); d = np.abs(a[m] - b[m])
    res['maps'][k] = dict(px=int(m.sum()), r=round(float(np.corrcoef(a[m], b[m])[0, 1]), 8), max_abs=float(d.max()), mean_abs=float(d.mean()),
                          within_1e3=round(float((d <= 1e-3).mean()), 7), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()))
    print(k, res['maps'][k], flush=True)
res['min_r'] = min(v['r'] for v in res['maps'].values()); res['max_abs'] = max(v['max_abs'] for v in res['maps'].values())
json.dump(res, open(OUT, 'w'), indent=1); print('min r', res['min_r'], 'max abs', res['max_abs'])
