"""Export smoke: the map from the exported job script against the reference 4070 map, as ahn_gate.py compares hn4_s0 native."""
import json, sys
import numpy as np
B0, NEW, GATE = sys.argv[1:4]
meta = json.load(open(f'{B0}/b0.json')); nc = meta['native_crop']
def on_raster(p):
    a = np.load(p, mmap_mode='r')
    if a.shape == tuple(meta['render_shape'][-2:]): return np.asarray(a[nc[0]:nc[1], nc[2]:nc[3]], dtype=np.float32)
    y0, y1, x0, x1 = json.load(open(p[:-4] + '.json'))['roi']
    out = np.full((nc[1] - nc[0], nc[3] - nc[2]), np.nan, np.float32); out[y0 - nc[0]:y1 - nc[0], x0 - nc[2]:x1 - nc[2]] = a; return out
old, new = on_raster(f'{B0}/runs/native/fwd_hn4_s0.npy'), on_raster(NEW)
fo, fn = np.isfinite(old), np.isfinite(new); k = fo & fn
r = float(np.corrcoef(old[k].astype(np.float64), new[k].astype(np.float64))[0, 1])
g = json.load(open(GATE))['members']['native/hn4_s0']
res = dict(r_vs_reference=r, gate_r_vs_reference=g['r'], r_diff_from_gate=abs(r - g['r']), max_abs=float(np.abs(old[k] - new[k]).max()),
           nan_mismatch=int((fo != fn).sum()), px=int(k.sum()), gate_px=g['px'])
res['pass'] = bool(r >= 0.99999 and res['nan_mismatch'] == 0 and res['px'] == g['px'])
print('SMOKE_CMP', json.dumps(res)); sys.exit(0 if res['pass'] else 1)
