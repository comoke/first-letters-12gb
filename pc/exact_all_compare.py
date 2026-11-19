"""Exact-recipe inference vs the shipped maps, for every map of the claim (TRACE, A-hn s0/s1/s2, decoy, ensA, prod, prodd).
Shipped: hf-data results/member_maps/fwd_trace_v3B.npy + fwd_hn4_*.npy and results/fused/fwd_{ensA,prod,prodd}.npy.
Per map, over pixels finite in both: Pearson r, mean / max abs diff, share within 1e-3, bit-equal share, on the whole raster and
inside the claim box (config_v7.json 'box', run.sh BOXPX formula). Each entry records both file paths.
Usage (in fl-claim-a7:final): python exact_all_compare.py <our dir with fwd_tr / fwd_hn4_* and fused/> <out.json>
"""
import json, math, os, sys
import numpy as np
O, OUT = sys.argv[1], sys.argv[2]
S = '/data/winner/hf-data/results'
pairs = {'tr': (f'{O}/fwd_tr.npy', f'{S}/member_maps/fwd_trace_v3B.npy')}
for m in ('s0', 's1', 's2', 'decoy'): pairs[m] = (f'{O}/fwd_hn4_{m}.npy', f'{S}/member_maps/fwd_hn4_{m}.npy')
for f in ('ensA', 'prod', 'prodd'): pairs[f] = (f'{O}/fused/fwd_{f}.npy', f'{S}/fused/fwd_{f}.npy')
b = json.load(open('/opt/fl/claim/config_v7.json'))['box']; K = 1000 / 8.64
BOX = (math.ceil((b['v0'] + 21) * K - 0.5), math.floor((b['v1'] + 21) * K - 0.5) + 1, math.ceil((b['u0'] + 36) * K - 0.5), math.floor((b['u1'] + 36) * K - 0.5) + 1)


def full(p):
    a = np.load(p).astype(np.float32); j = p[:-4] + '.json'
    if os.path.exists(j):                                    # infer.py --roi maps are frame-sized: embed in the render raster
        jj = json.load(open(j)); Hh, Ww = jj.get('render_shape', [0, 4900, 7780])[1:]
        if a.shape != (Hh, Ww):
            box = next((bb for bb in (jj.get('roi'), jj.get('frame')) if bb and a.shape == (bb[1] - bb[0], bb[3] - bb[2])), None)
            if box: o = np.full((Hh, Ww), np.nan, np.float32); o[box[0]:box[1], box[2]:box[3]] = a; a = o
    return a


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b); d = np.abs(a[m].astype(np.float64) - b[m])
    return dict(n=int(m.sum()), pearson=float(np.corrcoef(a[m], b[m])[0, 1]), mean_abs=float(d.mean()), max_abs=float(d.max()),
                within_1e3=float((d <= 1e-3).mean()), bit_equal=float((a[m] == b[m]).mean()))


res = dict(our_dir=O, claim_box_render_px=BOX, maps={})
for k, (po, ps) in pairs.items():
    if not (os.path.exists(po) and os.path.exists(ps)): res['maps'][k] = dict(missing=[p for p in (po, ps) if not os.path.exists(p)]); continue
    a, s = full(po), full(ps)
    if a.shape != s.shape: res['maps'][k] = dict(shape_ours=list(a.shape), shape_shipped=list(s.shape)); continue
    y0, y1, x0, x1 = BOX
    res['maps'][k] = dict(whole=stats(a, s), claim_box=stats(a[y0:y1, x0:x1], s[y0:y1, x0:x1]), files=[po, ps])
json.dump(res, open(OUT, 'w'), indent=1)
print(f"{'map':6} {'r':>10} {'mean|d|':>9} {'max|d|':>9} {'<=1e-3':>8} {'box r':>10} {'box max':>9}")
for k, v in res['maps'].items():
    if 'whole' in v:
        w, bx = v['whole'], v['claim_box']
        print(f"{k:6} {w['pearson']:10.7f} {w['mean_abs']:9.2e} {w['max_abs']:9.2e} {w['within_1e3']:8.5f} {bx['pearson']:10.7f} {bx['max_abs']:9.2e}")
    else: print(k, v)
