"""ahn_gate.py : the A-hn exactness gate on B0.  New member maps (ahn_fast.py) against the current 4070 maps, then the controls.

  python ahn_gate.py B0_DIR NEW_DIR OUT.json        (numpy only; B0_DIR = vesuvius-data/b0, NEW_DIR holds native/ and notext/)

Gate, fixed before any run:
  1. each member (hn4_s0, s1, s2, decoy) on the native window and on the no-text window: Pearson r >= 0.9999 against the current
     4070 map (b0/runs/<variant>/fwd_<member>.npy), and the same pixels finite;
  2. fused maps rebuilt with fuse_v7.py's arithmetic (ensA = float16 mean of s0..s2, prod = TRACE x ensA, prodd = TRACE x decoy; the
     TRACE map is the current run's, untouched) and scored as pc/b0/b0_score.py scores them:
       no-text control: the share of product pixels at or above the native top-1 % threshold stays 0.0;
       decoy control: prod-prodd and ensA-decoy top-1 % Jaccard within 0.005 of the current run's, on both windows;
       native: letter AP and AUC against the shipped product map within 0.001 of the current run's.
Prints one line per check and AHN_GATE_PASS / AHN_GATE_FAIL.
"""
import json, os, sys
import numpy as np

B0, NEW, OUT = sys.argv[1:4]
meta = json.load(open(f'{B0}/b0.json')); nc = meta['native_crop']
MEM = ['hn4_s0', 'hn4_s1', 'hn4_s2', 'hn4_decoy']
res = dict(members={}, controls={}); ok = True


def on_raster(p, var):
    """A map on its window's raster.  Native maps are either the whole render (after pad_frame.py; TRACE's is) or the roi as
    infer.py wrote it (one row and column short of native_crop): both end on the native_crop raster, NaN where there is no map."""
    a = np.load(p, mmap_mode='r')
    if var != 'native': return np.asarray(a, dtype=np.float32)
    if a.shape == tuple(meta['render_shape'][-2:]): return np.asarray(a[nc[0]:nc[1], nc[2]:nc[3]], dtype=np.float32)
    y0, y1, x0, x1 = json.load(open(p[:-4] + '.json'))['roi']
    if a.shape != (y1 - y0, x1 - x0): raise SystemExit(f'{p}: shape {a.shape} is neither the render nor its roi')
    out = np.full((nc[1] - nc[0], nc[3] - nc[2]), np.nan, np.float32)
    out[y0 - nc[0]:y1 - nc[0], x0 - nc[2]:x1 - nc[2]] = a
    return out


crop_ref = place_new = on_raster


def top(a, q=0.01):
    t = np.nanquantile(a, 1 - q); return a >= t, float(t)


def jac(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    x, _ = top(np.where(m, a, np.nan)); y, _ = top(np.where(m, b, np.nan))
    return float((x & y).sum() / max((x | y).sum(), 1))


def ap_auc(score, pos):
    o = np.argsort(-score, kind='stable'); p = pos[o]; tp = np.cumsum(p); n = p.size; P = int(tp[-1])
    ap = float((tp[p] / (np.flatnonzero(p) + 1)).sum() / max(P, 1))
    ranks = np.empty(n); ranks[o] = np.arange(n, 0, -1)
    return ap, float((ranks[pos].sum() - P * (P + 1) / 2) / max(P * (n - P), 1))


def fused(maps, tr):
    acc = None
    for m in MEM[:3]: acc = maps[m].copy() if acc is None else acc + maps[m]
    ensA = (acc / 3).astype(np.float16)
    prod = (tr * ensA.astype(np.float32)).astype(np.float16).astype(np.float32)
    prodd = (tr * maps['hn4_decoy']).astype(np.float16).astype(np.float32)
    return ensA.astype(np.float32), prod, prodd


def score(var, maps, tr):
    ensA, prod, prodd = fused(maps, tr); s = {}
    s['decoy_jaccard_top1'] = jac(prod, prodd); s['ensA_decoy_jaccard_top1'] = jac(ensA, maps['hn4_decoy'])
    k = np.isfinite(tr) & np.isfinite(ensA); s['trace_ensA_r'] = float(np.corrcoef(tr[k], ensA[k])[0, 1])
    if var == 'native':
        ref = np.load(f'{B0}/ref_prod_native.npy').astype(np.float32); y0, y1, x0, x1 = meta['box_in_crop']
        rb, sb = ref[y0:y1, x0:x1], prod[y0:y1, x0:x1]; m = np.isfinite(rb) & np.isfinite(sb)
        s['letter_ap'], s['letter_auc'] = ap_auc(sb[m], rb[m] >= np.quantile(rb[m], 0.95))
        s['prod_top1_threshold'] = top(prod)[1]
    return s, prod, ensA


thr = {}
for var in ('native', 'notext'):
    if not os.path.exists(f'{NEW}/{var}/fwd_hn4_s0.npy'): print(var, 'not run: skipped'); continue
    tr = crop_ref(f'{B0}/runs/{var}/fwd_tr.npy', var); new, old = {}, {}
    for m in MEM:
        src = MEM[(MEM.index(m) + 1) % 4] if os.environ.get('GATE_SELFTEST_SWAP') else m     # self-test: every member swapped for the next, must fail
        old[m] = crop_ref(f'{B0}/runs/{var}/fwd_{m}.npy', var); new[m] = place_new(f'{NEW}/{var}/fwd_{src}.npy', var)
        fo, fn = np.isfinite(old[m]), np.isfinite(new[m]); k = fo & fn
        r = float(np.corrcoef(old[m][k].astype(np.float64), new[m][k].astype(np.float64))[0, 1])
        row = dict(r=r, max_abs=float(np.abs(old[m][k] - new[m][k]).max()), mean_abs=float(np.abs(old[m][k] - new[m][k]).mean()),
                   nan_mismatch=int((fo != fn).sum()), px=int(k.sum()))
        row['pass'] = bool(r >= 0.9999 and row['nan_mismatch'] == 0); ok &= row['pass']
        res['members'][f'{var}/{m}'] = row; print(var, m, row, flush=True)
    sn, prod_n, ensA_n = score(var, new, tr); so, prod_o, ensA_o = score(var, old, tr)
    k = np.isfinite(prod_n) & np.isfinite(prod_o)
    c = dict(new=sn, current=so, prod_r=float(np.corrcoef(prod_n[k].astype(np.float64), prod_o[k].astype(np.float64))[0, 1]),
             ensA_r=float(np.corrcoef(ensA_n[k].astype(np.float64), ensA_o[k].astype(np.float64))[0, 1]))
    c['pass'] = bool(abs(sn['decoy_jaccard_top1'] - so['decoy_jaccard_top1']) <= 0.005
                     and abs(sn['ensA_decoy_jaccard_top1'] - so['ensA_decoy_jaccard_top1']) <= 0.005)
    if var == 'native':
        thr = dict(new=sn['prod_top1_threshold'], current=so['prod_top1_threshold'])
        c['pass'] &= bool(abs(sn['letter_ap'] - so['letter_ap']) <= 0.001 and abs(sn['letter_auc'] - so['letter_auc']) <= 0.001)
    else:
        t_new = thr.get('new'); t_cur = thr.get('current') or json.load(open(f'{B0}/runs/native/b0_score.json'))['prod_top1_threshold']
        c['false_ink_frac_new_map_vs_current_native_threshold'] = float(np.nanmean(prod_n >= t_cur))
        if t_new is not None: c['false_ink_frac_new_map_vs_new_native_threshold'] = float(np.nanmean(prod_n >= t_new))
        c['false_ink_frac_current'] = float(np.nanmean(prod_o >= t_cur))
        c['prod_p99'] = dict(new=float(np.nanquantile(prod_n, 0.99)), current=float(np.nanquantile(prod_o, 0.99)))
        c['prod_max'] = dict(new=float(np.nanmax(prod_n)), current=float(np.nanmax(prod_o)))
        c['pass'] &= bool(c['false_ink_frac_new_map_vs_current_native_threshold'] == 0.0
                          and c.get('false_ink_frac_new_map_vs_new_native_threshold', 0.0) == 0.0)
    ok &= c['pass']; res['controls'][var] = c; print(var, 'controls', json.dumps(c), flush=True)
res['pass'] = bool(ok and len(res['members']) > 0)
json.dump(res, open(OUT, 'w'), indent=1)
print('AHN_GATE_PASS' if res['pass'] else 'AHN_GATE_FAIL', flush=True)
sys.exit(0 if res['pass'] else 1)
