"""Depth-shuffle control on our end-to-end maps: how much of the claim signal survives when the render layers the models read
(8..31) are permuted (the winner's shuffle_control.py prep, seed 1). A model reading ink at a depth should lose its marks.
Inside the 4 cm2 claim box (render px 1639-2745 x 2431-6829), on pixels finite in both maps:
  per map (tr, each member, ensA, prod, prodd): Pearson r shuffled vs original;
  prod and ensA: top-1% Jaccard (each map's own top 1%) and the share of box pixels the shuffled map puts above the ORIGINAL map's
  top-1% threshold ('false ink' in the B0 table's sense; 1% means no change);
  TRACE ~ ensA agreement r, original and shuffled (the B0 shuffled crop with the shipped members gave 0.14 vs 0.75 native).
Usage (in fl-claim-a7:final): python shuffle_score.py <orig maps dir> <shuf maps dir> <out.json>
"""
import json, sys
import numpy as np
O, S, OUT = sys.argv[1:4]; y0, y1, x0, x1 = 1639, 2745, 2431, 6829
L = lambda d, k: np.load(f'{d}/{k}' if k.startswith('fused') else f'{d}/fwd_{k}.npy').astype(np.float64)[y0:y1, x0:x1]
names = {'tr': 'tr', 's0': 'hn4_s0', 's1': 'hn4_s1', 's2': 'hn4_s2', 'decoy': 'hn4_decoy',
         'ensA': 'fused/fwd_ensA.npy', 'prod': 'fused/fwd_prod.npy', 'prodd': 'fused/fwd_prodd.npy'}
res = dict(box_px=[y0, y1, x0, x1], shuffle='render layers 8..31 permuted, seed 1 (winner shuffle_control.py prep)', maps={})
for k, f in names.items():
    a, b = L(O, f), L(S, f); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    r = dict(px=int(m.sum()), r_shuf_vs_orig=round(float(np.corrcoef(a, b)[0, 1]), 4))
    if k in ('prod', 'ensA', 'prodd'):
        ta, tb = np.quantile(a, 0.99), np.quantile(b, 0.99); A, B = a > ta, b > tb
        r.update(top1_jaccard=round(float((A & B).sum() / max(1, (A | B).sum())), 4), shuf_above_orig_top1=round(float((b > ta).mean()), 5))
    res['maps'][k] = r
for side, d in (('orig', O), ('shuf', S)):
    t, e = L(d, 'tr'), L(d, 'fused/fwd_ensA.npy'); m = np.isfinite(t) & np.isfinite(e)
    res[f'trace_ensA_r_{side}'] = round(float(np.corrcoef(t[m], e[m])[0, 1]), 4)
json.dump(res, open(OUT, 'w'), indent=1)
p = res['maps']['prod']
print(f"prod r {p['r_shuf_vs_orig']} top1 J {p['top1_jaccard']} shuf above orig top1 {p['shuf_above_orig_top1']} | ensA J {res['maps']['ensA']['top1_jaccard']} | TRACE~ensA {res['trace_ensA_r_orig']} -> {res['trace_ensA_r_shuf']}")
