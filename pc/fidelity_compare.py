"""Compare our re-run maps on the PHerc0306B claim render with the maps shipped in fl-claim-a7:final.
Usage (inside the image): python fidelity_compare.py <our outdir>
Writes <outdir>/fidelity.json and prints a table. Compared on pixels finite in both maps.
"""
import json, os, sys
import numpy as np
O = sys.argv[1]
pairs = {  # ours -> shipped
    "tr": ("fwd_tr.npy", "/opt/fl/data/maps/fwd_tr.npy"),
    "hn4_s0": ("fwd_hn4_s0.npy", "/opt/fl/data/members/fwd_hn4_s0.npy"),
    "hn4_s1": ("fwd_hn4_s1.npy", "/opt/fl/data/members/fwd_hn4_s1.npy"),
    "hn4_s2": ("fwd_hn4_s2.npy", "/opt/fl/data/members/fwd_hn4_s2.npy"),
    "decoy": ("fwd_hn4_decoy.npy", "/opt/fl/data/members/fwd_decoy.npy"),
    "ensA": ("fused/fwd_ensA.npy", "/opt/fl/data/maps/fwd_ensA.npy"),
    "prod": ("fused/fwd_prod.npy", "/opt/fl/data/maps/fwd_prod.npy"),
    "prodd": ("fused/fwd_prodd.npy", "/opt/fl/data/maps/fwd_prodd.npy"),
}
res = {}
for k, (ours, ref) in pairs.items():
    p = os.path.join(O, ours)
    if not (os.path.exists(p) and os.path.exists(ref)):
        res[k] = {"missing": [x for x in (p, ref) if not os.path.exists(x)]}; continue
    a = np.load(p).astype(np.float32); b = np.load(ref).astype(np.float32)
    if a.shape != b.shape:
        res[k] = {"shape_mismatch": [list(a.shape), list(b.shape)]}; continue
    m = np.isfinite(a) & np.isfinite(b); d = np.abs(a[m] - b[m])
    res[k] = {"n_px": int(m.sum()), "ours_finite": float(np.isfinite(a).mean()), "ref_finite": float(np.isfinite(b).mean()),
              "pearson": float(np.corrcoef(a[m], b[m])[0, 1]), "mae": float(d.mean()), "p99_abs": float(np.percentile(d, 99)),
              "max_abs": float(d.max()), "bitwise_equal_frac": float((a[m] == b[m]).mean())}
json.dump(res, open(os.path.join(O, "fidelity.json"), "w"), indent=1)
print(f"{'map':8} {'pearson':>8} {'mae':>9} {'p99|d|':>9} {'max|d|':>9} {'equal':>7} {'px':>10}")
for k, r in res.items():
    if "pearson" in r:
        print(f"{k:8} {r['pearson']:8.5f} {r['mae']:9.2e} {r['p99_abs']:9.2e} {r['max_abs']:9.2e} {r['bitwise_equal_frac']:7.3f} {r['n_px']:10d}")
    else: print(k, r)
