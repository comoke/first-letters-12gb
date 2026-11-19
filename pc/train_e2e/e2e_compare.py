"""Compare a stage of our end-to-end train-mode run with the shipped artifact (metrics fixed in docs/train-e2e-plan-2026-11-15.md).
Usage (in fl-claim-a7:final): python e2e_compare.py <stage> <ours dir> <shipped dir> <out.json>
  stage masks   : <r>_masks.npz keys dist / qual / cov24p (+ H, W) per region: equality, max abs diff, NaN agreement; _masks.json
  stage teach   : <r>_<m>.npy for m in v3 trace rv2 v8: Pearson r, mean / max abs diff on finite px, finite-mask agreement
  stage packs   : 306_<r>.npz every key (img, lab_*, org_*, ink_*): equality / differing elements; org tile sets: only-ours / only-shipped
  stage exclusion: per-region tile counts, rule counts and distances vs exclusion_audit.json
  stage labels  : <r>_labels.npz main / sc / decoy: same-class fraction, ink Jaccard, changed px; fz r; stats.json side by side
Each entry records the two file paths it compared."""
import glob, json, os, sys
import numpy as np
STAGE, OURS, SHIP, OUT = sys.argv[1:5]
REG = ['A6', 'w052', 'w053', 'w059', 'w109']
res = dict(stage=STAGE, ours=OURS, shipped=SHIP, regions={})


def arrcmp(a, b):
    if a.shape != b.shape: return dict(shape_ours=list(a.shape), shape_shipped=list(b.shape), equal=False)
    if a.dtype.kind in 'fc':
        fa, fb = np.isfinite(a), np.isfinite(b); m = fa & fb; d = np.abs(a[m].astype(np.float64) - b[m].astype(np.float64))
        out = dict(equal=bool(np.array_equal(a, b, equal_nan=True)), finite_agree=float((fa == fb).mean()), n=int(m.sum()),
                   max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0)
        if d.size > 1 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
            s = slice(None, None, max(1, d.size // 2_000_000))
            out['pearson'] = float(np.corrcoef(a[m][s].astype(np.float64), b[m][s].astype(np.float64))[0, 1])
        return out
    return dict(equal=bool(np.array_equal(a, b)), n_diff=int((a != b).sum()), n=int(a.size))


for r in (REG if STAGE != 'exclusion' else []):
    e = {}
    if STAGE == 'masks':
        po, ps = f'{OURS}/{r}_masks.npz', f'{SHIP}/{r}_masks.npz'
        if not (os.path.exists(po) and os.path.exists(ps)): e['missing'] = [p for p in (po, ps) if not os.path.exists(p)]
        else:
            A, B = np.load(po), np.load(ps)
            e = {k: arrcmp(A[k], B[k]) for k in ('dist', 'qual', 'cov24p', 'H', 'W')}
            jo, js = f'{OURS}/{r}_masks.json', f'{SHIP}/{r}_masks.json'
            if os.path.exists(jo) and os.path.exists(js): e['json'] = dict(ours=json.load(open(jo)), shipped=json.load(open(js)))
            e['files'] = [po, ps]
    elif STAGE == 'teach':
        for m in ('v3', 'trace', 'rv2', 'v8'):
            po, ps = f'{OURS}/{r}_{m}.npy', f'{SHIP}/{r}_{m}.npy'
            e[m] = dict(arrcmp(np.load(po).astype(np.float32), np.load(ps).astype(np.float32)), files=[po, ps]) \
                if os.path.exists(po) and os.path.exists(ps) else dict(missing=[p for p in (po, ps) if not os.path.exists(p)])
    elif STAGE == 'labels':
        po, ps = f'{OURS}/{r}_labels.npz', f'{SHIP}/{r}_labels.npz'
        if not (os.path.exists(po) and os.path.exists(ps)): e['missing'] = [p for p in (po, ps) if not os.path.exists(p)]
        else:
            A, B = np.load(po), np.load(ps)
            for v in ('main', 'sc', 'decoy'):
                a, b = A[v], B[v]
                if a.shape != b.shape: e[v] = dict(shape_ours=list(a.shape), shape_shipped=list(b.shape)); continue
                ia, ib = a == 1, b == 1
                e[v] = dict(same_class_frac=float((a == b).mean()), changed_px=int((a != b).sum()),
                            ink_jaccard=float((ia & ib).sum() / max((ia | ib).sum(), 1)), ink_px_ours=int(ia.sum()), ink_px_shipped=int(ib.sum()))
            e['fz'] = arrcmp(A['fz'].astype(np.float32), B['fz'].astype(np.float32)); e['valid'] = arrcmp(A['valid'], B['valid'])
            e['files'] = [po, ps]
    elif STAGE == 'packs':
        po, ps = f'{OURS}/306_{r}.npz', f'{SHIP}/306_{r}.npz'
        if not (os.path.exists(po) and os.path.exists(ps)): e['missing'] = [p for p in (po, ps) if not os.path.exists(p)]
        else:
            A, B = np.load(po), np.load(ps)
            for k in sorted(set(A.files) | set(B.files)):
                if k == 'info' or k not in A.files or k not in B.files: e[k] = dict(in_ours=k in A.files, in_shipped=k in B.files); continue
                e[k] = arrcmp(A[k], B[k])
                if k.startswith('org_') and A[k].shape[1:] == B[k].shape[1:]:
                    sa, sb = set(map(tuple, A[k].tolist())), set(map(tuple, B[k].tolist()))
                    e[k].update(tiles_ours=len(sa), tiles_shipped=len(sb), only_ours=len(sa - sb), only_shipped=len(sb - sa))
            e['files'] = [po, ps]
    res['regions'][r] = e
if STAGE == 'exclusion':
    po = sorted(glob.glob(f'{OURS}/**/*.json', recursive=True)); ps = f'{SHIP}/exclusion_audit.json'
    S = json.load(open(ps)); O_ = next((json.load(open(p)) for p in po if 'regions' in json.load(open(p))), None)
    keys = ['tiles', 'rule_a_centre_corners_pad07_ok', 'rule_b_grid25_pad0_ok', 'labelled_px_sampled', 'rule_c_labelled_px_ok',
            'min_distance_mm_tile_grid', 'median_distance_mm_tile_grid']
    res['regions'] = {r: {k: dict(ours=(O_ or {}).get('regions', {}).get(r, {}).get(k), shipped=S['regions'].get(r, {}).get(k)) for k in keys} for r in REG}
    res['files'] = dict(ours=po, shipped=ps); res['top_level'] = {k: dict(ours=(O_ or {}).get(k), shipped=S.get(k)) for k in S if not isinstance(S[k], dict)}
if STAGE == 'labels':
    so, ss = f'{OURS}/stats.json', f'{SHIP}/stats.json'
    if os.path.exists(so) and os.path.exists(ss): res['stats'] = dict(ours=json.load(open(so)), shipped=json.load(open(ss)), files=[so, ss])
json.dump(res, open(OUT, 'w'), indent=1)
for r, e in res['regions'].items():
    print(r, json.dumps({k: ({kk: vv for kk, vv in v.items() if kk in ('equal', 'pearson', 'max_abs', 'n_diff', 'same_class_frac', 'ink_jaccard', 'missing')}
                             if isinstance(v, dict) else v) for k, v in e.items() if k not in ('files', 'json')}))
