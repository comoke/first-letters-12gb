"""Compare our training logs with the shipped run records (hf-model run_records/<m>/train_log.jsonl), per member.
Metrics (from the plan): Pearson r of the loss series over the steps both logged, plus first / final loss and the max abs difference.
Usage (in fl-claim-a7:final): python e2e_trainlog.py <our runs dir> <shipped run_records dir> <out.json> m1[:ours_name] m2 ...
"""
import json, os, sys
import numpy as np
RUNS, REC, OUT = sys.argv[1:4]; MEM = sys.argv[4:]
res = {}
for spec in MEM:
    m, ours = (spec.split(':') + [None])[:2]; ours = ours or m
    po, ps = f'{RUNS}/{ours}/train_log.jsonl', f'{REC}/{m}/train_log.jsonl'
    if not (os.path.exists(po) and os.path.exists(ps)): res[m] = dict(missing=[p for p in (po, ps) if not os.path.exists(p)]); continue
    lo = {r['step']: r for r in map(json.loads, open(po)) if 'loss' in r}; ls = {r['step']: r for r in map(json.loads, open(ps)) if 'loss' in r}
    st = sorted(set(lo) & set(ls)); a = np.array([lo[s]['loss'] for s in st]); b = np.array([ls[s]['loss'] for s in st])
    res[m] = dict(files=[po, ps], steps_common=len(st), steps_ours=len(lo), steps_shipped=len(ls),
                  pearson_loss=float(np.corrcoef(a, b)[0, 1]) if len(st) > 2 else None, max_abs_loss=float(np.abs(a - b).max()) if st else None,
                  first=dict(step=st[0], ours=float(a[0]), shipped=float(b[0])) if st else None,
                  final=dict(step=st[-1], ours=float(a[-1]), shipped=float(b[-1])) if st else None)
    print(m, {k: v for k, v in res[m].items() if k != 'files'})
json.dump(res, open(OUT, 'w'), indent=1)
