"""Retrained A-hn s0 vs the shipped seeds, next to the shipped seed-to-seed spread, plus run facts.
Usage (in fl-claim-a7:final): python retrain_compare.py <retrain dir under /data>
Reads <dir>/eval/fwd_hn4_s0_pc.npy, <dir>/probe/*.json, <dir>/runs/hn4_s0_pc/{train_log.jsonl,done.json}, <dir>/logs/gpu_train.csv,
<dir>/configs/hn4_s0_pc.json, <dir>/ref/hn4_s0.json. Writes <dir>/summary.json and <dir>/summary.md.
"""
import itertools, json, os, sys
import numpy as np
R = sys.argv[1]
SH = '/opt/fl/data/members/fwd_hn4_{}.npy'
maps = {f'shipped_{s}': SH.format(s) for s in ('s0', 's1', 's2')}
maps['retrained_s0'] = f'{R}/eval/fwd_hn4_s0_pc.npy'
maps['replay_shipped_s0'] = '/data/p0306b/fidelity_exact/fwd_hn4_s0.npy'   # our exact replay of the shipped checkpoint (repro floor)
miss = [p for k, p in maps.items() if k != 'replay_shipped_s0' and not os.path.exists(p)]
if miss: raise SystemExit(f'missing required maps: {miss}')
def load_full(p):
    """infer.py --roi writes frame-sized maps; embed them in the render raster using the sidecar json's frame"""
    a = np.load(p).astype(np.float32); j = p[:-4] + '.json'
    if os.path.exists(j):
        jj = json.load(open(j)); H, W = jj['render_shape'][1:]
        if a.shape != (H, W):
            y0, y1, x0, x1 = jj['frame']; out = np.full((H, W), np.nan, np.float32); out[y0:y1, x0:x1] = a; a = out
    return a


A = {k: load_full(p) for k, p in maps.items() if os.path.exists(p)}
shapes = {k: v.shape for k, v in A.items()}
if len(set(shapes.values())) != 1: raise SystemExit(f'maps on different rasters: {shapes}')


def cmp(a, b):
    m = np.isfinite(a) & np.isfinite(b); d = np.abs(a[m] - b[m])
    return dict(pearson=float(np.corrcoef(a[m], b[m])[0, 1]), mae=float(d.mean()), max_abs=float(d.max()),
                p99_abs=float(np.percentile(d, 99)), n_px=int(m.sum()))


res = {'pairs': {}}
for x, y in [('shipped_s0', 'shipped_s1'), ('shipped_s0', 'shipped_s2'), ('shipped_s1', 'shipped_s2'),
             ('retrained_s0', 'shipped_s0'), ('retrained_s0', 'shipped_s1'), ('retrained_s0', 'shipped_s2'),
             ('replay_shipped_s0', 'shipped_s0')]:
    if x in A and y in A: res['pairs'][f'{x}~{y}'] = cmp(A[x], A[y])
seed = [v for k, v in res['pairs'].items() if k.split('~')[0].startswith('shipped')]
if seed:
    res['seed_spread_r'] = [min(v['pearson'] for v in seed), max(v['pearson'] for v in seed)]
    res['seed_spread_mae'] = [min(v['mae'] for v in seed), max(v['mae'] for v in seed)]
rk = 'retrained_s0~shipped_s0'
if rk in res['pairs'] and seed:
    res['verdict'] = ('within seed spread' if res['pairs'][rk]['pearson'] >= res['seed_spread_r'][0] else 'below seed spread')


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


res['probe'] = jl(f'{R}/probe/probe.json'); res['bn_check'] = jl(f'{R}/probe/bn_check.json')
run = f'{R}/runs/hn4_s0_pc'; res['done'] = jl(f'{run}/done.json')
if os.path.exists(f'{run}/train_log.jsonl'):
    L = [json.loads(l) for l in open(f'{run}/train_log.jsonl') if l.strip()]
    res['train'] = dict(log_rows=len(L), last=L[-1] if L else None, peak_alloc_gb=max((r['peak_gb'] for r in L), default=None))
g = f'{R}/logs/gpu_train.csv'
if os.path.exists(g):
    mem = [int(l.split(',')[1].split()[0]) for l in open(g) if l.count(',') >= 2 and 'MiB' in l]
    res['gpu_train_peak_mib_total_card'] = max(mem) if mem else None
ts = f'{R}/logs/train_seconds'
res['train_wall_seconds'] = int(open(ts).read().split()[0]) if os.path.exists(ts) else None
a, b = jl(f'{R}/ref/hn4_s0.json'), jl(f'{R}/configs/hn4_s0_pc.json')


def flat(d, p=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, f'{p}{k}.'))
        else: o[f'{p}{k}'] = v
    return o


if a and b:
    fa, fb = flat(a), flat(b)
    res['config_diff'] = {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k) and k != 'note'}
json.dump(res, open(f'{R}/summary.json', 'w'), indent=1)
with open(f'{R}/summary.md', 'w') as f:
    f.write('# A-hn s0 retrain on the 4070 vs shipped\n\n| pair | r | mean abs | max abs | px |\n|---|---|---|---|---|\n')
    for k, v in res['pairs'].items():
        f.write(f"| {k} | {v['pearson']:.5f} | {v['mae']:.4g} | {v['max_abs']:.4g} | {v['n_px']} |\n")
    f.write(f"\nseed spread r: {res.get('seed_spread_r')}  verdict: {res.get('verdict')}\n")
    f.write(f"train wall s: {res['train_wall_seconds']}  peak alloc GB: {(res.get('train') or {}).get('peak_alloc_gb')}  "
            f"card peak MiB: {res.get('gpu_train_peak_mib_total_card')}\nconfig diff vs hn4_s0.json: {res.get('config_diff')}\n")
print(open(f'{R}/summary.md').read())
