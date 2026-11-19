"""Fuse the Mac (MPS) maps with the winner's own fuse_v7.py (unchanged) and compare the fused maps with the 4070's and the shipped ones.

Usage: python fuse_compare.py MAC_TRACE.npy S0.npy S1.npy S2.npy DECOY.npy OUTDIR
  - member maps are infer.py ROI-frame maps (with their .json sidecars); they are embedded into the full render raster (NaN outside
    the frame) as copies in OUTDIR/members/ (same placement as pc/pad_frame.py), then fuse_v7.py writes fwd_ensA / fwd_prod / fwd_prodd.
  - compared on the B0 native crop (and on the claim box inside it) against
      cuda    : the 4070's fused maps from the same recipe (b0/runs/native/fused/)
      shipped : the winner's shipped maps cut to the native crop (b0/ref_{ensA,prod,prodd}_native.npy)
    Pearson r, max abs diff, NaN mismatches, top-1 % Jaccard (each map thresholded at its own 99th percentile of finite pixels).
Writes OUTDIR/compare.json."""
import json, os, sys, subprocess
import numpy as np

MD = os.environ['MD']; FL = os.environ['FL_ROOT']
tr, s0, s1, s2, dec, out = sys.argv[1:7]
os.makedirs(f'{out}/members', exist_ok=True)


def pad(p):
    j = json.load(open(p[:-4] + '.json')); H, W = j['render_shape'][1:]; a = np.load(p)
    q = f'{out}/members/{os.path.basename(p)}'
    if a.shape != (H, W):
        fy0, fy1, fx0, fx1 = j['frame']; assert a.shape == (fy1 - fy0, fx1 - fx0), (p, a.shape, j['frame'])
        full = np.full((H, W), np.nan, a.dtype); full[fy0:fy1, fx0:fx1] = a; a = full
    np.save(q, a); return q


M = [pad(p) for p in (s0, s1, s2, dec)]
subprocess.run([sys.executable, f'{FL}/claim/fuse_v7.py', out, '--trace', tr, '--members', *M[:3], '--decoy', M[3]], check=True)


def stats(r_, m_):
    f = np.isfinite(r_) & np.isfinite(m_); d = np.abs(r_[f] - m_[f])
    t_r, t_m = np.quantile(r_[f], 0.99), np.quantile(m_[f], 0.99)
    A, B = (r_ >= t_r) & f, (m_ >= t_m) & f
    return dict(px=int(f.sum()), nan_mismatch=int((np.isfinite(r_) != np.isfinite(m_)).sum()),
                pearson=float(np.corrcoef(r_[f], m_[f])[0, 1]), max_abs=float(d.max()), mean_abs=float(d.mean()),
                top1_jaccard=float((A & B).sum() / max((A | B).sum(), 1)))


need = [f'{MD}/b0/b0.json'] + [f'{MD}/b0/runs/native/fused/fwd_{k}.npy' for k in ('ensA', 'prod', 'prodd')] + [f'{MD}/b0/ref_{k}_native.npy' for k in ('ensA', 'prod', 'prodd')]
missing = [n for n in need if not os.path.exists(n)]
if missing:   # the fusion above is done; the exactness comparison needs the PC's b0.json and the 4070's maps, which this bundle doesn't include
    print(f'fused maps written to {out}; PC reference files (b0.json, 4070 maps) not found ({len(missing)} missing, e.g. {missing[0]}): comparison skipped')
    sys.exit(0)
b0 = json.load(open(f'{MD}/b0/b0.json')); y0, y1, x0, x1 = b0['native_crop']; by0, by1, bx0, bx1 = b0['box_in_crop']

res = {}
for k in ('ensA', 'prod', 'prodd'):
    mac = np.load(f'{out}/fwd_{k}.npy').astype(np.float32)[y0:y1, x0:x1]
    cuda = np.load(f'{MD}/b0/runs/native/fused/fwd_{k}.npy').astype(np.float32)[y0:y1, x0:x1]
    ship = np.load(f'{MD}/b0/ref_{k}_native.npy').astype(np.float32)
    res[k] = {}
    for nm, ref in (('cuda', cuda), ('shipped', ship)):
        assert ref.shape == mac.shape, (k, nm, ref.shape, mac.shape)
        res[k][nm] = dict(crop=stats(ref, mac), claim_box=stats(ref[by0:by1, bx0:bx1], mac[by0:by1, bx0:bx1]))
    res[k]['cuda_vs_shipped_crop'] = stats(ship, cuda)
json.dump(res, open(f'{out}/compare.json', 'w'), indent=1)
print(json.dumps({k: {nm: {a: {m: round(v[m], 8) for m in ('pearson', 'max_abs', 'top1_jaccard', 'nan_mismatch')}
                           for a, v in (d.items() if isinstance(d, dict) and 'crop' in d else [('crop', d)])}
                      for nm, d in r.items()} for k, r in res.items()}, indent=1))
