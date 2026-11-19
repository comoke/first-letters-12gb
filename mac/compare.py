"""Mac (MPS) map vs the PC's CUDA map (and the shipped map) on the B0 native window.
Usage: python compare.py tr|s0|s1|s2|decoy MAC.npy [--cols x0 x1]   (member MAC map = ROI frame 1399 x 5600; tr = full raster)
Pearson r, max/mean abs diff, and quantised-level agreement over pixels finite in both. --cols limits to native-crop columns
(for a sub-ROI A-hn run: pass the interior columns)."""
import json, os, sys, argparse
import numpy as np
MD = os.environ['MD']
ap = argparse.ArgumentParser(); ap.add_argument('kind'); ap.add_argument('mac'); ap.add_argument('--cols', type=int, nargs=2)
ap.add_argument('--x0', type=int, default=None, help='s0 sub-ROI run: render column of the Mac map column 0 (interior = 64 px in)')
a = ap.parse_args()
b0 = json.load(open(f'{MD}/b0/b0.json')); y0, y1, x0, x1 = b0['native_crop']


def crop(arr):
    arr = arr.astype(np.float32)
    if arr.shape == tuple(b0['render_shape']): arr = arr[y0:y1, x0:x1]
    return arr


if a.kind == 'tr':
    cuda = crop(np.load(f'{MD}/b0/runs/native/fwd_tr.npy')); ship = np.load(f'{MD}/b0/ref_tr_native.npy').astype(np.float32)
else:
    j = json.load(open(f'{MD}/b0/runs/native/fwd_hn4_{a.kind}.json')); fy0, fy1, fx0, fx1 = j['frame']
    cuda = np.load(f'{MD}/b0/runs/native/fwd_hn4_{a.kind}.npy').astype(np.float32)
    cuda = cuda[fy0:fy1, fx0:fx1] if cuda.shape == tuple(j['render_shape'][1:]) else cuda
    ship = None
mac = np.load(a.mac).astype(np.float32)
if a.kind == 'tr': mac = crop(mac)
if a.kind != 'tr' and a.x0 is not None:          # sub-ROI run, full frame height: compare its interior columns only
    c0 = a.x0 - fx0; cuda = cuda[:, c0:c0 + mac.shape[1]]
    a.cols = [64, mac.shape[1] - 64]
if mac.shape != cuda.shape:
    raise SystemExit(f'shape {mac.shape} != cuda {cuda.shape}')
sl = (slice(None), slice(*a.cols)) if a.cols else (slice(None), slice(None))
res = dict(kind=a.kind, mac=a.mac, cols=a.cols)
for nm, ref in (('cuda', cuda), ('shipped', ship)):
    if ref is None: continue
    r_, m_ = ref[sl], mac[sl]; f = np.isfinite(r_) & np.isfinite(m_)
    d = np.abs(r_[f] - m_[f])
    res[nm] = dict(px=int(f.sum()), nan_mismatch=int((np.isfinite(r_) != np.isfinite(m_)).sum()),
                   pearson=float(np.corrcoef(r_[f], m_[f])[0, 1]), max_abs=float(d.max()), mean_abs=float(d.mean()),
                   p999_abs=float(np.quantile(d, 0.999)), same_u8=float((np.rint(r_[f] * 255) == np.rint(m_[f] * 255)).mean()))
print(json.dumps(res, indent=1))
