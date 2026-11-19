"""Score one B0 variant run (ink_run.sh output dir) with the research program's four numbers.
  1 reference agreement ("letter separation"): AP / AUC of the variant's product map (TRACE x A-hn) against the shipped product map
       binarized at the top REF_TOP of the claim box. Not ground truth: the release has no human labels for this area, so this
       measures how much of the shipped claim signal a variant keeps (near-circular for native, by design).
  2 decoy overlap: top-1% Jaccard of prod vs prodd (TRACE x decoy model) over the crop (lower is better; PHerc1203 0.46)
  3 throughput: variant area (cm2, physical) / wall hours of the GPU stages (all ink_run stage passes; not GPU-active time)
  4 crop sheet: 4 fixed windows (densest reference windows in the box; for notext, in the notext crop) x CT layer 20 of the input
       actually used | prod | prodd | TRACE, stretched with the reference's percentiles -> <run>/b0_sheet.png
Negative controls (shuf, notext): false-ink fraction = share of pixels above the native run's top-1% prod threshold. notext is
chosen as the lowest-signal window of the shipped map, so it is a descriptive control, not independent validation.
Rasters: native runs on the full render -> cropped to native_crop; deg936 -> resampled to the native crop raster.
Usage (in fl-claim-a7:final): python b0_score.py <b0 dir> <variant> <run dir> <ink_run.log>  -> <run dir>/b0_score.json (exit 1 if
a required input or metric is missing)
"""
import json, os, re, sys
import numpy as np, zarr
from PIL import Image
from scipy import ndimage
B0, VAR, RUN, LOG = sys.argv[1:5]
REF_TOP, TOP, S = 0.05, 0.01, 500
meta = json.load(open(f'{B0}/b0.json'))
nc, ntc = meta['native_crop'], meta['notext_crop']
nat_shape = (nc[1] - nc[0], nc[3] - nc[2]); full_shape = tuple(meta['render_shape'])
REG = 'notext' if VAR == 'notext' else 'native'                 # which reference crop / raster this variant lives on
for nm in ('prod', 'prodd', 'tr'):
    if not os.path.exists(f'{B0}/ref_{nm}_{REG}.npy'): raise SystemExit(f'missing reference ref_{nm}_{REG}.npy')
for p in ('fused/fwd_prod.npy', 'fused/fwd_prodd.npy', 'fwd_tr.npy'):
    if not os.path.exists(f'{RUN}/{p}'): raise SystemExit(f'missing run output {RUN}/{p}')


def load(p):
    a = np.load(p).astype(np.float32)
    if a.shape == full_shape and VAR == 'native': a = a[nc[0]:nc[1], nc[2]:nc[3]]
    if VAR not in ('native', 'notext') and a.shape != nat_shape:  # other pixel size (deg936, m1203b, ...): resample to the native crop raster (NaN-aware)
        f = (nat_shape[0] / a.shape[0], nat_shape[1] / a.shape[1]); fin = np.isfinite(a)
        v = ndimage.zoom(np.where(fin, a, 0), f, order=1)[:nat_shape[0], :nat_shape[1]]
        m = ndimage.zoom(fin.astype(np.float32), f, order=1)[:nat_shape[0], :nat_shape[1]]
        a = np.where(m > 0.5, v / np.maximum(m, 1e-6), np.nan)
    return a


def ap_auc(score, pos):
    o = np.argsort(-score, kind='stable'); p = pos[o]; tp = np.cumsum(p); n = p.size; P = int(tp[-1])
    ap = float((tp[p] / (np.flatnonzero(p) + 1)).sum() / max(P, 1))
    ranks = np.empty(n); ranks[o] = np.arange(n, 0, -1)
    auc = float((ranks[pos].sum() - P * (P + 1) / 2) / max(P * (n - P), 1))
    return ap, auc


def top(a, q):
    t = np.nanquantile(a, 1 - q); return a >= t, float(t)


res = dict(variant=VAR, run=RUN, letter_metric_basis='agreement with the shipped TRACE x A-hn product map (no ground truth)')
prod, prodd, tr = load(f'{RUN}/fused/fwd_prod.npy'), load(f'{RUN}/fused/fwd_prodd.npy'), load(f'{RUN}/fwd_tr.npy')
refs = {nm: np.load(f'{B0}/ref_{nm}_{REG}.npy').astype(np.float32) for nm in ('prod', 'prodd', 'tr')}
for nm, a in (('prod', prod), ('prodd', prodd), ('tr', tr)):
    if a.shape != refs['prod'].shape: raise SystemExit(f'{nm} raster {a.shape} != reference {refs["prod"].shape}')
if REG == 'native':
    y0, y1, x0, x1 = meta['box_in_crop']
    rb, sb = refs['prod'][y0:y1, x0:x1], prod[y0:y1, x0:x1]; m = np.isfinite(rb) & np.isfinite(sb)
    pos = rb[m] >= np.quantile(rb[m], 1 - REF_TOP)
    res['letter_ap'], res['letter_auc'] = ap_auc(sb[m], pos)
    res['ref_pearson_box'] = float(np.corrcoef(rb[m], sb[m])[0, 1]); res['box_px'] = int(m.sum())
else:
    y0, y1, x0, x1 = 0, refs['prod'].shape[0], 0, refs['prod'].shape[1]
m = np.isfinite(prod) & np.isfinite(prodd)
a, _ = top(np.where(m, prod, np.nan), TOP); b, _ = top(np.where(m, prodd, np.nan), TOP)
res['decoy_jaccard_top1'] = float((a & b).sum() / max((a | b).sum(), 1))
# revised false-positive metrics (B0 baseline 2026-11-15): cross-family agreement and TRACE-free decoy overlap
ensA = load(f'{RUN}/fused/fwd_ensA.npy'); dec = load(f'{RUN}/fwd_hn4_decoy.npy')
k = np.isfinite(tr) & np.isfinite(ensA); res['trace_ensA_r'] = float(np.corrcoef(tr[k], ensA[k])[0, 1])
k = np.isfinite(ensA) & np.isfinite(dec)
a, _ = top(np.where(k, ensA, np.nan), TOP); b, _ = top(np.where(k, dec, np.nan), TOP)
res['ensA_decoy_jaccard_top1'] = float((a & b).sum() / max((a | b).sum(), 1))
nat_json = f'{os.path.dirname(RUN.rstrip("/"))}/native/b0_score.json'
if VAR == 'native':
    res['prod_top1_threshold'] = top(prod, TOP)[1]
elif VAR in ('shuf', 'notext'):
    if not os.path.exists(nat_json): raise SystemExit('score native first (its top-1% threshold defines false ink)')
    t = json.load(open(nat_json))['prod_top1_threshold']
    res['false_ink_frac_vs_native_top1'] = float(np.nanmean(prod >= t)); res['native_top1_threshold'] = t
st = [(mm.group(2), int(mm.group(4))) for mm in re.finditer(r"\[(\d+)s\] (\S+)(?: pass \d+)? exit=(\S+).*?\((\d+)s\)", open(LOG).read())]
gpu_s = sum(s for n, s in st)
res['stage_wall_seconds'] = st; res['gpu_stage_wall_hours'] = gpu_s / 3600
res['area_cm2'] = meta['area_cm2_by_variant'][VAR]; res['cm2_per_gpu_h'] = res['area_cm2'] / (gpu_s / 3600) if gpu_s else None
if not res['cm2_per_gpu_h']: raise SystemExit('no stage timings parsed from ' + LOG)

# crop sheet: 4 densest reference windows inside the box (native family) or the notext crop; fixed by the reference
ref = refs['prod']; thr = np.nanquantile(ref[y0:y1, x0:x1], 0.95); cand = []
for yy in range(y0, max(y0 + 1, y1 - S + 1), 125):
    for xx in range(x0, max(x0 + 1, x1 - S + 1), 125):
        cand.append((float(np.nanmean(ref[yy:yy + S, xx:xx + S] >= thr)), yy, xx))
win = []
for s, yy, xx in sorted(cand, reverse=True):
    if all(abs(yy - a_) >= S or abs(xx - b_) >= S for a_, b_ in win): win.append((yy, xx))
    if len(win) == 4: break
res['sheet_windows_px_in_crop'] = win
src = VAR                                                       # the input this run actually saw
ct = np.asarray(zarr.open(f'{B0}/{src}/render.zarr', mode='r')['0'][20]).astype(np.float32)
if ct.shape != ref.shape: ct = ndimage.zoom(ct, (ref.shape[0] / ct.shape[0], ref.shape[1] / ct.shape[1]), order=1)[:ref.shape[0], :ref.shape[1]]
rows = []
for yy, xx in win:
    row = []
    for nm, arr in (('ct', ct), ('prod', prod), ('prodd', prodd), ('tr', tr)):
        r = ct if nm == 'ct' else refs[nm]
        lo, hi = np.nanpercentile(r[y0:y1, x0:x1], [1, 99.9])
        row.append(np.nan_to_num(np.clip((arr[yy:yy + S, xx:xx + S] - lo) / max(hi - lo, 1e-6), 0, 1)))
    rows.append(np.concatenate([np.pad(t, 4) for t in row], 1))
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(f'{RUN}/b0_sheet.png')
res['sheet'] = f'{RUN}/b0_sheet.png (rows = windows, cols = CT layer20 of the input used | prod | prodd | TRACE)'
json.dump(res, open(f'{RUN}/b0_score.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'stage_wall_seconds'}, indent=1))
