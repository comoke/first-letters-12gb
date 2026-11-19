"""Pixel diff of the claim-image variants: <ref>/variants/*.png (the package's, from run.sh final on the shipped maps) vs
<ours>/variants/*.png (run.sh final with our e2e maps). Per variant: shape, share of pixels that differ (any channel), mean and max abs
difference (0-255), and sha256 equality. Writes <out.json>; prints the lead variant's line last.
Usage (in fl-claim-a7:final): python stage9_diff.py <ref final dir> <ours final dir> <out.json>
"""
import glob, hashlib, json, os, sys
import numpy as np, cv2
REF, OURS, OUT = sys.argv[1:4]
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
res = {}
for f in sorted(glob.glob(f'{REF}/variants/*.png')):
    n = os.path.basename(f); g = f'{OURS}/variants/{n}'
    if not os.path.exists(g): res[n] = dict(missing_in_ours=True); continue
    a = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.int16); b = cv2.imread(g, cv2.IMREAD_UNCHANGED).astype(np.int16)
    if a.shape != b.shape: res[n] = dict(shape_ref=list(a.shape), shape_ours=list(b.shape)); continue
    d = np.abs(a - b); px = d.max(-1) if d.ndim == 3 else d
    res[n] = dict(shape=list(a.shape), byte_identical=sha(f) == sha(g), px_differ_share=round(float((px > 0).mean()), 6),
                  px_differ_gt8_share=round(float((px > 8).mean()), 6), mean_abs=round(float(d.mean()), 4), max_abs=int(d.max()))
lead = [x for x in os.listdir(OURS) if x.endswith('.png')]
json.dump(dict(ref=REF, ours=OURS, lead_png=lead, variants=res), open(OUT, 'w'), indent=1)
for n, v in res.items(): print(n, v)
print('lead', lead, res.get('TRACE_x_v8306H3_grey_page.png'))
