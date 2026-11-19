"""Embed ROI-sized A-hn member maps (infer.py --roi writes out_shape = frame size) into the full render raster (NaN outside the
frame), so fuse_v7.py can multiply them with TRACE's full-raster map. Idempotent: maps already at render size are left alone.
Usage (in fl-claim-a7:final): python pad_frame.py <map.npy> [...]   (each needs its infer.py sidecar <map>.json)
"""
import json, os, sys
import numpy as np
for p in sys.argv[1:]:
    j = json.load(open(p[:-4] + '.json')); H, W = j['render_shape'][1:]; a = np.load(p)
    if a.shape == (H, W): print(p, 'already full raster'); continue
    # the map covers the roi (no --grid-origin: 'frame' is the one-tile-padded admission frame) or the frame (clip_to_roi runs,
    # where the two coincide): place it by whichever box matches its shape
    box = next((b for b in (j.get('roi'), j.get('frame')) if b and a.shape == (b[1] - b[0], b[3] - b[2])), None)
    if box is None: raise SystemExit(f'{p}: shape {a.shape} matches neither roi {j.get("roi")} nor frame {j.get("frame")}')
    y0, y1, x0, x1 = box
    out = np.full((H, W), np.nan, a.dtype); out[y0:y1, x0:x1] = a
    np.save(p[:-4] + '.tmp.npy', out); os.replace(p[:-4] + '.tmp.npy', p)
    j['padded_to_render'] = True; json.dump(j, open(p[:-4] + '.json', 'w'), indent=1)
    print(p, a.shape, '->', out.shape)
