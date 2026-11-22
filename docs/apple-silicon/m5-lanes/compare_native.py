"""Mac two-lane A-hn maps (ROI frame 1399 x 5600, rows 1516.., cols 2038..) against the 4070's native maps (full render raster)."""
import json, sys
import numpy as np
D = sys.argv[1]; NAT = sys.argv[2]
y0, x0 = 1516, 2038
res = {}
for m in ('s0', 's1', 's2', 'decoy'):
    mac = np.load(f'{D}/{m}_lanes.npy').astype(np.float64)
    c = np.load(f'{NAT}/fwd_hn4_{m}.npy', mmap_mode='r')[y0:y0 + mac.shape[0], x0:x0 + mac.shape[1]].astype(np.float64)
    fm, fc = np.isfinite(mac), np.isfinite(c); k = fm & fc
    d = np.abs(mac[k] - c[k])
    res[m] = dict(px=int(k.sum()), nan_mismatch=int((fm != fc).sum()), pearson=float(np.corrcoef(mac[k], c[k])[0, 1]), max_abs=float(d.max()),
                  same_8bit=float((np.round(mac[k] * 255) == np.round(c[k] * 255)).mean()),
                  wall_s=json.load(open(f'{D}/{m}_lanes.json')).get('minutes_last_session'))
    print(m, res[m], flush=True)
json.dump(res, open(f'{D}/compare_native.json', 'w'), indent=1)
