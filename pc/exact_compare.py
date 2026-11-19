import json, sys, numpy as np
O = sys.argv[1]; res = {}
def cmp(k, a, b):
    m = np.isfinite(a) & np.isfinite(b); d = np.abs(a[m] - b[m])
    res[k] = dict(n_px=int(m.sum()), ours_finite=int(np.isfinite(a).sum()), ref_finite=int(np.isfinite(b).sum()),
                  pearson=float(np.corrcoef(a[m], b[m])[0, 1]), mae=float(d.mean()), p99_abs=float(np.percentile(d, 99)),
                  max_abs=float(d.max()), equal_frac=float((a[m] == b[m]).mean()), within_1e3=float((d <= 1e-3).mean()))
    print(k, json.dumps(res[k]))
tr = np.load(f'{O}/fwd_tr.npy').astype(np.float32)
cmp('tr_vs_image', tr, np.load('/opt/fl/data/maps/fwd_tr.npy').astype(np.float32))
s0c = np.load(f'{O}/fwd_hn4_s0.npy').astype(np.float32)
full = np.full((4900, 7780), np.nan, np.float32); full[115:115 + s0c.shape[0], 694:694 + s0c.shape[1]] = s0c
cmp('s0_vs_image_f16', full, np.load('/opt/fl/data/members/fwd_hn4_s0.npy').astype(np.float32))
ref = np.load('/data/winner/hf-data/results/member_maps/fwd_hn4_s0.npy').astype(np.float32)
print('hf member_maps s0 shape', ref.shape, 'ours', s0c.shape)
cmp('s0_vs_hf_member', s0c if ref.shape == s0c.shape else full, ref)
json.dump(res, open(f'{O}/exact_fidelity.json', 'w'), indent=1)
