"""w053_fz.py: recompute labels.py's fused z (float32, exactly as labels.make) at the decisive w053 pixel from our teacher maps vs the
shipped ones, and swap one teacher at a time to find which teacher's difference moves the pixel across NEG_T."""
import json, numpy as np
OURS = '/data/train_e2e/self'; SH = '/data/winner/hf-data/data'; M = ('v8', 'v3', 'trace', 'rv2'); NEG_T = 0.3
PY, PX = 1015, 1902   # bbox frame (render y 1015, x 1966)
valid = np.load(f'{OURS}/labels/w053_labels.npz')['valid']
lab_o = np.load(f'{OURS}/labels/w053_labels.npz')['main']; lab_s = np.load(f'{SH}/3_machine_labels/w053_labels.npz')['main']
To = {k: np.load(f'{OURS}/teach/w053_{k}.npy').astype(np.float32) for k in M}
Ts = {k: np.load(f'{SH}/2_teacher_maps/w053_{k}.npy').astype(np.float32) for k in M}
for k in M:   # labels.py: valid &= isfinite(T[k]); check same valid both ways
    assert np.array_equal(valid, valid & np.isfinite(To[k]) & np.isfinite(Ts[k]))
def fz_at(T):
    Z = {}; st = {}
    for k in M:
        v = T[k][valid]; mu, sd = v.mean(), max(v.std(), 1e-6); Z[k] = (T[k] - mu) / sd; st[k] = (float(mu), float(sd))
    f = np.mean([Z[k] for k in M], 0); v = f[valid]; mu, sd = v.mean(), max(v.std(), 1e-6)
    return float((f[PY, PX] - mu) / sd), {k: float(Z[k][PY, PX]) for k in M}, st, (float(mu), float(sd))
out = dict(pixel_bbox_frame=[PY, PX], label_main_ours=int(lab_o[PY, PX]), label_main_shipped=int(lab_s[PY, PX]), NEG_T=NEG_T)
out['teacher_values'] = {k: dict(ours=float(To[k][PY, PX]), shipped=float(Ts[k][PY, PX]), dtype=str(np.load(f'{OURS}/teach/w053_{k}.npy', mmap_mode='r').dtype),
                                 n_px_differ=int((To[k] != Ts[k]).sum()),
                                 n_valid_px_differ=int(((To[k] != Ts[k]) & valid).sum())) for k in M}
for nm, T in (('ours', To), ('shipped', Ts)):
    fz, zk, st, fst = fz_at(T); out[f'fz_{nm}'] = dict(fz=fz, fz_le_NEG_T=bool(fz <= NEG_T), z_per_teacher=zk, teacher_mean_std=st, fused_mean_std=fst)
for k in M:   # ours with only teacher k taken from shipped
    T = dict(To); T[k] = Ts[k]; fz = fz_at(T)[0]; out[f'ours_with_shipped_{k}'] = dict(fz=fz, bg=bool(fz <= NEG_T))
json.dump(out, open('/data/train_e2e/w053/decisive_pixel.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
