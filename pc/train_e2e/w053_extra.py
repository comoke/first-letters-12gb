"""w053_extra.py: identify the extra w053 tile (ours vs shipped packs) and the quantity in train_self.src_306 that decides it."""
import sys, json, numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/self/code')
import train_self as TS, v8306lib as L
print('v8306lib', L.__file__, 'TILE', L.TILE, 'PADM', TS.PADM, flush=True)
OURS = '/data/train_e2e/self'; SH = '/data/winner/hf-data/data'
T = L.TILE; PADM = TS.PADM; MIN_LAB = 0.2; out = {'tile_px': T, 'padm': PADM, 'min_lab': MIN_LAB, 'threshold_px': MIN_LAB * T * T}
po = np.load(f'{OURS}/packs/306_w053.npz'); ps = np.load(f'{SH}/3b_cloud_packs/306_w053.npz')
lo = np.load(f'{OURS}/labels/w053_labels.npz'); ls = np.load(f'{SH}/3_machine_labels/w053_labels.npz')
assert np.array_equal(lo['bbox'], ls['bbox']); y0, y1, x0, x1 = [int(v) for v in lo['bbox']]; out['bbox'] = [y0, y1, x0, x1]
out['valid_equal'] = bool(np.array_equal(lo['valid'], ls['valid']))
for v in ('main', 'sc', 'decoy'):
    so = {tuple(r) for r in po[f'org_{v}'].tolist()}; ss = {tuple(r) for r in ps[f'org_{v}'].tolist()}
    extra = sorted(so - ss); miss = sorted(ss - so)
    rec = dict(n_ours=len(so), n_shipped=len(ss), only_ours=extra, only_shipped=miss, tiles={})
    for (oy, ox) in extra + miss:
        Y, X = oy - PADM, ox - PADM   # bbox-frame top-left
        idx_o = [i for i, r in enumerate(po[f'org_{v}'].tolist()) if tuple(r) == (oy, ox)]
        t = dict(org_padded=[oy, ox], bbox_frame_yx=[Y, X], render_yx=[Y + y0, X + x0], index_in_ours=idx_o,
                 grid_index=[Y // 32, X // 32])
        for nm, lb in (('ours', lo), ('shipped', ls)):
            w = lb[v][Y:Y + T, X:X + T]
            t[nm] = dict(labelled_px=int((w != 255).sum()), ink_px=int((w == 1).sum()), bg_px=int((w == 0).sum()),
                        frac=float((w != 255).sum() / (T * T)), valid_px=int(lb['valid'][Y:Y + T, X:X + T].sum()))
        wo, ws = lo[v][Y:Y + T, X:X + T], ls[v][Y:Y + T, X:X + T]
        d = np.argwhere(wo != ws)
        t['diff_px_in_tile'] = int(len(d))
        trans = {}
        for a, b in zip(wo[wo != ws].tolist(), ws[wo != ws].tolist()): trans[f'{a}->{b}'] = trans.get(f'{a}->{b}', 0) + 1
        t['diff_transitions_ours_to_shipped'] = trans
        fo = lo['fz'][Y:Y + T, X:X + T].astype(np.float32); fs = ls['fz'][Y:Y + T, X:X + T].astype(np.float32)
        t['diff_px_fz'] = [dict(yx=[int(p[0]) + Y, int(p[1]) + X], ours=int(wo[p[0], p[1]]), shipped=int(ws[p[0], p[1]]),
                                fz_ours=float(fo[p[0], p[1]]), fz_shipped=float(fs[p[0], p[1]])) for p in d[:40]]
        rec['tiles'][f'{oy},{ox}'] = t
    out[v] = rec
# whole-region label diffs (main), by transition, and the fz around them
for v in ('main', 'sc'):
    m = lo[v] != ls[v]; tr = {}
    for a, b in zip(lo[v][m].tolist(), ls[v][m].tolist()): tr[f'{a}->{b}'] = tr.get(f'{a}->{b}', 0) + 1
    out[f'region_{v}_diff'] = dict(n=int(m.sum()), transitions=tr)
fo = lo['fz'].astype(np.float32); fs = ls['fz'].astype(np.float32); vv = lo['valid']
out['fz_equal_valid'] = int((fo[vv] == fs[vv]).sum()); out['fz_valid_n'] = int(vv.sum())
out['fz_maxabs_diff'] = float(np.nanmax(np.abs(fo - fs)))
# the threshold margin for every candidate tile on the grid, ours vs shipped
Il_o = L.integral(lo['main'] != 255); Il_s = L.integral(ls['main'] != 255)
H, Wd = lo['main'].shape
Y, X = np.meshgrid(np.arange(0, H - T + 1, 32), np.arange(0, Wd - T + 1, 32), indexing='ij'); Y, X = Y.ravel(), X.ravel()
bo = L.box_sum(Il_o, Y, X); bs = L.box_sum(Il_s, Y, X); thr = MIN_LAB * T * T
flip = np.nonzero((bo >= thr) != (bs >= thr))[0]
out['grid_main_tiles_crossing_minlab'] = [dict(yx=[int(Y[i]), int(X[i])], ours=int(bo[i]), shipped=int(bs[i])) for i in flip]
out['grid_main_tiles_labelled_changed'] = int((bo != bs).sum())
json.dump(out, open('/data/train_e2e/w053/extra_tile.json', 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in out if k not in ('main', 'sc', 'decoy')}, default=str)[:3000])
for v in ('main', 'sc', 'decoy'):
    r = out[v]; print(v, r['n_ours'], r['n_shipped'], r['only_ours'], r['only_shipped'])
    for k, t in r['tiles'].items(): print(' ', k, {kk: t[kk] for kk in t if kk != 'diff_px_fz'})
