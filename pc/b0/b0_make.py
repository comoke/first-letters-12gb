"""B0 benchmark inputs (research program 2026-11): crops of the PHerc0306B claim render, written as 41-layer OME-Zarr renders in
the same layout as the source (zarr v2, levels 0-5, chunks 41x128x128, uint8), so ink_v2.py / infer.py read them unchanged.
  native   : the claim box (config_v7.json 'box') in a window on the shipped tiling phase; the native RUN uses the full render
             with that window (TRACE --crop, A-hn --roi/--grid-origin 115 694/clip_to_roi); the native zarr feeds shuf and deg936
  shuf     : native with the 41 layers permuted (fixed seed): a depth-shuffle negative control
  notext   : a native-height, half-width window >= 0.5 cm from the native crop, >= 90% coverage, lowest shipped TRACE x A p99
  deg936   : native resampled to 9.36 um in y, x and z (zoom 8.64/9.36, linear; 41 layers about the surface layer 20,
             edge layers clamped), i.e. what an eligible 9.36 um scan of the same papyrus would look like (no noise model yet)
Also writes b0/b0.json (crop boxes in render px, box inside each crop, zoom) and the reference crops (shipped maps).
Usage (in fl-claim-a7:final, CPU only): python b0_make.py <render.zarr> <outdir>
"""
import json, math, os, sys
import numpy as np, zarr
from scipy import ndimage
SRC, OUT = sys.argv[1], sys.argv[2]
MARGIN, ZOOM = 150, 8.64 / 9.36
os.makedirs(OUT, exist_ok=True)
z0 = zarr.open(SRC, mode='r')['0']; L, H, W = z0.shape
b = json.load(open('/opt/fl/claim/config_v7.json'))['box']; K = 1000 / 8.64      # same formula as run.sh BOXPX
box = (math.ceil((b['v0'] + 21) * K - 0.5), math.floor((b['v1'] + 21) * K - 0.5) + 1,
       math.ceil((b['u0'] + 36) * K - 0.5), math.floor((b['u1'] + 36) * K - 0.5) + 1)
# Native window on the shipped tiling phase (a crop at an arbitrary origin re-tiles inference and breaks
# the pixelwise comparison with the shipped maps). The native variant runs on the FULL render with these windows:
#   TRACE (ink_v2.py) row tiles start at the crop's y0 in steps of --tile 1400; shipped crop y0 = 116, so y0 = 116 + 1400 k
#   A-hn: --grid-origin 115 694 (stride 21) + clip_to_roi; TRACE's hecate stride 16: x0 = 694 + lcm(16, 21) k = 694 + 336 k
TILE_T, LCM = 1400, 336
y0 = max(116 + TILE_T * k for k in range(4) if 116 + TILE_T * k <= box[0] - 100)
x0 = max(694 + LCM * k for k in range(30) if 694 + LCM * k <= box[2] - MARGIN)
# x1 = the shipped TRACE crop's x end (run.sh INK_CROP 116,4167,694,7639): each TRACE row tile reads columns x0-192 .. x1+192 and
# its input statistics depend on that extent (x1 = box + 200 gave r 0.9954 vs shipped; see docs/b0)
nat = (y0, min(H, y0 + TILE_T), x0, 7639)
assert nat[0] <= box[0] and nat[1] >= box[1] and nat[3] >= box[3], (nat, box)
h, w = nat[1] - nat[0], nat[3] - nat[2]

# no-text window: lowest p99 of the shipped product map among full-coverage windows that keep >= 1 cm from the native crop
prod = np.load('/opt/fl/data/maps/fwd_prod.npy', mmap_mode='r')
best = None; gap = int(5 * K)          # >= 0.5 cm from the native crop; shapes of about half the native crop's area
for nh, nw in [(h, w // 2), (700, 4700), (600, 5500), (550, 6000)]:
  for y in range(0, H - nh + 1, 50):
    for x in range(0, W - nw + 1, 100):
        if not (y + nh + gap <= nat[0] or y >= nat[1] + gap or x + nw + gap <= nat[2] or x >= nat[3] + gap): continue
        p = np.asarray(prod[y:y + nh:4, x:x + nw:4], np.float32)
        if np.isfinite(p).mean() < 0.90: continue
        s = float(np.nanpercentile(p, 99))
        if best is None or s < best[0]: best = (s, (y, y + nh, x, x + nw))
if best is None:
    raise SystemExit('no >=90%-coverage half-width no-text window at least 0.5 cm from the native crop')
nt = best[1]


def write(name, vol):
    """vol (41, h, w) uint8 -> OUT/name/render.zarr with levels 0-5 (mean 2x2 in y, x per level, as the source)"""
    p = f'{OUT}/{name}/render.zarr'
    if os.path.exists(f'{p}/5/.zarray'): print(name, 'exists, kept', flush=True); return
    g = zarr.open_group(p, mode='w', zarr_format=2)
    src_attrs = dict(zarr.open_group(SRC, mode='r').attrs)
    src_attrs['canvas_size'] = [vol.shape[2], vol.shape[1]]; src_attrs['b0'] = name
    g.attrs.update(src_attrs)
    a = vol
    for lev in range(6):
        g.create_array(str(lev), shape=a.shape, chunks=(41, 128, 128), dtype='u1', compressors=z0.compressors,
                       chunk_key_encoding={'name': 'v2', 'separator': '/'})[:] = a
        hh, ww = a.shape[1] // 2 * 2, a.shape[2] // 2 * 2
        a = a[:, :hh, :ww].reshape(41, hh // 2, 2, ww // 2, 2).astype(np.float32).mean((2, 4)).round().astype(np.uint8)
    print(name, vol.shape, flush=True)


native = np.asarray(z0[:, nat[0]:nat[1], nat[2]:nat[3]])
write('native', native)
perm = np.random.default_rng(0).permutation(41)
write('shuf', native[perm])
write('notext', np.asarray(z0[:, nt[0]:nt[1], nt[2]:nt[3]]))
# 9.36 um: in-plane zoom per layer, then z resample about layer 20 (output layer k samples input 20 + (k - 20) / ZOOM)
dh, dw = round(h * ZOOM), round(w * ZOOM)
zs = np.clip(20 + (np.arange(41) - 20) / ZOOM, 0, 40); k0 = np.floor(zs).astype(int); k1 = np.minimum(k0 + 1, 40)
cache = {}
def zl(k):                                   # in-plane zoomed input layer k (keeps at most the two layers in use)
    if k not in cache:
        for o in [o for o in cache if o < k - 1]: del cache[o]
        cache[k] = ndimage.zoom(native[k].astype(np.float32), (dh / h, dw / w), order=1)
    return cache[k]
deg = np.empty((41, dh, dw), np.uint8)
for k in range(41):                          # one output layer at a time (memory: ~3 float32 planes)
    f = zs[k] - k0[k]; deg[k] = ((1 - f) * zl(k0[k]) + f * zl(k1[k])).round().clip(0, 255).astype(np.uint8)
del cache
write('deg936', deg)

ref = {}
for nm, p in [('prod', '/opt/fl/data/maps/fwd_prod.npy'), ('prodd', '/opt/fl/data/maps/fwd_prodd.npy'),
              ('tr', '/opt/fl/data/maps/fwd_tr.npy'), ('ensA', '/opt/fl/data/maps/fwd_ensA.npy')]:
    if not os.path.exists(p): raise SystemExit(f'required reference map missing: {p}')
    if True:
        m = np.load(p, mmap_mode='r')
        np.save(f'{OUT}/ref_{nm}_native.npy', np.asarray(m[nat[0]:nat[1], nat[2]:nat[3]])); ref[nm] = p
        np.save(f'{OUT}/ref_{nm}_notext.npy', np.asarray(m[nt[0]:nt[1], nt[2]:nt[3]]))
meta = dict(source=SRC, claim_box_render_px=box, native_crop=nat, notext_crop=nt, notext_prod_p99=best[0], margin=MARGIN,
            box_in_crop=[box[0] - nat[0], box[1] - nat[0], box[2] - nat[2], box[3] - nat[2]], shuf_perm=perm.tolist(),
            deg936=dict(zoom=ZOOM, shape=[41, dh, dw], z_rule='out k <- in 20 + (k-20)/zoom, linear, clamped'),
            area_cm2=h * w * (8.64e-4) ** 2, refs=ref, render_shape=[H, W],
            area_cm2_by_variant=dict(native=h * w * (8.64e-4) ** 2, shuf=h * w * (8.64e-4) ** 2, deg936=h * w * (8.64e-4) ** 2,
                                     notext=(nt[1] - nt[0]) * (nt[3] - nt[2]) * (8.64e-4) ** 2),
            native_run=dict(render=SRC, trace_crop=f'{nat[0]},{nat[1]},{nat[2]},{nat[3]}', roi=[nat[0], nat[1] - 1, nat[2], nat[3] - 1],
                            grid_origin=[115, 694], clip_to_roi=True))
json.dump(meta, open(f'{OUT}/b0.json', 'w'), indent=1)
print(json.dumps(meta, indent=1))
