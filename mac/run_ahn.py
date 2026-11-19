"""A-hn member (v8-in fine-tune) on MPS with the winner's infer.py, unchanged except for the device and precision.
Same recipe as the PC's ink_run.sh native B0 run: --layer0 8 --tta 1 --nsig 3 --stride 21 --min-cov 0.25, --roi + --grid-origin,
clip_to_roi=True (infer_clip.py). Resumable through OUT.parts/ exactly like infer.py.
Usage: python run_ahn.py --ckpt CK --zarr Z --out OUT.npy [--roi y0 y1 x0 x1] [--grid-origin gy gx] [--batch 8] [--max-minutes M]
       [--max-tiles N]   (N > 0: only the first N tiles, for benchmarks; writes no map)
Env MAC_AMP = ref | fp32 | fp16 | bf16 (mps_shim.py).
"""
import os, sys, time, json, argparse, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mps_shim as S                                                          # noqa: E402  (before the reference code)
sys.path.insert(0, os.environ['AHN_DIR'])
import numpy as np, torch                                                     # noqa: E402
import infer                                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--zarr', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--roi', type=int, nargs=4, default=None); ap.add_argument('--grid-origin', type=int, nargs=2, default=(0, 0))
ap.add_argument('--batch', type=int, default=8); ap.add_argument('--max-minutes', type=float, default=1e9)
ap.add_argument('--part-tiles', type=int, default=2500); ap.add_argument('--max-tiles', type=int, default=0)
ap.add_argument('--no-clip', action='store_true')
a = ap.parse_args()
dev = S.DEVICE
T0 = time.time()
infer.install_sigterm()
V = infer.Volume(a.zarr)
model, info = infer.load_model(a.ckpt, 1.0, dev)

peak = [0.0]
_bs = infer.band_sums
def band_sums(model, st, pts, tta=4, nsig=3, batch=8, device='cuda'):
    if a.max_tiles: pts = pts[:a.max_tiles]
    S.sync(); t = time.time()
    r = _bs(model, st, pts, tta, nsig, batch, device)
    S.sync(); dt = time.time() - t
    peak[0] = max(peak[0], S.mem_gb()[1])
    print(f'    band: {len(pts)} tiles {dt:.1f}s {len(pts) / dt:.1f} tiles/s  mps driver {S.mem_gb()[1]:.2f} GB', flush=True)
    if a.max_tiles: raise SystemExit(json.dumps(dict(tiles=len(pts), s=dt, tiles_per_s=len(pts) / dt, amp=S.AMP, batch=batch,
                                                     mps_driver_gb=peak[0])))
    return r
infer.band_sums = band_sums
rc = infer.render(model, info, V, 8, a.out, reverse=False, roi=a.roi, tta=1, nsig=3, min_cov=0.25, stride=21,
                  grid_origin=tuple(a.grid_origin), batch=a.batch, part_tiles=a.part_tiles, deadline=T0 + a.max_minutes * 60,
                  device=dev, clip_to_roi=not a.no_clip,
                  extra_meta=dict(mac=dict(device=dev, amp=S.AMP, torch=torch.__version__, batch=a.batch)))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
print(json.dumps(dict(rc=rc, wall_s=round(time.time() - T0, 1), mps_driver_peak_gb=round(peak[0], 2), max_rss_gb=round(rss, 2))), flush=True)
sys.exit(rc)
