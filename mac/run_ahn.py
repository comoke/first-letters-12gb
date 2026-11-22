"""A-hn member (v8-in fine-tune) on MPS with the winner's infer.py, unchanged except for the device and precision.
Same recipe as the PC's ink_run.sh native B0 run: --layer0 8 --tta 1 --nsig 3 --stride 21 --min-cov 0.25, --roi + --grid-origin,
clip_to_roi=True (infer_clip.py). Resumable through OUT.parts/ exactly like infer.py.
Usage: python run_ahn.py --ckpt CK --zarr Z --out OUT.npy [--roi y0 y1 x0 x1] [--grid-origin gy gx] [--batch 8] [--max-minutes M]
       [--max-tiles N]   (N > 0: only the first N tiles, for benchmarks; writes no map)
Env MAC_AMP = ref | fp32 | fp16 | bf16 (mps_shim.py).
--backend stock (default; the winner's model as loaded) | fold (ahn_fold.FoldedAhn: the same weights, 2-D ops only) |
          coreml (--mlpackage from ahn_coreml.py, --units CPU_AND_NE | CPU_AND_GPU | ALL; tiles go one at a time) |
          lanes (the same Core ML program on the Neural Engine and the GPU at once, ahn_lanes.py; --lanes pattern, default GGNGN)
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
ap.add_argument('--backend', default='stock', choices=('stock', 'fold', 'coreml', 'lanes'))
ap.add_argument('--fold-mode', default='stack'); ap.add_argument('--mlpackage', default=None); ap.add_argument('--units', default='ALL')
ap.add_argument('--lanes', default='GGNGN', help='--backend lanes: which unit takes each tile in turn, N the Neural Engine, G the GPU')
a = ap.parse_args()
dev = S.DEVICE
T0 = time.time()
infer.install_sigterm()
V = infer.Volume(a.zarr)
model, info = infer.load_model(a.ckpt, 1.0, dev)
if a.backend == 'fold':
    import ahn_fold
    model = ahn_fold.fold(model.cpu().eval(), a.fold_mode).to(dev)
elif a.backend == 'coreml':
    import coremltools as ct

    class CoreMLNet(torch.nn.Module):
        """The Core ML program as band_sums calls a model: (B, 24, 64, 64) -> (B, 1, 64, 64) logits on the same device."""
        def __init__(self):
            super().__init__()
            self.ml = ct.models.MLModel(a.mlpackage, compute_units=getattr(ct.ComputeUnit, a.units))

        def forward(self, x):
            xs = x.float().cpu().numpy()
            return torch.from_numpy(np.concatenate([self.ml.predict({'tiles': xs[i:i + 1]})['logits'] for i in range(len(xs))])).to(x.device)
    model = CoreMLNet()
elif a.backend == 'lanes':
    import ahn_lanes

    class LanesNet(torch.nn.Module):
        """The Core ML program on the Neural Engine and the GPU at once, as band_sums calls a model."""
        def __init__(self):
            super().__init__()
            self.lanes = ahn_lanes.Lanes(a.mlpackage, a.lanes)

        def forward(self, x):
            return torch.from_numpy(self.lanes(x.float().cpu().numpy())).to(x.device)
    model = LanesNet()
info['backend'] = dict(lanes=a.lanes, backend=a.backend, fold_mode=a.fold_mode, mlpackage=a.mlpackage, units=a.units)

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
                  extra_meta=dict(mac=dict(device=dev, amp=S.AMP, torch=torch.__version__, batch=a.batch, **info['backend'])))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
print(json.dumps(dict(rc=rc, wall_s=round(time.time() - T0, 1), mps_driver_peak_gb=round(peak[0], 2), max_rss_gb=round(rss, 2))), flush=True)
sys.stdout.flush(); os._exit(rc)                                             # torch-MPS can segfault at exit (villa #2392)
