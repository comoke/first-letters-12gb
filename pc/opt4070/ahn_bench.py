"""ahn_bench.py : A-hn forward variants on real B0 tiles, on this GPU: tiles/s, peak VRAM, and agreement with the stock forward.

Runs inside the winner's image (fl-claim-a7:final), where infer.py and the weights live:
  python ahn_bench.py --ckpt CK.safetensors --zarr RENDER.zarr --out OUT.json [--tiles 256] [--variants a,b,..] [--batches 4,8]
                      [--profile] [--benchmark 0,1]
Tiles: the B0 native window exactly as infer.render cuts it (roi, grid origin 115 694, clip_to_roi), every k-th tile of the list.
Reference: stock at batch 4 with cudnn.benchmark off (the shipped call).  Each variant is timed after 3 warm-up batches with
torch.cuda.synchronize around the whole pass; its sigmoid probabilities are compared with the reference (pooled Pearson r, max abs).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

BENCH = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'
sys.path.insert(0, BENCH); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer                                                                   # noqa: E402
import ahn_variants as AV                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--zarr', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--roi', type=int, nargs=4, default=(1516, 2915, 2038, 7638)); ap.add_argument('--grid-origin', type=int, nargs=2, default=(115, 694))
ap.add_argument('--tiles', type=int, default=256); ap.add_argument('--variants', default='stock,cl3,fuse,fuse_cl3,half,half_cl3,fold2d')
ap.add_argument('--batches', default='4,8'); ap.add_argument('--benchmark', default='0,1'); ap.add_argument('--profile', default='', help='variant to profile with torch.profiler')
ap.add_argument('--vram', type=float, default=9.0)
a = ap.parse_args()
dev = 'cuda'; infer.vram_cap(a.vram)
res = dict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0), tiles=a.tiles, rows=[])


def save():
    json.dump(res, open(a.out + '.tmp', 'w'), indent=1); os.replace(a.out + '.tmp', a.out)


V = infer.Volume(a.zarr)
y0, y1, x0, x1 = a.roi
cov = V.read(20, 21, y0, y1, x0, x1)[0] > 0
org = ((a.grid_origin[0] - y0) % 21, (a.grid_origin[1] - x0) % 21)
pts = infer.positions(cov, 0.25, 21, org, (0, y1 - y0, 0, x1 - x0)); pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
res['tiles_in_window'] = int(len(pts))
pts = pts[np.linspace(0, len(pts) - 1, a.tiles).astype(int)]
t = time.time(); st = torch.from_numpy(infer.prep_stack(V.read(8, 32, y0, y1, x0, x1))).to(dev); res['read_s'] = round(time.time() - t, 1)
P = torch.from_numpy(np.ascontiguousarray(pts)).to(dev); ar = torch.arange(64, device=dev)


def tiles(i, b):
    q = P[i:i + b]; iy = q[:, 0:1] + ar[None]; ix = q[:, 1:2] + ar[None]
    return st[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3).float().mul_(1 / 255.)


@torch.inference_mode()
def run(f, b, timed=True):
    for i in range(0, 3 * b, b): f(tiles(i, b))                                # warm-up (cudnn.benchmark picks its algorithms here)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time(); out = []
    for i in range(0, len(pts), b): out.append(torch.sigmoid(f(tiles(i, b)).float())[:, 0])
    torch.cuda.synchronize(); dt = time.time() - t
    return torch.cat(out), dt, torch.cuda.max_memory_allocated() / 2 ** 30


model, info = infer.load_model(a.ckpt, 1.0, dev)
torch.backends.cudnn.benchmark = False
ref, dt, mem = run(AV.build('stock', model)[0], 4)
ref2, _, _ = run(AV.build('stock', model)[0], 4)
res['ref'] = dict(tiles_per_s=round(len(pts) / dt, 2), peak_gb=round(mem, 2), rerun_max_abs=float((ref - ref2).abs().max()))
print('ref', res['ref'], flush=True); save()
rf = ref.double().flatten(); rf = rf - rf.mean()


def agree(p):
    d = p.double().flatten(); d = d - d.mean()
    return float((d @ rf) / torch.sqrt((d @ d) * (rf @ rf))), float((p - ref).abs().max()), float((p - ref).abs().mean())


if a.profile:
    from torch.profiler import profile, ProfilerActivity
    f = AV.build(a.profile, model)[0]
    with torch.inference_mode():
        for i in range(0, 12, 4): f(tiles(i, 4))
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for i in range(0, 32, 4): torch.sigmoid(f(tiles(i, 4)).float())
            torch.cuda.synchronize()
    del f; torch.cuda.empty_cache()
    res['profile_by_op'] = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=40)
    res['profile_by_shape'] = prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total', row_limit=45, max_name_column_width=30,
                                                                              max_shapes_column_width=70)
    print(res['profile_by_op'], flush=True); print(res['profile_by_shape'], flush=True); save()

for bm in [int(v) for v in a.benchmark.split(',')]:
    torch.backends.cudnn.benchmark = bool(bm)
    for name in a.variants.split(','):
        try:
            f, mod = AV.build(name, model)
        except Exception as e:
            res['rows'].append(dict(variant=name, cudnn_benchmark=bm, error=repr(e)[:300])); print(res['rows'][-1], flush=True); save(); continue
        for b in [int(v) for v in a.batches.split(',')]:
            row = dict(variant=name, cudnn_benchmark=bm, batch=b)
            try:
                t = time.time(); p, dt, mem = run(f, b)
                r, mx, mean = agree(p)
                row.update(tiles_per_s=round(len(pts) / dt, 2), peak_gb=round(mem, 2), r=r, max_abs=mx, mean_abs=mean, total_s=round(time.time() - t, 1))
            except Exception as e:
                row['error'] = repr(e)[:300]; torch.cuda.empty_cache()
            res['rows'].append(row); print(row, flush=True); save()
        del f, mod; torch.cuda.empty_cache()
print('AHN_BENCH_DONE', flush=True)
