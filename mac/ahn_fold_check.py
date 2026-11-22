"""ahn_fold_check.py : is ahn_fold.FoldedAhn the same function as the winner's InkDetector? CPU only.

  1. float64, reduced upsample size (48 x 128 x 128: same layers and weights, 1/8 of the memory): max |logit diff| on real B0 tiles,
     for both 3x3x3 forms ('stack', 'sum'). Rounding alone should leave ~1e-12.
  2. float32, full size (96 x 256 x 256): max |logit diff| and |prob diff| on the same tiles; seconds per tile on this CPU.
  3. The winner's infer.predict_array (its exact tile grid, window and stitch) on a B0 window, with the stock and the folded model,
     against the 4070's CUDA map of the same member (b0/runs/native): r and max |diff| over the window's pixels.
Usage: python ahn_fold_check.py --data VESUVIUS_DATA --ckpt .../hn4_s0_final/model.safetensors --out result.json [--win y0 y1 x0 x1]
Env: AHN_DIR = the winner's aclick/bench (infer.py); V8_DIR = their third_party/v8in (ink8um).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['AHN_DIR'])
import infer                                                                         # noqa: E402  (winner's, unchanged)
from ink8um import InkDetector                                                       # noqa: E402
import ahn_fold                                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--win', type=int, nargs=4, default=None); ap.add_argument('--win-size', type=int, default=64)
ap.add_argument('--skip-pipeline', action='store_true')
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
res = dict(ckpt=a.ckpt, torch=torch.__version__, threads=torch.get_num_threads())


def log(*s):
    print(time.strftime('%H:%M:%S'), *s, flush=True)


def stock(small=False):
    cfg = json.load(open(os.path.join(os.environ['V8_DIR'], 'config.json')))
    m = InkDetector(**cfg)
    from safetensors.torch import load_file
    m.load_state_dict(load_file(a.ckpt), strict=True)
    if small:
        m.input_upsample_depth, m.input_upsample_hw = 48, 128
    return m.eval()


# B0 crop (native_crop of b0.json); the native run's frame is its first 1399 x 5600 px, grid origin (6, 0) in this frame
import zarr                                                                          # noqa: E402
Z = zarr.open(f'{a.data}/b0/native/render.zarr', mode='r')['0']
meta = json.load(open(f'{a.data}/b0/b0.json'))
run = json.load(open(f'{a.data}/b0/runs/native/fwd_hn4_s0.json'))
fy0, fy1, fx0, fx1 = run['frame']; nc = meta['native_crop']
assert (fy0, fx0) == (nc[0], nc[2]) and run['layer0'] == 8 and run['tta'] == 1 and run['nsig'] == 3 and run['clip_to_roi']
origin = ((run['grid_origin'][0] - fy0) % 21, (run['grid_origin'][1] - fx0) % 21)
ref_full = np.load(f'{a.data}/b0/runs/native/fwd_hn4_s0.npy', mmap_mode='r')
ref = np.asarray(ref_full[fy0:fy1, fx0:fx1], dtype=np.float32)
by0, by1, bx0, bx1 = meta['box_in_crop']
if a.win is None:                                                                    # the box window where the CUDA map varies most
    s, best = a.win_size, None
    for y in range(by0 + 64, by1 - 64 - s, 32):
        for x in range(bx0 + 64, bx1 - 64 - s, 32):
            v = ref[y:y + s, x:x + s]
            if np.isfinite(v).all() and (best is None or v.std() > best[0]): best = (float(v.std()), y, x)
    a.win = [best[1], best[1] + s, best[2], best[2] + s]
wy0, wy1, wx0, wx1 = a.win
res['window_in_crop'] = a.win; res['grid_origin_in_crop'] = list(origin)
log('window', a.win, 'grid origin', origin)

# two real tiles from the window, as band_sums builds them: layers 8..31, clipped at 200, / 255
vol = np.minimum(np.asarray(Z[8:32, fy0 - nc[0]:fy1 - nc[0], fx0 - nc[2]:fx1 - nc[2]]), 200)
tiles = np.stack([vol[:, wy0:wy0 + 64, wx0:wx0 + 64], vol[:, wy1 - 64:wy1, wx1 - 64:wx1]]).astype(np.float32) / 255.0
tiles = torch.from_numpy(tiles)

# 1. float64, reduced size
with torch.inference_mode():
    m = stock(small=True)
    f = {mode: ahn_fold.fold(m, mode, torch.float64) for mode in ('stack', 'sum')}
    m = m.double()
    ref64 = m(tiles.double())
    for mode, fm in f.items():
        out = fm(tiles.double())
        d = (out - ref64).abs().max().item()
        res[f'f64_small_{mode}_max_abs_logit_diff'] = d
        log(f'float64 small {mode}: max |logit diff| {d:.3e} (logit range {ref64.min().item():.2f}..{ref64.max().item():.2f})')
    del m, f, ref64

# 2. float32, full size
with torch.inference_mode():
    m = stock()
    fm = ahn_fold.fold(m, 'stack')
    for name, net in (('stock', m), ('folded', fm), ('folded_sum', ahn_fold.fold(m, 'sum'))):
        net(tiles[:1])                                                               # warm-up
        t = time.time(); out = net(tiles); dt = (time.time() - t) / len(tiles)
        res[f'f32_{name}_s_per_tile'] = round(dt, 2)
        if name == 'stock':
            o_ref = out
        else:
            res[f'f32_full_{name}_max_abs_logit_diff'] = (out - o_ref).abs().max().item()
            res[f'f32_full_{name}_max_abs_prob_diff'] = (torch.sigmoid(out) - torch.sigmoid(o_ref)).abs().max().item()
        log(f'{name}: {dt:.2f} s per tile (CPU fp32, {torch.get_num_threads()} threads)'
            + ('' if name == 'stock' else f'; max |logit diff| {res[f"f32_full_{name}_max_abs_logit_diff"]:.3e}, '
                                          f'max |prob diff| {res[f"f32_full_{name}_max_abs_prob_diff"]:.3e}'))

# 3. the winner's pipeline on the window: stock and folded (CPU fp32) against the 4070's CUDA map (fp16 autocast)
if not a.skip_pipeline:
    cov = np.asarray(Z[20, fy0 - nc[0]:fy1 - nc[0], fx0 - nc[2]:fx1 - nc[2]]) > 0
    vol24 = np.asarray(Z[8:32, fy0 - nc[0]:fy1 - nc[0], fx0 - nc[2]:fx1 - nc[2]])
    maps = {}
    for name, net in (('stock', m), ('folded', fm)):
        t = time.time()
        mp = infer.predict_array(net, vol24, cov, reverse=False, tta=1, nsig=3, min_cov=0.25, stride=21, origin=origin,
                                 roi=(wy0, wy1, wx0, wx1), batch=2, device='cpu')
        maps[name] = mp[wy0:wy1, wx0:wx1]
        res[f'pipe_{name}_s'] = round(time.time() - t, 1)
        log(f'pipeline {name}: {res[f"pipe_{name}_s"]} s')
    r_ = ref[wy0:wy1, wx0:wx1]
    res['pipe_tiles'] = int(len(infer.positions(cov, 0.25, 21, origin, (wy0, wy1, wx0, wx1))))
    for x, y, nm in ((maps['folded'], maps['stock'], 'folded_vs_stock_cpu'), (maps['stock'], r_, 'stock_cpu_vs_cuda'),
                     (maps['folded'], r_, 'folded_cpu_vs_cuda')):
        k = np.isfinite(x) & np.isfinite(y)
        res[f'{nm}_r'] = float(np.corrcoef(x[k], y[k])[0, 1]); res[f'{nm}_max_abs'] = float(np.abs(x[k] - y[k]).max())
        res[f'{nm}_px'] = int(k.sum()); res[f'{nm}_nan_mismatch'] = int((np.isfinite(x) != np.isfinite(y)).sum())
        log(f'{nm}: r {res[nm + "_r"]:.8f}, max |diff| {res[nm + "_max_abs"]:.2e} over {res[nm + "_px"]} px, NaN mismatches {res[nm + "_nan_mismatch"]}')
json.dump(res, open(a.out, 'w'), indent=1)
log('FOLDCHK_DONE', a.out)
