"""ahn_bench.py : speed and agreement of the A-hn backends on the Mac (M5 Pro). One member, real B0 tiles.

Backends (each timed as tiles/s over --n tiles after a warm-up, and compared with the stock model in float32 on the CPU):
  stock-mps   : the winner's InkDetector on MPS, fp16 autocast (Stage 0's baseline)
  fold-mps    : ahn_fold.FoldedAhn on MPS, 'stack' and 'sum' forms, fp16 (half weights) and fp32
  coreml-*    : the converted Core ML program (ahn_coreml.py) on CPU_AND_NE, CPU_AND_GPU and ALL, plus where each op runs
                (MLComputePlan, macOS 14.4+)
Usage: python ahn_bench.py --ckpt CK --zarr B0_RENDER --mlpackage DIR/hn4_s0_stack.mlpackage [--n 64] [--backends ...]
Env: V8_DIR (ink8um). Never holds the GPU long: each backend runs --n tiles (default 64, about a minute at stock speed).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['V8_DIR'])
from ink8um import InkDetector                                                       # noqa: E402
import ahn_fold                                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--zarr', required=True); ap.add_argument('--mlpackage', default=None)
ap.add_argument('--n', type=int, default=64); ap.add_argument('--out', default='ahn_bench.json')
ap.add_argument('--at', type=int, nargs=2, default=(2116, 3038),
                help='top-left of the 256 x 1024 px tile block; the default is inside the B0 claim box of the full claim render')
ap.add_argument('--backends', nargs='*', default=['stock-mps', 'fold-mps-stack-fp16', 'fold-mps-sum-fp16', 'fold-mps-stack-fp32',
                                                  'coreml-CPU_AND_NE', 'coreml-CPU_AND_GPU', 'coreml-ALL'])
a = ap.parse_args()


def stock():
    m = InkDetector(**json.load(open(os.path.join(os.environ['V8_DIR'], 'config.json'))))
    from safetensors.torch import load_file
    m.load_state_dict(load_file(a.ckpt), strict=True)
    return m.eval()


import zarr                                                                          # noqa: E402
Z = zarr.open(a.zarr, mode='r'); Z = Z['0'] if not hasattr(Z, 'shape') else Z
y0, x0 = a.at
vol = np.minimum(np.asarray(Z[8:32, y0:y0 + 64 * 4, x0:x0 + 64 * 16]), 200)          # a 4 x 16 grid of tiles from the B0 box
tiles = np.stack([vol[:, y:y + 64, x:x + 64] for y in range(0, 256, 64) for x in range(0, 1024, 64)]).astype(np.float32) / 255.0
tiles = np.concatenate([tiles] * (a.n // len(tiles) + 1))[:a.n]
with torch.inference_mode():
    ref = stock()(torch.from_numpy(tiles[:8])).numpy()                               # stock, float32, CPU: the agreement reference
res = dict(n=a.n, torch=torch.__version__, results={})


def record(name, f, batch=1):
    f(tiles[:batch])                                                                 # warm-up (and compile)
    t = time.time()
    outs = [f(tiles[i:i + batch]) for i in range(0, a.n, batch)]
    dt = time.time() - t
    out = np.concatenate(outs)[:8]
    p, pr = 1 / (1 + np.exp(-out)), 1 / (1 + np.exp(-ref))
    r = dict(tiles_per_s=round(a.n / dt, 2), batch=batch, max_abs_logit_vs_cpu_fp32=float(np.abs(out - ref).max()),
             max_abs_prob_vs_cpu_fp32=float(np.abs(p - pr).max()))
    res['results'][name] = r
    print(name, r, flush=True)


def torch_backend(net, half, batch=1, channels_last=False):
    dev = torch.device('mps')
    net = net.to(dev)
    if half:
        net = net.half()
    if channels_last:                                                               # MPSGraph's fast conv2d layout on the M5 GPU
        net = net.to(memory_format=torch.channels_last)

    def f(x):
        with torch.inference_mode():
            y = net(torch.from_numpy(x).to(dev).to(torch.float16 if half else torch.float32))
            return y.float().cpu().numpy()
    return f


for name in a.backends:
    try:
        if name == 'stock-mps':
            m = stock().to('mps')

            def f(x, m=m):
                with torch.inference_mode(), torch.autocast('mps', dtype=torch.float16):
                    return m(torch.from_numpy(x).to('mps')).float().cpu().numpy()
            record(name, f)
        elif name.startswith('fold-mps'):
            _, _, mode, prec, *layout = name.split('-')
            for batch in (1, 2, 4):
                record(f'{name}-b{batch}', torch_backend(ahn_fold.fold(stock(), mode), prec == 'fp16', batch, layout == ['cl']), batch)
        elif name.startswith('coreml'):
            import coremltools as ct
            units = name.split('-', 1)[1]
            t = time.time()
            ml = ct.models.MLModel(a.mlpackage, compute_units=getattr(ct.ComputeUnit, units))
            load_s = time.time() - t
            record(name, lambda x, ml=ml: np.concatenate([ml.predict({'tiles': x[i:i + 1]})['logits'] for i in range(len(x))]))
            res['results'][name]['load_s'] = round(load_s, 1)
            try:                                                                     # where each op runs
                from coremltools.models.compute_plan import MLComputePlan
                compiled = ml.get_compiled_model_path()
                plan = MLComputePlan.load_from_path(compiled, compute_units=getattr(ct.ComputeUnit, units))
                prog = plan.model_structure.program
                devs = {}
                for fn in prog.functions.values():
                    for op in fn.block.operations:
                        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                        d = type(u.preferred_compute_device).__name__ if u else 'none'
                        devs[d] = devs.get(d, 0) + 1
                res['results'][name]['ops_by_device'] = devs
                print(name, 'ops by device', devs, flush=True)
            except Exception as e:
                res['results'][name]['ops_by_device'] = f'n/a: {e!r:.200}'
    except Exception as e:
        res['results'][name] = dict(error=repr(e)[:400]); print(name, 'FAILED', repr(e)[:400], flush=True)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
json.dump(res, open(a.out, 'w'), indent=1)
print('AHNBENCH_DONE', a.out, flush=True)
os._exit(0)                                                                          # torch-MPS can segfault at exit (villa #2392)
