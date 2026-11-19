"""Forward-pass throughput of the A-hn network (v8-in arch) on MPS: precision x batch, plus a per-stage time split.
Usage: python bench_model.py [--batches 4 8 16] [--amps fp32 fp16 bf16] [--iters 6] [--profile]"""
import os, sys, time, json, argparse
sys.path.insert(0, os.environ['V8_DIR'])
import torch, torch.nn.functional as F
from ink8um import InkDetector
ap = argparse.ArgumentParser(); ap.add_argument('--batches', type=int, nargs='+', default=[4, 8, 16])
ap.add_argument('--amps', nargs='+', default=['fp32', 'fp16', 'bf16']); ap.add_argument('--iters', type=int, default=6)
ap.add_argument('--profile', action='store_true'); ap.add_argument('--device', default='mps')
a = ap.parse_args(); dev = a.device
sync = torch.mps.synchronize if dev == 'mps' else (lambda: None)
m = InkDetector.from_pretrained(os.environ['V8_DIR']).to(dev).eval()
DT = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}
res = []
for amp in a.amps:
    for B in a.batches:
        x = torch.rand(B, 24, 64, 64, device=dev) * (200 / 255)
        with torch.inference_mode(), torch.autocast(dev, dtype=DT[amp] or torch.float32, enabled=DT[amp] is not None):
            try:
                m(x); sync()
                t = time.time()
                for _ in range(a.iters): m(x)
                sync(); dt = (time.time() - t) / a.iters
            except Exception as e:
                print(amp, B, 'FAIL', repr(e)[:200]); continue
        r = dict(amp=amp, batch=B, ms_per_batch=round(dt * 1e3, 1), tiles_per_s=round(B / dt, 1),
                 driver_gb=round(torch.mps.driver_allocated_memory() / 2 ** 30, 2) if dev == 'mps' else 0)
        print(json.dumps(r), flush=True); res.append(r)
        if dev == 'mps': torch.mps.empty_cache()

if a.profile:     # stage split: time each top-level stage with syncs (fp16, batch 8)
    B = int(os.environ.get("PROF_B", 1)); x = torch.rand(B, 24, 64, 64, device=dev) * (200 / 255)
    amp = os.environ.get('PROF_AMP', 'fp16')
    with torch.inference_mode(), torch.autocast(dev, dtype=DT[amp] or torch.float32, enabled=DT[amp] is not None):
        bb = m.backbone; T = {}
        def tic(k, f, *args):
            sync(); t = time.time(); r = f(*args); sync(); T[k] = T.get(k, 0) + time.time() - t; return r
        for it in range(4):
            if it == 1: T = {}
            h = tic('instnorm', m.normalization, x[:, None])
            h = tic('upsample_trilinear', lambda z: F.interpolate(z, size=(96, 256, 256), mode='trilinear', align_corners=False), h)
            h = tic('stem_conv7x7x7', bb.conv1, h)
            h = tic('stem_bn_relu_pool', lambda z: bb.maxpool(bb.relu(bb.bn1(z))), h)
            f1 = tic('layer1', bb.layer1, h); f2 = tic('layer2', bb.layer2, f1); f3 = tic('layer3', bb.layer3, f2); f4 = tic('layer4', bb.layer4, f3)
            p = tic('depth_maxpool', lambda fs: [torch.max(f, dim=2)[0] for f in fs], [f1, f2, f3, f4])
            tic('decoder2d', m.decoder, p)
        tot = sum(T.values())
        print(json.dumps({'profile_amp': amp, 'batch': B, 'total_ms': round(tot / 3 * 1e3, 1),
                          'split_pct': {k: round(100 * v / tot, 1) for k, v in T.items()}}), flush=True)
        print('shapes: stem out', tuple(bb.conv1(F.interpolate(m.normalization(x[:1, None]), size=(96, 256, 256), mode='trilinear')).shape))
