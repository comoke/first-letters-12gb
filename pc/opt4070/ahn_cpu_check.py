"""ahn_cpu_check.py : the variants' arithmetic against the stock network in float32 on the CPU, one random tile (no GPU needed).
python ahn_cpu_check.py CK.safetensors   -> prints max abs logit difference per variant; exits 1 if any exceeds 1e-3."""
import os, sys, copy
import torch

BENCH = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'
sys.path.insert(0, BENCH); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer                                                                   # noqa: E402
import ahn_variants as AV, ahn_fold                                            # noqa: E402

torch.set_num_threads(4); torch.manual_seed(0)
model, _ = infer.load_model(sys.argv[1], 1.0, 'cpu')
x = (torch.rand(1, 24, 64, 64) * 200).round().clamp(0, 200) / 255.
bad = 0
with torch.inference_mode():
    ref = model(x)
    for name, mod in (('fuse', AV.fused(model)),
                      ('fuse_cl3', AV.cl3(AV.fused(model))),
                      ('split', AV.Split(AV.fused(model), False)),
                      ('split_cl3', AV.Split(AV.cl3(AV.fused(model)), True)),
                      ('fold2d', ahn_fold.fold(copy.deepcopy(model), 'stack').to(memory_format=torch.channels_last))):
        d = float((mod(x) - ref).abs().max()); bad += d > 1e-3
        print(f'{name}: max abs logit diff {d:.3e} (ref range {float(ref.min()):.2f}..{float(ref.max()):.2f})', flush=True)
print('CPU_CHECK_' + ('FAIL' if bad else 'OK'))
sys.exit(1 if bad else 0)
