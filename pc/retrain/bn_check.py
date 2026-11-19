"""BN running-stat check for train_cloud_hn.py's grad_ckpt='block' on identical data.
Checkpoint recompute runs each BN forward a second time in backward. The trainer snapshots the BN buffers after the forward
and restores them after backward. This checks that the restore leaves one BN update per micro-batch.
One optimizer step (ACCUM micro-batches, trainer loss incl. hard negatives) from the base v8-in on fixed micro-batches, three arms:
  plain (no checkpointing) | block (+ restore, = trainer) | block_norestore.
Usage (in fl-claim-a7:final): python bn_check.py <cloud_code dir> <packs dir> <out.json>
"""
import gc, json, os, sys
import numpy as np, torch
CC, PK, OUT = sys.argv[1:4]
sys.path.insert(0, CC)
import train_cloud_hn as T          # sets L.BASE_DIR from V8_DIR
T.PK = PK
L = T.L
ACCUM, N1447, W306, GAIN, SMOOTH = 4, 2, 0.5, 4.0, 0.25
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

anc, p306 = T.load_anchor('w058'), T.load_306('w109', 'main')
srcs = [anc, p306]
rng = np.random.default_rng(0)


def batches(micro):
    out = []
    for j in range(ACCUM):
        pr = [[0, rng.integers(len(anc['org']))] for _ in range(N1447)] + [[1, rng.integers(len(p306['org']))] for _ in range(micro - N1447)]
        out.append(L.make_micro(srcs, np.array(pr), 0, j, L.AUG_DEFAULT)[:3])
    return out


def wrap_block(model):                # copied from train_cloud_hn.py main() (GCK == 'block', ckpt_stem default True)
    import torch.utils.checkpoint as cp
    def wrap(mod):
        orig = mod.forward
        def fwd(*x): return cp.checkpoint(orig, *x, use_reentrant=False) if torch.is_grad_enabled() else orig(*x)
        mod.forward = fwd
    bb = model.backbone
    for nm in ('layer1', 'layer2', 'layer3', 'layer4'):
        for blk in getattr(bb, nm): wrap(blk)
    for c_ in model.decoder.convs: wrap(c_)
    class Stem(torch.nn.Module):
        def __init__(s_, b): super().__init__(); s_.b = [b]
        def forward(s_, x): b = s_.b[0]; return b.maxpool(b.relu(b.bn1(b.conv1(x))))
    stem = Stem(bb); wrap(stem)
    def bb_fwd(x, bb=bb, stem=stem):
        x = stem(x); x1 = bb.layer1(x); x2 = bb.layer2(x1); x3 = bb.layer3(x2); x4 = bb.layer4(x3); return [x1, x2, x3, x4]
    bb.forward = bb_fwd


def arm(gck, restore, B):
    torch.manual_seed(0)
    model = L.load_base('base').cuda().train()
    BN = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
    if gck: wrap_block(model)
    before = [(m.running_mean.clone(), m.running_var.clone(), int(m.num_batches_tracked)) for m in BN]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-6)
    torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    for x, y, k in B:
        x, y, k = x.cuda(), y.cuda(), k.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16): lg = model(x)
        lg = lg.float()
        if gck: after = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in BN]
        with torch.no_grad():
            pr = torch.sigmoid(lg); bg = (y == 0).float() * k.float()
            wa = 1 + GAIN * bg[:N1447] * pr[:N1447]; wp = 1 + GAIN * bg[N1447:] * pr[N1447:]
        la, _, _ = L.loss_fn(lg[:N1447], y[:N1447], k[:N1447], wa, smooth=SMOOTH)
        lp, _, _ = L.loss_fn(lg[N1447:], y[N1447:], k[N1447:], wp, smooth=SMOOTH)
        (((1 - W306) * la + W306 * lp) / ACCUM).backward()
        if gck and restore:
            for m, (rm, rv, nb) in zip(BN, after): m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    res = dict(bn=[(m.running_mean.detach().cpu(), m.running_var.detach().cpu(), int(m.num_batches_tracked)) for m in BN],
               before=[(a.cpu(), b.cpu(), c) for a, b, c in before],
               params={n: p.detach().cpu().clone() for n, p in model.named_parameters()},
               peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res


out = {}
for micro in (4, 2):
    B = batches(micro)
    try:
        R = {'plain': arm(False, False, B)}
    except torch.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache(); out[f'plain_micro{micro}'] = 'OOM'; print(f'plain arm OOM at micro {micro}', flush=True); continue
    R['block'] = arm(True, True, B); R['block_norestore'] = arm(True, False, B)
    out['micro'] = micro; P = R['plain']
    upd = max(float((a - b0).abs().max()) for (a, _, _), (b0, _, _) in zip(P['bn'], P['before']))
    out['plain_running_mean_change_max'] = upd
    for nm, r in R.items():
        dm = max(float((a[0] - b[0]).abs().max()) for a, b in zip(r['bn'], P['bn']))
        dv = max(float((a[1] - b[1]).abs().max()) for a, b in zip(r['bn'], P['bn']))
        nbt = sorted({a[2] - b[2] for a, b in zip(r['bn'], r['before'])})
        dp = max(float((r['params'][n] - P['params'][n]).abs().max()) for n in P['params'])
        out[nm] = dict(bn_updates_per_step=nbt, running_mean_maxdiff_vs_plain=dm, running_var_maxdiff_vs_plain=dv,
                       param_maxdiff_vs_plain=dp, peak_alloc_gb=round(r['peak_gb'], 2))
    b, n = out['block'], out['block_norestore']
    out['PASS'] = bool(b['bn_updates_per_step'] == [ACCUM] and max(n['bn_updates_per_step']) == 2 * ACCUM
                       and b['running_mean_maxdiff_vs_plain'] < 0.1 * n['running_mean_maxdiff_vs_plain'])
    break
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
print('BN_CHECK', 'PASS' if out.get('PASS') else 'FAIL', flush=True)
