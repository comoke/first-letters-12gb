"""ahn_fast.py : the winner's A-hn members on a render, with the forward chosen by --variant (ahn_variants.py) and every member in
one process.  The recipe, the tile grid, the stitch and the output files are infer.py's own (infer.render is called unchanged):

  python ahn_fast.py --zarr RENDER --outdir DIR [--members hn4_s0 hn4_s1 hn4_s2 hn4_decoy] [--variant half_cl3] [--batch 8]
                     [--roi y0 y1 x0 x1] [--grid-origin gy gx] [--clip] [--layer0 8] [--cudnn-benchmark] [--vram 9]

Writes DIR/fwd_<member>.npy + .json as pc/ink_run.sh does (ROI-sized with --roi; pad_frame.py embeds them for fuse_v7.py), and
DIR/ahn_fast.json with the wall seconds of each member (model load to map written).  --variant stock is the shipped forward: the
same-harness baseline.  Checkpoint sha256 values are checked against the release's (pc/ink_run.sh's table).
What is saved against pc/ink_run.sh besides the forward: one container and one import instead of four, and no 60 s poll per member.
"""
import os, sys, json, time, argparse
import torch

BENCH = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'
sys.path.insert(0, BENCH); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer                                                                   # noqa: E402
import ahn_variants as AV                                                      # noqa: E402

SHA = dict(hn4_s0='9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009',
           hn4_s1='5c2740c1f33ee5db73ca97425669607a2ffe4ae9af8b9d09d61f1c43165e60ad',
           hn4_s2='a294716b083769e8116d5f8d3cfa6a09378ca5c7e51c32225d04703266ad8ce7',
           hn4_decoy='b9cd0cbb61df991e2489c79bf085df1f1954143379abff044a641063ae31a1fe')


class Net(torch.nn.Module):
    """The chosen forward as infer.band_sums calls a model."""
    def __init__(self, call, mod):
        super().__init__(); self.call, self.mod = call, mod

    def forward(self, x):
        return self.call(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--ckdir', default='/data/winner/hf-model/checkpoints')
    ap.add_argument('--members', nargs='+', default=['hn4_s0', 'hn4_s1', 'hn4_s2', 'hn4_decoy'])
    ap.add_argument('--variant', default='half_cl3'); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--roi', type=int, nargs=4, default=None); ap.add_argument('--grid-origin', type=int, nargs=2, default=(0, 0))
    ap.add_argument('--clip', action='store_true', help='clip_to_roi (the shipped maps and the B0 native run use it)')
    ap.add_argument('--layer0', type=int, default=8); ap.add_argument('--cudnn-benchmark', action='store_true')
    ap.add_argument('--vram', type=float, default=9.0); ap.add_argument('--part-tiles', type=int, default=2500)
    a = ap.parse_args()
    T0 = time.time(); os.makedirs(a.outdir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    infer.vram_cap(a.vram); infer.install_sigterm()
    V = infer.Volume(a.zarr)
    rec = dict(variant=a.variant, batch=a.batch, cudnn_benchmark=bool(a.cudnn_benchmark), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), members={})
    for m in a.members:
        out = f'{a.outdir}/fwd_{m}.npy'
        if os.path.exists(out) and os.path.exists(out[:-4] + '.json'):
            print(f'{m}: exists, skip', flush=True); continue
        t = time.time()
        model, info = infer.load_model(f'{a.ckdir}/{m}_final/model.safetensors', 1.0, 'cuda')
        if m in SHA and info['ckpt_sha256'] != SHA[m]: raise SystemExit(f'{m}: checkpoint sha256 {info["ckpt_sha256"]} is not the release\'s')
        if a.variant != 'stock':
            call, mod = AV.build(a.variant, model); model = Net(call, mod)
        torch.cuda.reset_peak_memory_stats()
        rc = infer.render(model, info, V, a.layer0, out, reverse=False, roi=a.roi, tta=1, nsig=3, min_cov=0.25, stride=21,
                          grid_origin=tuple(a.grid_origin), batch=a.batch, part_tiles=a.part_tiles, deadline=None, device='cuda',
                          clip_to_roi=a.clip, extra_meta=dict(opt4070=dict(variant=a.variant, batch=a.batch, cudnn_benchmark=bool(a.cudnn_benchmark))))
        if rc: raise SystemExit(f'{m}: infer.render returned {rc}')
        tiles = json.load(open(out[:-4] + '.json'))['tiles']
        rec['members'][m] = dict(wall_s=round(time.time() - t, 1), tiles=tiles, tiles_per_s=round(tiles / (time.time() - t), 2),
                                 peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        print(f'{m}: {rec["members"][m]}', flush=True)
        del model; torch.cuda.empty_cache()
    rec['total_wall_s'] = round(time.time() - T0, 1)
    json.dump(rec, open(f'{a.outdir}/ahn_fast.json', 'w'), indent=1)
    print('AHN_FAST_DONE', json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
