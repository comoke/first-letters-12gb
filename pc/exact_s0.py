# A-hn s0 with the shipped member recipe (fwd_hn4_*.json): roi 115 4166 694 7638, grid origin 115 694, clip_to_roi True.
import os, sys
d = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'; sys.path.insert(0, d); os.chdir(d)
import infer
_r = infer.render
def render(*a, **k): k['clip_to_roi'] = True; return _r(*a, **k)
infer.render = render
Z, OUT = sys.argv[1], sys.argv[2]
CKPT = sys.argv[3] if len(sys.argv) > 3 else '/data/winner/hf-model/checkpoints/hn4_s0_final/model.safetensors'
sys.argv = ['infer.py', '--ckpt', CKPT, '--zarr', Z, '--layer0', '8',
            '--tta', '1', '--nsig', '3', '--stride', '21', '--min-cov', '0.25', '--roi', '115', '4166', '694', '7638',
            '--grid-origin', '115', '694', '--vram', '9', '--batch', '8', '--max-minutes', '600', '--out', OUT]
sys.exit(infer.main())
