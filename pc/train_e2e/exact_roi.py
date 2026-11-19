# A-hn member map with the shipped member recipe (layer0 8, TTA 1, nsig 3, stride 21, min_cov 0.25, grid origin 115 694,
# clip_to_roi True) on a smaller ROI: the depth-shuffle control's region (the winner's shuffle_control.py prep: the 4 cm2 box + 256 px,
# render px y 1383-3001, x 2175-7085). Same grid origin, so tiles align with the full-render maps.
# Usage (in fl-claim-a7:final): python exact_roi.py <render.zarr> <out.npy> <checkpoint> y0 y1 x0 x1
import os, sys
d = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'; sys.path.insert(0, d); os.chdir(d)
import infer
# CLIP_ROI=0 keeps infer.py's own default (no clip): the recipe of the winner's shipped shuffle maps (shuf_hn4_*.json: roi = the box,
# clip_to_roi false); CLIP_ROI=1 (default) forces clip_to_roi as in the shipped claim-map recipe.
if os.environ.get('CLIP_ROI', '1') == '1':
    _r = infer.render
    def render(*a, **k): k['clip_to_roi'] = True; return _r(*a, **k)
    infer.render = render
Z, OUT, CKPT = sys.argv[1:4]; y0, y1, x0, x1 = sys.argv[4:8]
sys.argv = ['infer.py', '--ckpt', CKPT, '--zarr', Z, '--layer0', '8', '--tta', '1', '--nsig', '3', '--stride', '21', '--min-cov', '0.25',
            '--roi', y0, y1, x0, x1, '--grid-origin', '115', '694', '--vram', '9', '--batch', '8', '--max-minutes', '600', '--out', OUT]
sys.exit(infer.main())
