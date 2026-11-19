# infer.py with clip_to_roi=True, which infer.py's CLI doesn't expose; every other argument passes through unchanged.
# The shipped A-hn member maps used clip_to_roi=True with --roi and --grid-origin (see exact_s0.py, docs/fidelity-2026-11-13.md).
import os, sys
d = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench'; sys.path.insert(0, d); os.chdir(d)
import infer
_r = infer.render
def render(*a, **k): k['clip_to_roi'] = True; return _r(*a, **k)
infer.render = render
sys.argv = ['infer.py'] + sys.argv[1:]
sys.exit(infer.main())
