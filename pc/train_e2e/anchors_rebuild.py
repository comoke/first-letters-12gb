"""anchors_rebuild.py : rebuild the PHerc1447 anchor packs a_w058.npz / a_w060.npz with the winner's OWN code.

Runs inside fl-claim-a7:final (python -I) with these mounts (see anchors_rebuild.sh):
  /home/liam/projects/vesuvius-work/data/pherc1447  (ro)  raw HF dataset files (same path the winner used, so info['path'] matches)
  /code                                              (ro)  winner repo submission/code/training/x_push3/m306
  /out                                               (rw)  output dir; packs land in /out/self/packs/a_<w>.npz

Recipe (unchanged winner code, nothing reimplemented):
  make_packs.py anchors(): save_anchor(w, v8306lib.src_1447(w), '1447') for w in (w058, w060), with src_1447 defaults
  (erode 0, decoy None, stride 48, label_mode 'soft').
Only two import shims, neither of which touches the anchor math:
  * ink8um (the v8-in model class, imported at v8306lib module load, unused by src_1447) is stubbed;
  * train_self (imported by make_packs for T.SELF = output root; its own imports need non-shipped exclusion code) is stubbed
    with SELF = /out/self.
"""
import sys, types, os

sys.modules['ink8um'] = types.SimpleNamespace(InkDetector=None)
sys.modules['train_self'] = types.SimpleNamespace(SELF='/out/self')
sys.path.insert(0, '/code/train')        # v8306lib.py (identical to denoise/cloud_code/v8306lib.py)
sys.path.insert(0, '/code/self/code')    # make_packs.py
import v8306lib as L                     # noqa: E402
import make_packs as M                   # noqa: E402

assert L.D1447 == '/home/liam/projects/vesuvius-work/data/pherc1447', L.D1447
print('PK', M.PK, 'D1447', L.D1447, 'TILE/DEPTH/CLIP/CTX', L.TILE, L.DEPTH, L.CLIP, L.CTX, flush=True)
for w in sys.argv[1:] or ['w058', 'w060']:
    if os.path.exists(f'{M.PK}/a_{w}.npz'):
        print('exists, skip', w, flush=True); continue
    M.save_anchor(w, L.src_1447(w), '1447')
print('BUILD_OK', flush=True)
