"""pack_p4.py <tag> [<tag> ...] : Scroll 1 (PHercParis4) segments prepared by fetch_paris4.py -> anchor packs a_p4<tag>.npz, with the
winner's own code, unchanged: make_packs.save_anchor(name, v8306lib.src_1447(tag), 'paris4'), src_1447 defaults (tile 64, depth 24,
clip 200, stride 48, soft labels). The prepared segments are mounted at the winner's D1447 path, so src_1447 reads
{D1447}/<tag>/labels/{inklabels,mask}.png and layers/00..23.tif (in its own 23 -> 0 order) exactly as for PHerc1447.
Same two import shims as pc/train_e2e/anchors_rebuild.py (ink8um model class, train_self output root); no anchor math touched.
Mounts (pc/finetune/ft_chain.sh): segments root -> D1447 (ro), winner m306 code -> /code (ro), output -> /out (packs in /out/self/packs).
"""
import os, sys, types
sys.modules['ink8um'] = types.SimpleNamespace(InkDetector=None)
sys.modules['train_self'] = types.SimpleNamespace(SELF='/out/self')
sys.path.insert(0, '/code/train'); sys.path.insert(0, '/code/self/code')
import v8306lib as L                     # noqa: E402
import make_packs as M                   # noqa: E402
os.makedirs(M.PK, exist_ok=True)
print('PK', M.PK, 'D1447', L.D1447, flush=True)
for tag in sys.argv[1:]:
    name = f'p4{tag}'
    if os.path.exists(f'{M.PK}/a_{name}.npz'): print('exists, skip', name, flush=True); continue
    M.save_anchor(name, L.src_1447(tag), 'paris4')
    print('packed', name, flush=True)
print('PACK_OK', flush=True)
