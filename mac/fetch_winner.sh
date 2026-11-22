#!/usr/bin/env bash
# Build the Mac data layout that mac/paths.sh expects, from the winner's own release (nothing is redistributed by this repo).
#
#   mac/fetch_winner.sh unpack IMAGE_TAR_ZST     no Docker: unpacks the image archive (a 'docker save' tar: one tar per layer plus
#                                                manifest.json) with zstd and Python's tarfile, applies the layers in order and
#                                                keeps only the inference code and the v8-in / TRACE weights. Needs zstd, python3
#                                                and about 14 GB free (the 12 GB unpacked archive, deleted afterwards, and 2 GB kept).
#   mac/fetch_winner.sh code   IMAGE_TAR_ZST|-   the same files through Docker (any OS, no GPU; the image is x86-64 but is only
#                                                copied from, never run). Loads fl-claim-a7:final if a tar.zst is given ('-' =
#                                                already loaded).
#   mac/fetch_winner.sh hf                       Hugging Face downloads: the A-hn member checkpoints and the claim-surface
#                                                render (level 0 + metadata).
#   mac/fetch_winner.sh link                     symlinks + checks (sha256 of the v8-in base weights).
#
# Env MD = data root (default ~/Workspace/vesuvius-data-mac). 'unpack' and 'code' give the same files; use either.
set -euo pipefail
MD=${MD:-$HOME/Workspace/vesuvius-data-mac}
BENCH=home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench
mkdir -p "$MD"
case "${1:-}" in
unpack)
  [ -f "${2:-}" ] || { echo "usage: $0 unpack fl-claim-a7_final.tar.zst"; exit 2; }
  tmp=$(mktemp -d "$MD/.image.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
  zstd -dc "$2" | tar -x -C "$tmp"                                    # docker save layout: manifest.json + layer tars
  "${PY:-python3}" - "$tmp" "$MD" "$BENCH" <<'EOF'
import json, os, shutil, sys, tarfile
tmp, md, bench = sys.argv[1:4]
want = {'opt/fl/claim': 'code', 'opt/fl/x_research2': 'code', 'opt/fl/x_zoom': 'code', 'opt/fl/x_deepread': 'code',
        'opt/villa/vesuvius/src': 'code', bench: 'code', 'opt/fl/third_party/v8in': 'fl', 'opt/fl/models/community/trace': 'fl'}
def dest(name):                                     # where an image path goes: code/<path>, or fl/<path below opt/fl>
    for w, root in want.items():
        if name == w or name.startswith(w + '/'):
            return os.path.join(md, 'code', name) if root == 'code' else os.path.join(md, 'fl', name[len('opt/fl/'):])
layers = json.load(open(os.path.join(tmp, 'manifest.json')))[0]['Layers']
for layer in layers:                                # in order: a later layer overrides an earlier one
    with tarfile.open(os.path.join(tmp, layer)) as t:
        members = t.getmembers()
        for m in members:                               # opaque whiteout: this layer hides everything lower layers put in the dir
            name = m.name.lstrip('./')
            if os.path.basename(name) == '.wh..wh..opq':
                d = dest(os.path.dirname(name))
                if d and os.path.isdir(d):
                    for e in os.listdir(d):
                        x = os.path.join(d, e)
                        shutil.rmtree(x) if os.path.isdir(x) and not os.path.islink(x) else os.remove(x)
        for m in members:
            name = m.name.lstrip('./')
            base = os.path.basename(name)
            if base == '.wh..wh..opq':
                continue
            if base.startswith('.wh.'):                 # whiteout: the file or directory was deleted in this layer
                d = dest(os.path.join(os.path.dirname(name), base[4:]))
                if d and os.path.lexists(d):
                    shutil.rmtree(d) if os.path.isdir(d) and not os.path.islink(d) else os.remove(d)
                continue
            d = dest(name)
            if d is None or m.isdev():
                continue
            if m.isdir():
                os.makedirs(d, exist_ok=True); continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.lexists(d):
                os.remove(d)
            if m.issym():
                os.symlink(m.linkname, d)
            else:                                       # a regular file, or a hard link (extractfile follows it)
                with t.extractfile(m) as f, open(d, 'wb') as o:
                    shutil.copyfileobj(f, o)
                os.chmod(d, m.mode & 0o777)
print('unpack: done', md)
EOF
  echo "unpack: done ($MD/code, $MD/fl)" ;;
code)
  [ "${2:-}" != "-" ] && zstd -dc "$2" | docker load
  c=$(docker create fl-claim-a7:final)
  cp_out() { docker cp "$c:$1" - | tar -x -C "$2"; }       # stream as tar: no host-path translation (Docker Desktop on WSL)
  trap 'docker rm -f "$c" >/dev/null' EXIT
  mkdir -p "$MD/code/opt/fl" "$MD/code/opt/villa/vesuvius" "$MD/code/$(dirname $BENCH)" "$MD/fl/third_party" "$MD/fl/models/community"
  for d in claim x_research2 x_zoom x_deepread; do cp_out "/opt/fl/$d" "$MD/code/opt/fl/"; done
  cp_out "/opt/villa/vesuvius/src" "$MD/code/opt/villa/vesuvius/"
  cp_out "/$BENCH" "$MD/code/$(dirname $BENCH)/"
  cp_out "/opt/fl/third_party/v8in" "$MD/fl/third_party/"
  cp_out "/opt/fl/models/community/trace" "$MD/fl/models/community/"
  echo "code: done ($MD/code, $MD/fl)" ;;
hf)
  "${PY:-python}" - "$MD" <<'EOF'
import sys
from huggingface_hub import snapshot_download
md = sys.argv[1]
snapshot_download('Prodigle/pherc0306b-ink-ahn', allow_patterns=['checkpoints/hn4_*_final/*'], local_dir=f'{md}/hf-ahn')
snapshot_download('Prodigle/pherc0306b-first-letters', repo_type='dataset',
                  allow_patterns=['results/claim_surface_render_41layers.zarr/.z*', 'results/claim_surface_render_41layers.zarr/0/*'],
                  local_dir=f'{md}/hf-data')
EOF
  mkdir -p "$MD/ckpt" "$MD/render"
  for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do ln -sfn "$MD/hf-ahn/checkpoints/${m}_final" "$MD/ckpt/${m}_final"; done
  ln -sfn "$MD/hf-data/results/claim_surface_render_41layers.zarr" "$MD/render/claim_surface_render_41layers.zarr"
  echo "hf: done" ;;
link)
  for d in claim x_research2 x_zoom x_deepread; do ln -sfn "$MD/code/opt/fl/$d" "$MD/fl/$d"; done
  want=3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd
  got=$(shasum -a 256 "$MD/fl/third_party/v8in/model.safetensors" | cut -d' ' -f1)
  [ "$got" = "$want" ] && echo "v8-in base weights: sha256 OK" || { echo "v8-in base weights: sha256 MISMATCH $got"; exit 1; }
  test -f "$MD/$([ -d "$MD/code" ] && echo code/$BENCH/infer.py)" && echo "A-hn infer.py: OK"
  test -f "$MD/fl/claim/ink_v2.py" && echo "TRACE ink_v2.py: OK" ;;
*) sed -n 2,17p "$0"; exit 2 ;;
esac
