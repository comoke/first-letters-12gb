#!/usr/bin/env bash
# Build the Mac data layout that mac/paths.sh expects, from the winner's own release (nothing is redistributed by this repo).
#
#   mac/fetch_winner.sh code   IMAGE_TAR_ZST|-   needs Docker (any OS, no GPU; the image is x86-64 but is only copied
#                                                from, never run). Loads fl-claim-a7:final if a tar.zst is given ('-' = already
#                                                loaded), copies the inference code and the v8-in / TRACE weights out of it.
#   mac/fetch_winner.sh hf                       Hugging Face downloads: the A-hn member checkpoints and the claim-surface
#                                                render (level 0 + metadata).
#   mac/fetch_winner.sh link                     symlinks + checks (sha256 of the v8-in base weights).
#
# Env MD = data root (default ~/Workspace/vesuvius-data-mac). Run 'code' on a machine with Docker and copy $MD/code and $MD/fl
# to the Mac if the Mac has no Docker.
set -euo pipefail
MD=${MD:-$HOME/Workspace/vesuvius-data-mac}
BENCH=home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench
mkdir -p "$MD"
case "${1:-}" in
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
*) sed -n 2,13p "$0"; exit 2 ;;
esac
