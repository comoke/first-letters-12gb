#!/bin/bash
# Fresh-folder test of apple-silicon.md "How to run it", 2026-11-22: the recipe's setup lines verbatim (only MD points into the test
# folder), from the exported public tree, then a 16-tile A-hn smoke run from the unpacked code (--max-tiles writes no map).
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
B=$HOME/Workspace/m5-recipe-test; cd $B/repo
echo "== start $(date -u +%T)"
set -x
export MD=$B/data
# the winner's image archive, 6.6 GB, from their Hugging Face dataset; then its sha256
curl -L -o fl-claim-a7_final.tar.zst \
  https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/main/docker/fl-claim-a7_final.tar.zst
echo "1832832330ab67b134795eaedd43d6d5c3dcf004596e1fd287d2e71b3af73536  fl-claim-a7_final.tar.zst" | shasum -a 256 -c
bash mac/fetch_winner.sh unpack fl-claim-a7_final.tar.zst   # the code and v8-in/TRACE weights, out of the image archive, no Docker
uv venv --python 3.13 $MD/.venv && uv pip install --python $MD/.venv/bin/python -r mac/requirements.txt
PY=$MD/.venv/bin/python bash mac/fetch_winner.sh hf    # A-hn members + claim-surface render (level 0)
bash mac/fetch_winner.sh link                          # links + sha256 check of the v8-in base weights
source mac/paths.sh
R=$MD/render/claim_surface_render_41layers.zarr; O=$MD/runs/native
set +x
echo "== smoke $(date -u +%T)"
python mac/run_ahn.py --ckpt $MD/ckpt/hn4_s0_final/model.safetensors --zarr $R --out $B/smoke.npy --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 1 --max-tiles 16 || true
du -sh $MD $B/repo/fl-claim-a7_final.tar.zst
echo "RECIPE_DONE $(date -u +%T)"
