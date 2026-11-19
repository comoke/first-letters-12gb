#!/bin/zsh
source ~/Workspace/projects/vesuvius/mac/paths.sh
cd ~/Workspace/vesuvius-data-mac/runs/native_full
exec caffeinate -i python ~/Workspace/projects/vesuvius/mac/run_ahn.py --ckpt $MD/ckpt/hn4_s0_final/model.safetensors \
  --zarr $MD/render/claim_surface_render_41layers.zarr --out s0_full.npy --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 1
