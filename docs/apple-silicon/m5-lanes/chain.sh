#!/bin/zsh
# A-hn members on the B0 native window through the two-lane Core ML backend (Neural Engine + GPU), same recipe as runs/native_full.
source ~/Workspace/vesuvius-ahn-port/mac/paths.sh
cd ~/Workspace/vesuvius-data-mac/runs/m5_lanes
A=~/Workspace/vesuvius-ahn-port/mac
R=(--zarr $MD/render/claim_surface_render_41layers.zarr --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 32)
for m in s0 decoy s1 s2; do
  pmset -g batt | head -1 | grep -q "AC Power" || { echo "on battery, stopping before $m"; exit 9; }
  P=$MD/runs/coreml/hn4_${m}_stack
  [ -d $P.mlpackage ] || python $A/ahn_coreml.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors --out $P --mode stack > convert_$m.log 2>&1
  echo "== $m start $(date -u +%T)"
  python $A/run_ahn.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors $R --out ${m}_lanes.npy --backend lanes --mlpackage $P.mlpackage --lanes GGNGN >> ${m}_lanes.log 2>&1
  echo "== $m end rc=$? $(date -u +%T) $(tail -1 ${m}_lanes.log)"
done
echo CHAIN_DONE
