#!/bin/zsh
# Members s1, s2, decoy on the B0 native window (same recipe as s0), then s0 with the parked BN-fold+stitch speed-up.
# Resumable: finished maps are skipped, unfinished ones resume from <out>.parts/.
source ~/Workspace/projects/vesuvius/mac/paths.sh
cd ~/Workspace/vesuvius-data-mac/runs/native_full
R=(--zarr $MD/render/claim_surface_render_41layers.zarr --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 1)
for m in s1 s2 decoy; do
  echo "== $m start $(date)"
  python ~/Workspace/projects/vesuvius/mac/run_ahn.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors $R --out ${m}_full.npy >> ${m}_full.log 2>&1
  echo "== $m end rc=$? $(date)"; tail -1 ${m}_full.log
done
echo "== s0fast start $(date)"
python $MD/runs/stage1/code/run_ahn.py --ckpt $MD/ckpt/hn4_s0_final/model.safetensors $R --out s0fast_full.npy --fast both >> s0fast_full.log 2>&1
echo "== s0fast end rc=$? $(date)"; tail -1 s0fast_full.log
echo CHAIN2_DONE
