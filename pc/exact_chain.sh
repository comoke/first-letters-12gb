#!/bin/bash
# After the fidelity chain: exact-recipe replay of TRACE (shipped crop) + A-hn s0 (shipped roi/grid/clip), then compare.
set -u
source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"; cp $PC/exact_s0.py $PC/exact_compare.py $D/scripts/; OUT=p0306b/fidelity_exact; Z=/data/winner/hf-data/results/claim_surface_render_41layers.zarr
DK() { $DK run --rm --gpus all --shm-size 4g -v $D:/data -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src \
       -e V8_DIR=/opt/fl/third_party/v8in -e OMP_NUM_THREADS=4 --entrypoint bash fl-claim-a7:final -c "$1"; }
ts() { date '+%F %T'; }
mkdir -p $D/$OUT
if [ ! -s $D/$OUT/fwd_tr.npy ]; then
  t=$(date +%s); DK "cd /opt/fl/claim && python ink_v2.py $Z /data/$OUT --models tr --crop 116,4167,694,7639 --vram 7.5" > $D/$OUT/trace.log 2>&1
  echo "$(ts) TRACE exit=$? ($(( $(date +%s) - t ))s)"; fi
for i in 1 2 3; do
  [ -s $D/$OUT/fwd_hn4_s0.npy ] && break
  t=$(date +%s); DK "python /data/scripts/exact_s0.py $Z /data/$OUT/fwd_hn4_s0.npy" >> $D/$OUT/s0.log 2>&1
  echo "$(ts) s0 pass $i exit=$? ($(( $(date +%s) - t ))s)"
done
DK "python /data/scripts/exact_compare.py /data/$OUT" 2>&1 | tee $D/$OUT/compare.log
[ -s $D/$OUT/exact_fidelity.json ] && echo "$(ts) EXACT_DONE" || echo "$(ts) EXACT_FAIL"
