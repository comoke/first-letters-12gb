#!/bin/bash
# exact_all_chain.sh : exact shipped-recipe inference for A-hn s1, s2 and decoy (s0 and TRACE were done 2026-11-13 in
# p0306b/fidelity_exact), then fuse_v7 on all of them, then exact_all_compare.py against every shipped map of the claim.
# Recipe (identical for all 4 members, shipped results/member_maps/fwd_hn4_*.json): roi 115 4166 694 7638, grid origin 115 694,
# clip_to_roi, layer0 8, TTA 1, nsig 3, stride 21 (pc/exact_s0.py with each member's shipped checkpoint).
# Host guard before each GPU step; ends with EXACT_ALL_DONE run=<id> | EXACT_ALL_FAIL run=<id> <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"; P=$PC; O=p0306b/fidelity_exact
Z=/data/winner/hf-data/results/claim_surface_render_41layers.zarr; CK=/data/winner/hf-model/checkpoints
ts() { date '+%F %T'; }; log() { echo "$(ts) $*"; }
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "EXACT_ALL_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "EXACT_ALL_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
source $P/hostguard.sh
cp $P/exact_s0.py $P/exact_all_compare.py $P/pad_frame.py $D/scripts/
DKG="$DK run --rm --gpus all --shm-size 4g --memory 8g --memory-swap 8g -v $D:/data -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src -e V8_DIR=/opt/fl/third_party/v8in --entrypoint bash fl-claim-a7:final -c"
log "EXACT_ALL chain start run=$RUNID"
mkdir -p $D/$O
if [ ! -s $D/$O/fwd_tr.npy ] || [ ! -s $D/$O/fwd_hn4_s0.npy ]; then   # TRACE + s0 come from exact_chain.sh: run it first
  log "TRACE or s0 missing in $O: running pc/exact_chain.sh"; bash $P/exact_chain.sh >> $D/$O/exact_chain.log 2>&1
fi
[ -s $D/$O/fwd_tr.npy ] && [ -s $D/$O/fwd_hn4_s0.npy ] || fail "exact TRACE / s0 missing in $O after exact_chain.sh ($O/exact_chain.log)"
for m in hn4_s1 hn4_s2 hn4_decoy; do
  for pass in 1 2 3; do
    [ -s $D/$O/fwd_$m.npy ] && break
    hg_check; [ -n "$HG_WHY" ] && fail "host guard before $m: $HG_WHY"
    t=$(date +%s); $DKG "python /data/scripts/exact_s0.py $Z /data/$O/fwd_$m.npy $CK/${m}_final/model.safetensors" >> $D/$O/$m.log 2>&1
    log "$m pass $pass exit=$? ($(( $(date +%s) - t ))s)"
  done
  [ -s $D/$O/fwd_$m.npy ] || fail "$m (see $O/$m.log)"
done
# fuse: fuse_v7 wants maps on one raster; exact_s0 writes frame-sized maps (roi = frame with clip_to_roi) -> pad_frame first
$DKG "python /data/scripts/pad_frame.py /data/$O/fwd_hn4_s0.npy /data/$O/fwd_hn4_s1.npy /data/$O/fwd_hn4_s2.npy /data/$O/fwd_hn4_decoy.npy && \
  cd /opt/fl/claim && python fuse_v7.py /data/$O/fused --trace /data/$O/fwd_tr.npy --members /data/$O/fwd_hn4_s0.npy /data/$O/fwd_hn4_s1.npy /data/$O/fwd_hn4_s2.npy --decoy /data/$O/fwd_hn4_decoy.npy" \
  > $D/$O/fuse_all.log 2>&1 || fail "pad/fuse (see $O/fuse_all.log)"
$DK run --rm --memory 6g -v $D:/data --entrypoint python fl-claim-a7:final /data/scripts/exact_all_compare.py /data/$O /data/$O/exact_all_compare.json \
  > $D/$O/exact_all_compare.log 2>&1 || fail "compare (see $O/exact_all_compare.log)"
MARKED=1; log "EXACT_ALL_DONE run=$RUNID"; cat $D/$O/exact_all_compare.log
