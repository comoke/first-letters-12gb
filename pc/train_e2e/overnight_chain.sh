#!/bin/bash
# overnight_chain.sh : closes two gaps of the end-to-end run on the home RTX 4070 (no rental), after the GPU is free.
#   C1. TRACE v3B regenerated inside the e2e data tree (maps_v2/fwd_tr.npy; exact shipped crop) with whole-card VRAM logged
#       (nvidia-smi memory.used every 500 ms), then one A-hn member map (e2e s0 checkpoint, exact recipe) with VRAM logged; refuse with
#       the regenerated TRACE and the e2e members, then every map vs shipped (exact_all_compare.py) -> compare/maps_v2.json
#   C2. depth-shuffle control on our maps: the winner's own shuffle_control.py prep (render layers 8..31 permuted, seed 1, box + 256 px),
#       TRACE + the four e2e members + fusion on it, then shuffle_score.py (ours vs original: r, top-1% Jaccard, TRACE ~ ensA, hot share)
# Marker: NIGHT_DONE run=<id> | NIGHT_FAIL run=<id> <why>. Host guard before each GPU step; resumable (finished outputs are skipped).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source "${PC:-$HERE/..}/env.sh"; P=$PC; E=$D/train_e2e
WAIT_SESSIONS=${WAIT_SESSIONS:-}   # optional: tmux sessions to wait for (GPU busy)
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}
log() { echo "$(date '+%F %T') $*"; }
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "NIGHT_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "NIGHT_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
source $P/hostguard.sh
TW=/home/liam/projects/vesuvius-work; SELF=$TW/.claude/worktrees/leadA-push3/x_push3/m306/self
Z=/data/winner/hf-data/results/claim_surface_render_41layers.zarr; V2=$E/maps_v2; SH=$E/shuf
mkdir -p $V2/fused $SH/maps/fused $E/scripts/night; cp $HERE/exact_roi.py $HERE/shuffle_score.py $HERE/shuf_vs_shipped.py $P/exact_s0.py $P/pad_frame.py $P/exact_all_compare.py $E/scripts/night/
cp -r $D/winner/repo/submission/code $E/scripts/night/wcode 2>/dev/null
DKG() { $DK run --rm --name night_$1 --gpus all --shm-size 4g --memory 10g --memory-swap 10g -v $D:/data -v $E/self/runs:$SELF/runs:ro \
        -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src -e V8_DIR=/opt/fl/third_party/v8in -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 \
        --entrypoint bash fl-claim-a7:final -c "$2" >> $E/logs/night_$1.log 2>&1; }
DKC() { $DK run --rm --memory 6g --memory-swap 6g -v $D:/data -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 --entrypoint bash fl-claim-a7:final -c "$1"; }
guard() { hg_check; while [ -n "$HG_WHY" ]; do log "host guard: $HG_WHY; waiting"; sleep 120; hg_check; done; }
vram() {   # vram NAME CMD... : run with whole-card memory.used logged every 500 ms to $E/logs/vram_NAME.csv; logs the peak
  local n=$1; shift; $NS --query-gpu=timestamp,memory.used --format=csv -lms 500 > $E/logs/vram_$n.csv 2>/dev/null & local s=$!
  local t=$(date +%s); "$@"; local rc=$?; kill $s 2>/dev/null
  log "$n exit=$rc ($(( $(date +%s) - t ))s) whole-card peak $(awk -F', ' 'NR>1 {v = $2; sub(/ MiB/, "", v); v += 0; if (v > m) m = v} END {print m}' $E/logs/vram_$n.csv) MiB"
  return $rc
}
log "NIGHT start run=$RUNID: waiting for the GPU ($WAIT_SESSIONS)"
for s in $WAIT_SESSIONS; do while tmux has-session -t $s 2>/dev/null; do touch $E/night_chain.log; sleep 300; done; done
log "GPU free; idle card $($NS --query-gpu=memory.used --format=csv,noheader)"
# ---- C1
guard; [ -s $V2/fwd_tr.npy ] || vram trace DKG trace "cd /opt/fl/claim && python ink_v2.py $Z /data/train_e2e/maps_v2 --models tr --crop 116,4167,694,7639 --vram 7.5"
[ -s $V2/fwd_tr.npy ] || fail "TRACE (logs/night_trace.log)"
guard; [ -s $V2/s0_check/fwd_hn4_s0.npy ] || { mkdir -p $V2/s0_check; vram map_s0 DKG map_s0 "python /data/train_e2e/scripts/night/exact_s0.py $Z /data/train_e2e/maps_v2/s0_check/fwd_hn4_s0.npy $SELF/runs/hn4_s0_e2e/final/model.safetensors"; }
[ -s $V2/s0_check/fwd_hn4_s0.npy ] || fail "s0 map (logs/night_map_s0.log)"
for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do cp -u $E/maps/fwd_$m.npy $E/maps/fwd_$m.json $V2/ 2>/dev/null; done
[ -s $V2/fused/fwd_prod.npy ] || DKC "cd /opt/fl/claim && python fuse_v7.py /data/train_e2e/maps_v2/fused --trace /data/train_e2e/maps_v2/fwd_tr.npy --members /data/train_e2e/maps_v2/fwd_hn4_s0.npy /data/train_e2e/maps_v2/fwd_hn4_s1.npy /data/train_e2e/maps_v2/fwd_hn4_s2.npy --decoy /data/train_e2e/maps_v2/fwd_hn4_decoy.npy" > $E/logs/night_fuse.log 2>&1 || fail "fuse v2"
DKC "python /data/train_e2e/scripts/night/exact_all_compare.py /data/train_e2e/maps_v2 /data/train_e2e/compare/maps_v2.json" > $E/logs/night_cmp_v2.log 2>&1 || fail "compare v2 (logs/night_cmp_v2.log)"
DKC "python /data/train_e2e/scripts/night/pad_frame.py /data/train_e2e/maps_v2/s0_check/fwd_hn4_s0.npy && python -c \"
import numpy as np, json
a = np.load('/data/train_e2e/maps/fwd_tr.npy').astype(np.float64); b = np.load('/data/train_e2e/maps_v2/fwd_tr.npy').astype(np.float64)
c = np.load('/data/train_e2e/maps/fwd_hn4_s0.npy').astype(np.float64); d = np.load('/data/train_e2e/maps_v2/s0_check/fwd_hn4_s0.npy').astype(np.float64)
f = lambda x, y: (lambda m: dict(r=float(np.corrcoef(x[m], y[m])[0, 1]), max_abs=float(np.abs(x[m] - y[m]).max()), bit_equal_share=float((x[m] == y[m]).mean())))(np.isfinite(x) & np.isfinite(y))
json.dump(dict(trace_regenerated_vs_earlier=f(a, b), s0_rerun_vs_e2e=f(c, d)), open('/data/train_e2e/compare/rerun_v2.json', 'w'), indent=1)
print(open('/data/train_e2e/compare/rerun_v2.json').read())\"" > $E/logs/night_rerun.log 2>&1 || fail "rerun compare"
log "C1 done: $(tr -d ' \n' < $E/compare/rerun_v2.json | cut -c1-300) | $(tail -9 $E/logs/night_cmp_v2.log | tr '\n' ' ' | cut -c1-500)"
# ---- C2 depth shuffle (the winner's prep: box + 256 px, layers 8..31, seed 1)
BOX="1383 3001 2175 7085"
[ -d $SH/shuf_render.zarr/0 ] || DKC "cd /data/train_e2e/scripts/night/wcode && python -c \"import shuffle_control as S; S.prep('$Z', '/data/train_e2e/shuf/shuf_render.zarr', 1)\"" > $E/logs/night_shufprep.log 2>&1 || fail "shuffle prep (logs/night_shufprep.log)"
guard; [ -s $SH/maps/fwd_tr.npy ] || vram trace_shuf DKG trace_shuf "cd /opt/fl/claim && python ink_v2.py /data/train_e2e/shuf/shuf_render.zarr /data/train_e2e/shuf/maps --models tr --crop 1383,3001,2175,7085 --vram 7.5"
[ -s $SH/maps/fwd_tr.npy ] || fail "TRACE shuf (logs/night_trace_shuf.log)"
for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do
  guard; [ -s $SH/maps/fwd_$m.npy ] || DKG shuf_$m "python /data/train_e2e/scripts/night/exact_roi.py /data/train_e2e/shuf/shuf_render.zarr /data/train_e2e/shuf/maps/fwd_$m.npy $SELF/runs/${m}_e2e/final/model.safetensors $BOX"
  [ -s $SH/maps/fwd_$m.npy ] || fail "shuf $m (logs/night_shuf_$m.log)"; log "shuf $m done"
done
[ -s $SH/maps/fused/fwd_prod.npy ] || DKC "python /data/train_e2e/scripts/night/pad_frame.py $(for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do echo -n "/data/train_e2e/shuf/maps/fwd_$m.npy "; done) && cd /opt/fl/claim && python fuse_v7.py /data/train_e2e/shuf/maps/fused --trace /data/train_e2e/shuf/maps/fwd_tr.npy --members /data/train_e2e/shuf/maps/fwd_hn4_s0.npy /data/train_e2e/shuf/maps/fwd_hn4_s1.npy /data/train_e2e/shuf/maps/fwd_hn4_s2.npy --decoy /data/train_e2e/shuf/maps/fwd_hn4_decoy.npy" > $E/logs/night_shuf_fuse.log 2>&1 || fail "shuf fuse (logs/night_shuf_fuse.log)"
DKC "python /data/train_e2e/scripts/night/shuffle_score.py /data/train_e2e/maps /data/train_e2e/shuf/maps /data/train_e2e/compare/shuffle.json" > $E/logs/night_shuf_score.log 2>&1 || fail "shuffle score (logs/night_shuf_score.log)"
log "C2 done: $(tail -1 $E/logs/night_shuf_score.log | cut -c1-400)"
# ---- C3 the winner's shipped shuffle maps reproduced: SHIPPED checkpoints on the same shuffled render, their recipe (roi = box,
#      clip_to_roi false: hf-data results/member_maps/shuf_hn4_*.json), then vs results/member_maps/shuf_* and results/fused/shuf_*
SK=$SH/shipped_ckpt; mkdir -p $SK/fused; cp -u $SH/maps/fwd_tr.npy $SK/
for m in hn4_s0 hn4_s1 hn4_s2; do
  guard; [ -s $SK/fwd_$m.npy ] || DKG shk_$m "CLIP_ROI=0 python /data/train_e2e/scripts/night/exact_roi.py /data/train_e2e/shuf/shuf_render.zarr /data/train_e2e/shuf/shipped_ckpt/fwd_$m.npy /data/winner/hf-model/checkpoints/${m}_final/model.safetensors 1639 2745 2431 6829"
  [ -s $SK/fwd_$m.npy ] || fail "shipped-ckpt shuf $m (logs/night_shk_$m.log)"; log "shipped-ckpt shuf $m done"
done
[ -s $SK/fused/fwd_prod.npy ] || DKC "python /data/train_e2e/scripts/night/pad_frame.py /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s0.npy /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s1.npy /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s2.npy && cd /opt/fl/claim && python fuse_v7.py /data/train_e2e/shuf/shipped_ckpt/fused --trace /data/train_e2e/shuf/shipped_ckpt/fwd_tr.npy --members /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s0.npy /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s1.npy /data/train_e2e/shuf/shipped_ckpt/fwd_hn4_s2.npy" > $E/logs/night_shk_fuse.log 2>&1 || fail "shipped-ckpt shuf fuse (logs/night_shk_fuse.log)"
DKC "python /data/train_e2e/scripts/night/shuf_vs_shipped.py /data/train_e2e/shuf/shipped_ckpt /data/train_e2e/compare/shuf_vs_shipped.json" > $E/logs/night_shk_cmp.log 2>&1 || fail "shuf vs shipped (logs/night_shk_cmp.log)"
MARKED=1; log "NIGHT_DONE run=$RUNID C2: $(tail -1 $E/logs/night_shuf_score.log | cut -c1-300) | C3: $(tail -1 $E/logs/night_shk_cmp.log | cut -c1-200)"
