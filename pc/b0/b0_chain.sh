#!/bin/bash
# b0_chain.sh : B0 baseline = the shipped pipeline (TRACE v3B x A-hn s0/s1/s2, decoy) on every B0 variant, then b0_score.py.
#   native : the FULL claim render, restricted to the B0 window on the shipped tiling (TRACE --crop, A-hn --roi + --grid-origin
#            115 694 + clip_to_roi), so its maps are pixel-registered with the shipped maps
#   shuf / notext / deg936 : their crop zarrs (only compared to native through thresholds or after resampling)
# Low VRAM (VRAM_T / VRAM_A / BATCH_A): under WSL, VRAM is charged to Windows commit, and C: has no room for pagefile growth.
# Watchdog every 30 s (pc/hostguard.sh): halt cleanly (stop ink_* and b0score_* containers) when the host is unsafe. Re-runnable. Ends with B0_DONE | B0_FAIL <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; P=$PC; B=$D/b0
export VRAM_T=${VRAM_T:-4.5} VRAM_A=${VRAM_A:-5} BATCH_A=${BATCH_A:-4} CMEM=${CMEM:-6g}
VARS=${VARS:-native shuf notext deg936}
ts() { date '+%F %T'; }; log() { echo "$(ts) $*"; }
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}      # markers carry run=<id>: appended logs can't wake a waiter with a stale marker
MARKED=0; WD=; trap 'st_=$?; [ -n "$WD" ] && kill $WD 2>/dev/null; [ $MARKED = 1 ] || log "B0_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "B0_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
halted() { [ -s $B/HALT ] && fail "$(cat $B/HALT) during $1 (containers stopped cleanly; rerun b0_chain.sh after fixing)"; }
source $P/hostguard.sh       # MIN_C_MB / MIN_COMMIT_MB floors, growth-aware commit check
watchdog() {   # never returns while the chain runs: after a halt it keeps stopping any ink_/b0score_ container (ink_run also refuses new stages)
  [ -s $B/host_mem.csv ] || echo "time,vm_avail_mb,c_free_mb,win_free_commit_mb,win_commit_total_mb" > $B/host_mem.csv
  while true; do
    hg_check
    echo "$(ts),$(free -m | awk '/^Mem:/ {print $7}'),$HG_ROW" >> $B/host_mem.csv
    if [ -n "$HG_WHY" ] && [ ! -s $B/HALT ]; then echo "$HG_WHY" > $B/HALT; log "watchdog: $HG_WHY -> halting"; fi
    [ -s $B/HALT ] && { $DK ps --filter name=ink_ -q; $DK ps --filter name=b0score_ -q; } | xargs -r $DK stop -t 60 >/dev/null
    sleep 30
  done
}
need() { py=$(grep -c "\"$2\"" $B/runs/$rv/b0_score.json 2>/dev/null); [ "${py:-0}" -ge 1 ] || fail "$1: b0_score.json lacks $2"; }

export HALT_FILE=$B/HALT
rm -f $B/HALT; mkdir -p $B/runs $B/scripts; cp $P/b0/b0_score.py $P/b0/b0_make.py $B/scripts/
# run a private copy of ink_run.sh: bash reads scripts incrementally, so editing the repo copy mid-run would corrupt a live run
INK=$B/scripts/ink_run.$$.sh; cp $P/ink_run.sh $INK
log "B0 chain start run=$RUNID: $VARS (VRAM_T $VRAM_T, VRAM_A $VRAM_A, batch $BATCH_A, cap $CMEM)"
watchdog & WD=$!
for v in $VARS; do
  rv=$v${RUNSUFFIX:-}                               # run dir name (RUNSUFFIX e.g. _adabn: same input, modified members via CKPT_DIR)
  [ -d $B/$v/render.zarr ] || fail "missing $B/$v/render.zarr (run b0_make.py)"
  if ! grep -q INK_DONE $B/runs/$rv.log 2>/dev/null; then
    log "$v: ink_run"
    if [ $v = native ]; then
      TC=$(grep -o '"trace_crop": "[0-9,]*"' $B/b0.json | cut -d'"' -f4)
      roi=$(tr -d '\n ' < $B/b0.json | grep -o '"roi":\[[0-9,]*\]' | grep -o '[0-9,]*' | tr ',' ' ')
      [ -n "${TC:-}" ] && [ -n "$roi" ] || fail "native: trace_crop/roi missing in b0.json"
      TRACE_CROP=$TC GRID_ORIGIN="115 694" CLIP_ROI=1 bash $INK winner/hf-data/results/claim_surface_render_41layers.zarr \
        b0/runs/native $roi >> $B/runs/native.log 2>&1
    else
      bash $INK b0/$v/render.zarr b0/runs/$rv >> $B/runs/$rv.log 2>&1
    fi
    halted $v
    grep -q INK_DONE $B/runs/$rv.log || fail "$v: $(grep -a INK_FAIL $B/runs/$rv.log | tail -1)"
  fi
  rm -f $B/runs/$rv/b0_score.json
  $DK run --rm --name b0score_$v --memory 4g --memory-swap 4g -v $D:/data --entrypoint python fl-claim-a7:final /data/b0/scripts/b0_score.py \
    /data/b0 $v /data/b0/runs/$rv /data/b0/runs/$rv.log > $B/runs/$rv.score.log 2>&1
  halted "$v scoring"
  [ -s $B/runs/$rv/b0_score.json ] || fail "$v: score failed (runs/$rv.score.log: $(tail -1 $B/runs/$rv.score.log))"
  need $v decoy_jaccard_top1; need $v cm2_per_gpu_h
  case $v in shuf|notext) need $v false_ink_frac_vs_native_top1;; *) need $v letter_ap;; esac
  need $v trace_ensA_r; need $v ensA_decoy_jaccard_top1
  log "$rv scored: $(grep -E '"(letter_ap|letter_auc|ref_pearson_box|decoy_jaccard_top1|trace_ensA_r|ensA_decoy_jaccard_top1|cm2_per_gpu_h|false_ink_frac_vs_native_top1)"' $B/runs/$rv/b0_score.json | tr -d ' \n')"
done
MARKED=1; log "B0_DONE run=$RUNID"
