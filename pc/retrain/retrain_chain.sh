#!/bin/bash
# retrain_chain.sh : A-hn s0 retrain on the 4070 (plan step 3), one detached chain:
#   probe  = pack sha256 vs hf-data SHA256SUMS -> BN check (bn_check.py) -> bench N optimizer steps (train_cloud_hn.py --bench)
#   gate   = bench exit 0 and whole-card nvidia-smi peak <= GATE_MIB      -> PROBE_OK | PROBE_FAIL <why> (stop)
#   train  = full run (resumable: state.pt every 5 min, exit 3 = chunk end / OOM -> rerun) -> runs/hn4_s0_pc/final
#   eval   = exact-recipe A-hn inference (exact_s0.py: shipped roi / grid origin / clip_to_roi) on the PHerc0306B claim render
#   compare= retrain_compare.py -> summary.{json,md}                        -> RETRAIN_DONE | RETRAIN_FAIL <stage>
# Re-runnable: finished stages are skipped, a still-running stage container is re-attached.
# Layout: $VESUVIUS_DATA/retrain/{scripts,configs,ref,probe,runs,eval,logs}. Launch with start_retrain.sh.
set -u
source "${PC:?start this with pc/retrain/start_retrain.sh}/env.sh"; R=$D/retrain; H=$D/winner/hf-data; IMG=fl-claim-a7:final
CC=/opt/fl/train/W/.claude/worktrees/leadA-push3/x_push3/m306/denoise/cloud_code
CFG=/data/retrain/configs/hn4_s0_pc.json; RUN=$R/runs/hn4_s0_pc
CMEM=${CMEM:-8g}   # host floors (MIN_C_MB, MIN_COMMIT_MB) come from pc/hostguard.sh
BENCH_STEPS=${BENCH_STEPS:-120}; GATE_MIB=${GATE_MIB:-11264}; GATE_MARGIN=${GATE_MARGIN:-300}; FULL_STEPS=600
mkdir -p $R/{probe,runs,eval,logs}
ts() { date '+%F %T'; }; log() { echo "$(ts) $*"; }
STAGE=probe; MARKED=0; LASTLINE=0
stop() { log "$*"; MARKED=1; exit 1; }
# every exit path writes a marker: an unexpected exit (set -u, a crash) logs PROBE_FAIL / RETRAIN_FAIL with the line and status
trap 'LASTLINE=${BASH_LINENO[0]:-$LINENO}' DEBUG
trap 'st_=$?; if [ $MARKED = 0 ]; then log "$([ $STAGE = probe ] && echo PROBE_FAIL || echo RETRAIN_FAIL) unexpected exit status $st_ near line $LASTLINE (stage $STAGE)"; fi' EXIT

# watchdog NAME : every 30 s log host memory (logs/host_mem.csv) and stop container NAME cleanly (SIGTERM -> the trainer saves
# state.pt and exits 3) when pc/hostguard.sh says the host is unsafe; leaves logs/HALT with the reason.
# Why: 2026-11-14 20:22 Windows ran out of commit, grew C:\pagefile.sys until C: was full, and the WSL VM went down.
source $PC/hostguard.sh
watchdog() {
  local n=$1 vm cm
  [ -s $R/logs/host_mem.csv ] || echo "time,container,vm_avail_mb,vm_swap_used_mb,container_mem,c_free_mb,win_free_commit_mb,win_commit_total_mb" > $R/logs/host_mem.csv
  while true; do
    vm=$(free -m | awk '/^Mem:/ {a=$7} /^Swap:/ {s=$3} END {print a "," s}')
    cm=$($DK stats --no-stream --format '{{.MemUsage}}' $n 2>/dev/null | cut -d/ -f1 | tr -d ' ')
    hg_check
    echo "$(ts),$n,$vm,${cm:-NA},$HG_ROW" >> $R/logs/host_mem.csv
    if [ -n "$HG_WHY" ]; then echo "$HG_WHY" > $R/logs/HALT; log "watchdog: $HG_WHY -> stopping $n"; $DK stop -t 180 $n >/dev/null 2>&1; return; fi
    sleep 30
  done
}

crun() {   # crun NAME LOGFILE CMD : run CMD in a detached container (or re-attach), poll, append its logs, return its exit code
  local n=$1 lf=$2 cmd=$3 st rc lp wd
  st=$($DK inspect -f '{{.State.Status}}' $n 2>/dev/null)
  if [ "$st" != running ]; then
    [ -n "$st" ] && $DK rm -f $n >/dev/null
    $DK run -d --name $n --gpus all --shm-size 4g --memory $CMEM --memory-swap $CMEM -v $D:/data -v $H/data/3b_cloud_packs:/selfroot/packs:ro -v $R/runs:/selfroot/runs \
      -e SELF_ROOT=/selfroot -e V8_DIR=/opt/fl/third_party/v8in -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src \
      -e OMP_NUM_THREADS=4 -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --entrypoint bash $IMG -c "$cmd" >/dev/null || return 125
  else log "re-attached to running container $n"; fi
  $DK logs -f $n >> $lf 2>&1 & lp=$!
  watchdog $n & wd=$!
  while [ "$($DK inspect -f '{{.State.Status}}' $n 2>/dev/null)" = running ]; do sleep 30; done
  rc=$($DK inspect -f '{{.State.ExitCode}} {{.State.OOMKilled}}' $n 2>/dev/null); sleep 2; kill $lp $wd 2>/dev/null
  echo "[$n exit/oomkilled: $rc]" >> $lf; $DK rm $n >/dev/null 2>&1
  [ -s $R/logs/HALT ] && stop "$(echo $STAGE | tr a-z A-Z)_FAIL $(cat $R/logs/HALT) (container stopped cleanly; free memory/space, then rerun start_retrain.sh)"
  return ${rc%% *}
}
gpumon() { while true; do $NS --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> $1 2>/dev/null; sleep 15; done; }
peak() { awk -F', ' '{split($2,a," "); if (a[1]+0>m) m=a[1]+0} END {print m+0}' $1 2>/dev/null; }

rm -f $R/logs/HALT
log "RETRAIN chain start (container cap $CMEM, halt if C: < ${MIN_C_MB:-n/a} MB or Windows commit < ${MIN_COMMIT_MB:-n/a} MB (n/a: host guard off outside WSL); bench $BENCH_STEPS steps, gate $GATE_MIB MiB)"
# ---------------------------------------------------------------- probe
if [ ! -s $R/probe/probe.json ]; then
  log "pack sha256 (8 files vs hf-data SHA256SUMS)"
  (cd $H && tr -d '\r' < SHA256SUMS | grep ' data/3b_cloud_packs/' | sha256sum -c -) > $R/probe/packs_sha.log 2>&1 \
    || stop "PROBE_FAIL packs sha256: $(grep -v ': OK' $R/probe/packs_sha.log | head -3 | tr '\n' ' ')"
  log "packs OK: $(grep -c ': OK' $R/probe/packs_sha.log) files"
  if [ ! -s $R/probe/bn_check.json ]; then
    t=$(date +%s); crun rt_bn $R/logs/bn_check.log "cd /data/retrain/scripts && python bn_check.py $CC /selfroot/packs /data/retrain/probe/bn_check.json"
    log "BN check exit=$? ($(( $(date +%s) - t ))s)"; fi
  grep -q '"PASS": true' $R/probe/bn_check.json 2>/dev/null || stop "PROBE_FAIL BN check (see logs/bn_check.log)"
  log "BN check PASS"
  if grep -aq '^BENCH' $R/logs/bench.log 2>/dev/null && grep -aq 'rt_bench exit/oomkilled: 0 ' $R/logs/bench.log; then
    rc=0; bt=$(grep -a '"step"' $R/logs/bench.log | tail -1 | grep -o '"t": [0-9]*' | grep -o '[0-9]*$'); bt=${bt:-null}; log "reusing the completed bench in logs/bench.log"
  else
    rm -f $R/logs/gpu_bench.csv; gpumon $R/logs/gpu_bench.csv & gm=$!
    # bench mode still writes state.pt every 5 min: keep it in a throwaway SELF_ROOT inside the container so the real run never resumes from it
    t=$(date +%s); crun rt_bench $R/logs/bench.log "mkdir -p /benchroot/runs/hn4_s0_pc && ln -s /selfroot/packs /benchroot/packs && SELF_ROOT=/benchroot python $CC/train_cloud_hn.py $CFG --bench $BENCH_STEPS"
    rc=$?; bt=$(( $(date +%s) - t ))
    kill $gm 2>/dev/null
  fi
  card=$(peak $R/logs/gpu_bench.csv); line=$(grep -a '^BENCH' $R/logs/bench.log | tail -1)
  sps=$(echo "$line" | awk '{print $2}'); alloc=$(echo "$line" | awk '{print $5}')
  proj=$(awk -v s="${sps:-0}" -v n=$FULL_STEPS 'BEGIN {printf "%.2f", s * n / 3600}')
  log "bench exit=$rc ${bt}s: $line; whole-card peak $card MiB; projected $FULL_STEPS steps ${proj} h"
  [ $rc = 0 ] || stop "PROBE_FAIL bench exit $rc$(grep -aq 'CUDA OOM' $R/logs/bench.log && echo ' (CUDA OOM at vram_gb 10.5, grad_ckpt block)')"
  # probe.json is the "gate passed" record: write it only after the gate passes (a rejected probe left
  # probe.json behind and the next launch skipped the gate); a failed gate writes probe_rejected.json instead.
  pj=$(printf '{"bench_steps": %s, "sec_per_step": %s, "peak_alloc_gb": %s, "card_peak_mib": %s, "gate_mib": %s, "projected_hours_%s_steps": %s, "bench_wall_s": %s}' \
    $BENCH_STEPS ${sps:-null} ${alloc:-null} ${card:-null} $GATE_MIB $FULL_STEPS $proj $bt)
  if [ "${card:-99999}" -gt $(( GATE_MIB - GATE_MARGIN )) ]; then echo "$pj" > $R/probe/probe_rejected.json
    stop "PROBE_FAIL card peak $card MiB within $GATE_MARGIN MiB of gate $GATE_MIB (try finer checkpointing)"; fi
  echo "$pj" > $R/probe/probe.json
fi
# re-check a stored probe against the current gate on every launch
pc=$(grep -o '"card_peak_mib": [0-9]*' $R/probe/probe.json | grep -o '[0-9]*$')
[ -n "$pc" ] && [ "$pc" -le $(( GATE_MIB - GATE_MARGIN )) ] || stop "PROBE_FAIL stored probe card peak '${pc}' MiB fails gate $GATE_MIB - $GATE_MARGIN"
log "PROBE_OK $(cat $R/probe/probe.json)"; STAGE=train
# ---------------------------------------------------------------- train
oom=0
for pass in $(seq 1 12); do
  [ -s $RUN/done.json ] && break
  gpumon $R/logs/gpu_train.csv & gm=$!
  t=$(date +%s); crun rt_train $R/logs/train.log "python $CC/train_cloud_hn.py $CFG --max-minutes 600"; rc=$?; el=$(( $(date +%s) - t ))
  kill $gm 2>/dev/null
  echo $(( $(cat $R/logs/train_seconds 2>/dev/null || echo 0) + el )) > $R/logs/train_seconds
  [ $rc = 0 ] || [ $rc = 3 ] || log "train pass $pass: last host_mem rows: $(tail -2 $R/logs/host_mem.csv | tr '\n' ' ')"
  log "train pass $pass exit=$rc (${el}s, total $(cat $R/logs/train_seconds)s) $(grep -a '"step"' $R/logs/train.log | tail -1)"
  if [ $rc = 3 ] && tail -5 $R/logs/train.log | grep -aq 'CUDA OOM'; then oom=$((oom + 1)); [ $oom -ge 3 ] && stop "RETRAIN_FAIL train: 3 OOM exits"; fi
  [ $rc = 0 ] || [ $rc = 3 ] || stop "RETRAIN_FAIL train exit $rc"
done
[ -s $RUN/done.json ] || stop "RETRAIN_FAIL train: no done.json after 12 passes"
log "TRAIN_DONE $(tr -d '\n ' < $RUN/done.json)"; STAGE=eval
# ---------------------------------------------------------------- eval (exact recipe) + compare
for pass in 1 2 3; do
  [ -s $R/eval/fwd_hn4_s0_pc.npy ] && break
  t=$(date +%s); crun rt_eval $R/logs/eval.log "python /data/retrain/scripts/exact_s0.py /data/winner/hf-data/results/claim_surface_render_41layers.zarr /data/retrain/eval/fwd_hn4_s0_pc.npy /data/retrain/runs/hn4_s0_pc/final/model.safetensors"
  log "eval pass $pass exit=$? ($(( $(date +%s) - t ))s)"
done
[ -s $R/eval/fwd_hn4_s0_pc.npy ] || stop "RETRAIN_FAIL eval"; STAGE=compare
rm -f $R/summary.json $R/summary.md          # never accept a stale summary
crun rt_cmp $R/logs/compare.log "python /data/retrain/scripts/retrain_compare.py /data/retrain"; rc=$?
[ $rc = 0 ] && [ -s $R/summary.json ] || stop "RETRAIN_FAIL compare exit $rc"
grep -q '"retrained_s0~shipped_s0"' $R/summary.json && grep -q '"verdict"' $R/summary.json || stop "RETRAIN_FAIL compare: summary lacks retrained-vs-shipped pair or verdict"
cat $R/summary.md
MARKED=1; log "RETRAIN_DONE"
