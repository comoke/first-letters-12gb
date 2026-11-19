#!/bin/bash
# TRACE v3B + A-hn (3 seeds, TTA 1) + product fuse, then the decoy control, on a 41-layer render,
# inside the winner's image fl-claim-a7:final.
# Usage (in WSL, detached in tmux):  ink_run.sh <render.zarr rel to vesuvius-data> <outdir rel> [y0 y1 x0 x1]
#
# Robust to the flaky WSL<->Docker Desktop link: every stage runs as a detached container (docker run -d);
# the driver only makes short, retried client calls. Re-running the script resumes: finished outputs are
# skipped and a still-running stage container is re-attached. A-hn exit 3 (time cap) is re-run automatically.
# Ends with: INK_DONE | INK_FAIL <stage> <reason>
set -u
REN=$1; OUT=$2; shift 2; ROI=("$@")              # optional ROI: 4 numbers
source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"; HOSTD=$D   # native CLI resolves bind sources inside the distro
cp $PC/infer_clip.py $PC/pad_frame.py $D/scripts/
IMG=fl-claim-a7:final   # native CLI: /usr/local/bin/docker on this PC is a symlink to docker.exe (flaky interop)
INFER=/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench/infer.py
CK=/data/winner/hf-model/checkpoints
TAG=$(basename "$OUT" | tr -c 'a-zA-Z0-9_.-' _ | cut -c1-40)
O=$D/$OUT; mkdir -p "$O/logs"
T0=$(date +%s); log() { echo "$(date '+%F %T') [$(( $(date +%s) - T0 ))s] $*"; }

dk() {  # short docker client call: timeout + 6 retries
  local i out
  for i in 1 2 3 4 5 6; do
    if out=$(timeout 120 $DK "$@" 2>&1); then printf '%s' "$out"; return 0; fi
    echo "$(date '+%T') docker $1 failed (try $i): $(echo "$out" | tail -1)" >&2; sleep 30
  done; return 1
}

stage() {  # stage <name> <bash command inside container>; echoes the container exit code
  local name=$1 cmd=$2 c=ink_${TAG}_$1 st rc
  # a caller's watchdog (b0_chain.sh) writes HALT_FILE when commit or C: runs low: start no new stage after that
  if [ -n "${HALT_FILE:-}" ] && [ -s "$HALT_FILE" ]; then log "stage $name refused: HALT ($(cat $HALT_FILE))" >&2; echo "125 halted"; return; fi
  st=$(dk ps -a --filter "name=^${c}\$" --format '{{.State}}'); [ -z "$st" ] && st=none
  if [ "$st" = none ]|| [ "$st" = exited ] || [ "$st" = created ] || [ "$st" = dead ]; then
    [ "$st" != none ] && dk rm -f "$c" >/dev/null
    dk run -d --name "$c" --gpus all --shm-size 4g --memory ${CMEM:-6g} --memory-swap ${CMEM:-6g} -v "$HOSTD:/data" \
       -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src -e V8_DIR=/opt/fl/third_party/v8in \
       -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 --entrypoint bash "$IMG" -c "$cmd" >/dev/null \
       || { # a client timeout can still create the container (the retries then fail on 'name already in use'): adopt it
            [ -n "$(dk ps -a --filter "name=^${c}\$" --format '{{.State}}')" ] || { echo "RUNFAIL"; return; }
            log "  docker run reported failure but $c exists: adopting it" >&2; }
    log "  container $c started" >&2
  else log "  re-attached to running container $c" >&2; fi
  while st=$(dk inspect -f '{{.State.Status}}' "$c") && [ "$st" = running ]; do sleep 60; done
  rc=$(dk inspect -f '{{.State.ExitCode}} oom={{.State.OOMKilled}}' "$c") || rc="unknown"
  dk logs "$c" >> "$O/logs/$name.log" 2>&1
  dk rm "$c" >/dev/null
  echo "$rc"
}

# GPU sampler (native WSL nvidia-smi, no interop)
( while true; do $NS --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> "$O/logs/gpu.csv" 2>/dev/null; sleep 30; done ) &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT

log "INK start: $REN -> $OUT ${ROI[*]:+ROI ${ROI[*]}}"
dk image inspect -f '{{.Id}}' "$IMG" >/dev/null || { log "INK_FAIL image $IMG missing"; exit 1; }

# checkpoint sha256 vs finalist.json
declare -A SHA=([hn4_s0]=9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009
                [hn4_s1]=5c2740c1f33ee5db73ca97425669607a2ffe4ae9af8b9d09d61f1c43165e60ad
                [hn4_s2]=a294716b083769e8116d5f8d3cfa6a09378ca5c7e51c32225d04703266ad8ce7
                [hn4_decoy]=b9cd0cbb61df991e2489c79bf085df1f1954143379abff044a641063ae31a1fe)
# CKPT_DIR (rel to vesuvius-data, e.g. b0/adabn/m1203b): modified members (H2 AdaBN...) in <dir>/<m>_final/model.safetensors;
# their sha256 is logged instead of checked against the shipped values
if [ -n "${CKPT_DIR:-}" ]; then
  CK=/data/$CKPT_DIR
  for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do f=$D/$CKPT_DIR/${m}_final/model.safetensors
    [ -f "$f" ] || { log "INK_FAIL missing $f"; exit 1; }; log "CKPT_DIR $m sha256 $(sha256sum "$f" | cut -c1-16)"; touch "$O/logs/sha_$m.ok"; done
fi
for m in hn4_s0 hn4_s1 hn4_s2 hn4_decoy; do
  f=$D/winner/hf-model/checkpoints/${m}_final/model.safetensors
  [ -f "$O/logs/sha_$m.ok" ] && continue
  echo "${SHA[$m]}  $f" | sha256sum -c --quiet && touch "$O/logs/sha_$m.ok" || { log "INK_FAIL sha256 mismatch $m"; exit 1; }
done
log "checkpoints sha256 ok"

# 1. TRACE (8x dihedral TTA, layers 19-21), whole render
if [ ! -s "$O/fwd_tr.npy" ]; then
  log "TRACE start"; t=$(date +%s)
  rc=$(stage trace "cd /opt/fl/claim && python ink_v2.py /data/$REN /data/$OUT --models tr --crop '${TRACE_CROP:-}' --c ${TRACE_C:-20} --vram ${VRAM_T:-7.5}")
  log "TRACE exit=$rc ($(( $(date +%s) - t ))s)"
  [ -s "$O/fwd_tr.npy" ] || { log "INK_FAIL trace exit=$rc (see logs/trace.log)"; exit 1; }
else log "TRACE: fwd_tr.npy exists, skip"; fi

# 2. A-hn members (+ decoy last), TTA 1, +-3 sigma, layers 8..31 stored order
ahn() {  # ahn <member>
  local m=$1 out=/data/$OUT/fwd_$1.npy rc t n=0 roiarg=""
  [ ${#ROI[@]} -eq 4 ] && roiarg="--roi ${ROI[*]}"
  # depth options (env): LAYER0 = A-hn first of 24 planes (default 8), TRACE_C = TRACE centre plane (default 20)
  # exact shipped-grid options (env, off by default): GRID_ORIGIN="y x" -> --grid-origin; CLIP_ROI=1 -> infer_clip.py (clip_to_roi)
  [ -n "${GRID_ORIGIN:-}" ] && roiarg="$roiarg --grid-origin $GRID_ORIGIN"
  local prog="cd $(dirname $INFER) && python infer.py"; [ "${CLIP_ROI:-0}" = 1 ] && prog="python /data/scripts/infer_clip.py"
  [ -s "$O/fwd_$m.npy" ] && { log "$m: exists, skip"; return 0; }
  log "$m start"; t=$(date +%s)
  while :; do
    n=$((n+1))
    rc=$(stage "$m" "$prog --ckpt $CK/${m}_final/model.safetensors --zarr /data/$REN \
          --layer0 ${LAYER0:-8} --tta 1 --nsig 3 --stride 21 --min-cov 0.25 --vram ${VRAM_A:-9} --batch ${BATCH_A:-8} --max-minutes 600 $roiarg --out $out")
    log "$m pass $n exit=$rc ($(( $(date +%s) - t ))s)"
    case "$rc" in "3 "*) [ $n -lt 6 ] && continue;; esac
    break
  done
  [ -s "$O/fwd_$m.npy" ] || { log "INK_FAIL $m exit=$rc (see logs/$m.log)"; exit 1; }
}
for m in hn4_s0 hn4_s1 hn4_s2; do ahn $m; done
padroi() {  # with an ROI, infer.py writes frame-sized maps: embed them in the render raster for fuse_v7 (pad_frame.py, idempotent)
  [ ${#ROI[@]} -eq 4 ] || return 0
  local rc; rc=$(stage pad "python /data/scripts/pad_frame.py $(for m in "$@"; do echo -n "/data/$OUT/fwd_$m.npy "; done)")
  [ "${rc%% *}" = 0 ] || { log "INK_FAIL pad exit=$rc"; exit 1; }
}
padroi hn4_s0 hn4_s1 hn4_s2

# 3. product rule (fuse_v7)
MEM="/data/$OUT/fwd_hn4_s0.npy /data/$OUT/fwd_hn4_s1.npy /data/$OUT/fwd_hn4_s2.npy"
rc=$(stage fuse "cd /opt/fl/claim && python fuse_v7.py /data/$OUT/fused --trace /data/$OUT/fwd_tr.npy --members $MEM")
log "FUSE exit=$rc"; ls "$O/fused" 2>/dev/null | sed 's/^/  fused: /'
[ "${rc%% *}" = 0 ] || { log "INK_FAIL fuse exit=$rc"; exit 1; }
log "PRIMARY_DONE (TRACE x A-hn product in $OUT/fused)"

# 4. control: decoy member, then re-fuse with --decoy (prodd)
ahn hn4_decoy
padroi hn4_decoy
rc=$(stage fuse_decoy "cd /opt/fl/claim && python fuse_v7.py /data/$OUT/fused --trace /data/$OUT/fwd_tr.npy --members $MEM --decoy /data/$OUT/fwd_hn4_decoy.npy")
log "FUSE+decoy exit=$rc"
[ "${rc%% *}" = 0 ] || { log "INK_FAIL fuse_decoy exit=$rc"; exit 1; }
log "INK_DONE"
