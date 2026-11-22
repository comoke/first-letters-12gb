#!/bin/bash
# dk_job.sh NAME LOG -- CMD... : one GPU job of the 4070 optimization work inside the winner's image.
# The container runs detached (the WSL<->Docker link drops client calls; pc/ink_run.sh does the same) and is polled every 5 s.
# Mounts: vesuvius-data at /data, this folder at /code (read-only).  Ends the log with  O4070_DONE <name> rc=0  or  O4070_FAIL <name> <why>.
# Env: CMEM (container memory, default 5g), IMG (default fl-claim-a7:final), SCR (where maps are written, mounted at /scratch;
# default $VESUVIUS_DATA/o4070-scratch).  VESUVIUS_DATA and DOCKER come from pc/env.sh.
set -u
NAME=${1:?name}; LOG=${2:?log}; shift 2; [ "$1" = -- ] && shift
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../env.sh"; IMG=${IMG:-fl-claim-a7:final}; C=o4070_$NAME
mkdir -p "$(dirname "$LOG")"; exec >> "$LOG" 2>&1
log() { echo "$(date '+%F %T') $*"; }
dk() { local i out; for i in 1 2 3 4 5 6; do if out=$(timeout 120 $DK "$@" 2>&1); then printf '%s' "$out"; return 0; fi
    echo "$(date '+%T') docker $1 failed (try $i): $(echo "$out" | tail -1)" >&2; sleep 20; done; return 1; }
MARKED=0; trap 'st=$?; [ $MARKED = 1 ] || log "O4070_FAIL $NAME unexpected exit status $st"' EXIT
log "start $NAME: $*"
st=$(dk ps -a --filter "name=^${C}\$" --format '{{.State}}'); [ -n "$st" ] && [ "$st" != running ] && dk rm -f "$C" >/dev/null
if [ "$st" != running ]; then
  SCR=${SCR:-$D/o4070-scratch}; mkdir -p "$SCR"
  dk run -d --name "$C" --gpus all --shm-size 2g --memory ${CMEM:-5g} --memory-swap ${CMEM:-5g} -v "$D:/data" -v "$HERE:/code:ro" -v "$SCR:/scratch" \
     -e FL_ROOT=/opt/fl -e VILLA_SRC=/opt/villa/vesuvius/src -e V8_DIR=/opt/fl/third_party/v8in -e PYTHONUNBUFFERED=1 \
     -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 --entrypoint "$1" "$IMG" "${@:2}" >/dev/null \
     || { [ -n "$(dk ps -a --filter "name=^${C}\$" --format '{{.State}}')" ] || { log "O4070_FAIL $NAME docker run"; MARKED=1; exit 1; }; }
fi
t0=$(date +%s); seen=0
while st=$(dk inspect -f '{{.State.Status}}' "$C") && [ "$st" = running ]; do
  n=$(timeout 60 $DK logs "$C" 2>&1 | wc -l); [ "$n" -gt "$seen" ] && { timeout 60 $DK logs "$C" 2>&1 | tail -n +$((seen + 1)); seen=$n; }
  sleep 5; done
n=$(timeout 60 $DK logs "$C" 2>&1 | wc -l); [ "$n" -gt "$seen" ] && timeout 60 $DK logs "$C" 2>&1 | tail -n +$((seen + 1))
rc=$(dk inspect -f '{{.State.ExitCode}} oom={{.State.OOMKilled}}' "$C") || rc="unknown"
dk rm "$C" >/dev/null
MARKED=1
if [ "${rc%% *}" = 0 ]; then log "O4070_DONE $NAME rc=$rc $(( $(date +%s) - t0 ))s"; exit 0; fi
log "O4070_FAIL $NAME rc=$rc $(( $(date +%s) - t0 ))s"; exit 1
