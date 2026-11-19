#!/bin/bash
# train_vram.sh : one A-hn training run (s0, the end-to-end packs, the 12 GB config) with the whole card logged every 500 ms
# (e2e_chain.sh train stage, SUFFIX=vram: run hn4_s0_vram next to hn4_s0_e2e, which it leaves untouched), Windows commit and
# container memory every 30 s (the chain's watchdog -> host_mem.csv), then a weight comparison with the e2e s0 run (same data,
# same seed: the difference is GPU non-determinism in training).
# Ends with TRAINVRAM_DONE run=<id> | TRAINVRAM_FAIL run=<id> <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; E=$D/train_e2e
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}; log() { echo "$(date '+%F %T') $*"; }
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "TRAINVRAM_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "TRAINVRAM_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
log "TRAINVRAM start run=$RUNID; idle card $($NS --query-gpu=memory.used --format=csv,noheader)"
log "host_mem.csv rows from here on belong to this run (first row after $(date '+%F %T'))"
STAGES=train MEMBERS=hn4_s0 SUFFIX=vram RUNID=$RUNID-train bash $PC/train_e2e/e2e_chain.sh || fail "train (see the e2e chain lines above)"
SELF=/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/self
$DK run --rm --memory 4g -v $D:/data --entrypoint python fl-claim-a7:final -c "
import json, torch
from safetensors.torch import load_file
a = load_file('/data/train_e2e/self/runs/hn4_s0_e2e/final/model.safetensors'); b = load_file('/data/train_e2e/self/runs/hn4_s0_vram/final/model.safetensors')
d = {k: float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].is_floating_point()}
n = sum(a[k].numel() for k in a if a[k].is_floating_point()); eq = sum(int((a[k] == b[k]).sum()) for k in a if a[k].is_floating_point())
out = dict(tensors=len(d), params=n, bit_equal_share=eq / n, max_abs=max(d.values()), tensors_identical=sum(v == 0 for v in d.values()))
json.dump(out, open('/data/train_e2e/compare/train_rerun_weights.json', 'w'), indent=1); print(json.dumps(out))" > $E/logs/train_rerun_weights.log 2>&1 || fail "weights compare (logs/train_rerun_weights.log)"
MARKED=1; log "TRAINVRAM_DONE run=$RUNID $(grep -a 'whole-card VRAM peak' $D/train_e2e/train_vram.log | tail -1 | cut -c1-200) | weights $(tail -1 $E/logs/train_rerun_weights.log | cut -c1-300)"
