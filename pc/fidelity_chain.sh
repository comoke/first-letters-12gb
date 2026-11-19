#!/bin/bash
# PHerc0306B fidelity chain: ink_run.sh on the shipped claim render -> compare with the shipped maps -> summary.
# Usage (WSL, detached in tmux): fidelity_chain.sh <outdir rel to vesuvius-data>
# Ends with: FID_DONE | FID_FAIL <step>
set -u
OUT=${1:-p0306b/fidelity}; source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"; cp $PC/fidelity_compare.py $PC/ink_summary.py $D/scripts/
REN=winner/hf-data/results/claim_surface_render_41layers.zarr
ts() { date '+%F %T'; }
mkdir -p $D/$OUT
echo "$(ts) FID start: $REN -> $OUT"
bash $PC/ink_run.sh "$REN" "$OUT" > $D/$OUT/ink_run.log 2>&1
tail -3 $D/$OUT/ink_run.log
grep -q INK_DONE $D/$OUT/ink_run.log || { echo "$(ts) FID_FAIL ink_run (see $OUT/ink_run.log)"; exit 1; }
$DK run --rm -v $D:/data --entrypoint python fl-claim-a7:final /data/scripts/fidelity_compare.py /data/$OUT \
  || { echo "$(ts) FID_FAIL compare"; exit 1; }
$DK run --rm -v $D:/data --entrypoint python fl-claim-a7:final /data/scripts/ink_summary.py /data/$OUT /data/$OUT/ink_run.log >/dev/null \
  || echo "$(ts) summary failed (non-fatal)"
echo "$(ts) FID_DONE"
