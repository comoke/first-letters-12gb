#!/bin/bash
# job_ahn_b0.sh VARIANT [BATCH] : one job. The four A-hn members on the B0 native window (the exact shipped recipe: roi, grid origin
# 115 694, clip_to_roi) and on the no-text window, with the forward VARIANT (ahn_variants.py), then the gate (ahn_gate.py) against the
# current 4070 maps in $VESUVIUS_DATA/b0/runs (pc/b0/b0_chain.sh).  Maps go to $SCR/b0_<VARIANT> (dk_job.sh), about 180 MB.
# Env: MEMBERS (default the four), WINDOWS (default "native notext"), BM=1 for cudnn.benchmark, GATE=0 to skip the gate (a stock
# same-harness timing run).  Log $VESUVIUS_DATA/opt4070/ahn_b0_<VARIANT>.log, ends O4070_DONE / O4070_FAIL.
V=${1:?variant}; B=${2:-8}; H=$(cd "$(dirname "$0")" && pwd); source "$H/../env.sh"; O=$D/opt4070
TAG=$(echo "$V" | tr -c 'a-zA-Z0-9_\n' _)${BM:+_bm}${RUNTAG:+_$RUNTAG}; M=${MEMBERS:-hn4_s0 hn4_s1 hn4_s2 hn4_decoy}; W=${WINDOWS:-native notext}
ROI=$(tr -d '\n ' < $D/b0/b0.json | grep -o '"roi":\[[0-9,]*\]' | grep -o '[0-9,]*' | tr ',' ' ')
[ -n "$ROI" ] || { echo "no roi in b0.json" >&2; exit 1; }
CMD="set -e; cd /code"
for w in $W; do
  if [ $w = native ]; then
    CMD="$CMD; python ahn_fast.py --zarr /data/winner/hf-data/results/claim_surface_render_41layers.zarr --outdir /scratch/b0_$TAG/native --roi $ROI --grid-origin 115 694 --clip --variant $V --batch $B --members $M ${BM:+--cudnn-benchmark}"
  else
    CMD="$CMD; python ahn_fast.py --zarr /data/b0/$w/render.zarr --outdir /scratch/b0_$TAG/$w --variant $V --batch $B --members $M ${BM:+--cudnn-benchmark}"
  fi
done
[ "${GATE:-1}" = 1 ] && CMD="$CMD; python ahn_gate.py /data/b0 /scratch/b0_$TAG /scratch/b0_$TAG/gate.json"
CMEM=${CMEM:-4500m} exec bash $H/dk_job.sh ahn_b0_$TAG $O/ahn_b0_$TAG.log -- bash -c "$CMD"
