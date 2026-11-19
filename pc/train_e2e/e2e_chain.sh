#!/bin/bash
# e2e_chain.sh : the winner's train mode, stage by stage, on this PC (plan: docs/train-e2e-plan-2026-11-15.md).
#   STAGES (comma list, in order): masks, teach_readers, teach_v8, labels, exclusion, packs, anchors_shipped, train, members, fuse
#   (stage names and numbering are ours; the image's `train` entrypoint is not invoked, its stage scripts are run one by one)
# Paths: pc/env.sh (VESUVIUS_DATA, DOCKER, NVIDIA_SMI).
# Every stage runs the winner's own script inside fl-claim-a7:final. The hf-data inputs are mounted read-only at the paths the scripts
# expect (/home/liam/projects/vesuvius-work/...), and writable D: folders are mounted over the output folders (m306/self/{pool,teach,
# labels,packs,runs}), so nothing lands in the Docker VHD on C:. After each stage, e2e_compare.py writes the planned metrics.
# Host guard (pc/hostguard.sh) checked before every stage; private script copies; ends with E2E_DONE run=<id> | E2E_FAIL run=<id> <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; P=$PC; E=$D/train_e2e; H=$D/winner/hf-data/data
STAGES=${STAGES:-masks}; CMEM=${CMEM:-10g}
SUFFIX=${SUFFIX:-e2e}   # run/config name <member>_<SUFFIX>; a second training run with SUFFIX=x leaves the e2e runs untouched
# TW / P3 / SELF are paths INSIDE the winner's image fl-claim-a7:final (where its scripts expect their inputs and outputs; the
# MOUNTS below map host folders onto them), not host paths.
TW=/home/liam/projects/vesuvius-work; P3=$TW/.claude/worktrees/leadA-push3/x_push3; SELF=$P3/m306/self
ts() { date '+%F %T'; }; log() { echo "$(ts) $*"; }
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}
MARKED=0; trap 'st_=$?; [ -n "${WD:-}" ] && kill $WD 2>/dev/null; [ $MARKED = 1 ] || log "E2E_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "E2E_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
source $P/hostguard.sh
mkdir -p $E/self/{pool,teach,labels,packs,runs} $E/compare $E/scripts $E/logs
cp $P/train_e2e/e2e_compare.py $P/train_e2e/e2e_trainlog.py $P/exact_s0.py $P/pad_frame.py $P/exact_all_compare.py $E/scripts/
rm -f $E/HALT; WD=
watchdog() {   # long GPU stages: halt cleanly (stop e2e_ containers, dke refuses new ones) when the host guard trips
  [ -s $E/host_mem.csv ] || echo "time,c_free_mb,win_free_commit_mb,win_commit_total_mb,containers" > $E/host_mem.csv
  while true; do hg_check; echo "$(ts),$HG_ROW,$($DK stats --no-stream --format '{{.Name}}={{.MemUsage}}' 2>/dev/null | grep e2e_ | tr '
' ' ')" >> $E/host_mem.csv
    if [ -n "$HG_WHY" ] && [ ! -s $E/HALT ]; then echo "$HG_WHY" > $E/HALT; log "watchdog: $HG_WHY -> halting"; fi
    [ -s $E/HALT ] && $DK ps --filter name=e2e_ -q | xargs -r $DK stop -t 120 >/dev/null; sleep 30; done; }
[ -s $E/self/pool/regions_pool.json ] || cp $H/0_inputs/regions_pool.json $E/self/pool/
MOUNTS="-v $D:/data \
 -v $H/0_inputs/renders/A6.zarr:$TW/x_focus/renders/A_own_r41.zarr:ro \
 -v $H/0_inputs/renders/w052.zarr:$TW/x_focus306/renders/w052.zarr:ro \
 -v $H/0_inputs/renders/w053.zarr:$TW/x_focus306/renders/w053.zarr:ro \
 -v $H/0_inputs/renders/w059.zarr:$TW/x_req/20261020b/F6Y/w059.zarr:ro \
 -v $H/0_inputs/renders/w109.zarr:$TW/x_refine/trials/P5_scour_0306B/crops/p3w109_esc.zarr:ro \
 -v $H/0_inputs/meshes/A6:$TW/fl/sqm/PHerc0306B_a306_row/auto_grown_20261012141245268:ro \
 -v $H/0_inputs/meshes/w052:$TW/x_vast_plan/results/PHerc0306B_z8400/results/flat/w052/tifxyz/flatten.tifxyz:ro \
 -v $H/0_inputs/meshes/w053:$TW/x_vast_plan/results/PHerc0306B_z8400/results/flat/w053/tifxyz/flatten.tifxyz:ro \
 -v $H/0_inputs/meshes/w059:$TW/x_vast_plan/results/PHerc0306B_z8400/results/flat/w059/tifxyz/flatten.tifxyz:ro \
 -v $H/0_inputs/meshes/w109:$TW/x_local_lanes/PHerc0306B_z9900/out/L0306bacw/p3/flat/w109/tifxyz/flatten.tifxyz:ro \
 -v $E/self/pool:$SELF/pool -v $E/self/teach:$SELF/teach -v $E/self/labels:$SELF/labels -v $E/self/packs:$SELF/packs -v $E/self/runs:$SELF/runs"
dke() {   # dke NAME GPU(0|1) CMD... : run CMD (bash -c) in the winner's image with the e2e mounts; logs to $E/logs/NAME.log
  local n=$1 g=$2; shift 2; local gp=""; [ $g = 1 ] && gp="--gpus all --shm-size 4g"
  [ -s $E/HALT ] && fail "halted: $(cat $E/HALT)"
  hg_check; [ -n "$HG_WHY" ] && fail "host guard before $n: $HG_WHY"
  $DK run --rm --name e2e_$n $gp --memory $CMEM --memory-swap $CMEM $MOUNTS -e FL_ROOT=/opt/fl -e V8_DIR=/opt/fl/third_party/v8in \
    -e VILLA_SRC=/opt/villa/vesuvius/src -e PYTHONPATH=/opt/villa/vesuvius/src -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 --entrypoint bash fl-claim-a7:final -c "$*" >> $E/logs/$n.log 2>&1
}
cmp_stage() {   # cmp_stage STAGE OURS_SUBDIR SHIPPED_SUBDIR
  dke cmp_$1 0 "python /data/train_e2e/scripts/e2e_compare.py $1 /data/train_e2e/self/$2 /data/winner/hf-data/data/$3 /data/train_e2e/compare/$1.json" \
    || fail "compare $1 (logs/cmp_$1.log)"
  log "compare $1: $(tail -5 $E/logs/cmp_$1.log | tr '\n' ' ' | cut -c1-700)"
}
log "E2E chain start run=$RUNID: stages $STAGES"
# one GPU job at a time: don't start this while another GPU chain runs
watchdog & WD=$!
MEMBERS=${MEMBERS:-hn4_s0 hn4_s1 hn4_s2 hn4_decoy}; CFGS=$P3/m306/denoise/configs; CC=$P3/m306/denoise/cloud_code
Z=/data/winner/hf-data/results/claim_surface_render_41layers.zarr
for st in ${STAGES//,/ }; do
  t=$(date +%s)
  case $st in
    masks)  dke masks 0 "cd $SELF/code && python masks.py" || fail "masks (logs/masks.log)"; cmp_stage masks pool 1_pool_masks;;
    teach_readers) for i in 1 2 3 4 5 6 7 8; do dke teach_readers 1 "cd $SELF/code && MAXMIN=600 VRAM_GB=9 python teachers.py readers"; rc=$?
                      [ $rc = 3 ] && continue; [ $rc = 0 ] && break; fail "teachers readers exit $rc (logs/teach_readers.log)"; done;;
    teach_v8) for i in 1 2 3 4 5 6 7 8; do dke teach_v8 1 "cd $SELF/code && MAXMIN=600 VRAM_GB=9 V8_STRIDE=32 python teachers.py v8"; rc=$?
                 [ $rc = 3 ] && continue; [ $rc = 0 ] && break; fail "teachers v8 exit $rc (logs/teach_v8.log)"; done
              cmp_stage teach teach 2_teacher_maps;;
    labels) dke labels 0 "cd $SELF/code && python labels.py" || fail "labels (logs/labels.log)"; cmp_stage labels labels 3_machine_labels;;
    exclusion) mkdir -p $E/exclusion; dke exclusion 0 "python /opt/fl/claim/exclusion_audit_v7.py recompute $P3/m306/denoise/configs/hn4_s0.json /data/train_e2e/exclusion --lane self --p3 $P3"                  || fail "exclusion (logs/exclusion.log)"
               dke cmp_exclusion 0 "python /data/train_e2e/scripts/e2e_compare.py exclusion /data/train_e2e/exclusion /data/winner/hf-data/data/0_exclusion_zone /data/train_e2e/compare/exclusion.json"                  || fail "compare exclusion"; log "compare exclusion: $(tail -3 $E/logs/exclusion.log | tr '
' ' ' | cut -c1-300)";;
    packs) dke packs 0 "cd $SELF/code && python make_packs.py 306" || fail "packs (logs/packs.log)"; cmp_stage packs packs 3b_cloud_packs;;
    anchors_shipped)   # the hn4 configs' anchors (w058, w060): the shipped packs, sha256-checked against the shipped MANIFEST
      for a in a_w058.npz a_w060.npz; do cp $H/3b_cloud_packs/$a $E/self/packs/$a.tmp && mv $E/self/packs/$a.tmp $E/self/packs/$a
        want=$(grep -A3 "\"$a\"" $H/3b_cloud_packs/MANIFEST.json | grep -o '"sha256": "[0-9a-f]*"' | cut -d'"' -f4); got=$(sha256sum $E/self/packs/$a | cut -c1-64)
        [ "$want" = "$got" ] || fail "anchor $a sha256 $got != MANIFEST $want"; log "anchor $a sha256 ok ($got)"; done;;
    train)   # shipped config with vram_gb 10.5 and grad_ckpt block (the trainer's own defaults are block / 9.0; the shipped hn4 configs
             # set false / 30.0); name <m>_$SUFFIX; seed, steps, data, loss unchanged. Whole-card VRAM logged every 500 ms per member.
      mkdir -p $E/configs
      for m in $MEMBERS; do dke cfg_$m 0 "python -c \"import json; c = json.load(open('$CFGS/$m.json')); c['name'] = '${m}_$SUFFIX'; c['training'].update(vram_gb=10.5, grad_ckpt='block'); json.dump(c, open('/data/train_e2e/configs/${m}_$SUFFIX.json', 'w'), indent=1)\"" || fail "config $m"
        $NS --query-gpu=timestamp,memory.used --format=csv -lms 500 >> $E/logs/vram_train_${m}_$SUFFIX.csv 2>/dev/null & VM=$!
        for pass in $(seq 1 10); do [ -s $E/self/runs/${m}_$SUFFIX/done.json ] && break
          t=$(date +%s); dke train_$m 1 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $CC/train_cloud_hn.py /data/train_e2e/configs/${m}_$SUFFIX.json --max-minutes 600"; rc=$?
          log "train $m pass $pass exit=$rc ($(( $(date +%s) - t ))s) $(grep -a '\"step\"' $E/logs/train_$m.log | tail -1)"
          [ -s $E/HALT ] && fail "halted during train $m: $(cat $E/HALT)"; [ $rc = 0 ] || [ $rc = 3 ] || fail "train $m exit $rc"; done
        kill $VM 2>/dev/null; log "train $m whole-card VRAM peak $(awk -F', ' '$2 ~ /MiB/ {v = $2; sub(/ MiB/, "", v); v += 0; if (v > p) p = v} END {print p}' $E/logs/vram_train_${m}_$SUFFIX.csv) MiB (logs/vram_train_${m}_$SUFFIX.csv)"
        [ -s $E/self/runs/${m}_$SUFFIX/done.json ] || fail "train $m: no done.json"; log "train $m done: $(tr -d '
 ' < $E/self/runs/${m}_$SUFFIX/done.json)"; done
      dke cmp_trainlog 0 "python /data/train_e2e/scripts/e2e_trainlog.py /data/train_e2e/self/runs /data/winner/hf-model/run_records /data/train_e2e/compare/trainlog$([ $SUFFIX = e2e ] || echo _$SUFFIX).json $(for m in $MEMBERS; do echo -n "$m:${m}_$SUFFIX "; done)" || fail "trainlog compare"
      log "compare trainlog: $(tail -4 $E/logs/cmp_trainlog.log | tr '
' ' ' | cut -c1-600)";;
    members)   # exact shipped member recipe (pc/exact_s0.py) with each e2e checkpoint, on the claim render; TRACE (a shipped model, not
               # trained in train mode) regenerated here with the shipped crop
      mkdir -p $E/maps
      [ -s $E/maps/fwd_tr.npy ] || dke trace 1 "cd /opt/fl/claim && python ink_v2.py $Z /data/train_e2e/maps --models tr --crop 116,4167,694,7639 --vram 7.5"
      [ -s $E/maps/fwd_tr.npy ] || fail "TRACE map missing after the trace step (logs/trace.log)"
      for m in $MEMBERS; do for pass in 1 2 3; do [ -s $E/maps/fwd_$m.npy ] && break
          t=$(date +%s); dke map_$m 1 "python /data/train_e2e/scripts/exact_s0.py $Z /data/train_e2e/maps/fwd_$m.npy $SELF/runs/${m}_$SUFFIX/final/model.safetensors"
          log "map $m pass $pass exit=$? ($(( $(date +%s) - t ))s)"; [ -s $E/HALT ] && fail "halted during map $m"; done
        [ -s $E/maps/fwd_$m.npy ] || fail "map $m (logs/map_$m.log)"; done;;
    fuse)   # fuse_v7 on our TRACE + e2e members (+ decoy), then every map vs the shipped claim maps
      dke fuse 0 "python /data/train_e2e/scripts/pad_frame.py $(for m in $MEMBERS; do echo -n "/data/train_e2e/maps/fwd_$m.npy "; done) && cd /opt/fl/claim && python fuse_v7.py /data/train_e2e/maps/fused --trace /data/train_e2e/maps/fwd_tr.npy --members /data/train_e2e/maps/fwd_hn4_s0.npy /data/train_e2e/maps/fwd_hn4_s1.npy /data/train_e2e/maps/fwd_hn4_s2.npy --decoy /data/train_e2e/maps/fwd_hn4_decoy.npy" || fail "fuse (logs/fuse.log)"
      dke cmp_maps 0 "python /data/train_e2e/scripts/exact_all_compare.py /data/train_e2e/maps /data/train_e2e/compare/maps.json" || fail "maps compare"
      log "compare maps: $(tail -9 $E/logs/cmp_maps.log | tr '
' ' ' | cut -c1-900)";;
    *) fail "unknown stage $st";;
  esac
  log "stage $st done ($(( $(date +%s) - t ))s)"
done
MARKED=1; log "E2E_DONE run=$RUNID stages $STAGES"
