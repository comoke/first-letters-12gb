#!/bin/bash
# fresh_clone_test.sh: the PUBLIC EXPORT (our export script's output, not included in this repository; /mnt/d/entry-fresh/export: no git, no private files)
# + a fresh data dir on C: (SSD),
# runs the masks stage and stage 9 with VESUVIUS_DATA pointing there, no PC set, cwd /tmp. The data dir holds a plain copy of only
# what those stages read (hard links fail on this drive): winner/hf-data/data/0_inputs (masks) and our four e2e maps (stage 9).
# Download step: the image archive's sha256 check (no 23 GB re-download). Results kept in /mnt/d/entry-fresh/; the C: copy is
# deleted at the end. Marker: FRESH_DONE run=<id> <summary> | FRESH_FAIL run=<id> <why>.
# FRESH_DIR overrides the test folder (each export round uses its own; default as for the first run).
set -u
F=${FRESH_DIR:-/mnt/d/entry-fresh}; R=$F/export; DD=/mnt/c/entry-fresh-data; L=$F/fresh3.log; OLD=/mnt/d/Workspace/vesuvius-data
RUNID=$(date +%Y%m%d-%H%M%S)-$$
log() { echo "$(date '+%F %T') $*" >> $L; }
MARKED=0; cleanup() { rm -rf "$DD"; }
trap 'st_=$?; cleanup; [ $MARKED = 1 ] || log "FRESH_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "FRESH_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
cfree=$(df -m /mnt/c | awk 'NR==2 {print $4}'); [ "$cfree" -gt 31000 ] || fail "C: has only $cfree MB free (needs 25 GB + 6 GB for the copy)"
log "FRESH start run=$RUNID: export $R ($(wc -l < $R/FILES.sha256) files, FILES.sha256 $(sha256sum $R/FILES.sha256 | cut -c1-12)), data $DD, C: free $cfree MB"
mkdir -p $DD/winner/hf-data/data $DD/train_e2e/maps/fused
t=$(date +%s); cp -r $OLD/winner/hf-data/data/0_inputs $OLD/winner/hf-data/data/1_pool_masks $DD/winner/hf-data/data/ || fail "copy inputs"
log "copied 0_inputs ($(du -sh $DD/winner/hf-data/data/0_inputs | cut -f1), $(( $(date +%s) - t ))s); C: free $(df -m /mnt/c | awk 'NR==2 {print $4}') MB"
cp $OLD/train_e2e/maps/fwd_tr.npy $DD/train_e2e/maps/ && for m in prod ensA prodd; do cp $OLD/train_e2e/maps/fused/fwd_$m.npy $DD/train_e2e/maps/fused/; done || fail "copy maps"
export VESUVIUS_DATA=$DD; unset PC; cd /tmp
log "1. download check: image archive sha256 against the dataset's .sha256 (archive read from the existing download)"
(cd $OLD/winner/hf-data/docker && sha256sum -c fl-claim-a7_final.tar.zst.sha256) >> $L 2>&1 || fail "image sha256"
bash -n $R/pc/download_winner.sh || fail "download_winner.sh does not parse"; log "download_winner.sh parses"
log "2. e2e_chain.sh STAGES=masks"; STAGES=masks bash $R/pc/train_e2e/e2e_chain.sh >> $L 2>&1 || fail "masks stage"
log "3. stage9.sh"; bash $R/pc/train_e2e/stage9.sh >> $L 2>&1 || fail "stage 9"
mkdir -p $F/out; cp $DD/train_e2e/compare/masks.json $DD/train_e2e/stage9/diff.json $DD/train_e2e/stage9/shipped.log $F/out/ 2>/dev/null
log "kept: $(ls $F/out | tr '\n' ' ')"
log "paths check: lines naming our checkouts or data dir outside this test's own copy steps:"
grep -n -E "/mnt/d/Workspace/(vesuvius-entry|vesuvius/|vesuvius-data/train_e2e)" $L | grep -v -E "copied|FRESH start|download check" | head -5 >> $L
MARKED=1; log "FRESH_DONE run=$RUNID masks $(grep -o '"identical[^,]*' $F/out/masks.json | head -1) stage9 $(grep -a 'MATCH' $L | tail -1 | cut -c1-80)"
