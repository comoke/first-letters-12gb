#!/bin/bash
# stage9.sh : the winner's stage 9 (claim images) on our end-to-end maps, CPU only, inside their image fl-claim-a7:final.
#   1. baseline: run.sh final on the shipped maps (must print MATCH: byte-identical to the package PNGs)
#   2. ours: run.sh final with MAPS = our e2e maps for the models the train mode regenerates (fwd_prod, fwd_ensA, fwd_prodd from
#      maps/fused; fwd_tr = our exact TRACE reproduction, as used in the e2e fusion) and the shipped maps for the models train mode does
#      not touch (rv2, v3, v8, v8g3, v8q); shipped CT and mesh
#   3. stage9_diff.py: pixel diff of every variant PNG (ours vs the package's), whole image and per channel
# Output: train_e2e/stage9/{shipped,ours}/final/, stage9/diff.json. Ends STAGE9_DONE | STAGE9_FAIL <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; T=$D/train_e2e; O=$T/stage9; E=$PC/train_e2e
log() { echo "$(date '+%F %T') $*"; }
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "STAGE9_FAIL unexpected exit $st_"' EXIT
fail() { log "STAGE9_FAIL $*"; MARKED=1; exit 1; }
mkdir -p $O/shipped $O/ours $O/maps
log "STAGE9 start"
for m in prod ensA prodd; do cp -u $T/maps/fused/fwd_$m.npy $O/maps/; done
cp -u $T/maps/fwd_tr.npy $O/maps/
$DK run --rm --memory 4g --entrypoint bash fl-claim-a7:final -c 'tar -C /opt/fl/data/maps -cf - fwd_rv2.npy fwd_v3.npy fwd_v8.npy fwd_v8g3.npy fwd_v8q.npy' | tar -C $O/maps -xf - || fail "copy shipped maps"
sha256sum $O/maps/*.npy > $O/maps.sha256
t=$(date +%s)
$DK run --rm --memory 4g --memory-swap 4g -v $O/shipped:/out fl-claim-a7:final final > $O/shipped.log 2>&1; rc=$?
log "shipped final: exit $rc ($(( $(date +%s) - t ))s): $(grep -E 'MATCH|DIFFERENT' $O/shipped.log | tail -1)"
t=$(date +%s)
$DK run --rm --memory 4g --memory-swap 4g -v $O/ours:/out -v $O/maps:/maps:ro -e MAPS=/maps fl-claim-a7:final final > $O/ours.log 2>&1 || fail "ours final (stage9/ours.log: $(tail -2 $O/ours.log | tr '\n' ' '))"
log "ours final: done ($(( $(date +%s) - t ))s)"
$DK run --rm --memory 4g -v $O:/s -v $E:/e:ro --entrypoint python fl-claim-a7:final -I /e/stage9_diff.py /s/shipped/final /s/ours/final /s/diff.json > $O/diff.log 2>&1 || fail "diff (stage9/diff.log)"
MARKED=1; log "STAGE9_DONE $(tail -1 $O/diff.log | cut -c1-600)"
