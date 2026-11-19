#!/usr/bin/env bash
# anchors_rebuild.sh : rebuild the PHerc1447 anchor packs a_w058.npz / a_w060.npz (used by the winner's hn4_s0/s1/s2/decoy
# A-hn configs) from the raw public PHerc1447 data with the winner's OWN code, then compare them with the shipped packs.
#   1. download w058 + w060 labels/{inklabels,mask}.png and layers/00..23.tif (~1.09 GB) from the HF dataset
#      YoussefMoNader/ink-8um-pherc1447-surfaces @ 7e4d918712a8d8642a6fdeee45b80217c1e98317 (the revision named in the
#      winner's docs/MODELS_AND_DATA.md and submission/method/model/data_sources.json); sha256-check each against the
#      dataset's SHA256SUMS.
#   2. in fl-claim-a7:final (CPU only, --memory 4g, --network none, python -I) run anchors_rebuild.py, which calls the
#      winner's make_packs.save_anchor(w, v8306lib.src_1447(w), '1447') unchanged (raw data mounted at the winner's
#      D1447 path so info['path'] matches).
#   3. anchors_compare.py: file sha256, per-key dtype/shape/n_diff/max_abs_diff, zip member CRCs and a zip-timestamp-
#      normalised sha256, against winner/hf-data/data/3b_cloud_packs/a_w0{58,60}.npz.
# Untrusted downloads live only in $O/raw; nothing from them is executed.  Log: $O/anchors_rebuild.log, end marker
# ANCHORS_DONE / ANCHORS_FAIL <why>.  Run detached:  tmux new -d -s anchors 'bash <this file>'
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
O=$D/train_e2e/anchors_rebuild
REV=7e4d918712a8d8642a6fdeee45b80217c1e98317
URL=https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/$REV
SHIP=$D/winner/hf-data/data/3b_cloud_packs
CODE=$D/winner/repo/submission/code/training/x_push3/m306
DRV=$(cd "$(dirname "$0")" && pwd)
IMG=fl-claim-a7:final
RAW=$O/raw/pherc1447
LOG=$O/anchors_rebuild.log
mkdir -p $RAW $O/out/self/packs $O/compare
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
fail() { log "ANCHORS_FAIL $*"; exit 1; }
T0=$(date +%s)
log "start rev=$REV"

# 1. download + verify
curl -sfL --retry 5 -o $RAW/SHA256SUMS $URL/SHA256SUMS || fail "SHA256SUMS download"
for w in w058 w060; do
  for f in labels/inklabels.png labels/mask.png $(for i in $(seq -w 0 23); do echo layers/$i.tif; done); do
    p=$w/$f; mkdir -p $(dirname $RAW/$p)
    want=$(awk -v p=$p '$2==p{print $1}' $RAW/SHA256SUMS); [ -n "$want" ] || fail "no SHA256SUMS entry for $p"
    if [ -s $RAW/$p ] && [ "$(sha256sum $RAW/$p | cut -c1-64)" = "$want" ]; then continue; fi
    curl -sfL --retry 5 -o $RAW/$p.part $URL/$p || fail "download $p"
    got=$(sha256sum $RAW/$p.part | cut -c1-64); [ "$got" = "$want" ] || fail "sha256 $p $got != $want"
    mv $RAW/$p.part $RAW/$p; log "got $p $(stat -c %s $RAW/$p) B"
  done
done
log "download ok: $(du -sb $RAW | cut -f1) B in $RAW"

# 2. build with the winner's code
rm -f $O/out/self/packs/a_w058.npz $O/out/self/packs/a_w060.npz
$DK run --rm --network none --memory 4g --cpus 4 --entrypoint python \
  -v $RAW:/home/liam/projects/vesuvius-work/data/pherc1447:ro -v $CODE:/code:ro -v $DRV:/drv:ro -v $O/out:/out \
  $IMG -I /drv/anchors_rebuild.py w058 w060 >> $LOG 2>&1 || fail "build (docker exit $?)"
grep -q BUILD_OK $LOG || fail "build: no BUILD_OK"

# 3. compare
for w in w058 w060; do
  $DK run --rm --network none --memory 4g --cpus 4 --entrypoint python \
    -v $O/out/self/packs:/reb:ro -v $SHIP:/ship:ro -v $DRV:/drv:ro -v $O/compare:/cmp \
    $IMG -I /drv/anchors_compare.py /reb/a_$w.npz /ship/a_$w.npz /cmp/a_$w.json >> $LOG 2>&1 || fail "compare $w (docker exit $?)"
done
log "wall $(( $(date +%s) - T0 )) s"
log "ANCHORS_DONE"
