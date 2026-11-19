#!/bin/bash
# data_chain.sh : fine-tune demo data (docs/finetune-protocol-2026-11-17.md), CPU only.
#   0. TAGS (default "w03 w09") are the segments to fetch, PACK (default = TAGS; empty = no packing) the ones to pack;
#   1. waits for the labels pre-fetch (tmux p4labels) to end;
#   2. fetch_paris4.py w03, then w09: label zarrs (cached), S3 level-2 range reads, 24 pooled layers, labels/mask from channel 10;
#   3. checks each segment against the gate-4 spec: 24 layer TIFFs, all the labels' shape, labels and mask in {0, 255}, mask non-empty;
#   4. packs both with the winner's own code (pack_p4.py: save_anchor(p4<tag>, src_1447(tag), 'paris4')) into $D/finetune/self/packs;
#   5. checks the packs (keys, tile counts, label range).
# Disk floors: the data drive ($VESUVIUS_DATA) >= 40 GB free; on WSL also C: >= 25 GB free (pagefile). Checked before each fetch. Resumable. Log: $D/finetune/data_chain.log.
# Marker: P4DATA_DONE run=<id> <summary> | P4DATA_FAIL run=<id> <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
TAGS=${TAGS:-w03 w09}; PACK=${PACK-$TAGS}
OUT=$D/finetune/paris4; SELFD=$D/finetune/self; W=$D/winner/repo/submission/code/training/x_push3/m306
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}; log() { echo "$(date '+%F %T') $*"; }
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "P4DATA_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "P4DATA_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
floors() { local d c=n/a; d=$(df -m "$D" | awk 'NR==2 {print $4}')
  [ "$d" -ge 40960 ] || fail "data drive ($D): $d MB free < 40 GB"
  if [ -d /mnt/c ]; then c=$(df -m /mnt/c | awk 'NR==2 {print $4}'); [ "$c" -ge 25600 ] || fail "C: $c MB free < 25 GB (Windows pagefile)"; fi
  log "disk: data drive $d MB, C: $c MB free"; }
mkdir -p $OUT $SELFD/packs
log "P4DATA start run=$RUNID"
while tmux has-session -t p4labels 2>/dev/null; do touch $D/finetune/data_chain.log 2>/dev/null; sleep 120; done
for t in $TAGS; do
  floors
  if [ ! -s $OUT/$t/layers/23.tif ] || [ ! -s $OUT/$t/labels/mask.png ]; then
    $DK run --rm --memory 8g -v $PC/finetune:/code:ro -v $OUT:/out --entrypoint python fl-claim-a7:final -u /code/fetch_paris4.py $t /out --every 500 \
      > $D/finetune/fetch_$t.log 2>&1
    grep -q "FETCH_DONE $t" $D/finetune/fetch_$t.log || fail "fetch $t: $(tail -2 $D/finetune/fetch_$t.log | tr '\n' ' ' | cut -c1-300)"
  fi
  log "fetch $t: $(grep -s "FETCH_DONE $t" $D/finetune/fetch_$t.log | tail -1 | cut -c1-200 || true)$([ -s $D/finetune/fetch_$t.log ] || echo "already in place, not fetched"); on disk $(du -sh $OUT/$t | cut -f1)"
  $DK run --rm --memory 8g -v $OUT:/out --entrypoint python fl-claim-a7:final -c "
import json, numpy as np, tifffile, cv2
t = '/out/$t'; lab = cv2.imread(t + '/labels/inklabels.png', 0); m = cv2.imread(t + '/labels/mask.png', 0)
L = [tifffile.imread(f'{t}/layers/{i:02d}.tif') for i in range(24)]
ok = dict(layers=len(L), shapes_equal=all(a.shape == lab.shape for a in L) and m.shape == lab.shape, dtype=str(L[0].dtype),
          lab_values=sorted(int(v) for v in np.unique(lab)), mask_values=sorted(int(v) for v in np.unique(m)),
          mask_frac=float((m > 0).mean()), ink_frac_in_mask=float((lab[m > 0] > 0).mean()), shape=list(lab.shape),
          layer_means=[round(float(a[m > 0].mean()), 1) for a in L])
json.dump(ok, open(t + '/check.json', 'w'), indent=1); print(json.dumps(ok))
assert ok['layers'] == 24 and ok['shapes_equal'] and set(ok['lab_values']) <= {0, 255} and set(ok['mask_values']) <= {0, 255} and ok['mask_frac'] > 0
" > $D/finetune/check_$t.log 2>&1 || fail "check $t: $(tail -3 $D/finetune/check_$t.log | tr '\n' ' ' | cut -c1-300)"
  log "check $t: $(head -c 300 $D/finetune/check_$t.log)"
done
floors; [ -z "$PACK" ] && { MARKED=1; log "P4DATA_DONE run=$RUNID fetched $TAGS (no packing)"; exit 0; }
$DK run --rm --network none --memory 10g --memory-swap 10g --cpus 4 -v $OUT:/home/liam/projects/vesuvius-work/data/pherc1447:ro \
  -v $W:/code:ro -v $PC/finetune:/drv:ro -v $SELFD:/out/self --entrypoint python fl-claim-a7:final -I /drv/pack_p4.py $PACK \
  > $D/finetune/pack.log 2>&1; grep -q PACK_OK $D/finetune/pack.log || fail "pack: $(tail -3 $D/finetune/pack.log | tr '\n' ' ' | cut -c1-300)"
$DK run --rm --memory 6g -v $SELFD/packs:/p --entrypoint python fl-claim-a7:final -c "
import json, numpy as np
out = {}
for n in [f'p4{t}' for t in '$PACK'.split()]:
    z = np.load(f'/p/a_{n}.npz', allow_pickle=True)
    out[n] = dict(keys=sorted(z.files), img=list(z['img'].shape), img_dtype=str(z['img'].dtype), lab=[float(z['lab'].min()), float(z['lab'].max())],
                  lab_mean=float(z['lab'].astype(np.float32).mean()), tiles=int(z['img'].shape[0]))
json.dump(out, open('/p/p4_packs_check.json', 'w'), indent=1); print(json.dumps(out))
" > $D/finetune/pack_check.log 2>&1 || fail "pack check: $(tail -3 $D/finetune/pack_check.log | tr '\n' ' ' | cut -c1-300)"
MARKED=1; log "P4DATA_DONE run=$RUNID packs $(cd $SELFD/packs && du -sh $(for t in $PACK; do echo a_p4$t.npz; done) | awk '{print $2": "$1}' | tr '\n' ' ') | $(head -c 400 $D/finetune/pack_check.log)"
