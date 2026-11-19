#!/bin/bash
# labels_chain.sh : CPU-only. Pre-fetches the ink_9um labels for TAGS (default w03 then w09) (sequential: HF pacing is per process) and
# writes <out>/<tag>/estimate.json (mask bbox, crop, volume chunks / GB a full run fetches). Log: <out>/labels_chain.log
source "$(dirname "$0")/../env.sh"
OUT=$D/finetune/paris4; LOG=$OUT/labels_chain.log; mkdir -p $OUT
for t in ${TAGS:-w03 w09}; do
  $DK run --rm --memory 6g -v $PC/finetune:/code:ro -v $OUT:/out --entrypoint python fl-claim-a7:final -u /code/fetch_paris4.py \
    $t /out --labels-only --every 1000 >> $LOG 2>&1
done
echo LABELS_CHAIN_END >> $LOG
