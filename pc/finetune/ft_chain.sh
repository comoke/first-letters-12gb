#!/bin/bash
# ft_chain.sh : the fine-tune demo's GPU runs, exactly as fixed in docs/finetune-protocol-2026-11-17.md (README, Demo: timeline).
#   0. needs P4DATA_DONE from pc/finetune/data_chain.sh in $DLOG (waits while tmux p4data runs); copies the shipped training packs (5 PHerc0306B regions + anchors w058, w060) next to
#      the Scroll 1 packs in $D/finetune/self/packs.
#   1. layer order, on A only: the shipped s0 zero-shot on w03 with the stored order fed reversed (= how src_1447 feeds training)
#      and not reversed; AUC against A's labels. KILL (FT_KILL) if both < 0.55. If the not-reversed order wins, the stored TIFFs are
#      flipped for both segments and the packs rebuilt, so training (src_1447 reads 23 -> 0) and evaluation (--reverse) agree.
#   2. training arms, 300 steps each from the shipped hn4_s0 final checkpoint, 12 GB recipe (grad_ckpt block, vram_gb 10.5,
#      expandable_segments): b1, b2 = shipped mix + anchor p4w03 (seeds 1, 2); c1, c2 = shipped mix only (seeds 1, 2).
#   3. maps on B (w09) for a (shipped s0), b1, b2, c1, c2 (eval_map.py, --reverse, mask-limited), then score.py (AUC, top-1%
#      precision, F0.5 @ 0.5, paired 512 px tile bootstrap) -> $D/finetune/results/score_B.json.
# Host guard before every GPU step; the 6 GPU-h cap is checked before each step from the chain's own GPU seconds.
# Marker: FT_DONE run=<id> <summary> | FT_KILL run=<id> <why> | FT_FAIL run=<id> <why>.
set -u
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; source $PC/hostguard.sh
F=$D/finetune; SELFD=$F/self; SEGS=$F/paris4
# pair parameters (defaults = pair 1, protocol 40e2eac; pair 2 = amendment 1: A_TAG=w06 B_TAG=w00 PFX=ft2 RES_NAME=results_p2)
A=${A_TAG:-w03}; B=${B_TAG:-w09}; PFX=${PFX:-ft}; RB=${RES_NAME:-results}; R=$F/$RB; DLOG=${DLOG:-$F/data_chain.log}; PROTO=${PROTO:-40e2eac}
TW=/home/liam/projects/vesuvius-work; P3=$TW/.claude/worktrees/leadA-push3/x_push3; SELF=$P3/m306/self   # paths inside the image
CFGS=$P3/m306/denoise/configs; CC=$P3/m306/denoise/cloud_code; W=$D/winner/repo/submission/code/training/x_push3/m306
S0=/data/winner/hf-model/checkpoints/hn4_s0_final/model.safetensors
RUNID=${RUNID:-$(date +%Y%m%d-%H%M%S)-$$}; log() { echo "$(date '+%F %T') $*"; }
MARKED=0; trap 'st_=$?; [ $MARKED = 1 ] || log "FT_FAIL run=$RUNID unexpected exit status $st_"' EXIT
fail() { log "FT_FAIL run=$RUNID $*"; MARKED=1; exit 1; }
GPUS=0; CAP=21600
guard() { hg_check; while [ -n "$HG_WHY" ]; do log "host guard: $HG_WHY; waiting"; sleep 120; hg_check; done
  [ $GPUS -lt $CAP ] || fail "GPU cap: $GPUS s used >= 6 h"; }
gpu() {   # gpu NAME CMD : one GPU container, whole-card VRAM logged, GPU seconds counted
  local n=$1; shift; guard; $NS --query-gpu=timestamp,memory.used --format=csv -lms 1000 > $R/logs/vram_$n.csv 2>/dev/null & local s=$!; local t=$(date +%s)
  $DK run --rm --name ft_$n --gpus all --shm-size 4g --memory 10g --memory-swap 10g -v $D:/data -v $SELFD/packs:$SELF/packs -v $SELFD/runs:$SELF/runs \
    -v $PC/finetune:/ft:ro -e FL_ROOT=/opt/fl -e V8_DIR=/opt/fl/third_party/v8in -e VILLA_SRC=/opt/villa/vesuvius/src -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=4 --entrypoint bash fl-claim-a7:final -c "$*" > $R/logs/$n.log 2>&1; local rc=$?
  kill $s 2>/dev/null; GPUS=$((GPUS + $(date +%s) - t))
  log "$n exit=$rc ($(( $(date +%s) - t ))s; chain GPU $GPUS s) card peak $(awk -F', ' '$2 ~ /MiB/ {v = $2; sub(/ MiB/, "", v); v += 0; if (v > p) p = v} END {print p}' $R/logs/vram_$n.csv) MiB"
  return $rc; }
mkdir -p $R/logs $SELFD/packs $SELFD/runs $F/configs
log "FT start run=$RUNID (protocol docs/finetune-protocol-2026-11-17.md @ $PROTO)"
while tmux has-session -t p4data 2>/dev/null; do touch $F/ft_chain.log 2>/dev/null; sleep 120; done
[ -s $DLOG ] || fail "no data chain log $DLOG: run pc/finetune/data_chain.sh first (TAGS=\"$A $B\"; see README, Demo)"
grep -q "P4DATA_DONE" $DLOG || fail "data chain did not finish ($DLOG): rerun pc/finetune/data_chain.sh with TAGS=\"$A $B\"; last: $(grep -a P4DATA_FAIL $DLOG | tail -1 | cut -c1-200)"
for f in 306_A6 306_w052 306_w053 306_w059 306_w109 a_w058 a_w060; do
  [ -s $SELFD/packs/$f.npz ] || { cp $D/winner/hf-data/data/3b_cloud_packs/$f.npz $SELFD/packs/$f.npz.tmp && mv $SELFD/packs/$f.npz.tmp $SELFD/packs/$f.npz; } || fail "copy $f"
done
log "training packs ready: $(ls $SELFD/packs | tr '\n' ' ')"
# ---- 1. layer order on A (shipped s0, zero-shot)
for o in rev fwd; do
  [ -s $R/A_s0_$o.npy ] || gpu orderA_$o "python /ft/eval_map.py /data/finetune/paris4/$A $S0 /data/finetune/$RB/A_s0_$o.npy $([ $o = rev ] && echo --reverse)" || fail "order check $o (logs/orderA_$o.log)"
done
$DK run --rm --memory 8g -v $D:/data -v $PC/finetune:/ft:ro --entrypoint python fl-claim-a7:final /ft/score.py /data/finetune/paris4/$A /data/finetune/$RB/order_A.json \
  rev=/data/finetune/$RB/A_s0_rev.npy fwd=/data/finetune/$RB/A_s0_fwd.npy > $R/logs/order_score.log 2>&1 || fail "order score"
read -r AR AF <<< "$(python3 -c "import json; d = json.load(open('$R/order_A.json'))['arms']; print(d['rev']['auc'], d['fwd']['auc'])")"
log "layer order on A: AUC reversed (training convention) $AR, not reversed $AF"
python3 -c "import sys; sys.exit(0 if max($AR, $AF) >= 0.55 else 1)" || { MARKED=1; log "FT_KILL run=$RUNID zero-shot AUC on A < 0.55 in both orders ($AR, $AF): inputs don't match the model (protocol kill)"; exit 0; }
# A segment is flipped at most once: FLIPPING is written before its 12 swaps and renamed to FLIPPED after them, and a rerun (guard
# trip, GPU cap, pre-emption, failure) that recomputes the same order score skips every segment with FLIPPED, so a finished flip is
# never undone. A segment left with FLIPPING (killed mid-swap) stops the chain: its order is unknown, so it must be fetched again.
# A's old-order pack is deleted before any swap, so a kill while B is flipped can't leave it behind.
if python3 -c "import sys; sys.exit(0 if $AF > $AR else 1)"; then
  todo=""; for t in $A $B; do [ -e $SEGS/$t/FLIPPED ] || todo="$todo $t"; done
  if [ -z "$todo" ]; then log "not-reversed order wins: $A and $B already flipped (FLIPPED markers); nothing to redo"
  else
    for t in $todo; do [ -e $SEGS/$t/FLIPPING ] && fail "$t: an earlier flip was interrupted mid-way, so its layer order is unknown; delete $SEGS/$t and rerun pc/finetune/data_chain.sh"; done
    log "not-reversed order wins: flipping the stored layers of$todo"
    case " $todo " in *" $A "*) rm -f $SELFD/packs/a_p4$A.npz;; esac
    for t in $todo; do d=$SEGS/$t/layers; echo "layer flip (renamed to FLIPPED when done)" > $SEGS/$t/FLIPPING
      for i in $(seq 0 11); do j=$((23 - i)); a=$(printf %02d $i); b=$(printf %02d $j)
        mv $d/$a.tif $d/swap.tif && mv $d/$b.tif $d/$a.tif && mv $d/swap.tif $d/$b.tif || fail "flip $t at layer $a"; done
      mv $SEGS/$t/FLIPPING $SEGS/$t/FLIPPED; done
    case " $todo " in *" $A "*) log "rebuilding the pack of $A"
      $DK run --rm --network none --memory 10g --memory-swap 10g --cpus 4 -v $SEGS:$TW/data/pherc1447:ro -v $W:/code:ro -v $PC/finetune:/drv:ro \
        -v $SELFD:/out/self --entrypoint python fl-claim-a7:final -I /drv/pack_p4.py $A > $R/logs/repack.log 2>&1 && grep -q PACK_OK $R/logs/repack.log || fail "repack";;
    esac
  fi
fi
[ -s $SELFD/packs/a_p4$A.npz ] || { $DK run --rm --network none --memory 10g --memory-swap 10g --cpus 4 -v $SEGS:$TW/data/pherc1447:ro -v $W:/code:ro \
  -v $PC/finetune:/drv:ro -v $SELFD:/out/self --entrypoint python fl-claim-a7:final -I /drv/pack_p4.py $A > $R/logs/pack.log 2>&1 && grep -q PACK_OK $R/logs/pack.log || fail "pack $A"; log "packed $A: $(du -sh $SELFD/packs/a_p4$A.npz | cut -f1)"; }
# ---- 2. training arms
for arm in b1 b2 c1 c2; do
  seed=${arm:1}; anc="[\"w058\", \"w060\", \"p4$A\"]"; [ ${arm:0:1} = c ] && anc='["w058", "w060"]'
  $DK run --rm --memory 2g -v $D:/data --entrypoint python fl-claim-a7:final -c "import json; c = json.load(open('$CFGS/hn4_s0.json')); c['name'] = '${PFX}_$arm'; c['init'] = '$S0'; c['anchor'] = $anc
c['training'].update(steps=300, seed=$seed, save_steps=[], vram_gb=10.5, grad_ckpt='block'); json.dump(c, open('/data/finetune/configs/${PFX}_$arm.json', 'w'), indent=1)" || fail "config $arm"
  for pass in 1 2 3 4 5; do [ -s $SELFD/runs/${PFX}_$arm/done.json ] && break
    gpu train_$arm "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $CC/train_cloud_hn.py /data/finetune/configs/${PFX}_$arm.json --max-minutes 600"; rc=$?
    [ $rc = 0 ] || [ $rc = 3 ] || fail "train $arm exit $rc (logs/train_$arm.log)"; done
  [ -s $SELFD/runs/${PFX}_$arm/done.json ] || fail "train $arm: no done.json"; log "train $arm done: $(tr -d '\n ' < $SELFD/runs/${PFX}_$arm/done.json | cut -c1-200)"
done
# ---- 3. maps on B, score
declare -A CK=([a]=$S0); for arm in b1 b2 c1 c2; do CK[$arm]=/data/finetune/self/runs/${PFX}_$arm/final/model.safetensors; done
for arm in a b1 b2 c1 c2; do
  [ -s $R/B_$arm.npy ] || gpu mapB_$arm "python /ft/eval_map.py /data/finetune/paris4/$B ${CK[$arm]} /data/finetune/$RB/B_$arm.npy --reverse" || fail "map B $arm (logs/mapB_$arm.log)"
done
$DK run --rm --memory 10g -v $D:/data -v $PC/finetune:/ft:ro --entrypoint python fl-claim-a7:final /ft/score.py /data/finetune/paris4/$B /data/finetune/$RB/score_B.json \
  $(for arm in a b1 b2 c1 c2; do echo -n "$arm=/data/finetune/$RB/B_$arm.npy "; done) > $R/logs/score_B.log 2>&1 || fail "score B (logs/score_B.log)"
MARKED=1; log "FT_DONE run=$RUNID GPU $GPUS s; $(python3 -c "import json; d = json.load(open('$R/score_B.json')); print({k: round(v['auc'], 4) for k, v in d['arms'].items()})")"
