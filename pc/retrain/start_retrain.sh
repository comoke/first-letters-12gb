#!/bin/bash
# start_retrain.sh : copy the repo's retrain scripts + config to $VESUVIUS_DATA/retrain and start retrain_chain.sh in tmux 'retrain'.
set -eu
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"; P=$PC; R=$D/retrain
mkdir -p $R/scripts $R/configs $R/logs
cp $P/retrain/bn_check.py $P/retrain/retrain_compare.py $P/retrain/retrain_chain.sh $P/exact_s0.py $R/scripts/
cp $P/retrain/hn4_s0_pc.json $R/configs/
tmux has-session -t retrain 2>/dev/null && { echo "tmux 'retrain' already running"; exit 1; }
tmux new-session -d -s retrain "PC=$PC VESUVIUS_DATA=$D DOCKER=$DK NVIDIA_SMI=$NS bash $R/scripts/retrain_chain.sh >> $R/chain.log 2>&1"   # pass the settings explicitly (a running tmux server keeps its own environment)
echo started; tmux ls
