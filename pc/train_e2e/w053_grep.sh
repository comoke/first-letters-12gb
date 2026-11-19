#!/bin/bash
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
$DK run --rm --memory 1g --entrypoint bash fl-claim-a7:final -c 'f=/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench/infer.py; grep -n "def predict_array" -A45 $f | grep -n "autocast\|half\|bfloat\|float16\|cudnn\|deterministic\|batch\|torch.no_grad\|inference_mode"; grep -n "cudnn\|deterministic\|autocast" $f | head; python -c "import torch;print(torch.__version__, torch.backends.cudnn.version())"'
