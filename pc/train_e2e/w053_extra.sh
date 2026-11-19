#!/bin/bash
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
TW=/home/liam/projects/vesuvius-work; P3=$TW/.claude/worktrees/leadA-push3/x_push3; SELF=$P3/m306/self
$DK run --rm --memory 4g --cpus 4 --name w053_extra -v $D:/data -v $PC/train_e2e:/an:ro \
  --entrypoint python fl-claim-a7:final /an/w053_extra.py
