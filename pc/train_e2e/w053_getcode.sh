#!/bin/bash
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
set -x
cid=$($DK create --entrypoint bash fl-claim-a7:final)
echo cid=$cid
$DK cp $cid:/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/self/code $D/train_e2e/w053/code/self
$DK cp $cid:/opt/fl/claim/exclusion_audit_v7.py $D/train_e2e/w053/code/
$DK rm $cid
ls $D/train_e2e/w053/code/self
