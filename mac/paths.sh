# source me: Mac data/code roots for the winner pipeline
export MD=${MD:-$HOME/Workspace/vesuvius-data-mac}
export FL_ROOT=$MD/fl V8_DIR=$MD/fl/third_party/v8in VILLA_SRC=$MD/code/opt/villa/vesuvius/src
export AHN_DIR=$MD/code/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/bench
export PYTORCH_ENABLE_MPS_FALLBACK=0
source $MD/.venv/bin/activate
