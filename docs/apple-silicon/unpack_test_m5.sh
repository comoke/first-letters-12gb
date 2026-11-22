#!/bin/zsh
# One-off test of 'fetch_winner.sh unpack' (no Docker) against the Mac's existing code tree. Read-only on the existing tree.
export PATH=/opt/homebrew/bin:$PATH
T=~/Workspace/vesuvius-data-mac/.native-test; rm -rf $T; mkdir -p $T/md; cp ~/Workspace/vesuvius-data-mac/.native-stage/fetch_winner.sh $T/
echo "== download $(date -u +%T)"
~/Workspace/vesuvius-data-mac/.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Prodigle/pherc0306b-first-letters', 'docker/fl-claim-a7_final.tar.zst', repo_type='dataset', local_dir='$T'))" || { echo NATIVE_DONE rc=dl; exit 1; }
ls -la $T/docker
/opt/homebrew/bin/zstd -dc $T/docker/fl-claim-a7_final.tar.zst | wc -c | xargs echo unpacked_bytes; echo "== unpack $(date -u +%T)"
MD=$T/md PY=/usr/bin/python3 bash $T/fetch_winner.sh unpack $T/docker/fl-claim-a7_final.tar.zst || { echo NATIVE_DONE rc=unpack; exit 1; }
echo "== compare $(date -u +%T)"
R=~/Workspace/vesuvius-data-mac
d1=$(diff -rq $T/md/code $R/code 2>&1 | grep -v __pycache__ | wc -l); diff -rq $T/md/code $R/code 2>&1 | grep -v __pycache__ 
diff -rq $T/md/fl/third_party/v8in $R/fl/third_party/v8in; d2=0
d3=$(diff -rq $T/md/fl/models/community/trace $R/fl/models/community/trace 2>&1 | grep -v __pycache__ | wc -l)
echo "differences: code $d1, v8in $d2, trace $d3"
shasum -a 256 $T/md/fl/third_party/v8in/model.safetensors
du -sh $T/md/code $T/md/fl
du -sh $T/docker; ls -la $T/docker; rm -rf $T
echo NATIVE_DONE rc=0
