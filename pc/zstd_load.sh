#!/bin/bash
# zstd_load.sh <image.tar.zst> : decompress a .tar.zst docker image to a .tar next to it (python-zstandard via uv, no sudo), then
# docker load -i. Usage: pc/zstd_load.sh "$VESUVIUS_DATA/winner/hf-data/docker/fl-claim-a7_final.tar.zst"
# Streaming into docker load over the WSL->Docker Desktop pipe timed out once, so go via a file.
set -euo pipefail
source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"
f=${1:?usage: zstd_load.sh <image.tar.zst>}; t=${f%.zst}
if [ ! -s "$t" ]; then
  uv run --quiet --no-project --with zstandard python -c "
import sys, zstandard
with open(sys.argv[1],'rb') as fh, open(sys.argv[2]+'.part','wb') as out:
    zstandard.ZstdDecompressor().copy_stream(fh, out, read_size=1<<24, write_size=1<<24)
" "$f" "$t"
  mv "$t.part" "$t"
fi
"$DK" load -i "$t"
