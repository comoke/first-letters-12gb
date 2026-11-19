#!/bin/bash
# download_winner.sh : everything the chains need from the winner, into $VESUVIUS_DATA/winner (pc/env.sh).
#   repo      github.com/SirProdigle/pherc0306b-first-letters (code, docs; shallow clone)
#   hf-model  Hugging Face model Prodigle/pherc0306b-ink-ahn: checkpoints and run records, about 6 GB
#   hf-data   Hugging Face dataset Prodigle/pherc0306b-first-letters, about 17 GB: renders, meshes, training data, shipped maps,
#             and the Docker image archive docker/fl-claim-a7_final.tar.zst (6.6 GB; its sha256 is checked below)
# Then load the image (writes a 12 GB .tar next to the archive, then docker load):
#   pc/zstd_load.sh "$VESUVIUS_DATA/winner/hf-data/docker/fl-claim-a7_final.tar.zst"
# Disk: about 23 GB of downloads + 12 GB unpacked tar on the data drive, plus about 9 GB of outputs for the full training chain;
# the loaded image takes 24 GB wherever Docker keeps its images (Docker Desktop: its VHD on C: unless moved).
# Needs git and uv (for the hf CLI). The hf CLI skips files already downloaded, so re-running resumes.
set -euo pipefail
source "${PC:-$(cd "$(dirname "$0")" && pwd)}/env.sh"
W=$D/winner; mkdir -p "$W" "$D/hf-home"
export HF_HOME=$D/hf-home HF_HUB_ENABLE_HF_TRANSFER=1
UVX=$(command -v uvx || echo "$HOME/.local/bin/uvx")
HF="$UVX --from huggingface_hub[cli,hf_transfer] hf"
cd "$W"
[ -d repo ] || git clone --depth 1 https://github.com/SirProdigle/pherc0306b-first-letters.git repo
$HF download Prodigle/pherc0306b-ink-ahn --local-dir hf-model; echo MODEL_DONE   # set -e stops the script before this line on a failure
$HF download Prodigle/pherc0306b-first-letters --repo-type dataset --local-dir hf-data; echo DATA_DONE
(cd hf-data/docker && sha256sum -c fl-claim-a7_final.tar.zst.sha256)
du -sh hf-model hf-data repo
