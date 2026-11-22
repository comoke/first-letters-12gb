#!/bin/bash
# job_ahnbench.sh [TAG] [extra ahn_bench.py args] : one job, A-hn forward variants on 256 real B0 tiles (hn4_s0), about 10 min of GPU.
# Output $VESUVIUS_DATA/opt4070/ahnbench_<TAG>.json and .log (ends O4070_DONE / O4070_FAIL).
TAG=${1:-1}; shift
H=$(cd "$(dirname "$0")" && pwd); source "$H/../env.sh"; O=$D/opt4070
CMEM=${CMEM:-5g} exec bash $H/dk_job.sh ahnbench_$TAG $O/ahnbench_$TAG.log -- python /code/ahn_bench.py \
  --ckpt /data/winner/hf-model/checkpoints/hn4_s0_final/model.safetensors \
  --zarr /data/winner/hf-data/results/claim_surface_render_41layers.zarr --out /data/opt4070/ahnbench_$TAG.json --vram 8 "$@"
