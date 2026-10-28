#!/usr/bin/env bash
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export EXPERIMENT_NAME=perf_opt_paramresident_optcpu
export PARAM_OFFLOAD=False
export OPTIMIZER_OFFLOAD=True
export OFFLOAD_POLICY=False
export REF_PARAM_OFFLOAD=False
nohup bash dsv41_grpo_1node_21.sh single 3 > ../logs/perf_opt_paramresident_optcpu.outer.log 2>&1 < /dev/null &
echo "launched pid $!"
