#!/usr/bin/env bash
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export EXPERIMENT_NAME=GRPO-DeepSeekV41-countbenchqa-lite-optlong-20261021
export VA_V41_PREFILL_LOGPROBS=0
export TRAIN_EXTRA_ARGS="actor_rollout_ref.rollout.load_format=auto"
nohup bash dsv41_grpo_countbenchqa_1node_21.sh 500 > ../logs/countbenchqa_optlong.outer.log 2>&1 < /dev/null &
echo "launched pid $!"
