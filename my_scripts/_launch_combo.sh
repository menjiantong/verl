#!/usr/bin/env bash
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export EXPERIMENT_NAME=perf_opt_combo_expand_adamwswap
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRAIN_EXTRA_ARGS="actor_rollout_ref.actor.optim.optimizer_impl=fsdp_turbo.memory.swap_optimizer actor_rollout_ref.actor.optim.optimizer=AdamWSwap actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
nohup bash dsv41_grpo_1node_21.sh single 3 > ../logs/perf_opt_combo_expand_adamwswap.outer.log 2>&1 < /dev/null &
echo "launched pid $!"
