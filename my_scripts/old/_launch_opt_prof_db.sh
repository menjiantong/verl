#!/usr/bin/env bash
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export EXPERIMENT_NAME=perf_opt_combo_profile_db
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRAIN_EXTRA_ARGS="actor_rollout_ref.actor.optim.optimizer_impl=fsdp_turbo.memory.swap_optimizer actor_rollout_ref.actor.optim.optimizer=AdamWSwap actor_rollout_ref.actor.fsdp_config.optimizer_offload=False global_profiler.tool=npu global_profiler.steps=[2] global_profiler.profile_continuous_steps=False global_profiler.save_path=/mnt/share/m00899630/workspace/dsv41-verl-copy/profiles/run_113_opt_combo_db actor_rollout_ref.actor.profiler.enable=True actor_rollout_ref.actor.profiler.all_ranks=False actor_rollout_ref.actor.profiler.ranks=[0] actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory] actor_rollout_ref.actor.profiler.tool_config.npu.level=level1 actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True actor_rollout_ref.ref.profiler.enable=False actor_rollout_ref.rollout.profiler.enable=False"
nohup bash dsv41_grpo_1node_21.sh single 2 > ../logs/perf_opt_combo_profile_db.outer.log 2>&1 < /dev/null &
echo "launched pid $!"
