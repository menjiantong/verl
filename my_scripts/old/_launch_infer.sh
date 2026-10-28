#!/usr/bin/env bash
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export EXPERIMENT_NAME=perf_infer_prof
export TRAIN_EXTRA_ARGS="actor_rollout_ref.rollout.disable_log_stats=False actor_rollout_ref.rollout.profiler.enable=True actor_rollout_ref.rollout.profiler.all_ranks=False actor_rollout_ref.rollout.profiler.ranks=[0,8] actor_rollout_ref.rollout.profiler.relocate_results=True actor_rollout_ref.rollout.profiler.tool_config.npu.contents=[npu,cpu] actor_rollout_ref.rollout.profiler.tool_config.npu.level=level1 actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=False global_profiler.tool=npu global_profiler.steps=[1] global_profiler.profile_continuous_steps=False global_profiler.save_path=/mnt/share/m00899630/workspace/dsv41-verl-copy/profiles/run_105_infer"
nohup bash dsv41_grpo_1node_21.sh single 1 > ../logs/perf_infer_prof.outer.log 2>&1 < /dev/null &
echo "launched pid $!"
