#!/usr/bin/env bash

set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-3}
source "$here/env_dsv41_yundao_930.sh"

export PYTHONPATH=/efs_rl/m00899630/workspace/dsv41-verl-refactor/mjt_workplace/task/1120_try_tnd/scratch/fsdp_rebase:${PYTHONPATH:-}
export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=2048
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS:-3}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  "$@"
