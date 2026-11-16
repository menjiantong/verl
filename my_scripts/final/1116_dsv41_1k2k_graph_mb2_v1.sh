#!/usr/bin/env bash
# [final] 1k→2k：图模式 + KV 3 GiB + actor mb2 / ref·logprob mb4（最快）
# 形状：prompt≤1024，response 2048；其余同上一个脚本
# 实测（2 步稳态）：204 382.1 s/步（同机比默认 mb 快 10%）；199 429.3 s/步（含残留进程干扰）
# 启动：STEPS=200 bash my_scripts/final/1116_dsv41_1k2k_graph_mb2_v1.sh
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env_dsv41.sh"

export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=2048
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  "$@"
