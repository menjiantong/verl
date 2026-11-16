#!/usr/bin/env bash
# [final] 2k→6k：图模式 + KV 3 GiB + actor mb1 / ref·logprob mb2（唯一不 OOM 的 6k 微批）
# 形状：prompt≤2048，response 6144；actor mb2 会在 update_actor OOM（LM head 全量 fp32 logits）
# 实测（step1）：199 1329.2 s/步（gen 276.3 s）；146 1356.9 s/步（gen 282.3 s）
# 启动：STEPS=200 bash my_scripts/final/1116_dsv41_2k6k_graph_mb1_v1.sh
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env_dsv41.sh"

export MAX_PROMPT_LEN=2048 MAX_RESPONSE_LEN=6144
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  "$@"
