#!/usr/bin/env bash
# [final] 1k→2k：图模式 + 显式 KV 3 GiB（默认 micro-batch 1/1/1）
# 形状：prompt≤1024，response 2048；数据 countbenchqa_lite，bsz 8×n 8，FSDP16/EP16/CP4，TP8×DP2
# 实测（2 步稳态）：199 410.1 s/步（gen 108.2 s）；204 426.0 s/步（gen 130.2 s）
# 启动：STEPS=200 bash my_scripts/final/1116_dsv41_1k2k_graph_v1.sh
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env_dsv41.sh"

export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=2048
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472                  # 3 GiB 显式 KV 池（gmu 被 fast path 忽略）
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  "$@"
