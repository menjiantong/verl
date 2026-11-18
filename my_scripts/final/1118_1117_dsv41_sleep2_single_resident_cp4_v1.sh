#!/usr/bin/env bash
# =============================================================================
# 1118_1117_dsv41_sleep2_single_resident_v1.sh
#   单机 16 卡基线（B 线：性能/基线）：sleep mode=2 + **参数/梯度驻卡、Adam 矩留主机**
#   （swapopt 布局），32 专家 top32 模型 + countbenchqa。
#
#   与另外两个脚本构成 2×2 控制变量（数据/形状除注明的 cap 外一致）：
#     1) 1117_dsv41_sleep2_single.sh           单机 + sleep2 + **offload 仍在**（本脚本的对照 a）
#     2) 1117_dsv41_4node_mixed_swapopt_v1.sh   四机 + sleep2 + 参数梯度驻卡（本脚本的对照 b）
#     3) 本脚本                                  单机 + sleep2 + 参数梯度驻卡
#   比较：单机 offload vs 单机驻卡 = offload 开销；单机驻卡 vs 四机驻卡 = 规模效应。
#
#   相对 1117_dsv41_sleep2_single.sh 的差异（其余逐字继承 env_dsv41.sh + 1116 参数）：
#     1) actor 三个 offload 关：offload_policy=False / param_offload=False / optimizer_offload=False
#     2) optimizer_impl=verl.utils.dsv41_swap_optim + optimizer=AdamWSwapSafe（Adam 矩换到主机）
#        （见 1117_dsv41_4node_mixed_swapopt_v1.sh 头部；save_freq 必须 -1）
#     3) actor micro-batch 2 → 1、rollout log_prob 4 → 2（驻卡后激活预算变小）
#     4) MAX_RESPONSE_LEN 2048 → 512、VERL_DSV41_MAX_SEQ_LEN 3072 → 1536：
#        单机 16 卡分 250.63B 参数，bf16 参数 31.3 GB + 梯度 31.3 GB = 62.6 GB/卡，
#        cap 2048 的激活放不下；512 是历史已验证「单机 top32 能涨」的配置
#        （verl/logs/longrun500_20261101_073304.log：countbenchqa + 512，150 步 reward +0.25 → +1.00）
#     5) 显式列出 sleep 两个开关（与 1117 相同，便于本文件单独读）
#     6) +actor_rollout_ref.ref...kv_allgather_parallel_size=4 —— **必须**：verl 先建 ref
#        引擎、而 ParallelState 是进程级单例（只认第一次构造），ref 缺这个键就会把 mesh
#        按 kv=1 建出来，导致 CP 静默关闭（见 C0 脚本头部的完整推导）
#
# 前置：FSDPTurbo model.py 已应用 apply_cpdebug_patch_v1.py --mode fix（CP=4 gather 反传修复）
#
# 结果：见 logs/1118_resident_single_long.log
# 记录：mjt_workplace/task/1118_fix_reward_not_increase/
# 代码：FSDPTurbo 201f413（CP gather 修复：gather_forward_split_backward + grad_scale="up"）
# =============================================================================
set -eo pipefail
F=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl/my_scripts/final

export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-15}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-DSV41-1node-resident-cb512}
# sleep mode=2
export MY_ENV_DSV41_SLEEP_LEVEL=${MY_ENV_DSV41_SLEEP_LEVEL:-2}
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
# swapopt 旋钮（与 1117_dsv41_4node_mixed_swapopt_v1.sh 默认值一致）
export MY_ENV_DSV41_SWAP_MEM_FRACTION=${MY_ENV_DSV41_SWAP_MEM_FRACTION:-0.25}
export MY_ENV_DSV41_SWAP_PARAM_PREFETCH=${MY_ENV_DSV41_SWAP_PARAM_PREFETCH:-0}
export MY_ENV_DSV41_SWAP_HOST_PINNED=${MY_ENV_DSV41_SWAP_HOST_PINNED:-0}

source "$F/env_dsv41.sh"

export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=512
export VERL_DSV41_MAX_SEQ_LEN=1536
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  trainer.save_freq=-1 \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.dsv41_swap_optim \
  actor_rollout_ref.actor.optim.optimizer=AdamWSwapSafe \
  +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.kv_allgather_parallel_size=4
