#!/usr/bin/env bash
# =============================================================================
# 1118_1117_dsv41_sleep2_single_resident_cp1_15step_v1.sh
#   性能/建议验证：与 .18 的驻卡长跑（CP=4 + 修复）**逐项相同**，
#   唯一差异是 **CP_SIZE=1**（关掉 KV-AllGather CP 及其集合通信）。
#
#   为什么：单机 16 卡分 250.63B 参数，参数/梯度/激活本来就放得下，CP=4 只是
#   把每个 rank 的序列切 1/4，但每次 attention 都要做 KV all-gather；
#   实测 C1（CP=4）update_actor 471 s vs A（CP 关闭）178 s，短序列下 CP 开销占主导。
#   本跑给出同数据/同 cap/同驻卡布局下 CP=1 的 step 时间与 reward，用于定最终长跑配置。
#
#   改动（相对 resident 脚本）：
#     1) source env_dsv41.sh 之后 export CP_SIZE=1（env 里是硬编码 4）
#     2) actor/ref 的 kv_allgather_parallel_size 都覆盖成 1
#     3) STEPS=15
#   注：CP_SIZE=1 时 `_kv_cp_group=None`，gather 分支不会执行（与历史单机行为一致）。
#
# 结果：见 logs/1118_resident_single_cp1_15step.log
# 代码：FSDPTurbo 201f413（CP gather 修复：gather_forward_split_backward + grad_scale="up"）
# =============================================================================
set -eo pipefail
F=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl/my_scripts/final

export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-15}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-DSV41-1node-resident-cb512-cp1}
export MY_ENV_DSV41_SLEEP_LEVEL=${MY_ENV_DSV41_SLEEP_LEVEL:-2}
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
export MY_ENV_DSV41_SWAP_MEM_FRACTION=${MY_ENV_DSV41_SWAP_MEM_FRACTION:-0.25}
export MY_ENV_DSV41_SWAP_PARAM_PREFETCH=${MY_ENV_DSV41_SWAP_PARAM_PREFETCH:-0}
export MY_ENV_DSV41_SWAP_HOST_PINNED=${MY_ENV_DSV41_SWAP_HOST_PINNED:-0}

source "$F/env_dsv41.sh"
export CP_SIZE=1   # 必须放在 source 之后

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
  actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.kv_allgather_parallel_size=1 \
  +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.kv_allgather_parallel_size=1
