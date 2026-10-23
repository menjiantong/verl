#!/usr/bin/env bash
# W3 long run, based on dsv41_grpo_2node_new.sh single. No perf optimizations.
# Usage: bash my_scripts/_launch_W3_longrun_1023_change_score_and_len.sh [STEPS]
set -eo pipefail

cd /mnt/share/m00899630/workspace/dsv41-verl/verl
BASE=my_scripts/dsv41_grpo_2node_new.sh
[ -f "$BASE" ] || { echo "[W3] missing $BASE"; exit 1; }
grep -Fq '${TRAIN_EXTRA_ARGS:-}' "$BASE" || { echo "[W3] $BASE missing TRAIN_EXTRA_ARGS"; exit 1; }

STEPS=${1:-500}
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/W3_longrun_${TS}.outer.log
RUN_LOG="${RUN_LOG:-logs/W3_long_run_countbenchqa2_2048_$(date +%Y%m%d_%H%M%S).log}"


# [2026-10-23 W3] 本次长跑在 120 上启动（119 仍被 W2 遗留 vLLM 进程占用，避开）。
export MY_IP=80.5.25.120
export NIC=enp48s3u1u1

export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MAX_PROMPT_LEN=1152
# [2026-10-23 W3] 按用户要求 response length 512 -> 2048；
# max_model_len = max_prompt 1152 + max_response 2048 = 3200。
# 新增可调项统一 MY_ENV_ 前缀，再映射到基座脚本读取的 MAX_RESPONSE_LEN。
export MY_ENV_MAX_RESPONSE_LEN=2048
export MAX_RESPONSE_LEN=${MY_ENV_MAX_RESPONSE_LEN}
# [2026-10-23 W3] 改用 countbenchqa2 打分器：严格解析 </think> 后的 response 通道，
# 只认末尾 boxed 整数格式。通过 verl custom_reward_function 注入，不改 __init__.py
# 的全局 data_source 路由（避免影响其他 countbenchqa 实验）。路径相对训练 cwd
# ($MIRROR_ROOT/verl)，即 verl/utils/reward_score/countbenchqa2.py。
export MY_ENV_REWARD_FN_PATH=verl/utils/reward_score/countbenchqa2.py
export USE_KL=0
export ROLLOUT_GPU_MEM_UTIL=0.6
export VERL_DSV41_VISION=1   # freeze defaults to 1 in fsdp_turbo_dsv41_impl.py
export EXPERIMENT_NAME=GRPO-DSV41-countbenchqa2-W3-longrun-${TS}

# Hydra extras: epoch cap, VLM load, agent/thinking, W3 sampler
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True \
reward.custom_reward_function.path=${MY_ENV_REWARD_FN_PATH} \
reward.custom_reward_function.name=compute_score"

# No performance optimizations
unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP

mkdir -p logs
echo "[W3] ip=$MY_IP nic=$NIC steps=$STEPS"
echo "[W3] log=$LOG inner=$RUN_LOG"
# Foreground: keeps the process attached to this terminal and shows output.
# For background: nohup bash my_scripts/_launch_W3_longrun_1023_change_score_and_len.sh > /tmp/W3.out 2>&1 &
bash "$BASE" single "$STEPS" 2>&1 | tee "$LOG"
