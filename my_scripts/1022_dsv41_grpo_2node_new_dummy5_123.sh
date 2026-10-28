#!/usr/bin/env bash
# 1022 W2 dummy 5-step A/B on 80.5.25.123.
#
# 目的（工作准则 3/4/7）：
#   与 119 上的 auto run 使用同一 25 段、同一数据/拓扑/超参，只把
#   actor_rollout_ref.rollout.load_format 从 auto 改成 dummy，跑 5 step，
#   对比 training/rollout_probs_diff_mean 与 pearson，验证 dummy 是否影响一致性。
#
# 基于 my_scripts/_launch_W3_longrun_1022.sh（W2 配置）；只改节点、load_format、日志名。
set -eo pipefail

cd /mnt/share/m00899630/workspace/dsv41-verl/verl
BASE=my_scripts/dsv41_grpo_2node_new.sh
[ -f "$BASE" ] || { echo "[dummy5] missing $BASE"; exit 1; }
grep -Fq '${TRAIN_EXTRA_ARGS:-}' "$BASE" || { echo "[dummy5] $BASE missing TRAIN_EXTRA_ARGS"; exit 1; }

# 安全：123 上如果已有 main_ppo，拒绝启动，避免 base single 模式的 pkill 误杀
if pgrep -f '[m]ain_ppo' >/dev/null 2>&1; then
  echo "[dummy5] main_ppo already running on this node; refuse to start"
  pgrep -af '[m]ain_ppo' | head -3
  exit 1
fi

STEPS=${1:-5}
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/W2_dummy5_123_${TS}.outer.log
export RUN_LOG="$PWD/logs/W2_dummy5_123_${TS}.inner.log"

# 节点直接指定（准则 7.1：不依赖命令行 export）
export MY_IP=80.5.25.123
export NIC=enp48s3u1u1
[ -d "/sys/class/net/$NIC" ] || { echo "[dummy5] NIC $NIC not found"; exit 1; }

# 与 119 auto run 相同的 W2 配置；唯一差异是下面的 load_format=dummy
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MAX_PROMPT_LEN=1152
export MAX_RESPONSE_LEN=512
export USE_KL=0
export ROLLOUT_GPU_MEM_UTIL=0.6
export VERL_DSV41_VISION=1   # freeze 默认 1（fsdp_turbo_dsv41_impl.py）
export EXPERIMENT_NAME=GRPO-DSV41-countbenchqa-W2-dummy5-123-${TS}

# Hydra extras：与 119 相同，只把 auto 换成 dummy
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=dummy \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"

# 不加性能优化
unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP

mkdir -p logs
echo "[dummy5] ip=$MY_IP nic=$NIC steps=$STEPS"
echo "[dummy5] log=$LOG inner=$RUN_LOG"
bash "$BASE" single "$STEPS" 2>&1 | tee "$LOG"
