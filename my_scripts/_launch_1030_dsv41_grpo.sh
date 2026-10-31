#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1 GRPO 长跑启动入口（128 单机 16 卡）
#   用法: bash my_scripts/_launch_1030_dsv41_grpo.sh [STEPS]   # 默认 500 步
#
# 分工：
#   环境变量 + Ray 启停 -> my_scripts/1030_ray_dsv41_env.sh
#   训练配置           -> examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
#   本脚本只负责：加载环境、启动 Ray、指定步数并启动训练。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

# 环境变量与 Ray 启动统一在 ray 脚本中，避免控制台 export 散落。
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh
dsv41_ray_start

STEPS=${1:-500}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/dsv41_grpo_longrun_${STEPS}steps_${TS}.log"
echo "[launch] steps=$STEPS ip=$MY_IP nic=$NIC log=$LOG"
echo "[launch] train config: examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    2>&1 | tee "$LOG"
