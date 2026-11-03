#!/usr/bin/env bash
# =============================================================================
# 1102 DSV4.1 CP 验证启动器 v1（基于 _launch_1030_dsv41_grpo.sh）
#
# 三个闸位（环境变量全部在本脚本内设定，不在命令行 export）：
#   g0a : CP=1, PAD=1  —— 基线，验证 .254 环境可用
#   g0b : CP=1, PAD=4  —— 补齐开启但不开 CP，验证补齐是 no-op
#   g1  : CP=2, PAD=4  —— CP 打通
#
# 用法：
#   bash my_scripts/1102_run_dsv41_cp_validate_v1.sh g0a [STEPS]
#
# 验证通过的 commit（2026-11-02，g1 CP=2 两步 RC=0）：
#   verl d7f6fa95  collect 只保留 kvallgather rank 0（修 Got duplicated keys）
#   verl aa3c1f1f  no_padding_2_padding 按真实内容末端切片（CP 补齐）
#   FSDPTurbo 8703d04  backbone gather 由 MY_ENV_DSV41_CP_GATHER_OUTPUT 控制
# 证据日志：logs/1102_cp_g1_cp2_pad4_2steps_20261102_133352.log
#          step1 257.3s / step2 205.1s，max_memory_allocated 15.98 GB，
#          cpu_memory_used 888.8/891.7 GB
#
# 注意：每次运行前强制 ray stop。环境变量只在 Ray worker 启动时继承一次，
#       复用旧集群会导致 MY_ENV_DSV41_CP_PAD_MULTIPLE 改了也不生效。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

GATE=${1:?usage: $0 <g0a|g0b|g1> [steps]}
STEPS=${2:-2}
case "$GATE" in
  g0a) CP_SIZE=1; PAD=1 ;;
  g0b) CP_SIZE=1; PAD=4 ;;
  g1)  CP_SIZE=2; PAD=4 ;;
  *)   echo "unknown gate $GATE (expect g0a|g0b|g1)" >&2; exit 2 ;;
esac
export CP_SIZE
export MY_ENV_DSV41_CP_PAD_MULTIPLE=$PAD
# CP 前向只实现训练路径：compute_log_prob 走 eval_mode 时必须把 training 标志留在 True，
# 否则 context_parallel.py:96 直接抛 "supports training mode only"（g1 实测）。
# CP=1 的闸位保持 0，保证基线行为与改动前一致。
if [ "$CP_SIZE" -gt 1 ]; then export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1; else export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0; fi
# verl 的 padded 输出路径（prepare_model_outputs 的 torch.nested.narrow/unbind）需要
# 整条序列的 logits，而 FSDPTurbo 单训练 CP loss 要求局部 logits。模型侧默认返回局部
# 切片（单训练语义），这里显式打开 gather 恢复 verl 需要的完整序列（g1-5 实测）。
# CP=1 的闸位保持 0，保证基线行为与改动前一致。
if [ "$CP_SIZE" -gt 1 ]; then export MY_ENV_DSV41_CP_GATHER_OUTPUT=1; else export MY_ENV_DSV41_CP_GATHER_OUTPUT=0; fi

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[cp-validate] gate=$GATE CP_SIZE=$CP_SIZE PAD_MULTIPLE=$PAD STEPS=$STEPS"

# 强制重建 Ray：环境变量只在 raylets 启动时继承一次
ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1102_cp_${GATE}_cp${CP_SIZE}_pad${PAD}_${STEPS}steps_${TS}.log"
echo "[cp-validate] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    2>&1 | tee "$LOG"
