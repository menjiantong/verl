#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 verl eval 验证启动器 v1
#
# 目的（任务3）：训练 STEPS 步，在 test 数据集（countbenchqa_lite_val.parquet）
#   上按 TEST_FREQ 评测，观察 val-core/<data_source>/reward/mean 的变化。
#   第 0 步（val_before_train=True）给基线，之后每 TEST_FREQ 步评测一次；
#   最后一步（is_last_step）无论 test_freq 都会评测（trainer_base.py:477）。
#
# 用法：bash my_scripts/1103_run_dsv41_eval_validate_v1.sh [STEPS] [TEST_FREQ] [CHIPS]
#   STEPS     默认 50
#   TEST_FREQ 默认 25（第 0/25/50 步各评测一次）
#   CHIPS     默认 16（rollout TP=8 -> 2 个 replica）
#
# 模型：top32-sft116-noengram（SFT 热启动权重的 noengram 软链农场）。
#   不用 top32-bf16-noengram：原始基座 never-EOS，得分恒为 0，看不出变化。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-50}
TEST_FREQ=${2:-25}
CHIPS=${3:-16}

# SFT 权重的 noengram 变体（config.json 的 engram_layer_ids=[]，46 shards 软链）
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-noengram
export NDEVICES_PER_NODE=$CHIPS
export FSDP_SIZE=$CHIPS
export EP_SIZE=8
export GEN_TP=8
export GEN_DP=1

# eval 闸位不启用 CP / 补齐（行为与改动前一致）
export CP_SIZE=1
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[eval-validate] STEPS=$STEPS TEST_FREQ=$TEST_FREQ CHIPS=$CHIPS MODEL_PATH=$MODEL_PATH"

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1103_eval_${STEPS}steps_freq${TEST_FREQ}_chips${CHIPS}_${TS}.log"
echo "[eval-validate] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train=True \
    trainer.test_freq="$TEST_FREQ" \
    2>&1 | tee "$LOG"
