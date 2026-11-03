#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 verl eval 验证启动器 v2
#
# v1 -> v2 演进：
#   * 增加 MODEL 选择：
#     base     = top32-bf16（原始基座 + engram，EP16/DP2/共享表）。历史日志
#                W3_longrun_20261029_171139 用同源 noengram 基座在 countbenchqa
#                上 critic/score/mean=0.703125，能真正看到 eval 分数变化；
#                sft116 权重在本数据集 reward 恒 0（512 token 全截断）。
#     engram   = top32-sft116（SFT 权重 + engram 表，验证拓扑同 1102_v3）
#     noengram = top32-bf16-noengram（基座 noengram，EP8/DP1）。历史日志
#                W3_longrun_20261029_171139 用它得到 reward 0.155(step1)->0.591(step4)，
#                且步时 440-625 s；engram 基座在本数据集贪心评测 reward=0，
#                故 eval 分数变化用该模型验证。
#   * 第 0 步评测（val_before_train=True）+ 每 TEST_FREQ 步评测；最后一步
#     无论 test_freq 都会评测（trainer_base.py:477）。
#
# 目的（任务3）：训练 STEPS 步，在 test 数据集（countbenchqa_lite_val.parquet）上
#   观察 val-core/<data_source>/reward/mean@N 的变化。
#
# 用法：bash my_scripts/1103_run_dsv41_eval_validate_v2.sh [STEPS] [TEST_FREQ] [CHIPS] [engram|noengram]
#   STEPS     默认 50
#   TEST_FREQ 默认 25（第 0/25/50 步各评测一次）
#   CHIPS     默认 16
#   MODEL     默认 base（能出非零 reward）
#
# 验证通过的 commit（2026-11-03）：
#   verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c
# 验证环境：80.5.25.125（16×Ascend910，npu-smi 26.0.rc1）
#   step0 val: val-core/countbenchqa_lite/acc/mean@1=0.04, reward=0.036
#   step1    : critic/score/mean=0.1828, response_length/mean=448.3,
#              rollout_probs_diff_mean=0.00812, pearson=0.99887, step=321.5s
#   日志 logs/1103_eval_v2_noengram_50steps_freq25_20261102_201719.log
# 注意：80.5.27.254 容器存在采样缺陷（同一引擎隔一个请求 logits 全 nan，
#   采样恒 token 0、logprob nan），eval/长跑必须放 80.5.25 节点；
#   证据见 wiki/worklog_20261103_dsv41_eval_and_254_sampling_bug.md。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-50}
TEST_FREQ=${2:-25}
CHIPS=${3:-16}
MODEL=${4:-base}

export NDEVICES_PER_NODE=$CHIPS
export FSDP_SIZE=$CHIPS
export GEN_TP=8

if [ "$MODEL" = "base" ]; then
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
    export EP_SIZE=16
    export GEN_DP=2
    export GEN_EP=16
    export VLLM_ENGRAM_DP_SHARED_MEMORY=1
elif [ "$MODEL" = "engram" ]; then
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116
    export EP_SIZE=16
    export GEN_DP=2
    export GEN_EP=16
    export VLLM_ENGRAM_DP_SHARED_MEMORY=1
elif [ "$MODEL" = "noengram" ]; then
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
    export EP_SIZE=8
    export GEN_DP=1
    export GEN_EP=8
    export VLLM_ENGRAM_DP_SHARED_MEMORY=0
else
    echo "unknown MODEL '$MODEL' (expect base|engram|noengram)" >&2
    exit 2
fi

# eval 闸位不启用 CP / 补齐（CP 由 1102 启动器单独验证）
export CP_SIZE=1
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[eval-v2] STEPS=$STEPS TEST_FREQ=$TEST_FREQ CHIPS=$CHIPS MODEL=$MODEL"
echo "[eval-v2] MODEL_PATH=$MODEL_PATH EP_SIZE=$EP_SIZE GEN_DP=$GEN_DP GEN_EP=$GEN_EP"
free -g | head -2

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1103_eval_v2_${MODEL}_${STEPS}steps_freq${TEST_FREQ}_${TS}.log"
echo "[eval-v2] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train=True \
    trainer.test_freq="$TEST_FREQ" \
    2>&1 | tee "$LOG"
