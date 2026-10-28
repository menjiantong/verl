#!/usr/bin/env bash
# DeepSeek-V4.1-Flash top32 · gsm8k-sft-3ep 新权重（无 engram）· 单机 16 卡 GRPO
#
#   MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1 \
#   AB_RUN_TAG=ne-epoch1 bash my_scripts/dsv41_grpo_1node_noengram.sh 20
#
# 说明：
# - 新权重 config 的 engram_layer_ids=[]（SFT 时已去 engram）→ 引擎/训练侧自动跳过 engram，
#   无 engram 表（省 ~366GB/节点 host）与 SWA 槽缓存；不需要任何 VERL_DSV41_ENGRAM_* 开关。
# - 单机拓扑与 SFT 脚本一致（FSDP16/EP8/TP1，引擎 TP8×DP2=16 卡）。
# - USE_KL=0 / ROLLOUT_IS=token / util 0.3：与已验证配置对齐，便于与旧权重对比 reward 曲线。
set -eo pipefail
STEPS=${1:-20}
export MODEL_PATH=${MODEL_PATH:?"usage: MODEL_PATH=<epoch dir> $0 [STEPS]"}
export USE_KL=${USE_KL:-0}
export ROLLOUT_IS=${ROLLOUT_IS:-token}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
export GEN_TP=${GEN_TP:-8} GEN_DP=${GEN_DP:-2} GEN_EP=${GEN_EP:-16}
export VERL_DSV41_HC_BF16=${VERL_DSV41_HC_BF16:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-noengram-$(basename "$MODEL_PATH")}
export AB_RUN_TAG=${AB_RUN_TAG:-noengram-$(basename "$MODEL_PATH")}
# 诊断 dump 一律关闭（本任务不需要）
unset VERL_DSV41_ENGRAM_DUMP_DIR VERL_DSV41_ENGRAM_DUMP_DIR_EX 2>/dev/null || true
echo "[noengram] model=$MODEL_PATH steps=$STEPS tag=$AB_RUN_TAG util=$ROLLOUT_GPU_MEM_UTIL kl=$USE_KL is=$ROLLOUT_IS"
exec bash "$(dirname "$0")/dsv41_grpo_2node_ab.sh" single "$STEPS"
