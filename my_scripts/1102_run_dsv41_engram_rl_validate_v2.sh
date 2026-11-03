#!/usr/bin/env bash
# =============================================================================
# 1102 DSV4.1 Engram RL 验证启动器 v2（在 v1 基础上加片数与共享表开关）
#
# v1 -> v2 的演进：
#   * 增加 CHIPS：先 8 卡单 replica 闸位（engram 主机表 393+393 GB），
#     再 16 卡双 replica 闸位（需 SHARED=1，否则 2x393 GB 可能超 2.4 TB 主机内存）；
#   * 增加 SHARED：透传 VLLM_ENGRAM_DP_SHARED_MEMORY，控制 vLLM 两个 replica
#     是否共用一份 TP 分片主机表（依赖 /dev/shm，本机 1 TB 足够）。
#
# 用法：bash my_scripts/1102_run_dsv41_engram_rl_validate_v2.sh [STEPS] [CP_SIZE] [CHIPS] [SHARED]
#   STEPS  默认 1
#   CP_SIZE 默认 1（2 = Engram + CP 集成闸位）
#   CHIPS  默认 8（8 或 16）
#   SHARED 默认 0（16 卡建议 1）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-1}
CP_SIZE=${2:-1}
CHIPS=${3:-8}
SHARED=${4:-0}

# 完整检查点：config.json 的 text_config.engram_layer_ids=[1,14]，engram 表 bf16
# [384006168,256] + [384016682,256]，合计 393 GB；TP=8 时 49.2 GB/rank。
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16

# vLLM 侧 engram 主机表共享开关（run 脚本据此决定是否传 engram_config 覆盖）
export VLLM_ENGRAM_DP_SHARED_MEMORY=$SHARED

# CP 开关与 CP 验证脚本保持一致（全部在脚本内设置，rule 7.1）
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0
if [ "$CP_SIZE" -gt 1 ]; then
    export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
    export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
    export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
fi

# 片数：8 卡时 FSDP=8、rollout TP=8、单 replica；16 卡时 FSDP=16、两个 TP=8 replica
export NDEVICES_PER_NODE=$CHIPS
export FSDP_SIZE=$CHIPS
export EP_SIZE=8
export GEN_TP=8
export GEN_DP=1

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[engram-rl-v2] STEPS=$STEPS CP_SIZE=$CP_SIZE CHIPS=$CHIPS SHARED=$SHARED MODEL_PATH=$MODEL_PATH"
free -g | head -2

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1102_engram_rl_v2_cp${CP_SIZE}_chips${CHIPS}_shared${SHARED}_${STEPS}steps_${TS}.log"
echo "[engram-rl-v2] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.kv_allgather_parallel_size=${CP_SIZE}" \
    2>&1 | tee "$LOG"
