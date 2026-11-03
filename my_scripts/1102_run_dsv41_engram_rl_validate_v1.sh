#!/usr/bin/env bash
# =============================================================================
# 1102 DSV4.1 Engram RL 验证启动器 v1
#
# 目的：用带 Engram 层的完整检查点（top32-bf16，40 层，engram_layer_ids=[1,14]）
#       跑通 RL 一步，验证 vLLM 侧 Engram cpu_offload（主机 UVA 表）与
#       FSDPTurbo 训练侧 host_offload Engram 表能同时工作。
#
# 用法：
#   bash my_scripts/1102_run_dsv41_engram_rl_validate_v1.sh [STEPS] [CP_SIZE]
#     STEPS   默认 1
#     CP_SIZE 默认 1；设为 2 即“Engram + CP”集成闸位
#
# 关键环境变量（全部在本脚本内设置，rule 7.1）：
#   MODEL_PATH      指向带 Engram 的完整检查点（494 GB，含 bf16 engram 表）
#   VLLM_ENGRAM_DP_SHARED_MEMORY=0  两个 vLLM replica 各自持有一份 TP 分片表；
#                   开启共享需要 /dev/shm + IPC，先用默认形态验证主链路
#   MY_ENV_DSV41_CP_*  仅在 CP_SIZE>1 时设置，与 CP 单训练验证同一套开关
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-1}
CP_SIZE=${2:-1}

# 完整检查点：config.json 的 text_config.engram_layer_ids=[1,14]
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
# 0 = 每个 replica 一份 TP 分片的主机表（cpu_offload 默认开启，走 NPU aclrtHostRegister 路径）
export VLLM_ENGRAM_DP_SHARED_MEMORY=0

# CP 开关与 CP 验证脚本保持一致：CP>1 时需要补齐到 4 的倍数、保留 training 标志、
# 并让模型把局部输出拼回整条序列（verl 的 padded 输出路径依赖完整序列）。
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0
if [ "$CP_SIZE" -gt 1 ]; then
    export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
    export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
    export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
fi

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[engram-rl] STEPS=$STEPS CP_SIZE=$CP_SIZE MODEL_PATH=$MODEL_PATH"

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1102_engram_rl_cp${CP_SIZE}_${STEPS}steps_${TS}.log"
echo "[engram-rl] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.kv_allgather_parallel_size=${CP_SIZE}" \
    2>&1 | tee "$LOG"
