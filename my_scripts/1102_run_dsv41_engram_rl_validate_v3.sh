#!/usr/bin/env bash
# =============================================================================
# 1102 DSV4.1 Engram RL 验证启动器 v3
#
# v2 -> v3 演进：
#   * 增加 MODEL 选择（sft|base）。v2 固定 top32-bf16 原始基座，never-EOS，
#     reward 恒 0，只能验证管线；v3 默认 top32-sft116 —— 它的 config 软链到
#     基座（engram_layer_ids=[1,14]），layers.{1,14}.engram.embed.weight 两个
#     大表 shard 也软链到基座，所以既有 SFT 收敛能力又带 engram 表。
#   * 16 卡改 GEN_DP=2：vLLM 单逻辑引擎 TP8×DP2=16 卡。
#     只有 DP2 才能触发 dp_shared_memory，让两个 replica 共用一份 TP 分片表
#     （host 366 GB）。否则(DP1 两个独立 replica)是 733 GB，加上训练侧
#     EP8×16rank 的 733 GB 与 CPU offload ≈890 GB，总量 ≈2.36 TB 会 OOM。
#     随之 GEN_EP 必须 = TP*DP = 16（verl config/rollout.py:295 断言）。
#   * 启动前做主机内存预检（rule 3.2），不足直接退出，避免 OOM 杀容器。
#
# 用法：bash my_scripts/1102_run_dsv41_engram_rl_validate_v3.sh [STEPS] [CP_SIZE] [CHIPS] [SHARED] [MODEL]
#   STEPS   默认 1
#   CP_SIZE 默认 1（2 = Engram + CP 集成闸位）
#   CHIPS   默认 16（8 或 16）
#   SHARED  默认 1（16 卡必须 1，否则两份表超内存）
#   MODEL   默认 sft（sft=top32-sft116 / base=top32-bf16）
#
# 验证通过的 commit（2026-11-02，step1 全链路 305.6s，日志
# logs/1102_engram_rl_v3_cp1_chips16_shared1_sft_1steps_20261102_172803.log）：
#   vllm-ascend 69de6ca8c  hash kernel 负地址钳 0（engram 推理不再 vector core exception）
#   verl 4844ed7b          本脚本提交前的基线（训练侧 host-offload engram 已在该基线内）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-1}
CP_SIZE=${2:-1}
CHIPS=${3:-16}
SHARED=${4:-1}
MODEL=${5:-sft}

case "$MODEL" in
  sft)  MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116 ;;
  base) MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16 ;;
  *)    echo "unknown MODEL '$MODEL' (expect sft|base)" >&2; exit 2 ;;
esac
export MODEL_PATH

# 16 卡必须共享 vLLM 侧表：两份 733 GB 放不下（见头部说明）。
if [ "$CHIPS" = "16" ] && [ "$SHARED" != "1" ]; then
  echo "[engram-rl-v3] CHIPS=16 requires SHARED=1 (two unshared 366 GB tables would OOM)" >&2
  exit 2
fi
export VLLM_ENGRAM_DP_SHARED_MEMORY=$SHARED

# [1103 Engram RL] EP_SIZE=16（16 卡时）：
#   1) 训练侧 engram 表按 EP 分片，EP8->EP16 让每 rank 表从 45.8 GiB 降到 22.9 GiB，
#      16 rank 合计 733 GB -> 366 GB，主机映射从 ~1.6 TB 降到 ~1.2 TB；
#      sleep 的 rtMallocHost/unmap 卡死与主机映射规模相关（见 worklog §2.2/2.3）。
#   2) GEN_EP(=TP8*DP2=16) 与 EP_SIZE 相等，run 脚本不会关掉 LOCAL_EXPERT_EXPORT，
#      权重同步走本地专家切片而不是每步 all-gather ~92 GB。
# 注意：EP_SIZE 必须与 GEN_EP 相等，否则本地专家导出会被自动禁用。
EP_SIZE=8
[ "$CHIPS" = "16" ] && EP_SIZE=16

# CP 开关与 1102_run_dsv41_cp_validate_v1.sh 保持一致（rule 7.1：全部写在脚本内）
export CP_SIZE  # run 脚本 284 行已追加 +kv_allgather_parallel_size=${CP_SIZE}，这里不能再重复
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0
if [ "$CP_SIZE" -gt 1 ]; then
  export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
  export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
  export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
fi

# 并行拓扑：8 卡 = TP8/DP1/EP8；16 卡 = TP8/DP2/EP16（见头部说明）
export NDEVICES_PER_NODE=$CHIPS
export FSDP_SIZE=$CHIPS
export EP_SIZE=$EP_SIZE
export GEN_TP=8
export GEN_DP=$((CHIPS / 8))
export GEN_EP=$((GEN_TP * GEN_DP))

# 主机内存预检（rule 3.2）：engram 表 + CPU offload 估算 ≈ 2.0 TB
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
echo "[engram-rl-v3] STEPS=$STEPS CP_SIZE=$CP_SIZE CHIPS=$CHIPS SHARED=$SHARED MODEL=$MODEL"
echo "[engram-rl-v3] MODEL_PATH=$MODEL_PATH GEN_TP=$GEN_TP GEN_DP=$GEN_DP GEN_EP=$GEN_EP FSDP=$FSDP_SIZE"
echo "[engram-rl-v3] MemAvailable=${AVAIL_GB} GiB"
if [ "$AVAIL_GB" -lt 2000 ]; then
  echo "[engram-rl-v3] ABORT: need >=2000 GiB available for engram RL" >&2
  exit 3
fi

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

# 环境变量只在 raylets 启动时继承一次：每次强制重建 Ray
ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1102_engram_rl_v3_cp${CP_SIZE}_chips${CHIPS}_shared${SHARED}_${MODEL}_${STEPS}steps_${TS}.log"
echo "[engram-rl-v3] log=$LOG"

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    2>&1 | tee "$LOG"
