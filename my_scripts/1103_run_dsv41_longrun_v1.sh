#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 全功能长跑启动器 v1
#
# 前置（用户要求“所有任务完成后启动长跑”）：
#   * 任务1 CP：单训练数值等价已验证；RL 侧 B4/B6 修复已入库。
#   * 任务2 Engram：EP16 分片/DP2 共享表/sleep/权重同步机制已验证。
#   * 任务3 eval：verl eval 管线在 80.5.25.125 上产出非零分数并随训练上升。
#   * 全功能集成闸位：1103_run_dsv41_engram_rl_validate_v4.sh 2 2 16 1 sft
#     在 80.5.25 节点上跑通（日志见 worklog）。
#
# 拓扑（16 卡单机）：
#   FSDP16 / EP16 / TP1，rollout TP8×DP2（GEN_EP=16），CP=2（KV-AllGather），
#   Engram 表 EP16 分片 22.9 GiB/rank，vLLM DP2 共享表 366 GB。
#
# 训练/评测：
#   STEPS=200，TEST_FREQ=50（step0 评测 + 每 50 步；最后一步必评）。
#   checkpoint：模型含 engram 表约 494 GB/次（engram 366 + dense 103 + buffer），
#   而 /mnt/share 仅剩 ~425 GB，且 rule 9 限定只写 /mnt/share/m00899630，
#   故空间不足时自动把 SAVE_FREQ 降为 -1（不存 checkpoint）；可用空间
#   ≥550 GB 时按 SAVE_FREQ 保存到 /mnt/share/m00899630/dsv41/ckpts/longrun_*。
#   [1103 修正] 内容选择键是 actor_rollout_ref.actor.checkpoint.save_contents；
#   误传 trainer.checkpoint.save_contents 会在 Hydra struct 校验阶段直接退出
#   （日志 1103_longrun_launch_engram_cp2_200steps_20261103_010921.log）。
#
# 用法：bash my_scripts/1103_run_dsv41_longrun_v1.sh [STEPS] [TEST_FREQ] [SAVE_FREQ] [MODEL] [CP_SIZE]
#   MODEL  默认 engram（top32-sft116）；base（top32-bf16）；noengram
#   CP_SIZE 默认 2（1 = 关 CP）
#
# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c
# 修复 commit（CP response_from_nested 切片，v1 trainer）：verl 8c1518c8
#   基线环境：80.5.25.125（transformers 5.21.0，16×Ascend910）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-200}
TEST_FREQ=${2:-50}
SAVE_FREQ=${3:-100}
MODEL=${4:-engram}
CP_SIZE=${5:-2}

# [1103] sleep 备份 weights 用 pageable 主机内存：121 上 rtMallocHost 在 engram
# 共享表 + FSDP offload 常驻后失败（207001）。pageable 绕过驱动 pinned 分配器；
# 需要回退 pinned 时显式传 1。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}

# [1103] transformers 5.21 叠加：127 等节点原生 5.10.4 缺 qwen4_exp，
# FSDPTurbo 导入链需要；125 原生 5.21，叠加同版本无行为差异。
export MY_ENV_DSV41_TF_OVERLAY=/mnt/share/m00899630/env/py521
if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
    export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
fi

export NDEVICES_PER_NODE=16
export FSDP_SIZE=16
export GEN_TP=8

case "$MODEL" in
  engram)
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116
    export EP_SIZE=16; export GEN_DP=2; export GEN_EP=16
    export VLLM_ENGRAM_DP_SHARED_MEMORY=1 ;;
  base)
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
    export EP_SIZE=16; export GEN_DP=2; export GEN_EP=16
    export VLLM_ENGRAM_DP_SHARED_MEMORY=1 ;;
  noengram)
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
    export EP_SIZE=8; export GEN_DP=1; export GEN_EP=8
    export VLLM_ENGRAM_DP_SHARED_MEMORY=0 ;;
  *) echo "unknown MODEL '$MODEL' (expect engram|base|noengram)" >&2; exit 2 ;;
esac

# CP 开关（与 1102_run_dsv41_engram_rl_validate_v3.sh 一致）
export CP_SIZE
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0
if [ "$CP_SIZE" -gt 1 ]; then
    export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
    export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
    export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
fi

# engram 主机内存预检（rule 3.2）：表 366 GB + vLLM 366 GB + offload ~890 GB
if [ "$MODEL" != "noengram" ]; then
    AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    echo "[longrun-v1] MemAvailable=${AVAIL_GB} GiB"
    if [ "$AVAIL_GB" -lt 2000 ]; then
        echo "[longrun-v1] ABORT: need >=2000 GiB available" >&2; exit 3
    fi
fi

# checkpoint 落工作目录内（rule 9：只写 /mnt/share/m00899630）。
# 模型含 engram 表 ≈494 GB/次，空间不足时自动停存，避免训练中途 ENOSPC。
export MY_ENV_DSV41_LONGRUN_CKPT=/mnt/share/m00899630/dsv41/ckpts/longrun_${MODEL}_cp${CP_SIZE}
mkdir -p "$MY_ENV_DSV41_LONGRUN_CKPT"
FREE_GB=$(df -Pk "$MY_ENV_DSV41_LONGRUN_CKPT" | awk 'NR==2{print int($4/1024/1024)}')
if [ "$SAVE_FREQ" -ge 0 ] && [ "$FREE_GB" -lt 550 ]; then
    echo "[longrun-v1] WARN: free=${FREE_GB}GiB < 550GiB (ckpt≈494GiB); SAVE_FREQ -> -1" >&2
    SAVE_FREQ=-1
fi

# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

echo "[longrun-v1] STEPS=$STEPS TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ MODEL=$MODEL CP=$CP_SIZE"
echo "[longrun-v1] MODEL_PATH=$MODEL_PATH EP=$EP_SIZE GEN_DP=$GEN_DP GEN_EP=$GEN_EP"
echo "[longrun-v1] CKPT=$MY_ENV_DSV41_LONGRUN_CKPT"
free -g | head -2

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1103_longrun_${MODEL}_cp${CP_SIZE}_${STEPS}steps_${TS}.log"
echo "[longrun-v1] log=$LOG"

# checkpoint 内容选择只在实际保存时传入：v1 配置里没有 trainer.checkpoint 节点
# （传入会在 Hydra struct 校验阶段报 "Key 'checkpoint' is not in struct"）。
# 正确路径是 actor_rollout_ref.actor.checkpoint（actor.yaml:135）。
CKPT_ARGS=()
if [ "$SAVE_FREQ" -ge 0 ]; then
    CKPT_ARGS=(actor_rollout_ref.actor.checkpoint.save_contents=[model])
fi

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train=True \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$MY_ENV_DSV41_LONGRUN_CKPT" \
    "${CKPT_ARGS[@]}" \
    2>&1 | tee "$LOG"
