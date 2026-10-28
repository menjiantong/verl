#!/usr/bin/env bash
# =============================================================================
# 2026-10-25 全量 566B 去 engram · 8 机 128 卡 GRPO
#
# 用法（8 台各 16 卡，容器内执行）：
#   头节点：bash 1025_W3_longrun_full_noengram_8node_v1.sh master [STEPS]
#   其余 7 台：bash 1025_W3_longrun_full_noengram_8node_v1.sh join <MASTER_IP>
#   （有外部租户的机器用 join：只 ray start，不 pkill；master 会清理本机残留）
#
# 拓扑（推导/显存账见 wiki/worklog_20261025_full_noengram_8_16node_fit.md）：
#   训练 FSDP=8*16=128；EP=128 × EFSDP=1 → 4.35B 参数/rank（host 1.13TB/节点）
#   rollout TP=8 × DP=16 = 128 卡/replica（8 机全用），8 机 = 1 个 replica
#   引擎每卡：专家 557.17B/128 + dense 9.11B/8 ≈ 11.4GiB bf16（util 0.6 可容纳）
#   EFSDP=1 且 GEN_EP==EP_SIZE=128 → LOCAL_EXPERT_EXPORT=1，sync 只传本 rank 的 3 个专家
#   训练 batch 16×8=128 条 = 每卡 1 条
#
# 前置：8 台容器可达且 NPU 空闲；当前 80.5.25.117-131 中 118/119/120/123/128
#       可达，跑 8 机前需先把另外 3 台容器拉起。
# =============================================================================
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/full_noengram_common_20261025.sh"

export NNODES=8 CPN=16
export MY_ENV_EP_SIZE=128         # 384 专家 / 128 = 3 专家/EP rank
export MY_ENV_EFSDP_SIZE=1        # 专家只在 EP 维分片（4.35B/rank），并启用本地专家导出
export MY_ENV_GEN_TP=8            # o_groups=8：vLLM TP 必须 ≤8
export MY_ENV_GEN_DP=16           # TP8×DP16=128 卡/replica（8 机全用）
export MY_ENV_GEN_EP=128          # = TP×DP；与训练 EP 对齐，触发 LOCAL_EXPERT_EXPORT
export TRAIN_BSZ=16 SAMPLE_N=8    # 128 条 = 128 卡
export EXPERIMENT_NAME=GRPO-DSV41-Full-NoEngram-8node-$(date +%m%d-%H%M)

MODE=${1:?usage: $0 master [STEPS] | $0 join <MASTER_IP>}
shift
case "$MODE" in
  master) bash "$HERE/dsv41_grpo_2node_new.sh" master "${1:-500}" ;;
  join)   bash "$HERE/dsv41_grpo_2node_new.sh" join "${1:?join 需要 master IP}" ;;
  *)      echo "unknown mode: $MODE (use master|join)"; exit 1 ;;
esac
