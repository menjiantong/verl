#!/usr/bin/env bash
# =============================================================================
# 2026-10-25 全量 566B 去 engram · 16 机 256 卡 GRPO
#
# 用法（16 台各 16 卡，容器内执行）：
#   头节点：bash 1025_W3_longrun_full_noengram_16node_v1.sh master [STEPS]
#   其余 15 台：bash 1025_W3_longrun_full_noengram_16node_v1.sh join <MASTER_IP>
#
# 拓扑：
#   训练 FSDP=16*16=256；EP=128 × EFSDP=1（edp=2，专家组复制 2 份）
#   rollout TP=8 × DP=16 = 128 卡/replica（8 机），256/128 = 2 个 replica
#   引擎每卡 ≈11.4GiB；EFSDP=1 且 GEN_EP==EP_SIZE=128 → LOCAL_EXPERT_EXPORT=1
#   训练 batch 32×8=256 条 = 每卡 1 条
#
# 关键结论：每卡/每节点内存与 8 机完全相同。专家参数只在 EP*EFSDP=128 内分片，
# 多出的节点成为 edp/rollout 副本，不会降低单卡权重；16 机只增加吞吐/副本数。
# 显存账见 wiki/worklog_20261025_full_noengram_8_16node_fit.md。
#
# ⚠️ 80.5.25.117-131 只有 15 个地址（且当前仅 5 台可达），16 机需要额外 1 台；
#    按用户约束不得使用 80.5.21 节点，因此 16 机暂不具备条件。
# =============================================================================
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/full_noengram_common_20261025.sh"

export NNODES=16 CPN=16
export MY_ENV_EP_SIZE=128         # 384 专家 / 128 = 3 专家/EP rank；256/128 = 2 个 edp 副本
export MY_ENV_EFSDP_SIZE=1        # 专家只在 EP 维分片（4.35B/rank），并启用本地专家导出
export MY_ENV_GEN_TP=8            # o_groups=8：vLLM TP 必须 ≤8
export MY_ENV_GEN_DP=16           # TP8×DP16=128 卡/replica（8 机）
export MY_ENV_GEN_EP=128          # = TP×DP；与训练 EP 对齐，触发 LOCAL_EXPERT_EXPORT
export TRAIN_BSZ=32 SAMPLE_N=8    # 256 条 = 256 卡
export EXPERIMENT_NAME=GRPO-DSV41-Full-NoEngram-16node-$(date +%m%d-%H%M)

MODE=${1:?usage: $0 master [STEPS] | $0 join <MASTER_IP>}
shift
case "$MODE" in
  master) bash "$HERE/dsv41_grpo_2node_new.sh" master "${1:-500}" ;;
  join)   bash "$HERE/dsv41_grpo_2node_new.sh" join "${1:?join 需要 master IP}" ;;
  *)      echo "unknown mode: $MODE (use master|join)"; exit 1 ;;
esac
