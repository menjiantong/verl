#!/usr/bin/env bash
# [验证结果] ❌ 不可用（2026-11-05 静态核算，未经实跑）：本模型 384 专家、bf16 权重
#   1.388 TiB，FSDP16 时 88.9 GiB/rank > 单卡 65.5 GB，单机 16 卡 HBM 合计 1.02 TiB
#   < 权重 1.39 TiB；训练状态 8 机实测 8.6 TB，单节点内存仅 2.01 TB。
#   **该模型无法在单机跑训练**。单机 E1 请用 1107_bench_dsv41_1node_engram_fetch_v1.sh
#   （同一份 _TorchHostFetch 代码的 EP16 微基准）。本文件保留仅作记录。
# =============================================================================
# 1107 DSV4.1 云道 **单机**（1 机 × 16 卡）：Engram fetch 分段探针（E1-单机）
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1107_run_dsv41_yundao_8node_engram_probe_v1.sh
#     → 1105_..._cp1_fadvise_v3.sh → … → 1103_..._engram_cp_v1.sh
#       → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# -----------------------------------------------------------------------------
# 一、E1-单机的唯一目的：把"跨机 all_to_all"这个变量摘出去
# -----------------------------------------------------------------------------
#   8 机实测：同拓扑、同批量，engram 版 update_actor 每 token 7.21 ms，
#   noengram 版 0.248 ms（29×）。
#   单机（16 rank）跑同样的 engram，如果：
#     A) 每 token 与 8 机的 7.21 ms 同量级 → 慢的主因**不是跨机通信**，
#        而是每 rank 都要做、且随 rank 数增长的主机侧工作（CPU gather / D2H / 表带宽）；
#     B) 每 token 掉到 1 ms 以下 → 主因就是 EP128 的跨机 all_to_all。
#   分段探针把答案直接给到段级：单机没有跨机 a2a 段，对照 8 机的
#   t_a2a_cnt_ms / t_a2a_ids_ms / t_a2a_val_ms 即可。
#
# -----------------------------------------------------------------------------
# 二、与 8 机探针脚本的唯一差异（拓扑 5 个 env + 探针 SYNC）
# -----------------------------------------------------------------------------
#   MY_ENV_DSV41_NNODES=1
#   MY_ENV_DSV41_FSDP_SIZE=16     1 机 16 卡 → FSDP16
#   MY_ENV_DSV41_EP_SIZE=16       384 专家 / EP16 = 24 专家/rank（8 机是 3 专家/rank）
#   MY_ENV_DSV41_GEN_DP=2         rollout TP8 × DP2 = 16 卡
#   MY_ENV_DSV41_GEN_EP=16
#   TRAIN_BSZ=16 保持与 8 机一致（每步喂进训练的 token 数不变，便于 per-token 对比）
#   MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=1
#     单机是我们自己验证、不与其他实验抢机，用同步版拿**最准**的分段时间。
#
#   ⚠️ 内存预期：单机时 vLLM 的 engram 共享表仍是 393 GiB（GEN_DP=2 → 节点内
#   `_engram_dp_shard_size=gcd(2,2)=2`，2 个 replica 共享一份 8×49 GiB 的 TP 分片表），
#   训练侧 engram 表按 EP16 按行切 → 393 GiB/16 = 24.6 GiB/rank × 16 = 393 GiB
#   （8 机时是 EP128 → 46 GiB/节点，单机会**更大**）。节点 2.01 TB，
#   MemAvailable 门槛沿用 1103 默认 1600 GiB；若预检 ABORT，说明该机空闲不足，
#   先清缓存再跑，不要改门槛。
#
# 用法（云道"启动命令"只填这一行，**1 机作业**）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1107_run_dsv41_yundao_1node_engram_probe_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_probe_1node_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 回滚：8 机跑 1107_..._8node_engram_probe_v1.sh，或 1105_..._cp1_fadvise_v3.sh。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh"

if [ ! -f "$BASE" ]; then
  echo "[probe1] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# ---- 1) 拓扑：1 机 16 卡（基线脚本第 69-77 行已改为 ${MY_ENV_...:-默认}）----
export MY_ENV_DSV41_NNODES=1
export MY_ENV_DSV41_NDEVICES_PER_NODE=16
export MY_ENV_DSV41_FSDP_SIZE=16
export MY_ENV_DSV41_EP_SIZE=16
export MY_ENV_DSV41_GEN_TP=8
export MY_ENV_DSV41_GEN_DP=2
export MY_ENV_DSV41_GEN_EP=16
# 每步训练 token 数与 8 机一致，便于 per-token 归一化对比
export MY_ENV_DSV41_TRAIN_BSZ=16

# ---- 2) Engram fetch 分段探针（单机用同步版，时间最准）----
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=1
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=0

# ---- 3) 标签 ----
export MY_ENV_DSV41_EXPERIMENT_TAG="probe1-cp1"
export MY_ENV_DSV41_RUN_TAG="probe1-cp1-math-dapo17k"

echo "[probe1] MY_DEBUG_LOG E1-单机 分段探针启动 nodes=1 fsdp=16 ep=16 gen_tp=8 gen_dp=2 gen_ep=16 bsz=16 probe=1 sync=1 base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
