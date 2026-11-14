#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-07 E1-单机 engram fetch 微基准启动器，尚未实跑。
# =============================================================================
# 1107 DSV4.1 **单机 16 卡**：Engram fetch 微基准（E1-单机的可行形态）
#
# -----------------------------------------------------------------------------
# 一、为什么不是"单机跑完整训练"
# -----------------------------------------------------------------------------
#   硬约束（实测，不是估算）：
#     权重 bf16 总量   = 1.388 TiB（268 个 safetensors 分片求和，763 B 参数）
#     8 机 FSDP128     → 11.1 GiB/rank
#     1 机 FSDP16      → 88.9 GiB/rank    > 单卡 HBM 65.5 GB
#     单机 16 卡 HBM 合计 1.02 TiB       < 权重 1.39 TiB
#     训练状态（含主机 offload）8 机实测 cpu_memory 1080 GiB/节点 × 8 = 8.6 TB
#     单节点内存 2.01 TB
#   → **这个模型在任何单机拓扑下都跑不了训练**（最少也要 8 机）。
#   E1 要的"去掉跨机通信"这一半，只能用微基准实现：同一份 `_TorchHostFetch`
#   代码、同样的表、同样的每 rank 行数，只把 EP 组从 128 卡跨机换成 16 卡片内。
#
# -----------------------------------------------------------------------------
# 二、跑什么
# -----------------------------------------------------------------------------
#   同一脚本内串行跑两档（可用 MY_ENV_DSV41_BENCH_SCALES 只跑其中一档）：
#     scale=1  全尺寸表，EP16 → 每 rank 24.6 GiB（2 张表），节点 393 GiB
#     scale=8  表缩 8 倍，EP16 → 每 rank 3.07 GiB ≈ 8 机 EP128 的 2.9 GiB ← 对照用这档
#   每档先 warmup 3 次，再计时 20 次（fwd + bwd 各一段），
#   模块内的 `[engram-probe]` 逐段打点同时输出（前 2 次调用，避免刷屏）。
#
#   判读：把本机 scale=8 的 `t_cpu_gather_ms` / `t_d2h_list_ms` 与 8 机探针的
#   同名段对比：
#     - 两者接近 → 这两段与网络无关，慢在主机侧（表带宽 / index_select）
#     - 8 机显著更大 → 慢在跨机 all_to_all（对照 t_a2a_* 三段）
#
# -----------------------------------------------------------------------------
# 三、环境变量（全部 MY_ENV_ 前缀，都有默认值）
# -----------------------------------------------------------------------------
#   MY_ENV_DSV41_BENCH_PYTHON      python 解释器，默认 python3
#   MY_ENV_DSV41_BENCH_FSDPTURBO   FSDPTurbo 仓库根（加进 PYTHONPATH）
#   MY_ENV_DSV41_BENCH_NPROC       进程数，默认 16（1 机 16 卡）
#   MY_ENV_DSV41_BENCH_NIC         通信网卡，默认自动探测
#   MY_ENV_DSV41_BENCH_PORT        rendezvous 端口，默认 29517
#   MY_ENV_DSV41_BENCH_SCALES      要跑的规模，默认 "1 8"
#   MY_ENV_DSV41_BENCH_ROWS        每次 fetch 的 row 数/rank，默认 8192
#   MY_ENV_DSV41_BENCH_ITERS       计时轮数，默认 20
#   MY_ENV_DSV41_BENCH_WARMUP      预热轮数，默认 3
#   MY_ENV_DSV41_BENCH_SYNC        1=探针逐段同步（时间最准，默认 1）
#
# 用法（**1 机 16 卡作业/裸机**，直接跑，不经过云道 8 机调度）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1107_bench_dsv41_1node_engram_fetch_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-engram_fetch_bench_1node_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 注意：脚本只读权重目录的 config 参数（行数/维度写死在 .py 里），
#       不读 safetensors、不写任何用户目录、不删任何文件。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_BENCH="${HERE}/1107_bench_dsv41_1node_engram_fetch_v1.py"

if [ ! -f "$PY_BENCH" ]; then
  echo "[bench] ABORT: python bench not found: $PY_BENCH" >&2
  exit 3
fi

MY_ENV_DSV41_BENCH_PYTHON=${MY_ENV_DSV41_BENCH_PYTHON:-python3}
MY_ENV_DSV41_BENCH_FSDPTURBO=${MY_ENV_DSV41_BENCH_FSDPTURBO:-/efs_rl/m00899630/workspace/dsv41-verl/FSDPTurbo}
MY_ENV_DSV41_BENCH_NPROC=${MY_ENV_DSV41_BENCH_NPROC:-16}
MY_ENV_DSV41_BENCH_PORT=${MY_ENV_DSV41_BENCH_PORT:-29517}
MY_ENV_DSV41_BENCH_SCALES=${MY_ENV_DSV41_BENCH_SCALES:-"1 8"}
MY_ENV_DSV41_BENCH_ROWS=${MY_ENV_DSV41_BENCH_ROWS:-8192}
MY_ENV_DSV41_BENCH_ITERS=${MY_ENV_DSV41_BENCH_ITERS:-20}
MY_ENV_DSV41_BENCH_WARMUP=${MY_ENV_DSV41_BENCH_WARMUP:-3}
MY_ENV_DSV41_BENCH_SYNC=${MY_ENV_DSV41_BENCH_SYNC:-1}

# CANN 环境（裸机上必须先 source；已 source 过的容器里重复 source 无副作用）
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
  set -u
fi

# 网卡：通信走业务网卡，取默认路由那张
if [ -z "${MY_ENV_DSV41_BENCH_NIC:-}" ]; then
  MY_ENV_DSV41_BENCH_NIC=$(ip -o -4 route show default 2>/dev/null | awk '{print $5; exit}')
  [ -z "$MY_ENV_DSV41_BENCH_NIC" ] && MY_ENV_DSV41_BENCH_NIC=eth0
fi
export HCCL_SOCKET_IFNAME="$MY_ENV_DSV41_BENCH_NIC"
export GLOO_SOCKET_IFNAME="$MY_ENV_DSV41_BENCH_NIC"

# fsdp_turbo 可导入
export PYTHONPATH="${MY_ENV_DSV41_BENCH_FSDPTURBO}${PYTHONPATH:+:${PYTHONPATH}}"

# 探针：逐段打点，只打前 2 次调用（每个 rank 每类 2 行，16 rank ≈ 128 行/档）
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC="$MY_ENV_DSV41_BENCH_SYNC"
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=2

MEM_GIB=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
NPU_CNT=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l)

echo "[bench] MY_DEBUG_LOG start host=$(hostname 2>/dev/null || echo unknown) mem=${MEM_GIB}GiB npu_dev=${NPU_CNT} nproc=${MY_ENV_DSV41_BENCH_NPROC} nic=${MY_ENV_DSV41_BENCH_NIC} scales='${MY_ENV_DSV41_BENCH_SCALES}' rows=${MY_ENV_DSV41_BENCH_ROWS} iters=${MY_ENV_DSV41_BENCH_ITERS} time=$(date '+%Y-%m-%d %H:%M:%S')"

# 规模 1 每节点要 393 GiB host 内存（2 张全尺寸表的 EP16 分片），先算预算
if [[ " ${MY_ENV_DSV41_BENCH_SCALES} " == *" 1 "* ]]; then
  NEED_GIB=450
  if [ "$MEM_GIB" -lt "$NEED_GIB" ]; then
    echo "[bench] ABORT: scale=1 需要 ≥${NEED_GIB}GiB 主机内存（每节点 engram 分片 393 GiB），本机 ${MEM_GIB}GiB" >&2
    exit 3
  fi
fi

RC=0
for SCALE in $MY_ENV_DSV41_BENCH_SCALES; do
  echo "[bench] MY_DEBUG_LOG ===== scale=${SCALE} begin $(date '+%Y-%m-%d %H:%M:%S') ====="
  set +e
  "$MY_ENV_DSV41_BENCH_PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$MY_ENV_DSV41_BENCH_NPROC" \
    --master_port="$MY_ENV_DSV41_BENCH_PORT" \
    "$PY_BENCH" \
    --scale "$SCALE" \
    --rows-per-rank "$MY_ENV_DSV41_BENCH_ROWS" \
    --iters "$MY_ENV_DSV41_BENCH_ITERS" \
    --warmup "$MY_ENV_DSV41_BENCH_WARMUP"
  RCSCALE=$?
  set -e
  echo "[bench] MY_DEBUG_LOG ===== scale=${SCALE} end rc=${RCSCALE} $(date '+%Y-%m-%d %H:%M:%S') ====="
  [ "$RCSCALE" -ne 0 ] && RC=$RCSCALE
  # 两档之间留 20 s 让 HCCL / 内存回收干净，避免下一档被上一档的峰值影响
  sleep 20
done

echo "[bench] MY_DEBUG_LOG done rc=${RC} $(date '+%Y-%m-%d %H:%M:%S')"
exit "$RC"
