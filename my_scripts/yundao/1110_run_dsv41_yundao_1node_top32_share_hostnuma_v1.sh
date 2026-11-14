#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-10 由用户提交（云道控制台：资源规格 1 机 × 16 卡）
# =============================================================================
# 1110 单机 top32 + P0 共用表：主机内存开关 A/B（THP / NUMA 交错）
#
# 基线（规则 7.3/7.4：本脚本只写差异，训练数学与拓扑一律不动）：
#   my_scripts/yundao/1109_run_dsv41_yundao_1node_top32_share_v1.sh
#
# 依据：mjt_workplace/worklog/worklog_20261109_dsv41_host_slowpath_probe.md
#   1) 启动器里的 VERL_DSV41_DISABLE_THP / VERL_DSV41_NUMA_INTERLEAVE 全仓库没有消费者，
#      是死开关；真实 1 机 P0 里 THP 一直是 [always]/[madvise]、NUMA 从未交错；
#   2) 真实 run 建优化器状态 16 rank wall 0.81–310.71 s，慢 rank 的页 fault 落在
#      只剩 4.9 GiB 的 node2 上（node2 free 全程 4.8->4.95 GiB，一条页没落）；
#   3) 单进程把 node2 填到 7.5 GiB 后，同样 900 次 zeros 从 1.09 s 变成 7.06 s，
#      compact_stall=compact_fail=58376。
#
# 档位（云道控制台启动命令只填一行；环境变量按规则 7.1 放在脚本里）：
#   MY_ENV_HN_ARM=A  bash …/1110_run_dsv41_yundao_1node_top32_share_hostnuma_v1.sh  # 现状（对照）
#   MY_ENV_HN_ARM=B  …                                                             # 关 THP
#   MY_ENV_HN_ARM=C  …                                                             # NUMA 交错
#   MY_ENV_HN_ARM=D  …                                                             # B+C，并开 per-op vmstat
#
# 预期判据：
#   slow_n=0；每 rank `[rt-probe] done` wall<10 s；`MY_DEBUG_LOG rt_probe_vmstat_slow`
#   的 compact_stall 增量=0；`numa_maps_mib` 不再偏向单一节点。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
BASE=$MIRROR_ROOT/verl/my_scripts/yundao/1109_run_dsv41_yundao_1node_top32_share_v1.sh
ARM=${MY_ENV_HN_ARM:-A}

case "$ARM" in
  A) ;;
  B) export MY_ENV_DSV41_DISABLE_THP=1 ;;
  C) export MY_ENV_DSV41_NUMA_INTERLEAVE=1 ;;
  D)
    export MY_ENV_DSV41_DISABLE_THP=1
    export MY_ENV_DSV41_NUMA_INTERLEAVE=1
    export MY_ENV_DSV41_OPT_STATE_VMSTAT=1
    ;;
  *)
    echo "[1110 hn] ABORT: MY_ENV_HN_ARM=$ARM (expect A|B|C|D)" >&2
    exit 2
    ;;
esac

if [ ! -f "$BASE" ]; then
  echo "[1110 hn] ABORT: base not found $BASE" >&2
  exit 3
fi

# 唯一标识符，方便一键检索；把实际生效的三个开关打进日志
echo "[1110 hn] MY_DEBUG_LOG hostnuma_arm=$ARM thp=${MY_ENV_DSV41_DISABLE_THP:-0} interleave=${MY_ENV_DSV41_NUMA_INTERLEAVE:-0} vmstat=${MY_ENV_DSV41_OPT_STATE_VMSTAT:-0} base=$BASE"

exec bash "$BASE"
