#!/usr/bin/env bash
# [验证结果] ❌ 失败（2026-11-05 12:15）：作业容器 /proc/sys 只读（errno=30 EROFS），drop_caches 守护为空操作，卡在 AdamW 状态创建，13:57 被 Ray keepalive 判死；已被 fadvise_v3 取代。
# =============================================================================
# 1105 DSV4.1 云道 8 机：Engram 开、CP 关（CP=1）+ 页缓存守护 v2
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_v1.sh
#     → 1104_..._memprobe_v1.sh（三路探针：host 5 s / npu 30 s / proc 60 s）
#       → 1104_..._noval_v2.sh（跳 step0 验证、TEST_FREQ=-1、STEPS=2）
#         → 1104_..._recompute_full_v1.sh（方案 B：整层重计算）
#           → 1103_..._engram_cp_v1.sh（Engram，CP 内部开关按 CP_SIZE 派生）
#             → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# 与基线唯一的差异：加一个“页缓存守护”（方案 A）。
#
# -----------------------------------------------------------------------------
# 为什么加它（11/04 14:49 那次的实测结论，数据在 logs/memprobe/）
# -----------------------------------------------------------------------------
#   那次死在第一次 update 建 AdamW 状态：单参数 torch.zeros_like 要 30–46 s，
#   128 个 rank 只有 6 个建完，最后 Ray keepalive 超时判 actor 不可用。
#   探针在 16:30 的横截面（8 台同型）：
#     cgroup 1.65–1.69 TB，MemFree 只剩 7.3–31.2 GB，MemAvailable 却还有 ~430 GiB。
#     8 个 NUMA 的 MemFree 分别是 3.97 / 2.40 / 0.91 / 0.55 / 0.54 / 0.58 / 0.54 / 0.54 GiB
#     —— 不是某个 NUMA 被打爆，是整机没有空洞页。
#   节点 MemTotal = 2011 GiB（/proc/meminfo 实测，HugePages_Total=0，没有大页预留），
#   其中 ~830 GiB 变成了 EFS 读 1.4 TB checkpoint 留下的 page cache。
#   内核靠“需要时回收页缓存”顶着，但这里的缓存主体是 NFS 页，回收路径慢，
#   于是每次 70 MB 的私有分配都要几十秒。
#
#   守护做的事：MemFree 低于阈值时 `echo 1 > /proc/sys/vm/drop_caches`，
#   只丢**干净的**页缓存（不动脏页、不动进程内存、不影响任何数值）。
#   预期效果：把 ~700–830 GiB 页缓存放掉，MemFree 从 10 GiB 回到 ~700 GiB 以上，
#   update 阶段的分配不再走慢速回收。
#
#   注意：drop_caches 是**节点级**（内核级）操作，会同时清掉同节点其它进程的干净页缓存。
#   这些是独占的训练节点，代价只是“下次读同一批文件要重新从 EFS 读”，可以接受。
#   训练中几乎不读盘（checkpoint 只在启动时读），所以代价基本为零。
#
# 开关（全部 MY_ENV_ 前缀）：
#   MY_ENV_DSV41_DROP_CACHE_GUARD=1        1=开守护，0=关（默认开）
#   MY_ENV_DSV41_DROP_CACHE_FREE_GIB=256   触发阈值：MemFree 低于这个值就 drop
#                                          （16:17 那次 MemFree 从 336 掉到 116 GiB，
#                                           256 能在撞墙前抓到；设太小就来不及）
#   MY_ENV_DSV41_DROP_CACHE_MIN_INTERVAL_S=60  两次 drop 的最小间隔，防止抖动
#
# 用法（云道“启动命令”只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_dropcache_v2.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_engram_cp1_dc_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1105_run_dsv41_yundao_8node_engram_cp1_v1.sh"

if [ ! -f "$BASE" ]; then
  echo "[dropcache] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

MY_ENV_DSV41_DROP_CACHE_GUARD=${MY_ENV_DSV41_DROP_CACHE_GUARD:-1}
MY_ENV_DSV41_DROP_CACHE_FREE_GIB=${MY_ENV_DSV41_DROP_CACHE_FREE_GIB:-256}
MY_ENV_DSV41_DROP_CACHE_MIN_INTERVAL_S=${MY_ENV_DSV41_DROP_CACHE_MIN_INTERVAL_S:-60}

if [ "$MY_ENV_DSV41_DROP_CACHE_GUARD" = "1" ]; then
  if [ ! -w /proc/sys/vm/drop_caches ]; then
    echo "[dropcache] WARN: /proc/sys/vm/drop_caches 不可写，守护不生效（训练继续）"
  else
    PROBE_DIR="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs/memprobe"
    mkdir -p "$PROBE_DIR"
    DC_LOG="${PROBE_DIR}/$(date -u -d '+8 hours' +%Y%m%d_%H%M%S)_$(hostname 2>/dev/null || echo unknown)_dropcache.log"
    MEMTOTAL_GIB=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
    FREESTART_GIB=$(( $(awk '/^MemFree:/{print $2}' /proc/meminfo) / 1048576 ))
    THRESH_KB=$(( MY_ENV_DSV41_DROP_CACHE_FREE_GIB * 1048576 ))
    echo "[dropcache] MY_DEBUG_LOG guard=on host=$(hostname 2>/dev/null || echo unknown) MemTotal=${MEMTOTAL_GIB}GiB MemFree=${FREESTART_GIB}GiB threshold=${MY_ENV_DSV41_DROP_CACHE_FREE_GIB}GiB min_interval=${MY_ENV_DSV41_DROP_CACHE_MIN_INTERVAL_S}s log=${DC_LOG}"
    (
      last=0
      while true; do
        free_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
        if [ "$free_kb" -lt "$THRESH_KB" ]; then
          now=$(date +%s)
          if [ $(( now - last )) -ge "$MY_ENV_DSV41_DROP_CACHE_MIN_INTERVAL_S" ]; then
            last=$now
            ts=$(date '+%F %T')
            cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo NA)
            cached_before=$(awk '/^Cached:/{print $2}' /proc/meminfo)
            if echo 1 > /proc/sys/vm/drop_caches 2>/dev/null; then
              free_after=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
              cached_after=$(awk '/^Cached:/{print $2}' /proc/meminfo)
              echo "$ts drop_ok free_before=$((free_kb/1048576))GiB free_after=$((free_after/1048576))GiB cached_before=$((cached_before/1048576))GiB cached_after=$((cached_after/1048576))GiB cgroup=$cg" >> "$DC_LOG"
              echo "[dropcache] $ts drop_ok free $((free_kb/1048576))GiB -> $((free_after/1048576))GiB, cached $((cached_before/1048576))GiB -> $((cached_after/1048576))GiB"
            else
              echo "$ts drop_FAILED free=${free_kb}KiB errno=?" >> "$DC_LOG"
              echo "[dropcache] $ts drop_FAILED（守护继续，训练不受影响）"
            fi
          fi
        fi
        sleep 5
      done
    ) &
    echo "[dropcache] MY_DEBUG_LOG guard_pid=$! base=$(basename "$BASE") time=$(date '+%Y-%m-%d %H:%M:%S')"

    # 附带：每 60 s 记一次内存构成，回答“cgroup 外那 400 多 GiB 到底是什么”。
    #   cg_anon/cg_file = 本 cgroup 的匿名内存/文件缓存（memory.stat，单位字节）
    #   Slab/Shmem      = 内核 slab / 共享内存（/proc/meminfo，单位 kB）
    # cgroup v2 用 anon/file/shmem，v1 用 total_rss/total_cache（本机是 v1），都兼容。
    read_memstat() {
      for k in "$@"; do
        v=$(awk -v k="$k" '$1==k{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
        [ -n "$v" ] && { echo "$v"; return; }
        v=$(awk -v k="$k" '$1==k{print $2}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
        [ -n "$v" ] && { echo "$v"; return; }
      done
      echo 0
    }
    (
      while true; do
        ts=$(date '+%F %T')
        cganon=$(read_memstat anon total_rss rss)
        cgfile=$(read_memstat file total_cache cache)
        cgshmem=$(read_memstat shmem total_shmem)
        slab=$(awk '/^Slab:/{print $2}' /proc/meminfo)
        shmem=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
        mtot=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
        echo "$ts breakdown MemTotal=$((mtot/1048576))GiB cg_anon=$((cganon/1073741824))GiB cg_file=$((cgfile/1073741824))GiB cg_shmem=$((cgshmem/1073741824))GiB slab=$((slab/1048576))GiB shmem=$((shmem/1048576))GiB" >> "$DC_LOG"
        sleep 60
      done
    ) &
  fi
fi

exec bash "$BASE"
