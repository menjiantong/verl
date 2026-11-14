#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-07 起由 1107_run_dsv41_yundao_1node_engram_memopt_v1.sh 自动拉起
# =============================================================================
# 1107 内存账探针 v2（**只读**：只读 /proc、/sys/fs/cgroup、/sys/devices/system/node、
# /dev/shm，不发信号、不启停作业）
#
# v1 的缺口：只抓 9 个 meminfo 字段，抓不到 Mlocked/Unevictable/Slab/SUnreclaim/
# PageTables/HugePages，也抓不到 cgroup memory.stat 与 per-NUMA free；导致 1.2 TiB
# 隐藏内存只能靠"总量减已知项"反推。
#
# v2 每 10 s 落一段六路分解（唯一标识符 MY_DEBUG_LOG memacct_v2_*）：
#   A 节点全字段   /proc/meminfo 全量（含 Mlocked/Unevictable/Slab/PageTables/
#                  HugePages/AnonHugePages/VmallocUsed/Percpu/Cma）
#   B cgroup      memory.stat 关键项（cache/rss/shmem/mapped_file/unevictable/
#                  hierarchical_memory_limit）+ usage/peak/failcnt/events
#   C NUMA        /sys/devices/system/node/node*/meminfo 的 MemFree
#   D 进程        按 VmRSS top-10，带 VmPin/VmRSS/RssAnon/RssFile/RssShmem/VmSwap
#   E 进程汇总    sum(VmRSS)、sum(VmPin)、进程数
#   F smaps       top-3 的 smaps_rollup（Rss/Pss/Pss_Anon/Pss_Shmem/Shared_Dirty/
#                  Private_Dirty/AnonHugePages/Locked）
#   G /dev/shm    df + 文件数（不做 du，366 GiB 的 shm du 太慢）
#
# 用法：
#   bash 1107_probe_mem_account_v2.sh [输出目录]
#   输出：<输出目录>/memacct_v2_<host>_<ts>.log
# =============================================================================
set -uo pipefail

OUTDIR=${1:-/efs_rl/m00899630/workspace/dsv41-verl/verl/logs}
HOST=$(hostname 2>/dev/null || echo unknown)
TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
OUT=$OUTDIR/memacct_v2_${HOST}_${TS}.log
mkdir -p "$OUTDIR" 2>/dev/null || true

log() { echo "[memacct_v2] $(date '+%F %T') $*" >> "$OUT"; }

log "MY_DEBUG_LOG memacct_v2_start host=$HOST pid=$$ out=$OUT"
log "MY_DEBUG_LOG memacct_v2_hierarchy use_hierarchy=$(cat /sys/fs/cgroup/memory/memory.use_hierarchy 2>/dev/null || echo n/a)"

while :; do
  NOW=$(date '+%F %T')

  # ---------- A 节点全字段 ----------
  # 逐项抓，避免 awk 拼 40 个字段时漏项；键名保持 /proc/meminfo 原样，便于事后 grep。
  {
    echo "[memacct_v2] $NOW node_begin"
    awk '{gsub(/:/,"",$1); print "[memacct_v2]   meminfo "$1"="$2}' /proc/meminfo 2>/dev/null
    echo "[memacct_v2] $NOW node_end"
  } >> "$OUT"

  # ---------- B cgroup v1/v2 ----------
  if [ -f /sys/fs/cgroup/memory/memory.stat ]; then
    {
      echo "[memacct_v2] $NOW cgroup_begin"
      grep -E '^(cache|rss|rss_huge|shmem|mapped_file|dirty|writeback|inactive_anon|active_anon|inactive_file|active_file|unevictable|hierarchical_memory_limit|hierarchical_memsw_limit|total_cache|total_rss|total_shmem|total_unevictable) ' \
        /sys/fs/cgroup/memory/memory.stat 2>/dev/null | sed 's/^/[memacct_v2]   cgstat /'
      echo "[memacct_v2]   cgusage limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null) cur=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null) peak=$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null) failcnt=$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)"
      echo "[memacct_v2]   cgevents $(tr '\n' ',' < /sys/fs/cgroup/memory/memory.events 2>/dev/null)"
      echo "[memacct_v2] $NOW cgroup_end"
    } >> "$OUT"
  elif [ -f /sys/fs/cgroup/memory.stat ]; then
    {
      echo "[memacct_v2] $NOW cgroup_begin"
      grep -E '^(anon|file|kernel|slab|shmem|file_mapped|file_dirty|inactive_anon|active_anon|inactive_file|active_file|unevictable) ' \
        /sys/fs/cgroup/memory.stat 2>/dev/null | sed 's/^/[memacct_v2]   cgstat /'
      echo "[memacct_v2]   cgusage max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)"
      echo "[memacct_v2]   cgevents $(tr '\n' ',' < /sys/fs/cgroup/memory.events 2>/dev/null)"
      echo "[memacct_v2] $NOW cgroup_end"
    } >> "$OUT"
  else
    echo "[memacct_v2] $NOW cgroup_begin mode=none cgroup_end" >> "$OUT"
  fi

  # ---------- C per-NUMA free ----------
  {
    echo "[memacct_v2] $NOW numa_begin"
    for f in /sys/devices/system/node/node*/meminfo; do
      [ -r "$f" ] || continue
      node=$(basename "$(dirname "$f")")
      awk -v n="$node" '/^Node [0-9]+ MemFree:/{print "[memacct_v2]   numa "n" MemFree_kB="$4} /^Node [0-9]+ MemUsed:/{print "[memacct_v2]   numa "n" MemUsed_kB="$4}' "$f" 2>/dev/null
    done
    echo "[memacct_v2] $NOW numa_end"
  } >> "$OUT"

  # ---------- D 进程 top-10 ----------
  TMP=$(mktemp 2>/dev/null || echo /tmp/memacct_v2.$$)
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ -r "$p/status" ] || continue
    rss=$(awk '/^VmRSS:/{print $2}' "$p/status" 2>/dev/null)
    [ -n "${rss:-}" ] || continue
    echo "$rss $pid"
  done | sort -rn | head -10 > "$TMP" 2>/dev/null

  {
    echo "[memacct_v2] $NOW proc_begin"
    while read -r rss pid; do
      [ -n "${pid:-}" ] || continue
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)
      st=$(awk '/^(VmPin|VmRSS|RssAnon|RssFile|RssShmem|VmSwap):/{printf "%s=%s ", $1, $2}' "/proc/$pid/status" 2>/dev/null)
      echo "[memacct_v2]   proc pid=$pid $st cmd=${cmd:-?}"
    done < "$TMP"
    # 汇总：全进程 VmRSS / VmPin 之和（mawk 没有数组排序，直接累加）
    SUMRSS=$(awk '/^VmRSS:/{s+=$2} END{print s+0}' /proc/[0-9]*/status 2>/dev/null)
    SUMPIN=$(awk '/^VmPin:/{s+=$2} END{print s+0}' /proc/[0-9]*/status 2>/dev/null)
    NPROC=$(ls -d /proc/[0-9]* 2>/dev/null | wc -l)
    echo "[memacct_v2]   sum proc_count=$NPROC sum_VmRSS_MiB=$((SUMRSS/1024)) sum_VmPin_MiB=$((SUMPIN/1024))"
    echo "[memacct_v2] $NOW proc_end"
  } >> "$OUT"

  # ---------- E smaps top-3 ----------
  {
    echo "[memacct_v2] $NOW smaps_begin"
    head -3 "$TMP" | while read -r rss pid; do
      [ -n "${pid:-}" ] || continue
      [ -r "/proc/$pid/smaps_rollup" ] || continue
      echo "[memacct_v2]   smaps pid=$pid rss_kB=$rss"
      awk '/^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous|AnonHugePages|ShmemPmdMapped|Locked|Swap):/{print "[memacct_v2]     "$0}' "/proc/$pid/smaps_rollup" 2>/dev/null
    done
    echo "[memacct_v2] $NOW smaps_end"
  } >> "$OUT"
  rm -f "$TMP" 2>/dev/null

  # ---------- G /dev/shm ----------
  echo "[memacct_v2] $NOW shm $(df -P /dev/shm 2>/dev/null | awk 'NR==2{printf "used_kB=%s total_kB=%s", $3, $2}') files=$(ls -1 /dev/shm 2>/dev/null | wc -l)" >> "$OUT"

  log "MY_DEBUG_LOG memacct_v2_tick"
  sleep 10
done
