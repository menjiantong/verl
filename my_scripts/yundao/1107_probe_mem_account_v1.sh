#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-07 起由 1107_run_dsv41_yundao_1node_align_v1.sh 自动拉起
# =============================================================================
# 1107 内存账探针 v1（**只读**：只读 /proc、/sys/fs/cgroup、/dev/shm，不发信号、不启停作业）
#
# 为什么需要它（上一轮的硬缺口）：
#   1 机 engram 作业临死时（14:00:16）节点 MemFree=9 GiB / MemAvail=6 GiB，
#   可同一个时刻 cgroup 计数只有：
#       memory.usage_in_bytes = 852.8 GiB   （peak 863.0 GiB）
#       memory.limit_in_bytes = 1893.4 GiB  failcnt=0 events 全 0
#   两者差 **~1140 GiB**。cgroup v1 在 memory.use_hierarchy=0 时父层级不聚合
#   子 cgroup，只读 memory.usage_in_bytes 会把账做丢。所以：
#     · 到底是谁占了那 1140 GiB（Ray? vLLM worker? trainer? 页缓存?）
#     · Shmem 366 GiB 之外还有多少匿名内存
#   这两个问题现在**没有答案**，而它们决定了下一步该砍谁。
#
# 本探针每 20 s 落一段四路分解：
#   A 节点视角   /proc/meminfo 的 MemTotal/MemFree/MemAvail/Cached/Shmem/AnonPages/Mapped
#   B cgroup 树  逐个列出 >1 GiB 的 cgroup（子目录），定位被父层级漏掉的那一块
#   C 进程视角   按 VmRSS 取 top-12，带 cmdline；top-1 再落整份 smaps_rollup
#   D /dev/shm   df + 按大小排序的文件列表（engram 共享表就住在这里）
#
# 用法：
#   bash 1107_probe_mem_account_v1.sh [输出目录]
#   输出：<输出目录>/memacct_<host>_<ts>.log
# =============================================================================
set -uo pipefail

OUTDIR=${1:-/efs_rl/m00899630/workspace/dsv41-verl/verl/logs}
HOST=$(hostname 2>/dev/null || echo unknown)
TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
OUT=$OUTDIR/memacct_${HOST}_${TS}.log
mkdir -p "$OUTDIR" 2>/dev/null || true

log() { echo "[memacct] $(date '+%F %T') $*" >> "$OUT"; }

log "start host=$HOST pid=$$ out=$OUT"
# 0 = 父 cgroup 不聚合子 cgroup 的用量（上次账做丢的直接原因之一）
log "cgroup_use_hierarchy=$(cat /sys/fs/cgroup/memory/memory.use_hierarchy 2>/dev/null || echo n/a)"

while :; do
  # ---------- A 节点视角 ----------
  # 注意：这里不用 awk 的 strftime（mawk/busybox awk 没有），时间戳由 shell 拼。
  NOW=$(date '+%F %T')
  MEM=$(awk '/^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^MemAvailable:/{a=$2} \
       /^Buffers:/{b=$2} /^Cached:/{c=$2} /^Shmem:/{s=$2} \
       /^AnonPages:/{an=$2} /^Mapped:/{m=$2} /^SReclaimable:/{sr=$2} \
       END{printf "MemTotal=%dGiB MemFree=%dGiB MemAvail=%dGiB Buffers=%dGiB Cached=%dGiB Shmem=%dGiB AnonPages=%dGiB Mapped=%dGiB SReclaim=%dGiB", \
           t/1048576, f/1048576, a/1048576, b/1048576, c/1048576, s/1048576, an/1048576, m/1048576, sr/1048576}' \
    /proc/meminfo 2>/dev/null)
  echo "[memacct] $NOW node ${MEM:-meminfo-unreadable}" >> "$OUT"

  # ---------- B cgroup 树：只报 >1 GiB 的 ----------
  for u in /sys/fs/cgroup/memory/*/memory.usage_in_bytes; do
    [ -r "$u" ] || continue
    v=$(cat "$u" 2>/dev/null) || continue
    [ -n "${v:-}" ] || continue
    [ "$v" -ge 1073741824 ] 2>/dev/null || continue
    echo "[memacct] $(date '+%F %T') cg $(dirname "$u") $((v/1073741824))GiB" >> "$OUT"
  done

  # ---------- C 进程视角 ----------
  TMP=$(mktemp 2>/dev/null || echo /tmp/memacct.$$)
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ -r "$p/status" ] || continue
    rss=$(awk '/^VmRSS:/{print $2}' "$p/status" 2>/dev/null)
    [ -n "${rss:-}" ] || continue
    echo "$rss $pid"
  done | sort -rn | head -12 > "$TMP" 2>/dev/null

  while read -r rss pid; do
    [ -n "${pid:-}" ] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-110)
    echo "[memacct] $(date '+%F %T') proc pid=$pid rss=$((rss/1024))MiB cmd=${cmd:-?}" >> "$OUT"
  done < "$TMP"

  top1=$(head -1 "$TMP" 2>/dev/null | awk '{print $2}')
  if [ -n "${top1:-}" ] && [ -r "/proc/$top1/smaps_rollup" ]; then
    echo "[memacct] $(date '+%F %T') smaps_rollup pid=$top1 begin" >> "$OUT"
    sed 's/^/[memacct]   /' "/proc/$top1/smaps_rollup" >> "$OUT" 2>/dev/null
    echo "[memacct] $(date '+%F %T') smaps_rollup pid=$top1 end" >> "$OUT"
  fi
  rm -f "$TMP" 2>/dev/null

  # ---------- D /dev/shm ----------
  echo "[memacct] $(date '+%F %T') devshm $(df -P /dev/shm 2>/dev/null | awk 'NR==2{printf "used=%dGiB total=%dGiB", $3/1048576, $2/1048576}')" >> "$OUT"
  ls -laS /dev/shm 2>/dev/null | head -6 | sed 's/^/[memacct]   shm /' >> "$OUT"

  log "tick_end"
  sleep 20
done
