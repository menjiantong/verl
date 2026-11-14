#!/usr/bin/env bash
# [验证结果] ✅ 成功：11/05 14:29 跑通 2 步；fadvise 8 台各回收 298-456 GiB 文件页；但 engram 使 update_actor 高达 1428.9 s（见 1106）。
# =============================================================================
# 1105 DSV4.1 云道 8 机：Engram 开、CP 关 + 页缓存回收守护 v3（posix_fadvise 版）
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_v1.sh
#     → 1104_..._memprobe_v1.sh（三路探针 host 5s / npu 30s / proc 60s）
#       → 1104_..._noval_v2.sh（跳过 step0 验证、TEST_FREQ=-1、STEPS=2）
#         → 1104_..._recompute_full_v1.sh（方案 B：recompute_plan=["model.layers.{*}"])
#           → 1103_..._engram_cp_v1.sh（Engram，CP 内部开关按 CP_SIZE 派生）
#             → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# 注意：**不串 v2**（1105_..._dropcache_v2.sh）。v2 的守护在作业容器里是空操作，
#   11/05 12:15 那次日志第一行就是：
#     [dropcache] WARN: /proc/sys/vm/drop_caches 不可写，守护不生效（训练继续）
#   ModelArts 作业容器把 /proc/sys 挂成只读（云道登录容器可写，两者不同），
#   所以 v2 整段（包括每 60 s 的内存构成明细）根本没跑。v3 换成容器内可用的机制，
#   并把明细日志改成无条件启动。
#
# -----------------------------------------------------------------------------
# 一、11/05 12:15 那次的结论（为什么 CP 不是变量）
# -----------------------------------------------------------------------------
#   配置确认无误：cp=1、dp_shared=1、load_format=safetensors、val_before_train=False、
#   test_freq=-1、steps=2、recompute plan=["model.layers.{*}"]。
#   结果与 11/04 14:49（CP=2）**逐项同型**：
#     - 第一次 sleep 13:13:26 → 13:26:39，13 分 13 秒；
#     - update 建 AdamW 状态：created=89–406/1192，last_dt=265.45 / 170.31 / 23.74 s，
#       大量 `<=== 30s 无进展`；
#     - 13:57 Ray `ActorUnavailableError ... keepalive watchdog timeout`，作业退出。
#   8 台 13:52 横截面（MemFree / cgroup）：
#     w0 10.8G/1643G  w1 72.5G/1619G  w2 71.9G/1577G  w3 8.7G/1639G
#     w4 16.6G/1650G  w5 12.8G/1643G  w6  7.2G/1670G  w7 19.1G/1683G
#   → CP 只吃显存（每层全序列 KV/indexer 缓冲），对主机内存零贡献，实测印证。
#
# -----------------------------------------------------------------------------
# 二、主机内存到底被谁占着（本次新增的定性证据）
# -----------------------------------------------------------------------------
#   SharedUvaBuffer（vllm-ascend/.../engram/npu.py:261）用
#   `multiprocessing.shared_memory.SharedMemory(create=True, size=...)`，
#   即 POSIX shm → **/dev/shm 上的 tmpfs 页**。每节点 8 个 TP 分片 × 2 张表
#   × 22.89 GiB = **366 GiB 常驻 tmpfs**，是匿名页，**不可回收、不可 swap**。
#   这解释了 /proc/meminfo 里 Cached 781 GiB 却只有 MemAvailable ~430 GiB：
#     Cached 781 ≈ Shmem 366（engram tmpfs，不可回收）+ ~415 文件页（可回收）
#     MemAvailable 430 ≈ MemFree 10 + 可回收文件页 ~420
#   也就是说：**能动的只有那 400 多 GiB 的文件页缓存，它来自读 1.4 TB checkpoint。**
#   把这部分放掉，节点 cgroup 从 1.62–1.64 TB 回到 ~1.2 TB —— 正好是 11/03 单机
#   worklog §2.2 记录的“主机映射降到 ~1.2 TB 后 rtMallocHost 恢复正常”那个量级。
#
# -----------------------------------------------------------------------------
# 三、v3 做的事
# -----------------------------------------------------------------------------
#   1) 启动自检（一次性）：读权重目录第一个 safetensors 的 1 GiB，量
#      Cached 读前/读后/fadvise 后的差，直接证明“这个容器里 fadvise 能不能放页”。
#      11/05 已在云道登录容器实测：读 3 GiB → Cached +3.14 GiB → fadvise 后精确回落。
#   2) 守护循环（每 5 s）：MemFree 低于阈值且距上次 ≥60 s 时，对
#      /efs_rl/weights/DeepSeek-V4.1-Flash-bf16/*.safetensors 逐个
#      posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)，并记 before/after。
#      - 只碰权重目录，**不碰数据集**（parquet 每步要重读，动了会变慢）；
#      - 只作用于**干净页**，不动脏页、不动任何进程内存，对数值零影响；
#      - 权重读完之后不会再读，所以代价 ≈ 0。
#   3) 内存构成明细（每 60 s，无条件启动）：MemTotal/MemFree/MemAvailable/
#      Cached/Shmem/Slab + cgroup current + cgroup 的 anon/file/shmem（v1/v2 都兼容）。
#      这条是用来验算“366 GiB tmpfs + 441 GiB sleep 备份 + 46 GiB 训练 engram”的。
#
#   预期（可证伪）：MemFree 10 GiB → ~400 GiB+，cgroup 1.63 TB → ~1.2 TB，
#   AdamW 状态创建的 `last_dt` 回到秒级。若自检里 dropped≈0，说明该 mount 的
#   文件页被 mmap 钉住，fadvise 无效 —— 那就直接看明细日志改走结构性方案。
#
# 开关（全部 MY_ENV_ 前缀）：
#   MY_ENV_DSV41_FADVISE_GUARD=1        1=开守护，0=关（默认开）
#   MY_ENV_DSV41_FADVISE_FREE_GIB=256   MemFree 低于该值触发回收（提前于撞墙）
#   MY_ENV_DSV41_FADVISE_MIN_INTERVAL_S=60  两次回收最小间隔
#   MY_ENV_DSV41_FADVISE_DIR=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16
#                                       要回收的目录（只回收 *.safetensors）
#   MY_ENV_DSV41_FADVISE_SELFTEST=1     1=启动时做一次 1 GiB 自检
#
# 用法（云道“启动命令”只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_engram_cp1_fadv_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 回滚：跑 1105_..._cp1_v1.sh（本脚本只加旁路守护，不动训练配置）。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1105_run_dsv41_yundao_8node_engram_cp1_v1.sh"

if [ ! -f "$BASE" ]; then
  echo "[fadvise] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

MY_ENV_DSV41_FADVISE_GUARD=${MY_ENV_DSV41_FADVISE_GUARD:-1}
MY_ENV_DSV41_FADVISE_FREE_GIB=${MY_ENV_DSV41_FADVISE_FREE_GIB:-256}
MY_ENV_DSV41_FADVISE_MIN_INTERVAL_S=${MY_ENV_DSV41_FADVISE_MIN_INTERVAL_S:-60}
MY_ENV_DSV41_FADVISE_DIR=${MY_ENV_DSV41_FADVISE_DIR:-/efs_rl/weights/DeepSeek-V4.1-Flash-bf16}
MY_ENV_DSV41_FADVISE_SELFTEST=${MY_ENV_DSV41_FADVISE_SELFTEST:-1}

# 读 cgroup 的某个 key（v2 用 anon/file/shmem，v1 用 total_rss/total_cache），单位字节
read_memstat() {
  for k in "$@"; do
    v=$(awk -v k="$k" '$1==k{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    [ -n "$v" ] && { echo "$v"; return; }
    v=$(awk -v k="$k" '$1==k{print $2}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
    [ -n "$v" ] && { echo "$v"; return; }
  done
  echo 0
}

# 对权重目录里所有 safetensors 做 POSIX_FADV_DONTNEED（只放干净页缓存）
run_fadvise() {
  python3 - "$MY_ENV_DSV41_FADVISE_DIR" <<'PY' 2>&1
import os, sys, time
d = sys.argv[1]
t0 = time.time()
try:
    names = sorted(os.listdir(d))
except OSError as e:
    print(f"ERR listdir {d}: {e}")
    sys.exit(0)
files = [os.path.join(d, n) for n in names if n.endswith(".safetensors")]
ok = err = 0
first_err = ""
for p in files:
    try:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        ok += 1
    except OSError as e:
        err += 1
        if not first_err:
            first_err = f"{os.path.basename(p)}: {e}"
print(f"advised={ok} err={err} files={len(files)} took={time.time()-t0:.2f}s"
      + (f" first_err={first_err}" if first_err else ""))
PY
}

# 启动自检：读 1 GiB → 量 Cached 变化 → fadvise → 再量。证明机制在本容器是否可用。
run_selftest() {
  python3 - "$MY_ENV_DSV41_FADVISE_DIR" <<'PY' 2>&1
import os, sys
def cached_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0
d = sys.argv[1]
p = None
for n in sorted(os.listdir(d)):
    if n.endswith(".safetensors"):
        p = os.path.join(d, n)
        break
if p is None:
    print(f"selftest SKIP: no safetensors under {d}")
    sys.exit(0)
n = 1 << 30
c0 = cached_kb()
fd = os.open(p, os.O_RDONLY)
off = 0
while off < n:
    b = os.pread(fd, min(64 << 20, n - off), off)
    if not b:
        break
    off += len(b)
c1 = cached_kb()
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
c2 = cached_kb()
print(f"selftest file={os.path.basename(p)} read={off/2**20:.0f}MiB "
      f"cached_before={c0/2**20:.2f}GiB after_read={c1/2**20:.2f}GiB "
      f"after_fadvise={c2/2**20:.2f}GiB dropped={(c1-c2)/2**20:.2f}GiB")
PY
}

if [ "$MY_ENV_DSV41_FADVISE_GUARD" = "1" ]; then
  PROBE_DIR="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs/memprobe"
  mkdir -p "$PROBE_DIR"
  FD_LOG="${PROBE_DIR}/$(date -u -d '+8 hours' +%Y%m%d_%H%M%S)_$(hostname 2>/dev/null || echo unknown)_fadvise.log"

  MEMTOTAL_GIB=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
  FREESTART_GIB=$(( $(awk '/^MemFree:/{print $2}' /proc/meminfo) / 1048576 ))
  echo "[fadvise] MY_DEBUG_LOG guard=on host=$(hostname 2>/dev/null || echo unknown) dir=${MY_ENV_DSV41_FADVISE_DIR} MemTotal=${MEMTOTAL_GIB}GiB MemFree=${FREESTART_GIB}GiB threshold=${MY_ENV_DSV41_FADVISE_FREE_GIB}GiB min_interval=${MY_ENV_DSV41_FADVISE_MIN_INTERVAL_S}s log=${FD_LOG}"

  # 开机诊断：drop_caches 到底为什么不可写。
  #   errno=30 EROFS  → /proc/sys 是只读挂载（ModelArts 作业容器的情形）
  #   errno=13 EACCES → 权限/能力不足
  #   errno=2  ENOENT → 该文件被屏蔽
  # 只 open 一次、不写任何内容，对内核零副作用。
  dc_diag=$(python3 - <<'PY' 2>&1
import errno, os
p = "/proc/sys/vm/drop_caches"
print(f"path_exists={os.path.exists(p)} w_ok={os.access(p, os.W_OK)}", end=" ")
try:
    fd = os.open(p, os.O_WRONLY)
    os.close(fd)
    print("open_wronly=OK")
except OSError as e:
    print(f"open_wronly=FAILED errno={e.errno}({errno.errorcode.get(e.errno, '?')})")
PY
)
  echo "[fadvise] MY_DEBUG_LOG drop_caches_diag uid=$(id -u) ${dc_diag}" | tee -a "$FD_LOG"

  # 1) 启动自检（同步执行，几秒；结果同时进主日志）
  if [ "$MY_ENV_DSV41_FADVISE_SELFTEST" = "1" ]; then
    st=$(run_selftest)
    echo "[fadvise] MY_DEBUG_LOG $(date '+%F %T') ${st}" | tee -a "$FD_LOG"
  fi

  # 2) 守护循环：MemFree 低于阈值才动，避免无谓地丢缓存
  (
    THRESH_KB=$(( MY_ENV_DSV41_FADVISE_FREE_GIB * 1048576 ))
    last=0
    while true; do
      free_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
      if [ "$free_kb" -lt "$THRESH_KB" ]; then
        now=$(date +%s)
        if [ $(( now - last )) -ge "$MY_ENV_DSV41_FADVISE_MIN_INTERVAL_S" ]; then
          last=$now
          ts=$(date '+%F %T')
          avail_b=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
          cached_b=$(awk '/^Cached:/{print $2}' /proc/meminfo)
          shmem_b=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
          res=$(run_fadvise)
          free_a=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
          avail_a=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
          cached_a=$(awk '/^Cached:/{print $2}' /proc/meminfo)
          shmem_a=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
          echo "$ts drop free ${free_kb}->${free_a}KiB avail ${avail_b}->${avail_a}KiB cached ${cached_b}->${cached_a}KiB shmem ${shmem_b}->${shmem_a}KiB ${res}" >> "$FD_LOG"
          echo "[fadvise] $ts free $((free_kb/1048576))->$((free_a/1048576))GiB avail $((avail_b/1048576))->$((avail_a/1048576))GiB cached $((cached_b/1048576))->$((cached_a/1048576))GiB ${res}"
        fi
      fi
      sleep 5
    done
  ) &
  echo "[fadvise] MY_DEBUG_LOG guard_pid=$! base=$(basename "$BASE") time=$(date '+%Y-%m-%d %H:%M:%S')"

  # 3) 内存构成明细：每 60 s 一条，**无条件启动**（v2 的教训：它被 drop_caches
  #    只读判断连带跳过了，导致整轮没有 anon/file/shmem 拆分）。
  (
    while true; do
      ts=$(date '+%F %T')
      memfree=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
      avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      cached=$(awk '/^Cached:/{print $2}' /proc/meminfo)
      shmem=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
      slab=$(awk '/^Slab:/{print $2}' /proc/meminfo)
      mtot=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
      cgcur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null \
              || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
      cganon=$(read_memstat anon total_rss rss)
      cgfile=$(read_memstat file total_cache cache)
      cgshmem=$(read_memstat shmem total_shmem)
      echo "$ts breakdown MemTotal=$((mtot/1048576))GiB free=$((memfree/1048576))GiB avail=$((avail/1048576))GiB cached=$((cached/1048576))GiB shmem=$((shmem/1048576))GiB slab=$((slab/1048576))GiB cg_cur=$((cgcur/1073741824))GiB cg_anon=$((cganon/1073741824))GiB cg_file=$((cgfile/1073741824))GiB cg_shmem=$((cgshmem/1073741824))GiB" >> "$FD_LOG"
      sleep 60
    done
  ) &
  echo "[fadvise] MY_DEBUG_LOG breakdown_pid=$! interval=60s"
else
  echo "[fadvise] MY_DEBUG_LOG guard=off（MY_ENV_DSV41_FADVISE_GUARD=0）"
fi

echo "[fadvise] MY_DEBUG_LOG base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
