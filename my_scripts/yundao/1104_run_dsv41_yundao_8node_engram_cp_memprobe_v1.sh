#!/usr/bin/env bash
# [验证结果] ✅ 成功：11/05 14:29 与 18:09 两轮 8 机跑通 2 步，三路探针（host 5s / npu 30s / proc 60s）正常产出 logs/memprobe。
# =============================================================================
# 1104 DSV4.1 云道 8 机 Engram+CP：显存 / 主机内存三路探针版 v1
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_noval_v2.sh
#     → 1104 recompute_full v1（方案 B 整层重计算）
#       → 1103 Engram+CP（CP2 / engram 表节点内共享 / 8 机 128 rank）
#         → 基线脚本
#   即：跳过 step0 验证、只跑 2 步、方案 B 不变、engram 配置不变。
#
# 与基线唯一的差异：把原来那个简单的 memprobe 换成三路探针，全部落在
#   ${PROBE_DIR}/<启动时间>_<主机名>_{host,npu,proc}.log
#
#   host（每 5 s）：/proc/meminfo 的 MemAvailable/MemFree/Cached/SwapFree、
#                   cgroup memory.current / memory.high、
#                   每个 NUMA 节点的 MemFree
#   npu （每 30 s）：npu-smi info 的 16 张卡 HBM 使用（used/total，单位 MB），
#                   顺序即 device 序号 0..15
#   proc（每 60 s）：按 RSS 排序的前 12 个进程（pid / rss_kB / comm）
#                   用来分清主机内存到底被谁占着（vLLM worker / 训练 worker / ray）
#
# 为什么加这三路（11/04 15:0x 的结论）：
#   11/04 10:22 那次死在第一次 update 建 AdamW 状态：单参数 `torch.zeros_like`
#   要 30–46 s，128 个 rank 只有 6 个建完，最后 Ray keepalive 超时。
#   但按 11/03 单机验证（top32 模型 / 2454 GiB / 实测峰值 1392 GB）对账，
#   8 机每节点 engram 反而更少（412 GiB vs 732 GB），所以"engram 放不下"不成立；
#   真实差异是 384 专家模型的非 engram 开销把整机推到 ~1.68 TB / 1988 GiB。
#   11/03 worklog 2.2 记录过同一类现象：主机映射 ~1.6 TB 时 Ascend 驱动
#   rtMallocHost 会自旋不返回，降到 ~1.2 TB 才恢复。
#   要判断这次是"内核回收"还是"驱动路径卡住"，必须同时看主机内存曲线和进程 RSS：
#     - MemAvailable 逼近 0、RSS 持续涨  → 内核回收/真不够
#     - MemAvailable 还有几百 GiB、RSS 不涨但分配卡住 → 驱动/分配路径
#   显存那一路是为了同时确认"设备侧到底剩多少"，避免再把主机问题和显存问题混在一起。
#
# 关掉探针：MY_ENV_DSV41_PROBE=0（只影响探针，不影响训练）
#
# 用法（云道"启动命令"只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_memprobe_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_cp_engram_probe_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_noval_v2.sh"

if [ ! -f "$BASE" ]; then
  echo "[memprobe] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 关掉 noval_v2 里那个简单探针，避免两份文件重复记。
export MY_ENV_DSV41_MEM_PROBE=0

MY_ENV_DSV41_PROBE=${MY_ENV_DSV41_PROBE:-1}
if [ "$MY_ENV_DSV41_PROBE" = "1" ]; then
  PROBE_DIR="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs/memprobe"
  mkdir -p "$PROBE_DIR"
  STAMP=$(date -u -d '+8 hours' +%Y%m%d_%H%M%S)
  HOST=$(hostname 2>/dev/null || echo unknown)
  PREFIX="${PROBE_DIR}/${STAMP}_${HOST}"

  # --- 1) 主机内存，每 5 s -------------------------------------------------
  (
    echo "# ts avail_kB free_kB cached_kB swapfree_kB cgroup_cur cgroup_high nodeN_free_kB..."
    while true; do
      ts=$(date '+%F %T')
      avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      memfree=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
      cached=$(awk '/^Cached:/{print $2}' /proc/meminfo)
      swapfree=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
      ccur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null \
             || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo NA)
      chi=$(cat /sys/fs/cgroup/memory.high 2>/dev/null \
            || cat /sys/fs/cgroup/memory/memory.soft_limit_in_bytes 2>/dev/null || echo NA)
      numa=$(for d in /sys/devices/system/node/node[0-9]*; do
               [ -r "$d/meminfo" ] || continue
               awk -v n="${d##*/}" '/MemFree:/{printf "%s_free_kB=%s ", n, $4}' "$d/meminfo"
             done)
      echo "$ts avail_kB=$avail free_kB=$memfree cached_kB=$cached swapfree_kB=$swapfree cgroup_cur=$ccur cgroup_high=$chi $numa"
      sleep 5
    done
  ) > "${PREFIX}_host.log" 2>&1 &

  # --- 2) NPU HBM，每 30 s ------------------------------------------------
  # npu-smi 在容器里缺 libc_sec.so，单独给这一路补 LD_LIBRARY_PATH，不影响训练进程环境。
  (
    export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
    echo "# ts hbm_MB=used/total 顺序为 device 0..15（npu-smi info 的 HBM-Usage 列）"
    while true; do
      ts=$(date '+%F %T')
      # 每张卡那一行里有两处 "N / M"（Memory-Usage 和 HBM-Usage），取最后一处。
      hbm=$(npu-smi info 2>/dev/null | awk '/0000:/ {
              line = $0; last = ""
              while (match(line, /[0-9]+ *\/ *[0-9]+/)) {
                last = substr(line, RSTART, RLENGTH)
                line = substr(line, RSTART + RLENGTH)
              }
              gsub(/ /, "", last)
              printf "%s ", last
            }')
      echo "$ts hbm_MB=${hbm:-NA}"
      sleep 30
    done
  ) > "${PREFIX}_npu.log" 2>&1 &

  # --- 3) 进程 RSS，每 60 s -----------------------------------------------
  (
    echo "# ts top12 by RSS（pid rss_kB comm）"
    while true; do
      ts=$(date '+%F %T')
      echo "== $ts =="
      ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -13
      sleep 60
    done
  ) > "${PREFIX}_proc.log" 2>&1 &

  echo "[memprobe] MY_DEBUG_LOG prefix=${PREFIX} host=${HOST} intervals=host5s/npu30s/proc60s time=$(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "[memprobe] MY_DEBUG_LOG base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
