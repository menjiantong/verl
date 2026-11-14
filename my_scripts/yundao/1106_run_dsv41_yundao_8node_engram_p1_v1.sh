#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次实跑（P3 云道 8 机四探针）
# =============================================================================
# 1106 P3：云道 8 机 128 卡，Engram A/B，四探针同时上
#
# 基线（规则 7.3/7.4：本脚本只写差异）：
#   engram  arm -> my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
#   noengram arm -> my_scripts/yundao/1106_run_dsv41_yundao_8node_noengram_cp1_fadvise_v1.sh
#
# 本脚本只做三件事，不碰训练逻辑：
#   1) 打开 P0 打好的两个探针（fetch 分段 + 优化器逐参数 wall/cpu/MB/MBps/VmPin）；
#   2) 起 cgroup 采样器（memory limit / events / pressure —— E1 缺的正是这个）；
#   3) 起「无进展即抓原生栈」的看门狗（判「谁在 devmm_svm_ioctl 里」）。
#
# 为什么需要第 3 项：E1 有 1812 s 的卡顿，但那一刻没有任何栈，
# 所以「rtMallocHost / process_events / futex / cgroup_reclaim」四条候选闭不上环。
# 这里用 [rt-probe] 的 i=N/1192 心跳做触发：30 s 不变就 dump 全节点 python 原生栈。
#
# 用法（云道「启动命令」**只填这一行，不带任何环境变量**，8 机作业每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1106_run_dsv41_yundao_8node_engram_p1_v1.sh
#
# 实验参数已写死在下面「本次实验锁定」一节，默认 arm=engram、steps=2。
# 需要换对照 arm 时才临时覆盖（规则 6.1 前缀不变）：
#   MY_ENV_P3_ARM=noengram bash .../1106_run_dsv41_yundao_8node_engram_p1_v1.sh
#
# 可覆盖开关（规则 6.1）：
#   MY_ENV_P3_ARM=engram|noengram          默认 engram
#   MY_ENV_P3_STEPS=2                      默认 2 步（E1 就是 2 步）
#   MY_ENV_P3_STALL_SEC=30                 心跳停多久算卡住
#   MY_ENV_P3_STACK_ROUNDS=8               每次卡住最多抓几轮
#   MY_ENV_P3_REUSE_HOST_BUF=0             P4-R1 的修复开关，基线保持 0
#   MY_ENV_P3_PIN_ENGRAM=1                 P4-R2 的表 pinning，基线保持 1
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

# ---------------------------- 本次实验锁定 -----------------------------------
# 启动命令只需要一行（见文件头），以下取值就是本次 P3 的实验配置。
# 之所以把默认值写死在这里：云道「启动命令」表单里少一个字符出错就整作业失败，
# 而且 8 台机器必须完全一致 —— 靠命令行传参容易漏传（E39 就踩过这个坑）。
ARM=${MY_ENV_P3_ARM:-engram}                    # 主实验 arm；对照换 noengram
STEPS=${MY_ENV_P3_STEPS:-2}                     # 与 E1 同为 2 步
STALL_SEC=${MY_ENV_P3_STALL_SEC:-30}            # 心跳 i=N/1192 停 30 s 即判定卡住并抓栈
STACK_ROUNDS=${MY_ENV_P3_STACK_ROUNDS:-8}       # 每次卡住最多抓 8 轮原生栈
# 下面两个是 P4 修复开关，本次基线保持"原行为"
REUSE_HOST_BUF=${MY_ENV_P3_REUSE_HOST_BUF:-0}
PIN_ENGRAM=${MY_ENV_P3_PIN_ENGRAM:-1}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname)
LOG=$LOGDIR/dsv4-yundao_1106_p3_${ARM}_${TS}.log
CGLOG=$LOGDIR/dsv4-yundao_1106_p3_${ARM}_${TS}.cg.log
STKLOG=$LOGDIR/dsv4-yundao_1106_p3_${ARM}_${TS}.stacks.log
PROBE_DIR=$LOGDIR/rtprobe_1106_p3_${ARM}_${TS}
mkdir -p "$PROBE_DIR"

echo "[p3] MY_DEBUG_LOG p3_start host=$HOST arm=$ARM steps=$STEPS ts=$TS log=$LOG" | tee -a "$LOG"

# ---------------- 探针开关（必须在 import 之前进环境，Ray worker 继承）---------
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
# SYNC=0：不逐段 device 同步，保持 E1 的时序可比。
# 真实等待由 t_d2h_list_ms / t_buf_alloc_ms 捕获，不依赖 SYNC。
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=0
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=0

export MY_ENV_DSV41_OPT_STATE_PROBE=1
export MY_ENV_DSV41_OPT_STATE_DUMP_SEC=${STALL_SEC}
export MY_ENV_DSV41_OPT_STATE_PROBE_DIR=$PROBE_DIR
export VERL_DSV41_OPT_STATE_DEBUG=0          # 用新探针替代老钩子，避免两套日志混淆

export MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF=$REUSE_HOST_BUF
export MY_ENV_DSV41_ENGRAM_PIN_MEMORY=$PIN_ENGRAM
# 云道脚本里那个历史死开关也一起设上，保证两侧口径一致
export VERL_DSV41_ENGRAM_PIN_MEMORY=$PIN_ENGRAM

echo "[p3] MY_DEBUG_LOG probe_env fetch_probe=1 sync=0 opt_probe=1 reuse_buf=$MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF pin_engram=$MY_ENV_DSV41_ENGRAM_PIN_MEMORY dir=$PROBE_DIR" | tee -a "$LOG"

# ---------------- 探针 3：cgroup 上限 / 节流计数 / PSI ------------------------
# E1 的 memprobe 只记了 cg_cur 峰值 1328 GiB，**没记上限**，所以判断不了是否触顶。
# memory.events 的 limit_in_bytes/max 计数与 memory.pressure 的 full avg10 是直接证据。
cgroup_probe() {
  local CG1=/sys/fs/cgroup/memory CG2=/sys/fs/cgroup mode=none
  [ -f "$CG2/memory.max" ] && mode=v2
  [ "$mode" = none ] && [ -f "$CG1/memory.limit_in_bytes" ] && mode=v1
  while :; do
    if [ "$mode" = v2 ]; then
      echo "[cg-probe] $(date '+%H:%M:%S') mode=v2 max=$(cat $CG2/memory.max 2>/dev/null) cur=$(cat $CG2/memory.current 2>/dev/null) events=$(tr '\n' ',' < $CG2/memory.events 2>/dev/null) pressure_full=$(grep '^full' $CG2/memory.pressure 2>/dev/null)" >> "$CGLOG"
    elif [ "$mode" = v1 ]; then
      echo "[cg-probe] $(date '+%H:%M:%S') mode=v1 limit=$(cat $CG1/memory.limit_in_bytes 2>/dev/null) cur=$(cat $CG1/memory.usage_in_bytes 2>/dev/null) peak=$(cat $CG1/memory.max_usage_in_bytes 2>/dev/null) failcnt=$(cat $CG1/memory.failcnt 2>/dev/null) events=$(tr '\n' ',' < $CG1/memory.events 2>/dev/null) pressure_full=$(grep '^full' $CG1/memory.pressure 2>/dev/null)" >> "$CGLOG"
    else
      echo "[cg-probe] $(date '+%H:%M:%S') mode=none" >> "$CGLOG"
    fi
    awk -v p="[cg-probe] $(date '+%H:%M:%S') node" '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{printf "%s MemTotal=%dGiB MemFree=%dGiB MemAvail=%dGiB Cached=%dGiB Shmem=%dGiB\n",p,t/1048576,f/1048576,a/1048576,c/1048576,s/1048576}' /proc/meminfo >> "$CGLOG"
    sleep 20
  done
}
cgroup_probe &
CG_PID=$!

# ---------------- 探针 4：卡住时抓原生栈 --------------------------------------
# 触发：主日志里 [rt-probe] 的心跳 i=N/1192 连续 STALL_SEC 秒不变。
# 只在那时抓，避免平时把 /efs_rl 写爆。
stall_watch() {
  local last="" last_ts rounds=0
  last_ts=$(date +%s)
  while :; do
    local cur now
    cur=$(grep -ao "\[rt-probe\]\[rank[0-9-]*\] i=[0-9]*/" "$LOG" 2>/dev/null | tail -1)
    now=$(date +%s)
    if [ "$cur" != "$last" ]; then
      last="$cur"; last_ts=$now
      [ "$cur" != "" ] && rounds=0
    fi
    if [ -n "$cur" ] && [ $((now - last_ts)) -ge "$STALL_SEC" ] && [ "$rounds" -lt "$STACK_ROUNDS" ]; then
      rounds=$((rounds + 1))
      {
        echo "===== STALL host=$HOST at=$(date '+%F %T') heartbeat=${cur:-none} idle=$((now - last_ts))s round=$rounds ====="
        pw=$(pgrep -c -f "python3.11" 2>/dev/null || echo 0)
        echo "----- python_procs=$pw -----"
        for p in $(pgrep -f "python3.11" 2>/dev/null | head -6); do
          echo "----- pid=$p wchan=$(cat /proc/$p/wchan 2>/dev/null) -----"
          grep -E "^(VmPin|VmRSS|VmSize|Threads):" /proc/$p/status 2>/dev/null
          timeout 25 py-spy dump --native --pid "$p" 2>&1 | head -60
        done
      } >> "$STKLOG" 2>&1
      echo "[p3] MY_DEBUG_LOG stall_stack_dumped host=$HOST round=$rounds heartbeat=$cur idle=$((now - last_ts))s file=$STKLOG" >> "$LOG"
      last_ts=$now
    fi
    sleep 10
  done
}
stall_watch &
STK_PID=$!

trap 'kill $CG_PID $STK_PID 2>/dev/null || true' EXIT

# ---------------- 训练本体 ----------------------------------------------------
case "$ARM" in
  engram)   BASE=1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh ;;
  noengram) BASE=1106_run_dsv41_yundao_8node_noengram_cp1_fadvise_v1.sh ;;
  *) echo "[p3] ABORT: MY_ENV_P3_ARM=$ARM (expect engram|noengram)" | tee -a "$LOG" >&2; exit 2 ;;
esac
BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/$BASE
if [ ! -f "$BASE_PATH" ]; then
  echo "[p3] ABORT: base not found $BASE_PATH" | tee -a "$LOG" >&2
  exit 3
fi

echo "[p3] MY_DEBUG_LOG base_start base=$BASE time=$(date '+%F %T')" | tee -a "$LOG"
bash "$BASE_PATH" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[p3] MY_DEBUG_LOG p3_done arm=$ARM rc=$RC time=$(date '+%F %T') log=$LOG stacks=$STKLOG cg=$CGLOG" | tee -a "$LOG"
exit "$RC"
