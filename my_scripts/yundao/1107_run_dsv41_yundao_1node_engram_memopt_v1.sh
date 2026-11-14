#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-07 由用户提交（云道控制台：资源规格 **1 机 × 16 卡**）
# =============================================================================
# 1107 单机 engram 内存归因 A/B（E1）：base / nopin / noengram
#
# 分支 / commit（规则 7.5）：
#   verl        dev_dsv41_rebase1030_yundao @ 10fdecf7
#   vllm-ascend from_9a38d36_1030           @ ae328b7f8
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#
# 基线（规则 7.3/7.4：本脚本只写差异，训练数学与拓扑一律不动）：
#   my_scripts/yundao/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
#     → 1106_8node_engram_p1 → 1105_cp1_fadvise_v3 → … → dsv41_.._final.sh
#
# 本脚本相对基线只加三件事：
#   1) 用 MY_ENV_P5_RUNG 选档（见下）；
#   2) 拉起 1107_probe_mem_account_v2.sh（全字段 meminfo / memory.stat /
#      per-NUMA / 进程 VmPin / smaps top-3 / /dev/shm，只读）；
#   3) 跑完自动调 1107_memopt_report_v1.py，把节点账、隐藏量、rt-probe、
#      engram-probe 打成一张表。
#
# 档位（同一套代码、同一份 top32-bf16、同样 3 步；只改 engram 表 pinning）：
#   base     engram + PIN_ENGRAM=1，复现当前配置，带 v2 探针。
#            预期：隐藏量 ~1200 GiB，MemAvail=0，update_actor 状态创建 >2000 s。
#   nopin    engram + PIN_ENGRAM=0，表改 pageable。
#            目的：判断隐藏的 732 GiB 是否随 pinning 移动；它是诊断档，不是修复档
#            （pageable 匿名页无 swap，仍然不可回收，总内存不变）。
#   noengram 同拓扑只换 top32-noengram 权重。预期 3/3 通过，隐藏量 ~166 GiB，
#            用来标定基线残余。
#
# 反向思考（为什么不在这一轮直接上"ref 共用表"）：
#   ref 共用需要改 FSDPTurbo 的 HostOffloadEngram.shard_ 与 verl 的 actor/ref
#   build 顺序，属于特性改动（规则 6.7 要单独子任务、单测、集成验证）。
#   本轮先用零代码改动的 nopin 把 732 GiB / 304 GiB 的归属钉死，再决定 P0 的实现面。
#
# 用法（云道控制台启动命令只填这一行，档位用前缀环境变量给）：
#   MY_ENV_P5_RUNG=nopin    bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1107_run_dsv41_yundao_1node_engram_memopt_v1.sh
#   MY_ENV_P5_RUNG=base     bash …/1107_run_dsv41_yundao_1node_engram_memopt_v1.sh
#   MY_ENV_P5_RUNG=noengram bash …/1107_run_dsv41_yundao_1node_engram_memopt_v1.sh
#
# 2026-11-06 p4 三档已经跑过 noengram(rc=0) / engram(rc=1) / relief(rc=1)，
# 所以本轮默认档改为 nopin：它是唯一还没跑、且能回答"残留 304 GiB"的档。
# base 只在需要 v2 全字段（Mlocked/PageTables/NUMA）时补跑；noengram 已有 3/3 通过样本。
#
# 提交前确认：MA_NUM_HOSTS 必须 = 1（基线脚本会自己校验，不符直接 ERROR 退出）。
# nopin 预计 1.5–2 h 内出结论；不要与其它档并行抢同一批节点。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
SCRIPTS=$MIRROR_ROOT/verl/my_scripts/yundao
mkdir -p "$LOGDIR" 2>/dev/null || true

RUNG=${MY_ENV_P5_RUNG:-nopin}
STEPS=${MY_ENV_P5_STEPS:-3}
TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
LOG=$LOGDIR/dsv4-yundao_1107_p5_${RUNG}_${TS}.log
T0=$(date +%s)

echo "[p5] MY_DEBUG_LOG p5_start host=$HOST rung=$RUNG steps=$STEPS branch=dev_dsv41_rebase1030_yundao@10fdecf7 MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} VC_TASK_INDEX=${VC_TASK_INDEX:-unset} log=$LOG" | tee -a "$LOG"

# ---------------------------- 档位 → 基线开关 ---------------------------------
case "$RUNG" in
  base)
    export MY_ENV_P3_ARM=engram
    export MY_ENV_P3_RESIDENT=1
    export MY_ENV_P3_PIN_ENGRAM=1
    ;;
  nopin)
    export MY_ENV_P3_ARM=engram
    export MY_ENV_P3_RESIDENT=1
    export MY_ENV_P3_PIN_ENGRAM=0
    ;;
  noengram)
    export MY_ENV_P3_ARM=noengram
    export MY_ENV_P3_RESIDENT=1
    export MY_ENV_P3_PIN_ENGRAM=1
    ;;
  *)
    echo "[p5] ABORT: MY_ENV_P5_RUNG=$RUNG (expect base|nopin|noengram)" | tee -a "$LOG" >&2
    exit 2
    ;;
esac
export MY_ENV_P3_STEPS=$STEPS

echo "[p5] MY_DEBUG_LOG p5_knobs arm=$MY_ENV_P3_ARM resident=$MY_ENV_P3_RESIDENT pin_engram=$MY_ENV_P3_PIN_ENGRAM steps=$STEPS" | tee -a "$LOG"

# ---------------------------- 预检（只读） ------------------------------------
# 与基线自己的 host_mem_check 不重复：这里只把 v2 探针关心的量先打一行，便于事后对表。
awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{printf "[p5] MY_DEBUG_LOG p5_preflight MemTotal=%dGiB MemFree=%dGiB MemAvail=%dGiB Cached=%dGiB Shmem=%dGiB\n",t/1048576,f/1048576,a/1048576,c/1048576,s/1048576}' /proc/meminfo 2>/dev/null | tee -a "$LOG"
echo "[p5] MY_DEBUG_LOG p5_shm_preflight $(df -P /dev/shm 2>/dev/null | awk 'NR==2{printf "used=%dGiB total=%dGiB",$3/1048576,$2/1048576}')" | tee -a "$LOG"

# ---------------------------- 内存账探针 v2 -----------------------------------
PROBE=$SCRIPTS/1107_probe_mem_account_v2.sh
MEM_PID=""
if [ -f "$PROBE" ]; then
  bash "$PROBE" "$LOGDIR" &
  MEM_PID=$!
  echo "[p5] MY_DEBUG_LOG memacct_v2_probe_start pid=$MEM_PID script=$PROBE" | tee -a "$LOG"
else
  echo "[p5] WARN: memacct v2 probe not found ($PROBE) —— 本次没有全字段内存分解" | tee -a "$LOG"
fi

cleanup() {
  if [ -n "${MEM_PID:-}" ]; then
    kill "$MEM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---------------------------- 训练本体 ---------------------------------------
BASE=$SCRIPTS/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
if [ ! -f "$BASE" ]; then
  echo "[p5] ABORT: base not found $BASE" | tee -a "$LOG" >&2
  exit 3
fi

# 反向思考：这里**不加** `| tee`。1106 基线链会起 fadvise/memprobe 后台守护，
# 它们继承同一个 stdout；主进程先退出、守护还在时管道不关，容器会卡成
# "运行中但不干活"的僵尸（1106 脚本第 260-264 行记的就是这个坑）。
echo "[p5] MY_DEBUG_LOG p5_base_start base=$(basename "$BASE") time=$(date '+%F %T')" | tee -a "$LOG"
bash "$BASE" >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
echo "[p5] MY_DEBUG_LOG p5_done rung=$RUNG rc=$RC elapsed=$((T1-T0))s time=$(date '+%F %T')" | tee -a "$LOG"

# ---------------------------- 结果汇总 ---------------------------------------
# 1106 基线把主日志写成 dsv4-yundao_1106_p3s_<arm>_<base_ts>.log；取本次启动后
# 最新的一份（排除 .cg/.stacks）。用 T0 而不是字符串时间，避免包装脚本与基线的 TS 差一秒。
TARGET=$(find "$LOGDIR" -maxdepth 1 -name 'dsv4-yundao_1106_p3s_*.log' \
  ! -name '*.cg.log' ! -name '*.stacks.log' -newermt "@$T0" -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | awk '{print $2}')
if [ -z "${TARGET:-}" ]; then
  TARGET=$(ls -t "$LOGDIR"/dsv4-yundao_1106_p3s_*.log 2>/dev/null | grep -vE '\.(cg|stacks)\.log$' | head -1)
fi

REPORT=$LOGDIR/report_1107_p5_${RUNG}_${TS}.txt
if [ -n "${TARGET:-}" ] && [ -f "$SCRIPTS/1107_memopt_report_v1.py" ]; then
  python3 "$SCRIPTS/1107_memopt_report_v1.py" "$TARGET" > "$REPORT" 2>&1
  echo "[p5] MY_DEBUG_LOG p5_report_done rung=$RUNG src=$TARGET out=$REPORT" | tee -a "$LOG"
  cat "$REPORT"
else
  echo "[p5] WARN: report skipped (target=${TARGET:-none})" | tee -a "$LOG"
fi

exit "$RC"
