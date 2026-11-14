#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-07 由用户提交（云道控制台：资源规格 **1 机 × 16 卡**）
# =============================================================================
# 1107 单机（1 机 × 16 卡）对齐 124 —— 统一启动器
#
# 分支 / commit（规则 7.5）：
#   verl        dev_dsv41_rebase1030_yundao @ 10fdecf7
#   vllm-ascend from_9a38d36_1030           @ ae328b7f8
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#
# 基线（规则 7.3/7.4：本脚本只写差异，训练数学与拓扑一律不动）：
#   my_scripts/yundao/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
#     → 1106_8node_engram_p1 → 1105_cp1_fadvise_v3 → … → 1103 → dsv41_.._final.sh
#
# 本脚本相对基线只加三件事：
#   1) 用 MY_ENV_P4_RUNG 选档（见下）；
#   2) 拉起 1107 内存账探针（节点 / cgroup 树 / 进程 RSS / /dev/shm 四路分解，
#      只读，用来回答"上次那 1140 GiB 到底在谁身上"）；
#   3) 跑完自动调 1107_align_report_v1.py，把结果与 124 的 P1 基线并排打成表。
#
# 档位（同一套代码、同一份 dapo512、同样 3 步；只改内存口径）：
#   noengram  同拓扑只把权重换成 top32-noengram。**先跑这个**：
#             过了 ⇒ 容器装得下这套训练，问题在 engram；
#             没过 ⇒ 问题与 engram 无关，是容器/拓扑本身，后面的档位都不用试。
#   engram    与 1106 逐项相同（124 口径：FREE_CACHE_ENGINE=False，vLLM 常驻）。
#             这一档的价值在于**配上增强探针再复现一次失败**，拿到内存分解。
#   relief    engram + 三个不改数值语义的内存开关：
#               MY_ENV_P3_REUSE_HOST_BUF=1        复用 pinned 中转缓冲（默认 0）
#               MY_ENV_VLLM_SLEEP_PIN_MEMORY=0    万一走 sleep，备份用 pageable
#               MY_ENV_DSV41_FADVISE_FREE_GIB=768 页缓存更早回收（默认 256）
#
# 反向思考（为什么不是"直接把 engram 表塞进显存"）：
#   cpu_offload=False 要求 dp_shared_memory=False（vllm/config/engram.py:59 的校验），
#   单机 TP8×DP2 就会各持一份 TP 分片 = 2 副本 × 366.2 GiB = **732 GiB HBM**，
#   16 卡共 1024 GiB，再叠 vLLM 权重 493 GiB 直接爆。所以这条路上不了，本脚本不提供。
#
# 用法（云道控制台启动命令只填这一行，档位用前缀环境变量给）：
#   1) 第一发：  MY_ENV_P4_RUNG=noengram bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1107_run_dsv41_yundao_1node_align_v1.sh
#   2) 第二发：  MY_ENV_P4_RUNG=engram   bash …/1107_run_dsv41_yundao_1node_align_v1.sh
#   3) 第三发：  MY_ENV_P4_RUNG=relief   bash …/1107_run_dsv41_yundao_1node_align_v1.sh
#
# 提交前确认：MA_NUM_HOSTS 必须 = 1（基线脚本会自己校验，不符直接 ERROR 退出）。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
SCRIPTS=$MIRROR_ROOT/verl/my_scripts/yundao
mkdir -p "$LOGDIR" 2>/dev/null || true

RUNG=${MY_ENV_P4_RUNG:-noengram}
STEPS=${MY_ENV_P4_STEPS:-3}
TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
LOG=$LOGDIR/dsv4-yundao_1107_p4_${RUNG}_${TS}.log

echo "[p4] MY_DEBUG_LOG p4_start host=$HOST rung=$RUNG steps=$STEPS branch=dev_dsv41_rebase1030_yundao@10fdecf7 MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} VC_TASK_INDEX=${VC_TASK_INDEX:-unset} log=$LOG" | tee -a "$LOG"

# ---------------------------- 档位 → 基线开关 ---------------------------------
case "$RUNG" in
  noengram)
    export MY_ENV_P3_ARM=noengram
    export MY_ENV_P3_RESIDENT=1
    ;;
  engram)
    export MY_ENV_P3_ARM=engram
    export MY_ENV_P3_RESIDENT=1
    ;;
  relief)
    export MY_ENV_P3_ARM=engram
    export MY_ENV_P3_RESIDENT=1
    export MY_ENV_P3_REUSE_HOST_BUF=1
    export MY_ENV_VLLM_SLEEP_PIN_MEMORY=0
    export MY_ENV_DSV41_FADVISE_FREE_GIB=768
    ;;
  *)
    echo "[p4] ABORT: MY_ENV_P4_RUNG=$RUNG (expect noengram|engram|relief)" | tee -a "$LOG" >&2
    exit 2
    ;;
esac
export MY_ENV_P3_STEPS=$STEPS

echo "[p4] MY_DEBUG_LOG p4_knobs arm=$MY_ENV_P3_ARM resident=$MY_ENV_P3_RESIDENT reuse_buf=${MY_ENV_P3_REUSE_HOST_BUF:-0} sleep_pin=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-1} fadvise_free_gib=${MY_ENV_DSV41_FADVISE_FREE_GIB:-256}" | tee -a "$LOG"

# ---------------------------- 内存账探针 --------------------------------------
PROBE=$SCRIPTS/1107_probe_mem_account_v1.sh
MEM_PID=""
if [ -f "$PROBE" ]; then
  bash "$PROBE" "$LOGDIR" &
  MEM_PID=$!
  echo "[p4] MY_DEBUG_LOG memacct_start pid=$MEM_PID script=$PROBE" | tee -a "$LOG"
else
  echo "[p4] WARN: memacct probe not found ($PROBE) —— 本次没有进程级内存分解" | tee -a "$LOG"
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
  echo "[p4] ABORT: base not found $BASE" | tee -a "$LOG" >&2
  exit 3
fi

# 反向思考：这里**不加** `| tee`。基线链会起后台守护，它们继承同一个 stdout，
# 一旦主进程先退出、守护还在，管道不关闭，容器会卡成"运行中但不干活"的僵尸
# （1106 脚本第 260-264 行记的就是这个坑）。直接继承 stdout，让它自然返回。
echo "[p4] MY_DEBUG_LOG base_start base=$(basename "$BASE") time=$(date '+%F %T')" | tee -a "$LOG"
T0=$(date +%s)
bash "$BASE"
RC=$?
T1=$(date +%s)
echo "[p4] MY_DEBUG_LOG p4_done rung=$RUNG rc=$RC elapsed=$((T1-T0))s time=$(date '+%F %T')" | tee -a "$LOG"

# ---------------------------- 结果汇总 ---------------------------------------
# 基线把日志写到 dsv4-yundao_1106_p3s_<arm>_<ts>.log，取本次新生成的那一份。
TARGET=$(ls -t "$LOGDIR"/dsv4-yundao_1106_p3s_*_"${TS}".log 2>/dev/null | head -1)
if [ -z "${TARGET:-}" ]; then
  TARGET=$(ls -t "$LOGDIR"/dsv4-yundao_1106_p3s_*.log 2>/dev/null | grep -vE '\.(cg|stacks)\.log$' | head -1)
fi

REPORT=$LOGDIR/report_1107_${RUNG}_${TS}.txt
if [ -n "${TARGET:-}" ] && [ -f "$SCRIPTS/1107_align_report_v1.py" ]; then
  python3 "$SCRIPTS/1107_align_report_v1.py" "$TARGET" > "$REPORT" 2>&1
  echo "[p4] MY_DEBUG_LOG report_done rung=$RUNG src=$TARGET out=$REPORT" | tee -a "$LOG"
  cat "$REPORT"
else
  echo "[p4] WARN: report skipped (target=${TARGET:-none})" | tee -a "$LOG"
fi

exit "$RC"
