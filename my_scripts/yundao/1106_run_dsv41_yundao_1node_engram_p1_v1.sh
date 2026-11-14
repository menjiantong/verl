#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次实跑（P3-单机：云道 1 机 × 16 卡 engram）
# =============================================================================
# 1106 P3-单机：云道 **1 机 × 16 卡**，Engram，四探针同开
#
# 基线（规则 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1106_run_dsv41_yundao_8node_engram_p1_v1.sh
#     → 1105_..._cp1_fadvise_v3.sh → 1105_..._cp1_v1.sh
#       → 1104_..._memprobe_v1.sh → 1104_..._noval_v2.sh
#         → 1104_..._recompute_full_v1.sh → 1103_..._engram_cp_v1.sh
#           → dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
#   本脚本相对 8 机 P1 只改四件事，不碰训练数学：
#     1) 拓扑 8 机×16 卡=128 rank → **1 机×16 卡=16 rank**（与 124 完全一致：
#        FSDP16 / EP16 / GEN_TP8 × GEN_DP2 / GEN_EP16 / CP1 / TRAIN_BSZ=16）；
#     2) **权重换成 top32**（32 专家 / 493.4 GiB），与 124 同一份模型；
#        8 机 P1 与 1103 默认用的是 384 专家 / 1421.7 GiB，二者不可混用（见 §二）；
#     3) 数据集 → dapo_math_train_512 + dapo_math_val_64（与 124 P1 那份逐字节同一文件，
#        云道上已有同名副本 /efs_rl/m00899630/dsv41/rl_data/）；
#     4) FREE_CACHE_ENGINE=False → vLLM 常驻，与 124 P1 同口径（也绕开 camem.sleep
#        的 rtMallocHost 卡死，否则 sleep 会变成主导变量）。
#   探针（engram 分段 / 优化器逐参数 / cgroup / 卡住抓栈）全部沿用 8 机 P1 那套。
#
# -----------------------------------------------------------------------------
# 一、为什么需要这一轮（要回答的问题）
# -----------------------------------------------------------------------------
#   已有的四个数（update_actor，单步）：
#     云道 8 机 EP128 engram   : 1937.9 s        (E1，2 步)
#     云道 8 机 EP128 noengram :   74.3 s        (E2)
#     124  1 机 EP16  engram   : 106.7 / 120.1 / 101.6 s   (P1，3 步)
#     124  1 机 EP16  noengram :  67.5 /  67.6 /  68.0 s   (P1，3 步)
#   124 上 engram/noengram = 1.62×；云道 8 机是 26.1×。**差 16 倍。**
#   这 16 倍只可能来自两件事，本轮把 B 单独摘出来：
#     A) 规模：EP128 跨 8 机（跨机 all_to_all + 每节点 8 个 vLLM worker 抢主机内存）
#     B) 宿主：单个云道节点的主机侧本身就比 124 慢（驱动 / 内存 / 分配器）
#   1 机 × 16 卡、EP16、同一份代码、同一份 dapo512：
#     update_actor ≈ 100 s 量级 → 排除 B，16 倍来自规模（A）；
#     update_actor ≈ 1900 s 量级 → B 成立，单个云道节点本身就有重尾。
#
# -----------------------------------------------------------------------------
# 二、内存可行性（本轮唯一有风险的地方，先说清楚依据）
# -----------------------------------------------------------------------------
#   [1106 修正] 权重用 top32：总量 493.4 GiB（6 256 tensors），其中 engram 表
#   2×183.1 GiB、非 engram 只有 127 GiB。对比 1103 默认的 384 专家版 1421.7 GiB，
#   单机内存压力小 2.9×，1893 GiB 的上限绰绰有余。
#   124 实测（同 top32 权重）整机占用 1400 GiB 里大头是 vLLM 与页缓存，不是权重。
#
#   1 机装下整个模型靠的是 FSDP-Turbo 的 CPU offload，不是 HBM。
#   124 实测（2026-11-06 02:52 那一轮，engram arm，logs/1106_nosleep_1node_engram_*.mem.log）：
#     MemTotal 2454 GiB；MemAvailable 2304 GiB → 最低 870 GiB ⇒ **整机实际占用 ≈ 1400 GiB**
#     （其中 Shmem 峰值 479 GiB = vLLM engram 共享表 366 + 原有 112）
#     actor/perf/max_memory_allocated_gb = 16.5 GB/rank ⇒ HBM 只放当前层
#   云道作业容器 cgroup 上限 = 1893 GiB（8 机作业实测 memory.limit_in_bytes=2032999997440，
#     见 1106 P3 的 *.cg.log）。1893 > 1400，余量 ~490 GiB；
#     且 cgroup 里的**文件页可回收**，1105 v3 的 fadvise 守护会持续放掉读 checkpoint 的干净页。
#   ⇒ 判定可行。若仍撞上限：日志会先出现 MemFree 快速下滑 + cg 逼近 limit，
#     此时用 MY_ENV_P3_RESIDENT=0 重跑（恢复 FREE_CACHE_ENGINE=True，vLLM 每步 sleep），
#     但要接受 camem.sleep 的 rtMallocHost 卡死成为新变量。
#
#   ⚠️ 与 1107_run_dsv41_yundao_1node_engram_probe_v1.sh 头部"单机跑不了训练"的结论相反：
#   那条是按 **HBM** 算的（1.39 TiB > 单机 16 卡 1.02 TiB），忽略了 CPU offload。
#   按主机内存算，124 已用同一拓扑、同一份权重跑通 3 步（2026-11-06 02:52）。
#
# -----------------------------------------------------------------------------
# 三、用法（云道控制台：资源规格 **1 机 × 16 卡**，启动命令只填这一行）
# -----------------------------------------------------------------------------
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
#
#   提交前确认：MA_NUM_HOSTS 必须 = 1，否则基线脚本第 281 行直接 ERROR 退出。
#
#   可覆盖开关（规则 6.1，全部有默认值，默认即本次实验）：
#     MY_ENV_P3_ARM=engram|noengram   默认 engram
#     MY_ENV_P3_STEPS=3               默认 3 步（与 124 P1 的 3 步对齐）
#     MY_ENV_P3_DATASET=dapo512|dapo17k  默认 dapo512（与 124 P1 同文件）
#     MY_ENV_P3_RESIDENT=1            1=vLLM 常驻（FREE_CACHE_ENGINE=False，与 124 同）
#     MY_ENV_P3_STALL_SEC=30          心跳停多久算卡住
#     MY_ENV_P3_STACK_ROUNDS=8        每次卡住最多抓几轮原生栈
#     MY_ENV_P3_REUSE_HOST_BUF=0      P4-R1 修复开关，基线 0
#     MY_ENV_P3_PIN_ENGRAM=1          P4-R2 表 pinning，基线 1（与 124 同）
#
#   回滚：8 机跑 1106_run_dsv41_yundao_8node_engram_p1_v1.sh。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

# ---------------------------- 本次实验锁定 -----------------------------------
ARM=${MY_ENV_P3_ARM:-engram}
STEPS=${MY_ENV_P3_STEPS:-3}
DATASET=${MY_ENV_P3_DATASET:-dapo512}
RESIDENT=${MY_ENV_P3_RESIDENT:-1}
STALL_SEC=${MY_ENV_P3_STALL_SEC:-30}
STACK_ROUNDS=${MY_ENV_P3_STACK_ROUNDS:-8}
REUSE_HOST_BUF=${MY_ENV_P3_REUSE_HOST_BUF:-0}
PIN_ENGRAM=${MY_ENV_P3_PIN_ENGRAM:-1}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
LOG=$LOGDIR/dsv4-yundao_1106_p3s_${ARM}_${TS}.log
CGLOG=$LOGDIR/dsv4-yundao_1106_p3s_${ARM}_${TS}.cg.log
STKLOG=$LOGDIR/dsv4-yundao_1106_p3s_${ARM}_${TS}.stacks.log
PROBE_DIR=$LOGDIR/rtprobe_1106_p3s_${ARM}_${TS}
mkdir -p "$PROBE_DIR"

echo "[p3s] MY_DEBUG_LOG p3s_start host=$HOST arm=$ARM steps=$STEPS dataset=$DATASET resident=$RESIDENT MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} VC_TASK_INDEX=${VC_TASK_INDEX:-unset} log=$LOG" | tee -a "$LOG"

# ---------------- 拓扑：1 机 × 16 卡 = 16 rank（逐项对齐 124 的 P1）----------
# 8 机基线是 FSDP128 / EP128 / GEN_DP16；这里全部改成 16 卡单机形态。
# 反向思考：EP16 会让每 rank 的 engram 表从 2.9 GiB 涨到 24.6 GiB（整张表 393 GiB
# 落在这一台机器上），这正是"1 机 vs 8 机"内存口径不同的唯一来源，必须在报告里写明。
export MY_ENV_DSV41_NNODES=1
export MY_ENV_DSV41_NDEVICES_PER_NODE=16
export MY_ENV_DSV41_FSDP_SIZE=16
export MY_ENV_DSV41_EP_SIZE=16
export MY_ENV_DSV41_GEN_TP=8
export MY_ENV_DSV41_GEN_DP=2
export MY_ENV_DSV41_GEN_EP=16
export MY_ENV_DSV41_TRAIN_BSZ=16
export MY_ENV_DSV41_CP_SIZE=1

# ---------------- 训练节奏：3 步、不存盘、不评估（与 124 P1 同）-------------
export MY_ENV_DSV41_STEPS=$STEPS
export MY_ENV_DSV41_SAVE_FREQ=-1
export MY_ENV_DSV41_TEST_FREQ=-1
export MY_ENV_DSV41_VAL_BEFORE_TRAIN=False

# ---------------- 数据集：dapo512（与 124 P1 逐字节同一份文件）---------------
# 为什么不用全量 dapo-math-17k：124 的 P1 用的是 512 条子集，engram 表的 uniq 行数
# 直接决定 all_to_all 载荷与 CPU gather 规模，两边不同就没法比较 update_actor。
case "$DATASET" in
  dapo512)
    export MY_ENV_DSV41_TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_train_512.parquet
    export MY_ENV_DSV41_TEST_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
    ;;
  dapo17k)
    export MY_ENV_DSV41_TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
    export MY_ENV_DSV41_TEST_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
    ;;
  *) echo "[p3s] ABORT: MY_ENV_P3_DATASET=$DATASET (expect dapo512|dapo17k)" | tee -a "$LOG" >&2; exit 2 ;;
esac

# ---------------- 引擎：vLLM 常驻（与 124 P1 的 free_cache_engine=False 同）---
if [ "$RESIDENT" = "1" ]; then
  export MY_ENV_DSV41_FREE_CACHE_ENGINE=False
else
  export MY_ENV_DSV41_FREE_CACHE_ENGINE=True
fi

# ---------------- 对照 arm：noengram 只换权重，其余逐项不变 -----------------
# [1106 修正] 权重必须用 **top32**（32 专家，493.4 GiB），不能用 1103 默认的
# /efs_rl/weights/DeepSeek-V4.1-Flash-bf16（384 专家，1421.7 GiB）。
# 依据：124 单机跑的是 /mnt/share/.../DeepSeek-V4.1-Flash-top32-bf16，
#   两者 engram 表完全相同（2×183.1 GiB），但非 engram 部分差 8.3×（127 vs 1055 GiB），
#   专家数差 12×（32 vs 384）。用 384 专家版做"单机 vs 124"的对照是错的：
#   E1 里最慢的形状 DTensor(384, 4608, 5120) 第一维就是专家数，top32 上不存在。
if [ "$ARM" = "noengram" ]; then
  export MY_ENV_DSV41_MODEL_OVERRIDE=${MY_ENV_P3_MODEL:-/efs_rl/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram}
  export MY_ENV_DSV41_REQUIRE_ENGRAM=0
  export MY_ENV_DSV41_ENGRAM_DP_SHARED=0
elif [ "$ARM" = "engram" ]; then
  export MY_ENV_DSV41_MODEL_OVERRIDE=${MY_ENV_P3_MODEL:-/efs_rl/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
elif [ "$ARM" != "engram" ]; then
  echo "[p3s] ABORT: MY_ENV_P3_ARM=$ARM (expect engram|noengram)" | tee -a "$LOG" >&2
  exit 2
fi

# [1106] 1103 的预检会断言 n_routed_experts：top32 是 32，默认仍 384。
# 不设时 1103 用 384，行为与 8 机路径完全一致。
export MY_ENV_DSV41_EXPECT_EXPERTS=${MY_ENV_P3_EXPERTS:-32}
# [1106] fadvise 守护要指向**真正在用**的权重目录：1105 v3 的默认值是
# /efs_rl/weights/DeepSeek-V4.1-Flash-bf16（384 专家那份）。单机用 top32 时若不改，
# 守护会去放另一份模型的页缓存 —— 无害但完全无用，还会让"回收了多少页"的日志误导。
export MY_ENV_DSV41_FADVISE_DIR=$MY_ENV_DSV41_MODEL_OVERRIDE

# ---------------- 探针开关（必须在 import 之前进环境，Ray worker 继承）-------
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=0
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=0

export MY_ENV_DSV41_OPT_STATE_PROBE=1
export MY_ENV_DSV41_OPT_STATE_DUMP_SEC=${STALL_SEC}
export MY_ENV_DSV41_OPT_STATE_PROBE_DIR=$PROBE_DIR
export VERL_DSV41_OPT_STATE_DEBUG=0

export MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF=$REUSE_HOST_BUF
export MY_ENV_DSV41_ENGRAM_PIN_MEMORY=$PIN_ENGRAM
export VERL_DSV41_ENGRAM_PIN_MEMORY=$PIN_ENGRAM

echo "[p3s] MY_DEBUG_LOG probe_env fetch_probe=1 sync=0 opt_probe=1 reuse_buf=$MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF pin_engram=$MY_ENV_DSV41_ENGRAM_PIN_MEMORY dir=$PROBE_DIR" | tee -a "$LOG"
echo "[p3s] MY_DEBUG_LOG topo world=16 fsdp=16 ep=16 gen_tp=8 gen_dp=2 gen_ep=16 bsz=16 cp=1 free_cache_engine=$MY_ENV_DSV41_FREE_CACHE_ENGINE train_file=$MY_ENV_DSV41_TRAIN_FILE" | tee -a "$LOG"

# ---------------- 探针 3：cgroup 上限 / 节流计数 / 内存构成 ------------------
cgroup_probe() {
  local CG1=/sys/fs/cgroup/memory CG2=/sys/fs/cgroup mode=none
  [ -f "$CG2/memory.max" ] && mode=v2
  [ "$mode" = none ] && [ -f "$CG1/memory.limit_in_bytes" ] && mode=v1
  while :; do
    if [ "$mode" = v2 ]; then
      echo "[cg-probe] $(date '+%H:%M:%S') mode=v2 max=$(cat $CG2/memory.max 2>/dev/null) cur=$(cat $CG2/memory.current 2>/dev/null) events=$(tr '\n' ',' < $CG2/memory.events 2>/dev/null)" >> "$CGLOG"
    elif [ "$mode" = v1 ]; then
      echo "[cg-probe] $(date '+%H:%M:%S') mode=v1 limit=$(cat $CG1/memory.limit_in_bytes 2>/dev/null) cur=$(cat $CG1/memory.usage_in_bytes 2>/dev/null) peak=$(cat $CG1/memory.max_usage_in_bytes 2>/dev/null) failcnt=$(cat $CG1/memory.failcnt 2>/dev/null) events=$(tr '\n' ',' < $CG1/memory.events 2>/dev/null)" >> "$CGLOG"
    else
      echo "[cg-probe] $(date '+%H:%M:%S') mode=none" >> "$CGLOG"
    fi
    awk -v p="[cg-probe] $(date '+%H:%M:%S') node" '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{printf "%s MemTotal=%dGiB MemFree=%dGiB MemAvail=%dGiB Cached=%dGiB Shmem=%dGiB\n",p,t/1048576,f/1048576,a/1048576,c/1048576,s/1048576}' /proc/meminfo >> "$CGLOG"
    sleep 20
  done
}
cgroup_probe &
CG_PID=$!

# ---------------- 探针 4：卡住时抓原生栈 ------------------------------------
# 触发：主日志里 [rt-probe] 的心跳 i=N/1192 连续 STALL_SEC 秒不变。
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
      echo "[p3s] MY_DEBUG_LOG stall_stack_dumped host=$HOST round=$rounds heartbeat=$cur idle=$((now - last_ts))s file=$STKLOG" >> "$LOG"
      last_ts=$now
    fi
    sleep 10
  done
}
stall_watch &
STK_PID=$!

trap 'kill $CG_PID $STK_PID 2>/dev/null || true' EXIT

# ---------------- 训练本体 ---------------------------------------------------
# 走 8 机 P1 的同一条链（fadvise 守护 + noval + recompute_full + engram），
# 拓扑与数据集已由上面的 export 覆盖，链上不再重复设。
BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
if [ ! -f "$BASE_PATH" ]; then
  echo "[p3s] ABORT: base not found $BASE_PATH" | tee -a "$LOG" >&2
  exit 3
fi

echo "[p3s] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$LOG"
# [1106 修复] 这里原来用 `| tee -a "$LOG"`。问题：整条链（1104 memprobe / 1105 fadvise）
# 会起**后台守护进程**，它们继承同一个 stdout 管道；主脚本失败退出后守护仍在跑，
# 管道不关，tee 就永远不返回，作业容器于是卡在"运行中"却什么都不做
# （2026-11-06 12:11 那次预检失败就是这样变成僵尸的，p3s_done 一直不出现）。
# 改成直接重定向：日志照样实时落盘，但 wrapper 只等 base 本身返回。
bash "$BASE_PATH" >> "$LOG" 2>&1
RC=$?
echo "[p3s] MY_DEBUG_LOG p3s_done arm=$ARM rc=$RC time=$(date '+%F %T') log=$LOG stacks=$STKLOG cg=$CGLOG" | tee -a "$LOG"
exit "$RC"
