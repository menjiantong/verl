#!/usr/bin/env bash
# =============================================================================
# 1105 对单个 profiling rank 目录做离线文本解析（v1）
#
# 用途：1105_run_dsv41_profiling_engram_v1.sh 采到的每个 *_ascend_pt 目录都是
#   独立的 rank profiling 目录，用本脚本按需补导出 ASCEND_PROFILER_OUTPUT/*.csv。
#
# 为什么需要它：采集时 NPU profiler 用的是 ExportType.Db，落盘只有 .db；
#   rollout 侧退出时也提示需要离线解析。要 kernel_details.csv 必须再用
#   export_type='text' 解析一次（小写，torch_npu 2.10 的 Constant.Text == 'text'；
#   传 'Text' 会被判定非法并静默回退成采集时的 'db'）。
#
# 用法：bash my_scripts/pref_dsv41/1105_analyse_rank_dir_v1.sh <rank 目录> [tag]
#   <rank 目录>：形如 .../train/node-xxx_<pid>_<ts>_ascend_pt 或
#                 .../agent_loop_rollout_replica_0/dp0_pp0_tp1_..._rank1_<pid>_<ts>_ascend_pt
#   [tag]：日志里的标识，默认取目录名里的 rank 信息
#
# 安全：不删除文件；analyse 只在 <rank 目录>/ASCEND_PROFILER_OUTPUT 内追加产物。
# 注意：解析器会清理 ASCEND_PROFILER_OUTPUT 下的旧 db 并重新生成，属其正常行为。
#
# 日志：logs/1105_analyse_rank_<tag>_<TS>.log
# 脚本 commit：verl a7b9eac3（首次提交）
# =============================================================================
# [1105 修复] 不能用 set -u：source 的链路里 /etc/bash.bashrc:7 引用未定义的 PS1，
# 非交互 shell 下 set -u 会让脚本在 source 处静默退出。
set -o pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO" || exit 1

RANK_DIR=${1:?用法: $0 <rank 目录> [tag]}
RANK_DIR=${RANK_DIR%/}
TAG=${2:-$(basename "$RANK_DIR" | grep -o "rank[0-9]*\|node-[0-9-]*" | head -1)}
[ -d "$RANK_DIR" ] || { echo "[analyse-rank] ABORT: 目录不存在 $RANK_DIR" >&2; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1105_analyse_rank_${TAG}_${TS}.log"
echo "[analyse-rank] dir=$RANK_DIR" | tee "$LOG"
echo "[analyse-rank] log=$LOG" | tee -a "$LOG"

# [1105 修复] 不把 source 输出丢到 /dev/null，否则错误会被吞掉
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

python3 -u - "$RANK_DIR" "$TAG" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, time
from torch_npu.profiler.profiler import analyse

rank_dir, tag = sys.argv[1], sys.argv[2]
out = os.path.join(rank_dir, "ASCEND_PROFILER_OUTPUT")
kd = os.path.join(out, "kernel_details.csv")
if os.path.isfile(kd) and os.path.getsize(kd) > 0:
    print(f"[analyse-rank] {tag}: 已存在 kernel_details.csv ({os.path.getsize(kd)} B)，跳过")
    sys.exit(0)

t0 = time.time()
print(f"[analyse-rank] {tag}: 开始离线 analyse(export_type='text') -> {rank_dir}")
analyse(rank_dir, max_process_number=32, export_type="text")
dt = time.time() - t0
ok = os.path.isfile(kd) and os.path.getsize(kd) > 0
print(f"[analyse-rank] {tag}: 用时 {dt:.1f}s, kernel_details.csv={'OK' if ok else 'MISSING'}")
if os.path.isdir(out):
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p):
            print(f"[analyse-rank] {tag}:   {f}  {os.path.getsize(p)} B")
print(f"MY_DEBUG_LOG analyse_rank_{tag}={'1' if ok else '0'}")
sys.exit(0 if ok else 1)
PY

rc=$?
echo "[analyse-rank] exit=$rc" | tee -a "$LOG"
exit $rc
