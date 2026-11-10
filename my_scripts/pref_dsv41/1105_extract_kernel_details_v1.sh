#!/usr/bin/env bash
# =============================================================================
# 1105 DSV4.1 Engram RL profiling —— 只提取 kernel_details.csv（v1）
#
# 背景：
#   1105_run_dsv41_profiling_engram_v1.sh 在 80.5.25.130 跑 2 步、采集第 2 步、
#   actor/ref/rollout 只采 rank0。训练侧 actor 的 analyse=True 用的是
#   ExportType.Db，落盘只有 analysis.db + ascend_pytorch_profiler_0.db，
#   没有 kernel_details.csv；rollout 侧 trace 在退出时提示需离线解析。
#
# 本脚本只做一件事：
#   对「训练 rank0」和「rollout rank0」两个 *_ascend_pt 目录调用
#       torch_npu.profiler.profiler.analyse(<rank 目录>, export_type="Text")
#   让 ASCEND_PROFILER_OUTPUT/ 下生成 kernel_details.csv 等文本产物。
#   已存在 kernel_details.csv 的目录直接跳过（幂等）。
#
# 不做：MFU / score / Excel（用户明确要求只取 kernel detail）。
# 不删除任何文件（rule 9.2）：analyse 只在 ASCEND_PROFILER_OUTPUT 内追加产物。
#
# 用法：bash my_scripts/pref_dsv41/1105_extract_kernel_details_v1.sh <profiling 根目录>
#   例：bash ... 1105_extract_kernel_details_v1.sh \
#         /mnt/share/m00899630/workspace/dsv41-verl-rebase/output/profiling/engram_cp1_step2_20261105_033401
# 日志：logs/1105_extract_kernel_details_<TS>.log
# 脚本 commit：verl a7b9eac3（首次提交）
# =============================================================================
# [1105 修复] 不能用 set -u：1030_ray_dsv41_env.sh 间接 source 的 /etc/bash.bashrc
# 第 7 行引用未定义的 PS1，在非交互 shell 下 set -u 会直接让脚本静默退出
# （首版 v1 就是这样在 source 处无声退出，日志只剩 4 行）。
set -o pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO" || exit 1

ROOT=${1:?用法: $0 <profiling 根目录>}
ROOT=${ROOT%/}

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1105_extract_kernel_details_${TS}.log"

echo "[extract-kd] root=$ROOT" | tee "$LOG"
echo "[extract-kd] log=$LOG" | tee -a "$LOG"

# 只在 root 下找 rank0 目录：training 目录名统一带 _rank0 或 node-*（单 rank），
# rollout 目录名带 tp0_..._rank0_。
TRAIN_DIR=$(find "$ROOT/train" -maxdepth 1 -type d -name "*_ascend_pt" 2>/dev/null | head -1)
ROLLOUT_DIR=$(find "$ROOT/agent_loop_rollout_replica_0" -maxdepth 1 -type d -name "*tp0_*_rank0_*_ascend_pt" 2>/dev/null | head -1)

echo "[extract-kd] train=$TRAIN_DIR" | tee -a "$LOG"
echo "[extract-kd] rollout=$ROLLOUT_DIR" | tee -a "$LOG"
[ -n "$TRAIN_DIR" ] || { echo "[extract-kd] ABORT: 未找到 train rank0 目录" | tee -a "$LOG"; exit 2; }
[ -n "$ROLLOUT_DIR" ] || { echo "[extract-kd] ABORT: 未找到 rollout rank0 目录" | tee -a "$LOG"; exit 2; }

# torch_npu / CANN 环境（PATH + 环境变量），与 profiling 运行时一致。
# [1105 修复] 不再把 source 的输出丢到 /dev/null：首版静默失败就是因为错误被吞掉。
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

# [1105] python3 -u：stdout 走管道时默认块缓冲，-u 保证进度实时落日志
python3 -u - "$TRAIN_DIR" "$ROLLOUT_DIR" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, time
from torch_npu.profiler.profiler import analyse

def ensure_csv(rank_dir, tag):
    out = os.path.join(rank_dir, "ASCEND_PROFILER_OUTPUT")
    kd = os.path.join(out, "kernel_details.csv")
    if os.path.isfile(kd) and os.path.getsize(kd) > 0:
        print(f"[extract-kd] {tag}: 已存在 kernel_details.csv ({os.path.getsize(kd)} B)，跳过")
        return True
    t0 = time.time()
    print(f"[extract-kd] {tag}: 开始离线 analyse(export_type='Text') -> {rank_dir}")
    # [1105 提取] 显式 export_type='text'：只跑文本解析，产出 kernel_details.csv 等。
    # 注意取值必须是小写（torch_npu 2.10 的 Constant.Text == 'text'）；传 'Text' 会被
    # analyse 判为非法并静默回退成 None，回退后按采集时的 'db' 口径解析，只出 analysis.db。
    analyse(rank_dir, max_process_number=32, export_type="text")
    dt = time.time() - t0
    ok = os.path.isfile(kd) and os.path.getsize(kd) > 0
    print(f"[extract-kd] {tag}: analyse 用时 {dt:.1f}s, kernel_details.csv={'OK' if ok else 'MISSING'}")
    if os.path.isdir(out):
        for f in sorted(os.listdir(out)):
            p = os.path.join(out, f)
            if os.path.isfile(p):
                print(f"[extract-kd] {tag}:   {f}  {os.path.getsize(p)} B")
    return ok

a = ensure_csv(sys.argv[1], "train_rank0")
b = ensure_csv(sys.argv[2], "rollout_rank0")
print(f"MY_DEBUG_LOG extract_kernel_details_train={int(a)} rollout={int(b)}")
sys.exit(0 if (a and b) else 1)
PY

rc=$?
echo "[extract-kd] exit=$rc" | tee -a "$LOG"
exit $rc
