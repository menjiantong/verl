#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填 rc。
# =============================================================================
# 1109 矩阵 A1 补跑（用户选择：不改代码、不动 A2/B2/B3，原样重跑 A1）
#
# 逻辑：等待 1109_chain_matrix_engram_cp_v1.sh 退出（A2/B2/B3 全部结束），
#      再调用 1109_exp_A1_engram_cp1_3step.sh（engram + CP=1，3 步）。
#      不改任何训练语义；A1 仍是原始 CP=1 配置。
#
# 启动：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid -f bash my_scripts/1109_rerun_A1_after_matrix_v1.sh \
#     > logs/1109_rerun_A1_nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/1109_rerun_A1_${TS}.nohup.out"
echo "[rerun-A1] MY_DEBUG_LOG wait_matrix_start time=$(date '+%F %T') host=$(hostname) log=$LOG" | tee -a "$LOG"

# 最长等 12 h；每 30 s 检查一次 master 是否还在。
for _ in $(seq 1 1440); do
  if ! pgrep -f "1109_chain_matrix_engram_cp_v1\.sh" >/dev/null 2>&1; then
    echo "[rerun-A1] MY_DEBUG_LOG matrix_finished time=$(date '+%F %T')" | tee -a "$LOG"
    break
  fi
  sleep 30
done

# master 退出后给 Ray/pinned 一点回收时间；A1 自己的 chain 还会再等 MemAvailable>=2000GiB。
sleep 15
echo "[rerun-A1] MY_DEBUG_LOG rerun_start time=$(date '+%F %T')" | tee -a "$LOG"

bash my_scripts/1109_exp_A1_engram_cp1_3step.sh >> "$LOG" 2>&1
RC=$?
echo "[rerun-A1] MY_DEBUG_LOG rerun_done rc=$RC time=$(date '+%F %T')" | tee -a "$LOG"
exit "$RC"
