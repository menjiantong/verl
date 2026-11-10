#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填矩阵汇总。
# =============================================================================
# 1109 单变量实验矩阵串联：A1 -> A2 -> B2 -> B3（每个 3 步）
#
#   A1  engram    CP=1   基线
#   A2  noengram  CP=1   只变 engram（vs A1）
#   B2  engram    CP=2   只变 CP（vs A1）
#   B3  engram    CP=4   只变 CP（vs A1）
#
# 每个实验脚本自己导出全部环境变量（rule 7.1），并复用 chain v3 的
# fadvise + MemAvailable>=2000GiB 等待；本脚本只负责顺序编排与索引。
#
# 启动：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid -f bash my_scripts/1109_chain_matrix_engram_cp_v1.sh \
#     > logs/1109_matrix_nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

MATRIX_DIR="logs/1109_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MATRIX_DIR"
INDEX="$MATRIX_DIR/index.txt"

{
  echo "[matrix] MY_DEBUG_LOG matrix_start time=$(date '+%F %T') host=$(hostname) dir=$MATRIX_DIR"
  echo "[matrix] MY_DEBUG_LOG matrix_design exps=A1(engram,cp1),A2(noengram,cp1),B2(engram,cp2),B3(engram,cp4) steps=3"
} | tee -a "$INDEX"

RC_ALL=0
for EXP in A1_engram_cp1 A2_noengram_cp1 B2_engram_cp2 B3_engram_cp4; do
  echo "[matrix] MY_DEBUG_LOG exp_begin exp=$EXP time=$(date '+%F %T')" | tee -a "$INDEX"
  bash "my_scripts/1109_exp_${EXP}_3step.sh" > "$MATRIX_DIR/${EXP}.out" 2>&1
  RC=$?
  [ "$RC" -ne 0 ] && RC_ALL=1
  echo "[matrix] MY_DEBUG_LOG exp_end exp=$EXP rc=$RC time=$(date '+%F %T')" | tee -a "$INDEX"
done

echo "[matrix] MY_DEBUG_LOG matrix_done rc=$RC_ALL time=$(date '+%F %T') dir=$MATRIX_DIR" | tee -a "$INDEX"
exit "$RC_ALL"
