#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填日志路径与 rc。
# =============================================================================
# 1109 实验矩阵 B2：Engram 有表 + CP=2
#
# 单变量：相对 A1 只把 CP_SIZE 从 1 改为 2，其余（engram、dapo512、3 步、
#   USE_KL=1、SHARE=1）完全一致。
#   CP=2 的 PAD 必须为 4：validate_sequence_length 要求
#   length % cp_size == 0 且 (length/cp_size) % max_compress_ratio(2) == 0，
#   即 length % 4 == 0；与 1102 CP 闸位 g1 档一致。
#
# 基线（rule 7.3/7.4）：
#   my_scripts/1107_chain_engram_share_ab_v3.sh
#     -> my_scripts/1107_run_dsv41_130_engram_share_ab_v2.sh（USE_KL=1 + SHARE=1）
#     -> my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh
#
# 分支/commit（rule 7.5）：
#   verl        dev_dsv41_rebase1030_cp_engram @ e18fe6ec
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#   vllm-ascend from_9a38d36_1030           @ e3f220b05
#   vllm        main                        @ fd46ca5ada
#
# 启动：
#   setsid -f bash my_scripts/1109_exp_B2_engram_cp2_3step.sh \
#     > logs/1109_exp_B2.nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

export MY_ENV_DSV41_EXP_TAG=B2_engram_cp2

# ---- 变量 1：有 Engram 表（与 A1 相同）----
export MY_ENV_DSV41_AB_ARM=engram

# ---- 变量 2：CP=2 ----
export CP_SIZE=2
export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
export MY_ENV_DSV41_CP_GATHER_OUTPUT=1

# ---- 固定项：Engram P0 共用表 + 3 步 ----
export MY_ENV_DSV41_SHARE_ARMS=on
export MY_ENV_DSV41_SHARE_STEPS=3
export MY_ENV_DSV41_SHARE_WAIT_MAX_S=1800

echo "[exp-B2] MY_DEBUG_LOG exp_start tag=$MY_ENV_DSV41_EXP_TAG arm=$MY_ENV_DSV41_AB_ARM cp=$CP_SIZE pad=$MY_ENV_DSV41_CP_PAD_MULTIPLE keep_train=$MY_ENV_DSV41_CP_KEEP_TRAIN_MODE gather=$MY_ENV_DSV41_CP_GATHER_OUTPUT steps=$MY_ENV_DSV41_SHARE_STEPS time=$(date '+%F %T') host=$(hostname)"

exec bash my_scripts/1107_chain_engram_share_ab_v3.sh
