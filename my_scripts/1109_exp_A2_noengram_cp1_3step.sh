#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填日志路径与 rc。
# =============================================================================
# 1109 实验矩阵 A2：Engram 无表（noengram）+ CP=1
#
# 单变量：相对 A1 只把模型从 top32-bf16（engram_layer_ids=[1,14]）换成
#   top32-bf16-noengram（engram_layer_ids=[]），其余（CP=1、dapo512、3 步、
#   USE_KL=1、P0 开关 SHARE=1）完全一致。
#   反向思考：noengram 没有表，SHARE=1 在该 arm 上是空操作；保持同一个值是为了
#   让 A1/A2 的脚本只差权重路径这一个变量。
#
# 基线（rule 7.3/7.4）：
#   my_scripts/1107_chain_engram_share_ab_v3.sh
#     -> my_scripts/1107_run_dsv41_130_engram_share_ab_v2.sh（USE_KL=1 + SHARE=1）
#     -> my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh（ARM=noengram 分支）
#
# 分支/commit（rule 7.5）：
#   verl        dev_dsv41_rebase1030_cp_engram @ e18fe6ec
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#   vllm-ascend from_9a38d36_1030           @ e3f220b05
#   vllm        main                        @ fd46ca5ada
#
# 启动：
#   setsid -f bash my_scripts/1109_exp_A2_noengram_cp1_3step.sh \
#     > logs/1109_exp_A2.nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

export MY_ENV_DSV41_EXP_TAG=A2_noengram_cp1

# ---- 变量 1：无 Engram 表 ----
# 1106 的 noengram 分支会自动选 /mnt/share/m00899630/weights/
# DeepSeek-V4.1-Flash-top32-bf16-noengram，并设 VLLM_ENGRAM_DP_SHARED_MEMORY=0。
export MY_ENV_DSV41_AB_ARM=noengram

# ---- 变量 2：CP=1（与 A1 相同）----
export CP_SIZE=1
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0

# ---- 固定项：3 步；SHARE=1 对 noengram 是空操作（保持与 A1 同值）----
export MY_ENV_DSV41_SHARE_ARMS=on
export MY_ENV_DSV41_SHARE_STEPS=3
export MY_ENV_DSV41_SHARE_WAIT_MAX_S=1800

echo "[exp-A2] MY_DEBUG_LOG exp_start tag=$MY_ENV_DSV41_EXP_TAG arm=$MY_ENV_DSV41_AB_ARM cp=$CP_SIZE pad=$MY_ENV_DSV41_CP_PAD_MULTIPLE keep_train=$MY_ENV_DSV41_CP_KEEP_TRAIN_MODE gather=$MY_ENV_DSV41_CP_GATHER_OUTPUT steps=$MY_ENV_DSV41_SHARE_STEPS time=$(date '+%F %T') host=$(hostname)"

exec bash my_scripts/1107_chain_engram_share_ab_v3.sh
