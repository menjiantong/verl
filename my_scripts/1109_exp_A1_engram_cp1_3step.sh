#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填日志路径与 rc。
# =============================================================================
# 1109 实验矩阵 A1：Engram 有表 + CP=1（单变量基线）
#
# 单变量设计（rule 3）：
#   A1 vs A2：只差 MY_ENV_DSV41_AB_ARM=engram|noengram（两者 CP=1、SHARE=1）；
#   A1 vs B2/B3：只差 CP_SIZE=1|2|4（三者 engram、SHARE=1）。
#   数据集 dapo512、权重 top32-bf16、3 步、rollout/vLLM 配置完全一致。
#
# 基线（rule 7.3/7.4，本脚本只写差异）：
#   my_scripts/1107_chain_engram_share_ab_v3.sh
#     -> my_scripts/1107_run_dsv41_130_engram_share_ab_v2.sh（USE_KL=1 + SHARE=1）
#     -> my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh（top32-bf16/dapo512/3步）
#
# 分支/commit（rule 7.5）：
#   verl        dev_dsv41_rebase1030_cp_engram @ e18fe6ec
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#   vllm-ascend from_9a38d36_1030           @ e3f220b05
#   vllm        main                        @ fd46ca5ada
#
# 启动：
#   setsid -f bash my_scripts/1109_exp_A1_engram_cp1_3step.sh \
#     > logs/1109_exp_A1.nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

# 实验标签只用于日志名（不改变训练语义）。
export MY_ENV_DSV41_EXP_TAG=A1_engram_cp1

# ---- 变量 1：模型是否带 Engram 表（A1=engram / A2=noengram）----
export MY_ENV_DSV41_AB_ARM=engram

# ---- 变量 2：CP 大小（A1/B2/B3 = 1/2/4）----
# CP=1 时关闭 CP 相关开关，PAD=1（无补齐），与 1102 CP 闸位 g0a 一致。
export CP_SIZE=1
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0

# ---- 固定项：Engram P0 共用表 + 3 步 + 内存等待 ----
export MY_ENV_DSV41_SHARE_ARMS=on
export MY_ENV_DSV41_SHARE_STEPS=3
export MY_ENV_DSV41_SHARE_WAIT_MAX_S=1800

echo "[exp-A1] MY_DEBUG_LOG exp_start tag=$MY_ENV_DSV41_EXP_TAG arm=$MY_ENV_DSV41_AB_ARM cp=$CP_SIZE pad=$MY_ENV_DSV41_CP_PAD_MULTIPLE keep_train=$MY_ENV_DSV41_CP_KEEP_TRAIN_MODE gather=$MY_ENV_DSV41_CP_GATHER_OUTPUT steps=$MY_ENV_DSV41_SHARE_STEPS time=$(date '+%F %T') host=$(hostname)"

exec bash my_scripts/1107_chain_engram_share_ab_v3.sh
