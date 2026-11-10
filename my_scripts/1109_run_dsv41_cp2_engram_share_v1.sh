#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（80.5.27.169）。成功后回填日志路径与结论。
# =============================================================================
# 1109 DSV4.1 单机 16 卡：CP=2 + Engram P0（actor/ref 共用 frozen 表）单档
#
# 依据（rule 3）：mjt_workplace/plan/plan_20261109_dsv41_cp2_engram_169_v1.md
# 基线（rule 7.3/7.4，本脚本只写差异）：
#   my_scripts/1107_chain_engram_share_ab_v3.sh（只跑 on 档）
#     -> my_scripts/1107_run_dsv41_130_engram_share_ab_v2.sh（USE_KL=1 + SHARE=1）
#     -> my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh（top32-bf16/dapo512/3步）
#
# 相对基线的唯一差异：打开 CP（KV-AllGather，CP_SIZE=2）。
#   1102 CP 单训/RL 记录：CP>1 时数据侧必须补齐到 cp_size*4 的倍数，
#   并保留 train mode / gather output；与 1102_run_dsv41_cp_validate_v1.sh 的 g1 档一致。
#   CP 只作用于训练侧 actor 的序列切分（base 脚本只给 actor 追加
#   +kv_allgather_parallel_size），rollout(vLLM) 与 ref 布局不变。
#
# 启动：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid -f bash my_scripts/1109_run_dsv41_cp2_engram_share_v1.sh \
#     > logs/1109_cp2_engram_nohup.out 2>&1 < /dev/null
#
# 分支/commit（rule 7.5）：
#   verl        dev_dsv41_rebase1030_cp_engram @ e18fe6ec
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b
#   vllm-ascend from_9a38d36_1030           @ e3f220b05
#   vllm        main                        @ fd46ca5ada
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

# [1109 CP] 与 1102 CP 闸位 g1 档一致：CP=2、序列补齐 4 的倍数、保留 train mode、gather 输出。
# 反向思考：这些开关不改变训练数学，只让 CP 的切片/补齐与 v1 trainer 对齐；
# CP_SIZE=1 时与 1106 基线逐字节一致。
export CP_SIZE=${CP_SIZE:-2}
export MY_ENV_DSV41_CP_PAD_MULTIPLE=${MY_ENV_DSV41_CP_PAD_MULTIPLE:-4}
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=${MY_ENV_DSV41_CP_KEEP_TRAIN_MODE:-1}
export MY_ENV_DSV41_CP_GATHER_OUTPUT=${MY_ENV_DSV41_CP_GATHER_OUTPUT:-1}

# [1109] 只跑 on 档（Engram P0 共用表，SHARE=1）；off 档已有 130 的 A/B 结果，不再重复。
export MY_ENV_DSV41_SHARE_ARMS=${MY_ENV_DSV41_SHARE_ARMS:-on}
# 与 1107 A/B 相同的 3 步，便于和历史数据对齐。
export MY_ENV_DSV41_SHARE_STEPS=${MY_ENV_DSV41_SHARE_STEPS:-3}
# 前序作业刚退出，pinned 回收可能慢；等待 MemAvailable>=2000GiB 的上限放宽到 30 min。
export MY_ENV_DSV41_SHARE_WAIT_MAX_S=${MY_ENV_DSV41_SHARE_WAIT_MAX_S:-1800}

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/1109_cp2_engram_${TS}.nohup.out
echo "[cp2-engram] MY_DEBUG_LOG cp2_engram_start time=$(date '+%F %T') host=$(hostname) cp=$CP_SIZE pad=$MY_ENV_DSV41_CP_PAD_MULTIPLE keep_train=$MY_ENV_DSV41_CP_KEEP_TRAIN_MODE gather=$MY_ENV_DSV41_CP_GATHER_OUTPUT arms=$MY_ENV_DSV41_SHARE_ARMS steps=$MY_ENV_DSV41_SHARE_STEPS log=$LOG" | tee -a "$LOG"

exec bash my_scripts/1107_chain_engram_share_ab_v3.sh >> "$LOG" 2>&1
