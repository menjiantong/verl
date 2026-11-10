#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次创建（80.5.25.130）。成功后回填日志路径与结论。
# =============================================================================
# 1107 DSV4.1 单机 16 卡 Engram P0 A/B 串联：先 off（旧行为），再 on（actor 复用 ref 表）
#
# 基线（规则 7.3/7.4）：my_scripts/1106_chain_reuse_arm_v1.sh（同样的串联思路）
#   → my_scripts/1107_run_dsv41_130_engram_share_ab_v1.sh（单档）
#
# 设计：
#   - 两个 arm 之间不手动干预；每档开头由单档脚本做 hard ray stop + 重新拉起；
#   - 前档失败也继续跑后档（off 失败时 on 的对照更有价值），两档 rc 都记录；
#   - 两档之间最多等 120 s 让 trainer 进程自然退出，只观测、不 kill（规则 9.2）。
#
# 预计：单档 ~35–45 min（含 USE_KL=1 的 ref build + 两次表加载），两档 ~1.5 h。
# 启动命令（规则 7.1：脚本内不 export，直接 nohup 拉起本脚本即可）：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid nohup bash my_scripts/1107_chain_engram_share_ab_v1.sh \
#     > logs/1107_chain_nohup.out 2>&1 < /dev/null &
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

DRV="logs/1107_chain_share_ab_$(date +%Y%m%d_%H%M%S).driver.log"
echo "[share-chain] MY_DEBUG_LOG chain_start time=$(date '+%F %T') host=$(hostname) log=$DRV" | tee -a "$DRV"

# 两档共用的步数（其余开关全部在单档脚本里）。
export MY_ENV_DSV41_SHARE_STEPS=${MY_ENV_DSV41_SHARE_STEPS:-3}

export MY_ENV_DSV41_SHARE_ARM=off
echo "[share-chain] MY_DEBUG_LOG arm_start arm=off time=$(date '+%F %T')" | tee -a "$DRV"
bash my_scripts/1107_run_dsv41_130_engram_share_ab_v1.sh >> "$DRV" 2>&1
RC_OFF=$?
echo "[share-chain] MY_DEBUG_LOG arm_done arm=off rc=$RC_OFF time=$(date '+%F %T')" | tee -a "$DRV"

# 等 trainer 进程自然退出（观测，不 kill）。规则 9.2：不删除/不强杀用户资源。
for _i in $(seq 1 24); do
  pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1 || break
  sleep 5
done
if pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
  echo "[share-chain] MY_DEBUG_LOG WARN trainer_still_alive after 120s arm=off" | tee -a "$DRV"
fi
awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{printf "[share-chain] MY_DEBUG_LOG between_arms MemTotal=%dGiB MemFree=%dGiB MemAvailable=%dGiB Cached=%dGiB Shmem=%dGiB\n",t/1048576,f/1048576,a/1048576,c/1048576,s/1048576}' /proc/meminfo | tee -a "$DRV"

export MY_ENV_DSV41_SHARE_ARM=on
echo "[share-chain] MY_DEBUG_LOG arm_start arm=on time=$(date '+%F %T')" | tee -a "$DRV"
bash my_scripts/1107_run_dsv41_130_engram_share_ab_v1.sh >> "$DRV" 2>&1
RC_ON=$?
echo "[share-chain] MY_DEBUG_LOG arm_done arm=on rc=$RC_ON time=$(date '+%F %T')" | tee -a "$DRV"

echo "[share-chain] MY_DEBUG_LOG chain_done rc_off=$RC_OFF rc_on=$RC_ON time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"

[ "$RC_OFF" -eq 0 ] && [ "$RC_ON" -eq 0 ]
