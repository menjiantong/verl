#!/usr/bin/env bash
# 等当前 A/B driver 跑完（/tmp/dsv41_ab_driver.done），再串行跑 reuse arm。
# 目的：不打断正在跑的 noengram 对照，同时保证三个 arm 用同机同口径。
set -u
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO" || exit 1

echo "[chain] MY_DEBUG_LOG chain_wait_start time=$(date '+%T')"
for i in $(seq 1 360); do
  [ -f /tmp/dsv41_ab_driver.done ] && break
  sleep 10
done
if [ ! -f /tmp/dsv41_ab_driver.done ]; then
  echo "[chain] MY_DEBUG_LOG chain_abort reason=driver_not_done time=$(date '+%T')"
  exit 1
fi
echo "[chain] MY_DEBUG_LOG chain_wait_done time=$(date '+%T')"
sleep 10

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/1106_drv_reuse_outer_${TS}.log
bash my_scripts/1106_run_engram_reuse_arm_v1.sh > "$LOG" 2>&1
echo "[chain] MY_DEBUG_LOG chain_reuse_rc=$? log=$LOG time=$(date '+%T')"
