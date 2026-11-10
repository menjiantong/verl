#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次创建（80.5.25.130）。
# =============================================================================
# 1107 v2 DSV4.1 单机 16 卡 Engram P0 A/B 串联（修正 v1 的"上一档刚退出就起下一档"）
#
# v1 现象（2026-11-06 12:54:15，logs/1107_chain_share_ab_20261106_121707.driver.log）：
#   off 档在 vLLM 拉起阶段失败（pinned 207001，见 worklog §5.1），进程刚退，
#   pinned/页缓存还没回收，chain 立刻起 on 档 → precheck 看到 MemAvailable=735GiB
#   < 2000GiB → on 档 rc=3 直接跳过。
#
# v2 做法：
#   1) 每档启动前等待 MemAvailable 回到 ≥ 2000 GiB（最多 15 min，每 10 s 打一行）；
#   2) 默认只跑 on 档（off 已在 12:17 跑过：2 次 loaded + vLLM init 207001）；
#      需要复跑两档时：MY_ENV_DSV41_SHARE_ARMS="off on" bash 本脚本；
#   3) 其它逻辑不变，仍调用单档脚本 1107_run_dsv41_130_engram_share_ab_v1.sh。
#
# 启动：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid -f bash my_scripts/1107_chain_engram_share_ab_v2.sh \
#     > logs/1107_chain_v2_nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

ARMS=${MY_ENV_DSV41_SHARE_ARMS:-on}
MIN_AVAIL_GB=${MY_ENV_DSV41_SHARE_MIN_AVAIL_GB:-2000}
WAIT_MAX_S=${MY_ENV_DSV41_SHARE_WAIT_MAX_S:-900}

DRV="logs/1107_chain_share_ab_v2_$(date +%Y%m%d_%H%M%S).driver.log"
echo "[share-chain-v2] MY_DEBUG_LOG chain_v2_start time=$(date '+%F %T') host=$(hostname) arms='$ARMS' min_avail=${MIN_AVAIL_GB}GiB log=$DRV" | tee -a "$DRV"

# 等待主机内存回到门槛以上：上一档的 pinned 释放和页缓存回收是异步的。
wait_mem() {
  local _arm=$1 _t0=$(date +%s) _avail _waited
  while :; do
    _avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    _waited=$(( $(date +%s) - _t0 ))
    if [ "$_avail" -ge "$MIN_AVAIL_GB" ]; then
      echo "[share-chain-v2] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=1" | tee -a "$DRV"
      return 0
    fi
    if [ "$_waited" -ge "$WAIT_MAX_S" ]; then
      echo "[share-chain-v2] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=0 timeout=1" | tee -a "$DRV"
      return 1
    fi
    echo "[share-chain-v2] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=0" | tee -a "$DRV"
    sleep 10
  done
}

RC_ALL=0
for ARM in $ARMS; do
  case "$ARM" in
    off|on) ;;
    *) echo "[share-chain-v2] ABORT: bad arm '$ARM'" >&2; exit 2 ;;
  esac
  if ! wait_mem "$ARM"; then
    echo "[share-chain-v2] MY_DEBUG_LOG arm_skip arm=$ARM reason=mem_timeout" | tee -a "$DRV"
    RC_ALL=1
    continue
  fi
  export MY_ENV_DSV41_SHARE_ARM=$ARM
  echo "[share-chain-v2] MY_DEBUG_LOG arm_start arm=$ARM time=$(date '+%F %T')" | tee -a "$DRV"
  bash my_scripts/1107_run_dsv41_130_engram_share_ab_v1.sh >> "$DRV" 2>&1
  RC=$?
  echo "[share-chain-v2] MY_DEBUG_LOG arm_done arm=$ARM rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"
  [ "$RC" -ne 0 ] && RC_ALL=1
done

echo "[share-chain-v2] MY_DEBUG_LOG chain_v2_done rc=$RC_ALL time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC_ALL"
