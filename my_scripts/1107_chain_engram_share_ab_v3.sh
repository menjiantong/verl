#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次创建（80.5.25.130）。
# =============================================================================
# 1107 v3 DSV4.1 单机 16 卡 Engram P0 A/B 串联（off → on，带页缓存看护）
#
# 演进：
#   v1 chain：off 档退出后立刻起 on，MemAvailable=735GiB < 2000 → on 被预检跳过（编排问题）；
#   v2 chain：每档前等 MemAvailable ≥ 2000GiB；on 单独跑。
#     on 结果：P0 生效（第二次表加载 0.1 s、2 of 2 reused），但 rollout 里
#     小 pinned 分配 207001（MemAvailable=702GiB、MemFree=60GiB、Cached=1099GiB）。
#     原因：checkpoint 读出来的 ~643GiB 干净页缓存，驱动 pinned 分配不会回收。
#   v3 chain（本脚本）：
#     1) 默认跑 off + on 两档（off 也需要复跑：上次死在 vLLM init，没有 step 数据）；
#     2) 每档启动前先对权重目录 fadvise(DONTNEED)（让两档从同样的权重缓存起点开始），
#        再等 MemAvailable ≥ 2000GiB；不用全局 drop_caches（本机约 20 GiB/min，太慢）；
#     3) 调用单档 v2 脚本（1107_run_dsv41_130_engram_share_ab_v2.sh），
#        它在模型初始化后再丢一次页缓存，并在 MemFree < 120GiB 时补丢（最多 2 次）。
#
# 启动：
#   cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
#   setsid -f bash my_scripts/1107_chain_engram_share_ab_v3.sh \
#     > logs/1107_chain_v3_nohup.out 2>&1 < /dev/null
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
mkdir -p logs

ARMS=${MY_ENV_DSV41_SHARE_ARMS:-off on}
MIN_AVAIL_GB=${MY_ENV_DSV41_SHARE_MIN_AVAIL_GB:-2000}
WAIT_MAX_S=${MY_ENV_DSV41_SHARE_WAIT_MAX_S:-900}
# 用户指定 32 专家权重；fadvise 和单档脚本共用这个路径。
WEIGHTS=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16

# [1109 matrix] 实验标签只影响日志名；未设置时保持原文件名。
TAG=${MY_ENV_DSV41_EXP_TAG:-share_ab_v3}
DRV="logs/1107_chain_${TAG}_$(date +%Y%m%d_%H%M%S).driver.log"
echo "[share-chain-v3] MY_DEBUG_LOG chain_v3_start time=$(date '+%F %T') host=$(hostname) arms='$ARMS' min_avail=${MIN_AVAIL_GB}GiB log=$DRV" | tee -a "$DRV"

# 等待主机内存回到门槛以上：上一档的 pinned 释放和页缓存回收是异步的。
wait_mem() {
  local _arm=$1 _t0=$(date +%s) _avail _waited
  while :; do
    _avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    _waited=$(( $(date +%s) - _t0 ))
    if [ "$_avail" -ge "$MIN_AVAIL_GB" ]; then
      echo "[share-chain-v3] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=1" | tee -a "$DRV"
      return 0
    fi
    if [ "$_waited" -ge "$WAIT_MAX_S" ]; then
      echo "[share-chain-v3] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=0 timeout=1" | tee -a "$DRV"
      return 1
    fi
    echo "[share-chain-v3] MY_DEBUG_LOG wait_mem arm=$_arm MemAvailable=${_avail}GiB waited=${_waited}s ok=0" | tee -a "$DRV"
    sleep 10
  done
}

RC_ALL=0
for ARM in $ARMS; do
  case "$ARM" in
    off|on) ;;
    *) echo "[share-chain-v3] ABORT: bad arm '$ARM'" >&2; exit 2 ;;
  esac
  # [1107 v3] 每档从同样的权重页缓存起点开始：只对权重目录 fadvise(DONTNEED)，
  # 秒级回收 checkpoint 干净页；不用全局 drop_caches（本机实测约 20 GiB/min，太慢）。
  /usr/local/python3.11.15/bin/python3 my_scripts/1107_fadvise_weights_v1.py "$WEIGHTS" 2>/dev/null | tail -1 | tee -a "$DRV"
  echo "[share-chain-v3] MY_DEBUG_LOG pre_arm_fadvise arm=$ARM MemFree=$(awk '/^MemFree:/{print int($2/1024/1024)}' /proc/meminfo)GiB MemAvailable=$(awk '/^MemAvailable:/{print int($2/1024/1024)}' /proc/meminfo)GiB" | tee -a "$DRV"
  if ! wait_mem "$ARM"; then
    echo "[share-chain-v3] MY_DEBUG_LOG arm_skip arm=$ARM reason=mem_timeout" | tee -a "$DRV"
    RC_ALL=1
    continue
  fi
  export MY_ENV_DSV41_SHARE_ARM=$ARM
  echo "[share-chain-v3] MY_DEBUG_LOG arm_start arm=$ARM time=$(date '+%F %T')" | tee -a "$DRV"
  bash my_scripts/1107_run_dsv41_130_engram_share_ab_v2.sh >> "$DRV" 2>&1
  RC=$?
  echo "[share-chain-v3] MY_DEBUG_LOG arm_done arm=$ARM rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"
  [ "$RC" -ne 0 ] && RC_ALL=1
done

echo "[share-chain-v3] MY_DEBUG_LOG chain_v3_done rc=$RC_ALL time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC_ALL"
