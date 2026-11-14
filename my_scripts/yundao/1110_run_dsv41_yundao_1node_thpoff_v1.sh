#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-10 创建（云道 1 机 × 16 卡 top32 + P0 共用表 + THP off）
# =============================================================================
# 1110 单机 thpoff 档：把 188 已验证的 THP off 搬到云道单机路径
#
# 基线（规则 7.3/7.4：本脚本只写差异，训练数学与拓扑一律不动）：
#   my_scripts/yundao/1109_run_dsv41_yundao_1node_top32_share_v1.sh
#
# 依据（只读同步 80.5.27.188 rebase 工作区）：
#   mjt_workplace/worklog/worklog_20261110_dsv41_188_thp_rootfix.md
#   E1（188 单机 16 卡）：THP off 后 144 s 的 update_actor 尾消失，
#   最慢 step 291→238 s（−6.9%），gen 更稳（128/128/127 s vs 166/126/134 s），
#   compact_stall 全程仅 +8（基线同期 +55,312）；代价 update_actor 68→77~98 s。
#
# 与基线脚本的差异（只有一处行为变化）：
#   MY_ENV_DSV41_DISABLE_THP=1 —— 训练侧 Worker.__init__ 与 vLLM NPUWorker.__init__
#   两侧都在任何 engram 大表分配之前 prctl(PR_SET_THP_DISABLE)，关掉本进程 THP。
#   代码来自 188 rebase：verl/utils/host_numa.py（apply_host_numa_env）+
#   vllm_ascend/host_numa.py（disable_thp_if_requested）+ 两处 worker 钩子。
#
# 判据：
#   a) 训练侧 `host_numa_applied ... "thp_off": true`、vLLM 侧
#      `vllm_host_numa_env ... thp_off=1 thp_ok=True`（16 worker）；
#   b) 3 步跑完、pearson 与基线同量级；对比 update_actor 与 timing_s/step；
#   c) 0 条 Communication_Error_Bind_IP_Port / EngineDeadError。
#
# 用法（云道控制台：资源规格 **1 机 × 16 卡**，启动命令只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1110_run_dsv41_yundao_1node_thpoff_v1.sh
#
# 可覆盖开关：
#   MY_ENV_DSV41_DISABLE_THP=0（回基线 A/B，其余不变）
#   其余继承基线：MY_ENV_P41_SHARE / MY_ENV_P41_STEPS / MY_ENV_P41_DATASET 等
#
# 回滚：不设 MY_ENV_DSV41_DISABLE_THP 直接跑基线脚本即可；
#       代码 cp mjt_workplace/backup/*.bak_20261110_thpoff 还原。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
BASE=$MIRROR_ROOT/verl/my_scripts/yundao/1109_run_dsv41_yundao_1node_top32_share_v1.sh
VERL_HOST_NUMA=$MIRROR_ROOT/verl/verl/utils/host_numa.py
VLLM_HOST_NUMA=$MIRROR_ROOT/vllm-ascend/vllm_ascend/host_numa.py
VLLM_WORKER=$MIRROR_ROOT/vllm-ascend/vllm_ascend/worker/worker.py

if [ "${MA_NUM_HOSTS:-0}" -ne 1 ]; then
  echo "[1110 thp1] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 1-node script (expect 1)" >&2
  exit 2
fi
for _f in "$BASE" "$VERL_HOST_NUMA" "$VLLM_HOST_NUMA" "$VLLM_WORKER"; do
  if [ ! -f "$_f" ]; then
    echo "[1110 thp1] ABORT: required file missing: $_f" >&2
    exit 3
  fi
done
_missing=0
grep -q "def apply_host_numa_env" "$VERL_HOST_NUMA" || { echo "[1110 thp1] ABORT: apply_host_numa_env 缺失 in verl/utils/host_numa.py（先同步 188 版本）" >&2; _missing=1; }
grep -q "disable_thp_if_requested" "$VLLM_HOST_NUMA" || { echo "[1110 thp1] ABORT: THP 开关缺失 in vllm_ascend/host_numa.py（先同步 188 版本）" >&2; _missing=1; }
grep -q "disable_thp_if_requested()" "$VLLM_WORKER" || { echo "[1110 thp1] ABORT: NPUWorker 未调用 disable_thp_if_requested（先同步 188 worker.py）" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[1110 thp1] 提示：188 同步文件在 $MIRROR_ROOT/mjt_workplace/backup/ 有备份" >&2
  exit 3
fi

export MY_ENV_DSV41_DISABLE_THP=${MY_ENV_DSV41_DISABLE_THP:-1}

echo "[1110 thp1] MY_DEBUG_LOG thpoff_start ts=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S') thp_off=$MY_ENV_DSV41_DISABLE_THP base=$(basename "$BASE") MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} vllm_host_numa=$(md5sum "$VLLM_HOST_NUMA" | awk '{print $1}') worker=$(md5sum "$VLLM_WORKER" | awk '{print $1}')"

exec bash "$BASE" "$@"
