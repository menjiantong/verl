#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-10 创建（8 机全尺寸 + Engram P0 + THP off）
# =============================================================================
# 1110 v1：云道 8 机 × 16 卡 thpoff 档（关闭 THP，规避低水位节点的直接压缩慢路径）
#
# 依据（只读同步 80.5.27.188 rebase 工作区）：
#   mjt_workplace/worklog/worklog_20261110_dsv41_188_thp_rootfix.md
#   E1（188 单机）：THP off 后 144 s 的 update_actor 尾消失，最慢 step 291→238 s
#   （−6.9%），compact_stall 全程仅 +8（基线同期 +55,312）；
#   代价：健康节点走 4K fault，update_actor 68→77~98 s。
#   E3（188）：sleep level 2 在首个 rollout 报 507057（与云道 v4 同根因），已否决。
#
# 与 1109 v4（sleep2d）的差异：
#   1) 不设 MY_ENV_VLLM_SLEEP_LEVEL（回默认 level=1），不设 MY_ENV_ENGINE_HCCL_PER_RANK；
#   2) 代码从 188 rebase 同步：新增 vllm_ascend/host_numa.py，覆盖 verl host_numa /
#      single_controller worker / vllm_async_server / vllm-ascend worker，
#      删除按 replica / 按 rank 的 HCCL 端口窗口（level=1 基线实测 0 条 bind 报错）；
#   3) 新增 MY_ENV_DSV41_DISABLE_THP=1，标签 sleep2d → thpoff，前缀 p85 → p86。
#
# 判据（2 步）：
#   a) vllm_host_numa_env ... thp_off=1 thp_ok=True（16 worker/节点）
#      与 host_numa_applied ... thp_off=1 thp_ok=True（训练侧 16 rank）；
#   b) 0 条 Communication_Error_Bind_IP_Port、0 条 EngineDeadError；
#   c) 第 1 步跑完，pearson ≈0.9996~0.9998、loss 不炸；
#   d) 与 1109 基线对比 update_actor / timing_s/step。
#
# 用法（云道控制台：资源规格 **8 机 × 16 卡**）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1110_run_dsv41_yundao_8node_thpoff_v1.sh
#
# 可覆盖开关：
#   MY_ENV_P86_PROFILE=validate|longrun（默认 validate：2 步、不存盘）
#   MY_ENV_P86_STEPS / MY_ENV_P86_SAVE_FREQ / MY_ENV_P86_MAX_CKPT_KEEP
#   MY_ENV_P86_SHARE=0|1（默认 1）
#   MY_ENV_DSV41_DISABLE_THP=0（回基线，用于 A/B）
#   MY_ENV_P86_MIN_MEM_GB / MY_ENV_P86_MIN_SHM_GB
#
# 回滚：代码 cp mjt_workplace/backup/*.bak_20261110_thpoff 还原；
#       行为不设 MY_ENV_DSV41_DISABLE_THP 即可（默认 0）。
# =============================================================================
set -eo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
HOST_OFFLOAD=$MIRROR_ROOT/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
DSV41_IMPL=$MIRROR_ROOT/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
VERL_HOST_NUMA=$MIRROR_ROOT/verl/verl/utils/host_numa.py
VLLM_HOST_NUMA=$MIRROR_ROOT/vllm-ascend/vllm_ascend/host_numa.py
VLLM_WORKER=$MIRROR_ROOT/vllm-ascend/vllm_ascend/worker/worker.py
ASYNC_SERVER=$MIRROR_ROOT/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
ENGRAM_MODEL=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
VAL_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

PROFILE=${MY_ENV_P86_PROFILE:-validate}
case "$PROFILE" in
  validate) DEF_STEPS=2;   DEF_SAVE=-1;  DEF_KEEP=2 ;;
  longrun)  DEF_STEPS=500; DEF_SAVE=110; DEF_KEEP=2 ;;
  *) echo "[p86] ABORT: MY_ENV_P86_PROFILE=$PROFILE (expect validate|longrun)" >&2; exit 2 ;;
esac
STEPS=500
SAVE_FREQ=${MY_ENV_P86_SAVE_FREQ:-$DEF_SAVE}
MAX_CKPT_KEEP=${MY_ENV_P86_MAX_CKPT_KEEP:-$DEF_KEEP}
SHARE=${MY_ENV_P86_SHARE:-1}
MIN_MEM_GB=${MY_ENV_P86_MIN_MEM_GB:-1600}
MIN_SHM_GB=${MY_ENV_P86_MIN_SHM_GB:-450}
THP_OFF=${MY_ENV_DSV41_DISABLE_THP:-1}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
NODE_RANK=${VC_TASK_INDEX:-0}
DRV=$LOGDIR/1110_8node_thpoff_${TS}_node${NODE_RANK}.driver.log

echo "[p86] MY_DEBUG_LOG p86_start host=$HOST node=$NODE_RANK ts=$TS profile=$PROFILE share=$SHARE steps=$STEPS save_freq=$SAVE_FREQ keep=$MAX_CKPT_KEEP thp_off=$THP_OFF MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} log=$DRV" | tee -a "$DRV"

if [ "${MA_NUM_HOSTS:-0}" -ne 8 ]; then
  echo "[p86] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 8-node script (expect 8)" | tee -a "$DRV" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 代码预检：P0 锚点 + THP 钩子（两侧）
# -----------------------------------------------------------------------------
for _f in "$HOST_OFFLOAD" "$DSV41_IMPL" "$BASE_PATH" "$ASYNC_SERVER" "$VLLM_WORKER" "$VERL_HOST_NUMA" "$VLLM_HOST_NUMA"; do
  if [ ! -f "$_f" ]; then
    echo "[p86] ABORT: required file missing: $_f" | tee -a "$DRV" >&2
    exit 3
  fi
done
_missing=0
grep -q "MY_ENV_DSV41_ENGRAM_SHARE_MODEL" "$HOST_OFFLOAD" || { echo "[p86] ABORT: P0 switch missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "supports_table_sharing" "$HOST_OFFLOAD" || { echo "[p86] ABORT: P0 supports_table_sharing missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_shared_hit" "$DSV41_IMPL" || { echo "[p86] ABORT: P0 shared-hit loader branch missing in fsdp_turbo_dsv41_impl.py" >&2; _missing=1; }
grep -q "MY_ENV_DSV41_DISABLE_THP" "$VERL_HOST_NUMA" || { echo "[p86] ABORT: THP 开关缺失 in verl/utils/host_numa.py（先同步 188 版本）" >&2; _missing=1; }
grep -q "def apply_host_numa_env" "$VERL_HOST_NUMA" || { echo "[p86] ABORT: apply_host_numa_env 缺失 in verl/utils/host_numa.py" >&2; _missing=1; }
grep -q "disable_thp_if_requested" "$VLLM_HOST_NUMA" || { echo "[p86] ABORT: THP 开关缺失 in vllm_ascend/host_numa.py（先同步 188 版本）" >&2; _missing=1; }
grep -q "disable_thp_if_requested()" "$VLLM_WORKER" || { echo "[p86] ABORT: NPUWorker 未调用 disable_thp_if_requested（先同步 188 worker.py）" >&2; _missing=1; }
grep -q "MY_ENV_ENGINE_HCCL_PER_RANK" "$VLLM_WORKER" && { echo "[p86] ABORT: worker.py 仍含已否决的 HCCL 端口窗口补丁（先同步 188 worker.py）" >&2; _missing=1; }
grep -q "MY_ENV_VLLM_SLEEP_LEVEL" "$ASYNC_SERVER" && { echo "[p86] ABORT: async_server 仍含已否决的 sleep-level 覆写（先同步 188 版本）" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[p86] 提示：188 同步文件在 $MIRROR_ROOT/mjt_workplace/backup/ 有备份" | tee -a "$DRV" >&2
  exit 3
fi
echo "[p86] MY_DEBUG_LOG code_check_ok host_offload=$(md5sum "$HOST_OFFLOAD" | awk '{print $1}') impl=$(md5sum "$DSV41_IMPL" | awk '{print $1}') async_server=$(md5sum "$ASYNC_SERVER" | awk '{print $1}') worker=$(md5sum "$VLLM_WORKER" | awk '{print $1}') verl_host_numa=$(md5sum "$VERL_HOST_NUMA" | awk '{print $1}') vllm_host_numa=$(md5sum "$VLLM_HOST_NUMA" | awk '{print $1}')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 权重预检
# -----------------------------------------------------------------------------
python3 - "$ENGRAM_MODEL" <<'PY' 2>&1 | tee -a "$DRV"
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cfg = json.loads((root / "config.json").read_text())
text = cfg.get("text_config", cfg)
ids = text.get("engram_layer_ids") or []
nexp = text.get("n_routed_experts")
assert ids, f"{root}: engram_layer_ids empty, not an engram checkpoint"
assert nexp == 384, f"{root}: n_routed_experts={nexp}, this script needs the full 384-expert checkpoint"
w = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
tables = [k for k in w if "engram" in k and k.endswith("embed.weight")]
assert len(tables) == len(ids), f"{root}: engram tables {tables} vs ids {ids}"
print(f"[p86] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY
for _f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$_f" ] || { echo "[p86] ABORT: dataset missing: $_f" | tee -a "$DRV" >&2; exit 3; }
done

# -----------------------------------------------------------------------------
# 主机内存 / /dev/shm / 磁盘预检
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[p86] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB need_shm=${MIN_SHM_GB}GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[p86] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SHM_AVAIL_GB" -lt "$MIN_SHM_GB" ]; then
  echo "[p86] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < ${MIN_SHM_GB}GiB（engram 共享表约 366 GiB）" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SAVE_FREQ" -ge 0 ]; then
  FREE_T=$(df -Pk /efs_rl | awk 'NR==2{printf "%.1f", $4/1024/1024/1024}')
  NEED_T=$(awk -v k="$MAX_CKPT_KEEP" 'BEGIN{printf "%.1f", k*1.5}')
  echo "[p86] MY_DEBUG_LOG ckpt_space_check free=${FREE_T}T need=${NEED_T}T keep=${MAX_CKPT_KEEP}" | tee -a "$DRV"
  if awk -v f="$FREE_T" -v n="$NEED_T" 'BEGIN{exit !(f < n)}'; then
    echo "[p86] ABORT: /efs_rl free ${FREE_T}T < ${NEED_T}T for ${MAX_CKPT_KEEP} engram checkpoints（可设 MY_ENV_P86_SAVE_FREQ=-1 不存盘）" | tee -a "$DRV" >&2
    exit 3
  fi
fi

# -----------------------------------------------------------------------------
# 传给 1105/1103 链的差异
# -----------------------------------------------------------------------------
export MY_ENV_DSV41_MODEL_OVERRIDE="$ENGRAM_MODEL"
export MY_ENV_DSV41_REQUIRE_ENGRAM=1
export MY_ENV_DSV41_EXPECT_EXPERTS=384
export MY_ENV_DSV41_ENGRAM_SHARE_MODEL="$SHARE"
export MY_ENV_DSV41_STEPS="$STEPS"
export MY_ENV_DSV41_SAVE_FREQ="$SAVE_FREQ"
export MY_ENV_DSV41_MAX_CKPT_KEEP="$MAX_CKPT_KEEP"
export MY_ENV_DSV41_TEST_FREQ=-1
export MY_ENV_DSV41_VAL_BEFORE_TRAIN=False
export MY_ENV_DSV41_CP_SIZE=1
export MY_ENV_DSV41_LOAD_FORMAT=safetensors
export MY_ENV_DSV41_ENGRAM_DP_SHARED=1
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=0
# [1110 thpoff] 训练/vLLM 两侧进程关 THP（默认 on，可用 MY_ENV_DSV41_DISABLE_THP=0 回基线）
export MY_ENV_DSV41_DISABLE_THP="$THP_OFF"
export MY_ENV_DSV41_FADVISE_DIR="$ENGRAM_MODEL"
export MY_ENV_DSV41_EXPERIMENT_TAG="engram-share-cp1-thpoff"
export MY_ENV_DSV41_RUN_TAG="engram-share-cp1-thpoff-math-dapo17k"

echo "[p86] MY_DEBUG_LOG share_env MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$MY_ENV_DSV41_ENGRAM_SHARE_MODEL model=$ENGRAM_MODEL cp=1 dp_shared=1 load_format=safetensors sleep_pin=0 sleep_level=1 thp_off=$THP_OFF fadvise_dir=$MY_ENV_DSV41_FADVISE_DIR" | tee -a "$DRV"

echo "[p86] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$DRV"
bash "$BASE_PATH" >> "$DRV" 2>&1
RC=$?
echo "[p86] MY_DEBUG_LOG base_done rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# rank0 证据汇总
# -----------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
  RUN_LOG=$(ls -t "$LOGDIR"/dsv41-8node-engram-share-cp1-thpoff-math-dapo17k-*/train-math-dapo17k.log 2>/dev/null | head -1 || true)
  if [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ]; then
    {
      echo "[p86] == evidence =="
      echo "[p86] run_log=$RUN_LOG"
      echo "[p86] p0_lines=$(grep -c '\[1109 P0\]' "$RUN_LOG" 2>/dev/null || true) loaded_lines=$(grep -c 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p86] thp_off_vllm=$(grep -c 'vllm_host_numa_env.*thp_off=1 thp_ok=True' "$RUN_LOG" 2>/dev/null || true) thp_off_train=$(grep -c 'host_numa_applied.*thp_off=1 thp_ok=True' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p86] sleep_level1_lines=$(grep -c 'Sleep mode (level=1)' "$RUN_LOG" 2>/dev/null || true) sleep_max_seconds=$(grep -h 'to fall asleep' "$RUN_LOG" 2>/dev/null | grep -o 'took [0-9.]* seconds' | awk '{print $2}' | sort -n | tail -1)"
      echo "[p86] bind_port_errors=$(grep -c 'Communication_Error_Bind_IP_Port' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p86] engine_dead_errors=$(grep -c 'EngineDeadError' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p86] shm_broadcast_warns=$(grep -c 'No available shared memory broadcast' "$RUN_LOG" 2>/dev/null || true)"
      grep -h 'Training Progress' "$RUN_LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*/[0-9]* \[[^]]*\]' | tail -1
      grep -h 'training/global_step' "$RUN_LOG" 2>/dev/null | tail -1 | grep -o 'training/global_step:[0-9]*\|timing_s/update_actor:[0-9.]*\|timing_s/step:[0-9.]*\|cpu_memory_used_gb:[0-9.]*\|rollout_actor_probs_pearson_corr:[-0-9.]*' | tr '\n' ' '
      echo
    } | tee -a "$DRV"
  else
    echo "[p86] WARN: 未找到 thpoff 运行日志（$LOGDIR/dsv41-8node-engram-share-cp1-thpoff-math-dapo17k-*）" | tee -a "$DRV"
  fi
fi

echo "[p86] MY_DEBUG_LOG p86_done host=$HOST node=$NODE_RANK profile=$PROFILE share=$SHARE thp_off=$THP_OFF rc=$RC time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC"
