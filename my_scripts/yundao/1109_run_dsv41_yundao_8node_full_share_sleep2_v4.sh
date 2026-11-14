#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 创建（8 机全尺寸 + Engram P0 + sleep level=2 + 全局唯一 worker 端口窗口）
# =============================================================================
# 1109 v4：修正 v3 的 rank 语义错误——用 dp_rank*TP+tp_rank 作为全局索引
#
# v3（TS=20261109_200451）的结论：
#   1) sleep level=2 正常：Sleep mode (level=2)，sleep_max=12.83 s，MemAvail ~890 GiB；
#   2) 但 21:10:55 仍报 bind 60200（4 条 / 3 个节点 IP），随后 51 条 EngineDeadError，
#      0/2 步；per_worker_windows 只出现 rank=0/2 两个窗口；
#   3) 根因：v3 用 self.rank 切窗口，而 self.rank 是 **DP 组内的 TP rank（0..7）**
#      —— init_distributed_environment 的 world_size = TP*PP = 8，rank 必须 < 8。
#      所有 DP 的 TP0 worker 都拿到 60200，同一节点 DP12_TP0 与 DP13_TP0 照样撞。
#
# v4 的修复（代码补丁 mjt_workplace/patches/1109_patch_hccl_per_rank_window_v3.py）：
#   全局唯一索引 idx = data_parallel_rank*TP*PP + self.rank（0..127）：
#     host 60200 + idx*16 .. +15   → 60200-62231
#     npu  63200 + idx*16 .. +15   → 63200-65247
#   同一节点 16 个 worker 的 idx 必不相同；与 trainer 60000-60150 / 61000-61150 不重叠。
#
# 与 v3 的差异：
#   1) 代码补丁改为 dp_rank*TP+tp_rank 索引（worker.py md5 0991b5d1…）；
#   2) 运行标签 sleep2c → sleep2d、驱动日志 sleep2v3 → sleep2v4、前缀 p84 → p85；
#   3) 证据汇总保留 per_rank_port_windows 计数。
#
# 判据（2 步）：
#   a) 0 条 Communication_Error_Bind_IP_Port、0 条 EngineDeadError；
#   b) Sleep mode (level=2)，sleep_max_s 几十秒；
#   c) 第 1 步跑完，pearson ≈0.9996~0.9998、loss 不炸；
#   d) 每节点 MemAvailable 比 level=1 多 ~440 GiB。
#
# 用法（云道控制台：资源规格 **8 机 × 16 卡**）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1109_run_dsv41_yundao_8node_full_share_sleep2_v4.sh
#
# 可覆盖开关（同 v3）：
#   MY_ENV_P81_PROFILE=validate|longrun（默认 validate：2 步、不存盘）
#   MY_ENV_P81_STEPS / MY_ENV_P81_SAVE_FREQ / MY_ENV_P81_MAX_CKPT_KEEP
#   MY_ENV_P81_SHARE=0|1（默认 1）
#   MY_ENV_VLLM_SLEEP_LEVEL=2（改 1 即回 level=1，用于 A/B）
#
# 回滚：
#   代码：cp mjt_workplace/backup/worker.py.bak_20261109_sleep2v3  vllm-ascend/vllm_ascend/worker/worker.py
#   行为：MY_ENV_ENGINE_HCCL_PER_RANK=0
# =============================================================================
set -eo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
HOST_OFFLOAD=$MIRROR_ROOT/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
DSV41_IMPL=$MIRROR_ROOT/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
ASYNC_SERVER=$MIRROR_ROOT/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
VLLM_WORKER=$MIRROR_ROOT/vllm-ascend/vllm_ascend/worker/worker.py
ENGRAM_MODEL=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
VAL_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

PROFILE=${MY_ENV_P81_PROFILE:-validate}
case "$PROFILE" in
  validate) DEF_STEPS=2;   DEF_SAVE=-1;  DEF_KEEP=2 ;;
  longrun)  DEF_STEPS=500; DEF_SAVE=110; DEF_KEEP=2 ;;
  *) echo "[p85] ABORT: MY_ENV_P81_PROFILE=$PROFILE (expect validate|longrun)" >&2; exit 2 ;;
esac
STEPS=${MY_ENV_P81_STEPS:-$DEF_STEPS}
SAVE_FREQ=${MY_ENV_P81_SAVE_FREQ:-$DEF_SAVE}
MAX_CKPT_KEEP=${MY_ENV_P81_MAX_CKPT_KEEP:-$DEF_KEEP}
SHARE=${MY_ENV_P81_SHARE:-1}
MIN_MEM_GB=${MY_ENV_P81_MIN_MEM_GB:-1600}
MIN_SHM_GB=${MY_ENV_P81_MIN_SHM_GB:-450}
SLEEP_LEVEL=${MY_ENV_VLLM_SLEEP_LEVEL:-2}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
NODE_RANK=${VC_TASK_INDEX:-0}
DRV=$LOGDIR/1109_8node_sleep2v4_${TS}_node${NODE_RANK}.driver.log

echo "[p85] MY_DEBUG_LOG p85_start host=$HOST node=$NODE_RANK ts=$TS profile=$PROFILE share=$SHARE steps=$STEPS save_freq=$SAVE_FREQ keep=$MAX_CKPT_KEEP sleep_level=$SLEEP_LEVEL MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} log=$DRV" | tee -a "$DRV"

if [ "${MA_NUM_HOSTS:-0}" -ne 8 ]; then
  echo "[p85] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 8-node script (expect 8)" | tee -a "$DRV" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 代码预检：P0 锚点 + sleep level 覆写 + HCCL 端口窗口
# -----------------------------------------------------------------------------
for _f in "$HOST_OFFLOAD" "$DSV41_IMPL" "$BASE_PATH" "$ASYNC_SERVER" "$VLLM_WORKER"; do
  if [ ! -f "$_f" ]; then
    echo "[p85] ABORT: required file missing: $_f" | tee -a "$DRV" >&2
    exit 3
  fi
done
_missing=0
grep -q "MY_ENV_DSV41_ENGRAM_SHARE_MODEL" "$HOST_OFFLOAD" || { echo "[p85] ABORT: P0 switch missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "supports_table_sharing" "$HOST_OFFLOAD" || { echo "[p85] ABORT: P0 supports_table_sharing missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_shared_hit" "$DSV41_IMPL" || { echo "[p85] ABORT: P0 shared-hit loader branch missing in fsdp_turbo_dsv41_impl.py" >&2; _missing=1; }
grep -q "MY_ENV_VLLM_SLEEP_LEVEL" "$ASYNC_SERVER" || { echo "[p85] ABORT: sleep-level env override missing in vllm_async_server.py（先打 1109_sleep_level2_yundao_v1.patch）" >&2; _missing=1; }
grep -q "replica_rank % 16" "$ASYNC_SERVER" || { echo "[p85] ABORT: HCCL 端口窗口补丁缺失 in vllm_async_server.py（先打 1109_hccl_port_window_yundao_v1.patch）" >&2; _missing=1; }
grep -q "per-worker HCCL port window" "$VLLM_WORKER" || { echo "[p85] ABORT: 每 worker 端口窗口补丁缺失 in vllm-ascend/worker/worker.py（先跑 mjt_workplace/patches/1109_patch_hccl_per_rank_window_v2.py）" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[p85] 提示：补丁在 $MIRROR_ROOT/mjt_workplace/patches/" | tee -a "$DRV" >&2
  exit 3
fi
if [ -n "${MY_ENV_ENGINE_HCCL_HOST_RANGE:-}" ]; then
  echo "[p85] WARN: MY_ENV_ENGINE_HCCL_HOST_RANGE=$MY_ENV_ENGINE_HCCL_HOST_RANGE 会覆盖按 replica 的端口窗口，同一节点的 2 个 server 又会共用同一段（不建议）" | tee -a "$DRV" >&2
fi
echo "[p85] MY_DEBUG_LOG code_check_ok host_offload=$(md5sum "$HOST_OFFLOAD" | awk '{print $1}') impl=$(md5sum "$DSV41_IMPL" | awk '{print $1}') async_server=$(md5sum "$ASYNC_SERVER" | awk '{print $1}') worker=$(md5sum "$VLLM_WORKER" | awk '{print $1}')" | tee -a "$DRV"

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
print(f"[p85] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY
for _f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$_f" ] || { echo "[p85] ABORT: dataset missing: $_f" | tee -a "$DRV" >&2; exit 3; }
done

# -----------------------------------------------------------------------------
# 主机内存 / /dev/shm / 磁盘预检
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[p85] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB need_shm=${MIN_SHM_GB}GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[p85] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SHM_AVAIL_GB" -lt "$MIN_SHM_GB" ]; then
  echo "[p85] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < ${MIN_SHM_GB}GiB（engram 共享表约 366 GiB）" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SAVE_FREQ" -ge 0 ]; then
  FREE_T=$(df -Pk /efs_rl | awk 'NR==2{printf "%.1f", $4/1024/1024/1024}')
  NEED_T=$(awk -v k="$MAX_CKPT_KEEP" 'BEGIN{printf "%.1f", k*1.5}')
  echo "[p85] MY_DEBUG_LOG ckpt_space_check free=${FREE_T}T need=${NEED_T}T keep=${MAX_CKPT_KEEP}" | tee -a "$DRV"
  if awk -v f="$FREE_T" -v n="$NEED_T" 'BEGIN{exit !(f < n)}'; then
    echo "[p85] ABORT: /efs_rl free ${FREE_T}T < ${NEED_T}T for ${MAX_CKPT_KEEP} engram checkpoints（可设 MY_ENV_P81_SAVE_FREQ=-1 不存盘）" | tee -a "$DRV" >&2
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
# [1109 sleep2-v3] 每 worker 按全局 rank 切 HCCL 端口窗口（修 60200 bind 竞态）
export MY_ENV_ENGINE_HCCL_PER_RANK=1
export MY_ENV_VLLM_SLEEP_LEVEL="$SLEEP_LEVEL"
export MY_ENV_DSV41_FADVISE_DIR="$ENGRAM_MODEL"
export MY_ENV_DSV41_EXPERIMENT_TAG="engram-share-cp1-sleep2d"
export MY_ENV_DSV41_RUN_TAG="engram-share-cp1-sleep2d-math-dapo17k"

echo "[p85] MY_DEBUG_LOG share_env MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$MY_ENV_DSV41_ENGRAM_SHARE_MODEL model=$ENGRAM_MODEL cp=1 dp_shared=1 load_format=safetensors sleep_pin=0 sleep_level=$SLEEP_LEVEL hccl_window=per_worker_rank fadvise_dir=$MY_ENV_DSV41_FADVISE_DIR" | tee -a "$DRV"

echo "[p85] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$DRV"
bash "$BASE_PATH" >> "$DRV" 2>&1
RC=$?
echo "[p85] MY_DEBUG_LOG base_done rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# rank0 证据汇总
# -----------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
  RUN_LOG=$(ls -t "$LOGDIR"/dsv41-8node-engram-share-cp1-sleep2d-math-dapo17k-*/train-math-dapo17k.log 2>/dev/null | head -1 || true)
  if [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ]; then
    {
      echo "[p85] == evidence =="
      echo "[p85] run_log=$RUN_LOG"
      echo "[p85] p0_lines=$(grep -c '\[1109 P0\]' "$RUN_LOG" 2>/dev/null || true) loaded_lines=$(grep -c 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p85] sleep_level2_lines=$(grep -c 'Sleep mode (level=2)' "$RUN_LOG" 2>/dev/null || true) sleep_level1_lines=$(grep -c 'Sleep mode (level=1)' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p85] sleep_max_seconds=$(grep -h 'to fall asleep' "$RUN_LOG" 2>/dev/null | grep -o 'took [0-9.]* seconds' | awk '{print $2}' | sort -n | tail -1)"
      echo "[p85] bind_port_errors=$(grep -c 'Communication_Error_Bind_IP_Port' "$RUN_LOG" 2>/dev/null || true) per_rank_port_windows=$(grep -c 'per-worker HCCL port window' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p85] engine_dead_errors=$(grep -c 'EngineDeadError' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p85] shm_broadcast_warns=$(grep -c 'No available shared memory broadcast' "$RUN_LOG" 2>/dev/null || true)"
      grep -h 'Training Progress' "$RUN_LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*/[0-9]* \[[^]]*\]' | tail -1
      grep -h 'training/global_step' "$RUN_LOG" 2>/dev/null | tail -1 | grep -o 'training/global_step:[0-9]*\|timing_s/update_actor:[0-9.]*\|timing_s/step:[0-9.]*\|cpu_memory_used_gb:[0-9.]*\|rollout_actor_probs_pearson_corr:[-0-9.]*' | tr '\n' ' '
      echo
    } | tee -a "$DRV"
  else
    echo "[p85] WARN: 未找到 sleep2d 运行日志（$LOGDIR/dsv41-8node-engram-share-cp1-sleep2d-math-dapo17k-*）" | tee -a "$DRV"
  fi
fi

echo "[p85] MY_DEBUG_LOG p83_done host=$HOST node=$NODE_RANK profile=$PROFILE share=$SHARE sleep_level=$SLEEP_LEVEL rc=$RC time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC"
