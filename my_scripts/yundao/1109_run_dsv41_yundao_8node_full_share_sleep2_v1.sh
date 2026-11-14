#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 创建（8 机全尺寸 + Engram P0 + vLLM sleep level=2）
# =============================================================================
# 1109 DSV4.1 云道 8 机（8 × 16 = 128 rank）全尺寸 384 专家 Engram P0 + sleep2
#
# 目的：把 vLLM 的 sleep 从 level=1 改成 level=2，解决长跑里最大的一块无效开销。
#
# 为什么 level=2 才是对的（实测 + vLLM 官方文档）：
#   1) verl 每次 rollout 开始前都会全量灌权重（update_weights）。本作业实测：
#        15:24:34 done, time cost: 20.41s
#        16:15:39 done, time cost: 14.49s（1550 tensors / 24.27 GiB）
#   2) 而 level=1 在 sleep 时把同一份权重（27.5 GiB/卡）备份到主机 CPU，唤醒后
#      1.1 s 再拷回来 —— 紧接着就被 update_weights 覆盖，等于白做。
#   3) vLLM 官方 docs/features/sleep_mode.md：
#        "Level 2 sleep is good for ... update the model, where previous model
#         weights are not needed, e.g. RLHF weight update"
#      推荐流程正是 sleep(2) → wake_up(["weights"]) → update_weights → wake_up(["kv_cache"])，
#      与 verl _sleep_hybrid/wake 的顺序完全一致。
#   4) 代价（可证伪）：level=1 每次 sleep 每节点要 16 × 27.5 GiB = 440 GiB 主机内存，
#      且个别卡 D2H 长尾 8~9 min（2026-11-09 长跑 slept 8m51s / 15m58s）。
#      level=2 不备份 → 主机应多出 ~440 GiB，sleep 应回到几十秒。
#
# 与 1109_run_dsv41_yundao_8node_full_share_v1.sh 的差异（只有三处）：
#   1) export MY_ENV_VLLM_SLEEP_LEVEL=2
#      （需要 mjt_workplace/patches/1109_sleep_level2_yundao_v1.patch 已打进
#        verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py；本脚本启动前会校验）
#   2) 修掉 STEPS 写死 500 的 bug：STEPS=${MY_ENV_P81_STEPS:-$DEF_STEPS}
#      （云道上在跑的 v1 是 STEPS=500 写死版，所以 profile=validate 也跑了 500 步、
#        且 save_freq=-1 不存盘）
#   3) driver/run 日志名加 sleep2 标记，便于和 v1 的 level=1 日志区分
#
# 前置条件（本作业配置已满足，脚本会校验代码锚点）：
#   - MTP：enable/enable_rollout/enable_train 全为 False（MTP drafter 权重不由
#     update_weights 恢复，若开启 MTP rollout 必须留在 level=1）；
#   - lora_rank=0（LoRA 只更新 adapter，必须 level=1）。
#   这两条正是 _resolve_sleep_level 原有的两个 return 1 分支，补丁保留了它们。
#
# 基线（规则 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
#     → 1105_..._cp1_v1.sh（CP=1）
#       → 1104_..._memprobe_v1.sh（host 5s / npu 30s / proc 60s 三路探针）
#         → 1104_..._recompute_full_noval_v2.sh（VAL=False / TEST=-1 / STEPS 透传）
#           → 1104_..._recompute_full_v1.sh（recompute_plan=["model.layers.{*}"])
#             → 1103_..._engram_cp_v1.sh（全尺寸 engram 预检 + 环境）
#               → dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# 待验证的判据（跑 2 步即可看出）：
#   a) 日志出现 `Sleep mode (level=2)`，且不再有 `(level=1)`；
#   b) `It took N seconds to fall asleep` 的最大值从 525/954 s 降到几十秒；
#   c) 训练期间每节点 MemAvailable 比 level=1 多 ~440 GiB（memprobe 可直接看）；
#   d) step 1 的 pearson 与 level=1 同量级（≈0.9996~0.9998），loss 不炸 ——
#      这是"权重确实被 update_weights 完整恢复"的最终判据。
#
# 用法（云道控制台：资源规格 **8 机 × 16 卡**，启动命令只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1109_run_dsv41_yundao_8node_full_share_sleep2_v1.sh
#
# 可覆盖开关（全部 MY_ENV_ 前缀、都有默认值）：
#   MY_ENV_P81_PROFILE=validate|longrun
#       validate（默认）：2 步、不存盘、不评估；longrun：500 步、每 110 步存、keep=2
#   MY_ENV_P81_STEPS / MY_ENV_P81_SAVE_FREQ / MY_ENV_P81_MAX_CKPT_KEEP
#       显式覆盖 profile 的步数/保存；SAVE_FREQ<0 表示不存
#   MY_ENV_P81_SHARE=0|1          默认 1（1=actor/ref 共用表；0=对照）
#   MY_ENV_P81_MIN_MEM_GB=1600    启动前每节点最低 MemAvailable
#   MY_ENV_P81_MIN_SHM_GB=450     /dev/shm 最低余量
#   MY_ENV_VLLM_SLEEP_LEVEL=2     固定为 2；改 1 即回到 1109 v1 的行为（A/B 用）
#
# 回滚：
#   代码：cp mjt_workplace/backup/vllm_async_server.py.bak_20261109_sleep2 \
#            verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
#   行为：MY_ENV_VLLM_SLEEP_LEVEL=1 或不设（未打补丁时默认 NPU=1）
# =============================================================================
set -eo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
HOST_OFFLOAD=$MIRROR_ROOT/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
DSV41_IMPL=$MIRROR_ROOT/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
ASYNC_SERVER=$MIRROR_ROOT/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
ENGRAM_MODEL=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
VAL_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

PROFILE=${MY_ENV_P81_PROFILE:-validate}
case "$PROFILE" in
  validate) DEF_STEPS=2;   DEF_SAVE=-1;  DEF_KEEP=2 ;;
  longrun)  DEF_STEPS=500; DEF_SAVE=110; DEF_KEEP=2 ;;
  *) echo "[p82] ABORT: MY_ENV_P81_PROFILE=$PROFILE (expect validate|longrun)" >&2; exit 2 ;;
esac
STEPS=500
SAVE_FREQ=${MY_ENV_P81_SAVE_FREQ:-$DEF_SAVE}
MAX_CKPT_KEEP=${MY_ENV_P81_MAX_CKPT_KEEP:-$DEF_KEEP}
SHARE=${MY_ENV_P81_SHARE:-1}
MIN_MEM_GB=${MY_ENV_P81_MIN_MEM_GB:-1600}
MIN_SHM_GB=${MY_ENV_P81_MIN_SHM_GB:-450}
SLEEP_LEVEL=${MY_ENV_VLLM_SLEEP_LEVEL:-2}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
NODE_RANK=${VC_TASK_INDEX:-0}
DRV=$LOGDIR/1109_8node_sleep2_${TS}_node${NODE_RANK}.driver.log

echo "[p82] MY_DEBUG_LOG p82_start host=$HOST node=$NODE_RANK ts=$TS profile=$PROFILE share=$SHARE steps=$STEPS save_freq=$SAVE_FREQ keep=$MAX_CKPT_KEEP sleep_level=$SLEEP_LEVEL MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} log=$DRV" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 1. 节点数预检
# -----------------------------------------------------------------------------
if [ "${MA_NUM_HOSTS:-0}" -ne 8 ]; then
  echo "[p82] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 8-node script (expect 8)" | tee -a "$DRV" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 2. 代码预检：P0 锚点 + sleep level 补丁锚点
# -----------------------------------------------------------------------------
for _f in "$HOST_OFFLOAD" "$DSV41_IMPL" "$BASE_PATH" "$ASYNC_SERVER"; do
  if [ ! -f "$_f" ]; then
    echo "[p82] ABORT: required file missing: $_f" | tee -a "$DRV" >&2
    exit 3
  fi
done
_missing=0
grep -q "MY_ENV_DSV41_ENGRAM_SHARE_MODEL" "$HOST_OFFLOAD" || { echo "[p82] ABORT: P0 switch missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_SHARED_ENGRAM_TABLES" "$HOST_OFFLOAD" || { echo "[p82] ABORT: P0 registry missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "supports_table_sharing" "$HOST_OFFLOAD" || { echo "[p82] ABORT: P0 supports_table_sharing missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_shared_hit" "$DSV41_IMPL" || { echo "[p82] ABORT: P0 shared-hit loader branch missing in fsdp_turbo_dsv41_impl.py" >&2; _missing=1; }
grep -q "MY_ENV_VLLM_SLEEP_LEVEL" "$ASYNC_SERVER" || { echo "[p82] ABORT: sleep-level env override missing in vllm_async_server.py（先打 1109_sleep_level2_yundao_v1.patch）" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[p82] 提示：补丁在 $MIRROR_ROOT/mjt_workplace/patches/1109_sleep_level2_yundao_v1.patch（P0 见 1109_p0_engram_share_yundao_{host,impl}_v1.patch）" | tee -a "$DRV" >&2
  exit 3
fi
echo "[p82] MY_DEBUG_LOG code_check_ok host_offload=$(md5sum "$HOST_OFFLOAD" | awk '{print $1}') impl=$(md5sum "$DSV41_IMPL" | awk '{print $1}') async_server=$(md5sum "$ASYNC_SERVER" | awk '{print $1}')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 3. 权重预检：全尺寸 384 专家 + engram 表
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
print(f"[p82] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY
for _f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$_f" ] || { echo "[p82] ABORT: dataset missing: $_f" | tee -a "$DRV" >&2; exit 3; }
done

# -----------------------------------------------------------------------------
# 4. 主机内存 / /dev/shm / 磁盘预检
#    level=2 不再需要 440 GiB/节点 的 sleep 权重备份，但启动前口径保持与 v1 一致，
#    避免把"睡醒后主机更宽裕"误判成启动门槛。
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[p82] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB need_shm=${MIN_SHM_GB}GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[p82] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SHM_AVAIL_GB" -lt "$MIN_SHM_GB" ]; then
  echo "[p82] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < ${MIN_SHM_GB}GiB（engram 共享表约 366 GiB）" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SAVE_FREQ" -ge 0 ]; then
  FREE_T=$(df -Pk /efs_rl | awk 'NR==2{printf "%.1f", $4/1024/1024/1024}')
  NEED_T=$(awk -v k="$MAX_CKPT_KEEP" 'BEGIN{printf "%.1f", k*1.5}')
  echo "[p82] MY_DEBUG_LOG ckpt_space_check free=${FREE_T}T need=${NEED_T}T keep=${MAX_CKPT_KEEP}" | tee -a "$DRV"
  if awk -v f="$FREE_T" -v n="$NEED_T" 'BEGIN{exit !(f < n)}'; then
    echo "[p82] ABORT: /efs_rl free ${FREE_T}T < ${NEED_T}T for ${MAX_CKPT_KEEP} engram checkpoints（可设 MY_ENV_P81_SAVE_FREQ=-1 不存盘）" | tee -a "$DRV" >&2
    exit 3
  fi
fi

# -----------------------------------------------------------------------------
# 5. 传给 1105/1103 链的差异
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
# level=2 下 pin 无意义（不备份权重），保留 0 以便 A/B 回 level=1 时仍是验证过的路径。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=0
# 本次的核心差异：sleep 时不备份权重，唤醒后由 update_weights 重新灌入。
export MY_ENV_VLLM_SLEEP_LEVEL="$SLEEP_LEVEL"
export MY_ENV_DSV41_FADVISE_DIR="$ENGRAM_MODEL"
# 日志/实验名标记：与 v1（level=1）区分。
export MY_ENV_DSV41_EXPERIMENT_TAG="engram-share-cp1-sleep2"
export MY_ENV_DSV41_RUN_TAG="engram-share-cp1-sleep2-math-dapo17k"

echo "[p82] MY_DEBUG_LOG share_env MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$MY_ENV_DSV41_ENGRAM_SHARE_MODEL model=$ENGRAM_MODEL cp=1 dp_shared=1 load_format=safetensors sleep_pin=0 sleep_level=$SLEEP_LEVEL fadvise_dir=$MY_ENV_DSV41_FADVISE_DIR" | tee -a "$DRV"

echo "[p82] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$DRV"
# 不用 tee：1105 v3 的 fadvise 守护是后台进程，继承 stdout 会让 tee 永不返回。
bash "$BASE_PATH" >> "$DRV" 2>&1
RC=$?
echo "[p82] MY_DEBUG_LOG base_done rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 6. rank0 证据汇总：P0 + 表加载 + sleep level / 睡眠耗时 / shm 警告
# -----------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
  RUN_LOG=$(ls -t "$LOGDIR"/dsv41-8node-engram-share-cp1-sleep2-math-dapo17k-*/train-math-dapo17k.log 2>/dev/null | head -1 || true)
  if [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ]; then
    {
      echo "[p82] == evidence =="
      echo "[p82] run_log=$RUN_LOG"
      echo "[p82] p0_lines=$(grep -c '\[1109 P0\]' "$RUN_LOG" 2>/dev/null || true) shared_lines=$(grep -c 'Engram table shared' "$RUN_LOG" 2>/dev/null || true) loaded_lines=$(grep -c 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null || true)"
      grep -h '\[1109 P0\]' "$RUN_LOG" 2>/dev/null | sed -n '1,3p'
      grep -h 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null | sed -n '1,2p'
      echo "[p82] sleep_level2_lines=$(grep -c 'Sleep mode (level=2)' "$RUN_LOG" 2>/dev/null || true) sleep_level1_lines=$(grep -c 'Sleep mode (level=1)' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p82] sleep_max_seconds=$(grep -h 'to fall asleep' "$RUN_LOG" 2>/dev/null | grep -o 'took [0-9.]* seconds' | awk '{print $2}' | sort -n | tail -1)"
      echo "[p82] shm_broadcast_warns=$(grep -c 'No available shared memory broadcast' "$RUN_LOG" 2>/dev/null || true)"
      echo "[p82] forced_level_logs=$(grep -c '\[1109 sleep2\] sleep level forced' "$RUN_LOG" 2>/dev/null || true)"
      grep -h '\[1109 sleep2\] sleep level forced' "$RUN_LOG" 2>/dev/null | sed -n '1,2p'
    } | tee -a "$DRV"
  else
    echo "[p82] WARN: 未找到 sleep2 运行日志（$LOGDIR/dsv41-8node-engram-share-cp1-sleep2-math-dapo17k-*）" | tee -a "$DRV"
  fi
fi

echo "[p82] MY_DEBUG_LOG p82_done host=$HOST node=$NODE_RANK profile=$PROFILE share=$SHARE sleep_level=$SLEEP_LEVEL rc=$RC time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC"
