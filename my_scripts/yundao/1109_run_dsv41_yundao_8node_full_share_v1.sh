#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（云道 8 机 × 16 卡 全尺寸 384 专家 + P0）
# =============================================================================
# 1109 DSV4.1 云道 8 机（8 机 × 16 卡 = 128 rank）全尺寸 384 专家 Engram P0
#
# 目的：在最新代码（P0 已移植到云道 FSDPTurbo/verl 工作树）上，跑全尺寸
#   384 专家 + Engram 的 8 机路径，验证 actor/ref 共用同一份表在 EP128 下
#   同样生效，并给出与 1106 8 机 P1 同链的对照数据。
#
# 对照基线（1106 8 机 P1 engram，2026-11-06，日志名时间 11:08，同一条 1105 链、2 步）：
#   日志 logs/dsv4-yundao_1106_p3_engram_20261106_110832.log：
#     step1 update_actor 1030.1 s / step 2707.0 s / cpu_mem 1588.8 GiB
#     step2 update_actor  413.1 s / step 1017.1 s / cpu_mem 1582.1 GiB
#     cg 日志 min MemAvailable 376 GiB
#   → P0 在 EP128 下每节点省 2 层 × 2 引擎 → 1 引擎的 ~45.8 GiB pinned，
#     预期 cpu_mem 降到 ~1542 GiB、min MemAvailable 升到 ~420 GiB，
#     且第二次表加载从数十秒降到秒级（130 单机实测 37.6 s → 2.1 s）。
#
# 基线（规则 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
#     → 1105_..._cp1_v1.sh（CP=1）
#       → 1104_..._memprobe_v1.sh（host 5s / npu 30s / proc 60s 三路探针）
#         → 1104_..._recompute_full_noval_v2.sh（VAL=False / TEST=-1 / STEPS 透传）
#           → 1104_..._recompute_full_v1.sh（recompute_plan=["model.layers.{*}"])
#             → 1103_..._engram_cp_v1.sh（全尺寸 engram 预检 + 环境）
#               → dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#   这条链就是 E1 之后验证过的 8 机 engram 路径（fadvise 守护 + 整层重算 + CP1）。
#
# 与 1106 8 机 P1 的差异（只有三项，不碰训练数学）：
#   1) 打开 MY_ENV_DSV41_ENGRAM_SHARE_MODEL（默认 1；=0 即 130 的 SHARE=0 对照）；
#      8 机 EP128 下每 rank 每层表 1.43 GiB，2 层 × 2 引擎 → 1 引擎，
#      每节点省 ~45.8 GiB 隐藏 pinned 内存，同时省掉第二次 checkpoint 回填；
#   2) 不挂 fetch / opt-state 探针（P1 是定位慢点用的，长跑/验证不挂）；
#      保留链上原有的 memprobe + fadvise 守护 + cgroup 明细；
#   3) 步数与保存可配（默认 validate：2 步、不存盘，与 1106 P1 的 2 步同口径）。
#      /efs_rl 当前只剩 3.6T；全尺寸含 engram 单份 ckpt 约 1.4T，长跑用 keep=2。
#
# 用法（云道控制台：资源规格 **8 机 × 16 卡**，启动命令只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1109_run_dsv41_yundao_8node_full_share_v1.sh
#
# 可覆盖开关（全部 MY_ENV_ 前缀、都有默认值）：
#   MY_ENV_P81_PROFILE=validate|longrun
#       validate（默认）：2 步、不存盘、不评估（与 1106 P1 同口径）
#       longrun        ：500 步、每 110 步存、keep=2（需要 /efs_rl ≥3T 空闲）
#   MY_ENV_P81_STEPS / MY_ENV_P81_SAVE_FREQ / MY_ENV_P81_MAX_CKPT_KEEP
#       显式覆盖 profile 的步数/保存；SAVE_FREQ<0 表示不存
#   MY_ENV_P81_SHARE=0|1          默认 1（1=actor/ref 共用表；0=对照）
#   MY_ENV_P81_MIN_MEM_GB=1600    启动前每节点最低 MemAvailable（1103 默认）
#   MY_ENV_P81_MIN_SHM_GB=450     /dev/shm 最低余量（vLLM 节点内共享表约 366 GiB）
#
# 回滚：SHARE=0 重跑，或 bash 1106_run_dsv41_yundao_8node_engram_p1_v1.sh。
# =============================================================================
set -eo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
HOST_OFFLOAD=$MIRROR_ROOT/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
DSV41_IMPL=$MIRROR_ROOT/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
ENGRAM_MODEL=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
VAL_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

PROFILE=${MY_ENV_P81_PROFILE:-validate}
case "$PROFILE" in
  validate) DEF_STEPS=2;   DEF_SAVE=-1;  DEF_KEEP=2 ;;
  longrun)  DEF_STEPS=500; DEF_SAVE=110; DEF_KEEP=2 ;;
  *) echo "[p81] ABORT: MY_ENV_P81_PROFILE=$PROFILE (expect validate|longrun)" >&2; exit 2 ;;
esac
STEPS=500
SAVE_FREQ=${MY_ENV_P81_SAVE_FREQ:-$DEF_SAVE}
MAX_CKPT_KEEP=${MY_ENV_P81_MAX_CKPT_KEEP:-$DEF_KEEP}
SHARE=${MY_ENV_P81_SHARE:-1}
MIN_MEM_GB=${MY_ENV_P81_MIN_MEM_GB:-1600}
MIN_SHM_GB=${MY_ENV_P81_MIN_SHM_GB:-450}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
NODE_RANK=${VC_TASK_INDEX:-0}
DRV=$LOGDIR/1109_8node_full_share_${TS}_node${NODE_RANK}.driver.log

echo "[p81] MY_DEBUG_LOG p81_start host=$HOST node=$NODE_RANK ts=$TS profile=$PROFILE share=$SHARE steps=$STEPS save_freq=$SAVE_FREQ keep=$MAX_CKPT_KEEP MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} log=$DRV" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 1. 节点数预检（最终基线第 250 行也会校验，这里先失败以免白等）
# -----------------------------------------------------------------------------
if [ "${MA_NUM_HOSTS:-0}" -ne 8 ]; then
  echo "[p81] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 8-node script (expect 8)" | tee -a "$DRV" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 2. 代码预检：P0 必须在（开关、注册表、shard_ 分支、loader 跳过分支）
#    8 台共享 /efs_rl，同一份代码；任一锚点缺失说明工作树不是本轮验证过的那份。
# -----------------------------------------------------------------------------
for _f in "$HOST_OFFLOAD" "$DSV41_IMPL" "$BASE_PATH"; do
  if [ ! -f "$_f" ]; then
    echo "[p81] ABORT: required file missing: $_f" | tee -a "$DRV" >&2
    exit 3
  fi
done
_missing=0
grep -q "MY_ENV_DSV41_ENGRAM_SHARE_MODEL" "$HOST_OFFLOAD" || { echo "[p81] ABORT: P0 switch missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_SHARED_ENGRAM_TABLES" "$HOST_OFFLOAD" || { echo "[p81] ABORT: P0 registry missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "supports_table_sharing" "$HOST_OFFLOAD" || { echo "[p81] ABORT: P0 supports_table_sharing missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_shared_hit" "$DSV41_IMPL" || { echo "[p81] ABORT: P0 shared-hit loader branch missing in fsdp_turbo_dsv41_impl.py" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[p81] 提示：补丁在 $MIRROR_ROOT/mjt_workplace/patches/1109_p0_engram_share_yundao_{host,impl}_v1.patch" | tee -a "$DRV" >&2
  exit 3
fi
echo "[p81] MY_DEBUG_LOG code_check_ok host_offload=$(md5sum "$HOST_OFFLOAD" | awk '{print $1}') impl=$(md5sum "$DSV41_IMPL" | awk '{print $1}')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 3. 权重预检：全尺寸 384 专家 + engram 表（与 1103 同款断言）
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
print(f"[p81] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY
for _f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$_f" ] || { echo "[p81] ABORT: dataset missing: $_f" | tee -a "$DRV" >&2; exit 3; }
done

# -----------------------------------------------------------------------------
# 4. 主机内存 / /dev/shm / 磁盘预检（每个节点都会执行）
#    预算（单节点，1103 头部的实测口径）：noengram 基线 cpu_memory ≈1049 GB
#    + vLLM 共享 engram 表 366 GiB + trainer engram 表 EP128 分片 2×22.9 GiB。
#    P0 把 trainer 的 2 份合成 1 份，省 ~45.8 GiB/节点。
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[p81] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB need_shm=${MIN_SHM_GB}GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[p81] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SHM_AVAIL_GB" -lt "$MIN_SHM_GB" ]; then
  echo "[p81] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < ${MIN_SHM_GB}GiB（engram 共享表约 366 GiB）" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SAVE_FREQ" -ge 0 ]; then
  FREE_T=$(df -Pk /efs_rl | awk 'NR==2{printf "%.1f", $4/1024/1024/1024}')
  NEED_T=$(awk -v k="$MAX_CKPT_KEEP" 'BEGIN{printf "%.1f", k*1.5}')
  echo "[p81] MY_DEBUG_LOG ckpt_space_check free=${FREE_T}T need=${NEED_T}T keep=${MAX_CKPT_KEEP}" | tee -a "$DRV"
  if awk -v f="$FREE_T" -v n="$NEED_T" 'BEGIN{exit !(f < n)}'; then
    echo "[p81] ABORT: /efs_rl free ${FREE_T}T < ${NEED_T}T for ${MAX_CKPT_KEEP} engram checkpoints（可设 MY_ENV_P81_SAVE_FREQ=-1 不存盘）" | tee -a "$DRV" >&2
    exit 3
  fi
fi

# -----------------------------------------------------------------------------
# 5. 传给 1105/1103 链的差异（全部写在本脚本内，不在命令行 export）
#    注意：1105 cp1 链会把 CP 固定为 1；1104 noval v2 会把 TEST_FREQ 固定为 -1、
#    VAL_BEFORE_TRAIN 固定为 False（与 final 基线默认一致，长跑口径不变）。
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
export MY_ENV_DSV41_FADVISE_DIR="$ENGRAM_MODEL"
# 日志/实验名标记：便于在 logs/ 与 ckpts/ 下区分本实验。
export MY_ENV_DSV41_EXPERIMENT_TAG="engram-share-cp1"
export MY_ENV_DSV41_RUN_TAG="engram-share-cp1-math-dapo17k"

echo "[p81] MY_DEBUG_LOG share_env MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$MY_ENV_DSV41_ENGRAM_SHARE_MODEL model=$ENGRAM_MODEL cp=1 dp_shared=1 load_format=safetensors sleep_pin=0 fadvise_dir=$MY_ENV_DSV41_FADVISE_DIR" | tee -a "$DRV"

echo "[p81] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$DRV"
# 不用 tee：1105 v3 的 fadvise 守护是后台进程，继承 stdout 会让 tee 永不返回。
bash "$BASE_PATH" >> "$DRV" 2>&1
RC=$?
echo "[p81] MY_DEBUG_LOG base_done rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 6. rank0 证据汇总：找最新的 engram-share 运行目录，统计 P0 与表加载日志
#    判据：应出现 "Engram table shared"（ref 分配后登记）与
#    "[1109 P0] N of N engram table(s) reused"（actor 命中，不再回填）。
# -----------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
  RUN_LOG=$(ls -t "$LOGDIR"/dsv41-8node-engram-share-cp1-math-dapo17k-*/train-math-dapo17k.log 2>/dev/null | head -1 || true)
  if [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ]; then
    {
      echo "[p81] == evidence =="
      echo "[p81] run_log=$RUN_LOG"
      echo "[p81] p0_lines=$(grep -c '\[1109 P0\]' "$RUN_LOG" 2>/dev/null || true) shared_lines=$(grep -c 'Engram table shared' "$RUN_LOG" 2>/dev/null || true) loaded_lines=$(grep -c 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null || true)"
      grep -h '\[1109 P0\]' "$RUN_LOG" 2>/dev/null | sed -n '1,3p'
      grep -h 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null | sed -n '1,2p'
    } | tee -a "$DRV"
  else
    echo "[p81] WARN: 未找到 engram-share 运行日志（$LOGDIR/dsv41-8node-engram-share-cp1-math-dapo17k-*）" | tee -a "$DRV"
  fi
fi

echo "[p81] MY_DEBUG_LOG p81_done host=$HOST node=$NODE_RANK profile=$PROFILE share=$SHARE rc=$RC time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC"
