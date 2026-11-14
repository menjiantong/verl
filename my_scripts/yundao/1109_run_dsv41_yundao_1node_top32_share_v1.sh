#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-09 首次创建（云道 1 机 × 16 卡 top32 + P0 共用表）
# =============================================================================
# 1109 DSV4.1 云道单机（1 机 × 16 卡，32 专家）Engram P0：actor/ref 共用同一份表
#
# 目的：把 80.5.25.130 上已验证的 P0（worklog_20261106_dsv41_engram_p0_130.md）
#   原样搬到云道单机路径：1 机 × 16 卡、top32 权重、dapo512、3 步；唯一
#   实验变量是 MY_ENV_DSV41_ENGRAM_SHARE_MODEL=1 —— ref 先建表并登记，actor
#   命中同一 key 后复用同一个 storage，跳过第二次 torch.empty(pin_memory=True)
#   与第二次 checkpoint 回填。
#
# 依据一（云道同配置的死现场，1106 单机 P3，2026-11-06 17:16–19:19）：
#   top32 engram、EP16、USE_KL=1、dapo512；日志
#   logs/dsv4-yundao_1106_p3s_engram_20261106_171614.log 与 report_1107_engram_*.txt：
#     没有任何一步完成；死在 update_actor 建优化器状态；
#     cg 日志末条 MemFree=12GiB MemAvail=8GiB Cached=366GiB Shmem=366GiB，
#     **MemAvail 最低 0 GiB**（124 P1 同配置是 870 GiB）。
#   → 云道单机不是"慢"，是主机内存被打穿。P0 把它预计省下的 ~518 GiB
#     （130 实测差值）还给 update_actor，是这台机器能否出第一步的关键。
#
# 依据二（130 A/B，2026-11-06 14:15–16:35，同代码/同权重/同数据/同 USE_KL=1）：
#   SHARE=0：第二次表加载 37.6 s；第 3 步 vLLM pinned 207001 死掉；
#            min MemAvailable 153 GiB，cpu_memory_used 2232.7 GiB；
#   SHARE=1：第二次表加载 2.1 s；日志 "[1107 P0] 2 of 2 ... reused"；
#            3/3 步 rc=0；min MemAvailable 675 GiB，cpu_memory_used 1708.2 GiB。
#
# 基线（规则 7.3/7.4：本脚本只写差异，其余全部沿用基线）：
#   my_scripts/yundao/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
#     → 1105_..._cp1_fadvise_v3.sh → 1105_..._cp1_v1.sh
#       → 1104_..._memprobe_v1.sh → 1104_..._noval_v2.sh
#         → 1104_..._recompute_full_v1.sh → 1103_..._engram_cp_v1.sh
#           → dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# 与 1106 单机基线相比，本脚本只做两件事：
#   1) 打开 MY_ENV_DSV41_ENGRAM_SHARE_MODEL（默认 1；=0 即 130 的 SHARE=0 对照）；
#   2) 启动前检查代码里确实有 P0（开关 + 进程级注册表 + loader 的 _shared_hit
#      分支），缺失就明确报错 —— 避免"脚本以为开了、代码没打"的假实验。
#
# 用法（云道控制台：资源规格 **1 机 × 16 卡**，启动命令只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1109_run_dsv41_yundao_1node_top32_share_v1.sh
#
# 提交前确认：MA_NUM_HOSTS 必须 = 1（基线第 250 行也会再校验一次）。
#
# 可覆盖开关（全部 MY_ENV_ 前缀、都有默认值，默认即本次实验）：
#   MY_ENV_P41_SHARE=0|1          默认 1（1=actor/ref 共用表；0=对照）
#   MY_ENV_P41_STEPS=3            默认 3 步（与 130 A/B / 124 P1 对齐）
#   MY_ENV_P41_DATASET=dapo512    默认 dapo512（dapo17k 可切全量）
#   MY_ENV_P41_MIN_MEM_GB=1400    启动前单节点最低 MemAvailable（fresh 节点约 1939）
#   MY_ENV_P41_MIN_SHM_GB=450     /dev/shm 最低余量（vLLM 节点内共享表 366 GiB）
#
# 回滚：SHARE=0 重跑，或 bash 1106_run_dsv41_yundao_1node_engram_p1_v1.sh。
# =============================================================================
set -uo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
LOGDIR=$MIRROR_ROOT/verl/logs
mkdir -p "$LOGDIR" 2>/dev/null || true

BASE_PATH=$MIRROR_ROOT/verl/my_scripts/yundao/1106_run_dsv41_yundao_1node_engram_p1_v1.sh
HOST_OFFLOAD=$MIRROR_ROOT/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
DSV41_IMPL=$MIRROR_ROOT/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
MODEL=/efs_rl/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_train_512.parquet
VAL_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

SHARE=${MY_ENV_P41_SHARE:-1}
STEPS=500
DATASET=${MY_ENV_P41_DATASET:-dapo512}
MIN_MEM_GB=${MY_ENV_P41_MIN_MEM_GB:-1400}
MIN_SHM_GB=${MY_ENV_P41_MIN_SHM_GB:-450}

TS=$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')
HOST=$(hostname 2>/dev/null || echo unknown)
DRV=$LOGDIR/1109_1node_top32_share_${TS}.driver.log
mkdir -p "$LOGDIR" 2>/dev/null || true

echo "[p41] MY_DEBUG_LOG p41_start host=$HOST ts=$TS share=$SHARE steps=$STEPS dataset=$DATASET MA_NUM_HOSTS=${MA_NUM_HOSTS:-unset} VC_TASK_INDEX=${VC_TASK_INDEX:-unset} log=$DRV" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 1. 节点数预检（单机脚本误投成 8 机时立刻失败，不白等加载）
# -----------------------------------------------------------------------------
if [ "${MA_NUM_HOSTS:-0}" -ne 1 ]; then
  echo "[p41] ABORT: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, this is the 1-node script (expect 1)" | tee -a "$DRV" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 2. 代码预检：P0 必须在（开关、注册表、shard_ 分支、loader 跳过分支）
#    反向思考：只查环境变量会漏掉"代码没打补丁"；四个锚点任一缺失都说明
#    工作树不是本轮验证过的那份，直接退出比跑出一份假 A/B 便宜得多。
# -----------------------------------------------------------------------------
for _f in "$HOST_OFFLOAD" "$DSV41_IMPL" "$BASE_PATH"; do
  if [ ! -f "$_f" ]; then
    echo "[p41] ABORT: required file missing: $_f" | tee -a "$DRV" >&2
    exit 3
  fi
done
_missing=0
grep -q "MY_ENV_DSV41_ENGRAM_SHARE_MODEL" "$HOST_OFFLOAD" || { echo "[p41] ABORT: P0 switch missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_SHARED_ENGRAM_TABLES" "$HOST_OFFLOAD" || { echo "[p41] ABORT: P0 registry missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "supports_table_sharing" "$HOST_OFFLOAD" || { echo "[p41] ABORT: P0 supports_table_sharing missing in host_offload_engram.py" >&2; _missing=1; }
grep -q "_shared_hit" "$DSV41_IMPL" || { echo "[p41] ABORT: P0 shared-hit loader branch missing in fsdp_turbo_dsv41_impl.py" >&2; _missing=1; }
if [ "$_missing" -ne 0 ]; then
  echo "[p41] 提示：补丁在 $MIRROR_ROOT/mjt_workplace/patches/1109_p0_engram_share_yundao_{host,impl}_v1.patch" | tee -a "$DRV" >&2
  exit 3
fi
echo "[p41] MY_DEBUG_LOG code_check_ok host_offload=$(md5sum "$HOST_OFFLOAD" | awk '{print $1}') impl=$(md5sum "$DSV41_IMPL" | awk '{print $1}')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 3. 权重预检：top32（32 专家）+ 2 层 engram 表；与 1103 的 384 专家断言区分开
# -----------------------------------------------------------------------------
python3 - "$MODEL" <<'PY' 2>&1 | tee -a "$DRV"
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cfg = json.loads((root / "config.json").read_text())
text = cfg.get("text_config", cfg)
ids = text.get("engram_layer_ids") or []
nexp = text.get("n_routed_experts")
assert ids, f"{root}: engram_layer_ids empty, not an engram checkpoint"
assert nexp == 32, f"{root}: n_routed_experts={nexp}, this script needs the top32 checkpoint"
w = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
tables = [k for k in w if "engram" in k and k.endswith("embed.weight")]
assert len(tables) == len(ids), f"{root}: engram tables {tables} vs ids {ids}"
print(f"[p41] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "[p41] ABORT: model preflight failed for $MODEL" | tee -a "$DRV" >&2
  exit 3
fi
for _f in "$TRAIN_FILE" "$VAL_FILE"; do
  [ -f "$_f" ] || { echo "[p41] ABORT: dataset missing: $_f" | tee -a "$DRV" >&2; exit 3; }
done

# -----------------------------------------------------------------------------
# 4. 主机内存 / /dev/shm 预检（每个节点都执行；单机作业只有 1 个节点）
#    预算：1106 单机实测整机占用 ~1400 GiB（top32、vLLM 常驻、Shmem 峰值 479）；
#    P0 再把 trainer 的 2 份表合成 1 份，预期再省 ~518 GiB，余量更大。
#    门槛仍按未省钱的情况设 1400 GiB，宁可早失败也不要在加载 1 小时后才撞墙。
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[p41] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB need_shm=${MIN_SHM_GB}GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[p41] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB（上一轮残留？等容器回收或换节点）" | tee -a "$DRV" >&2
  exit 3
fi
if [ "$SHM_AVAIL_GB" -lt "$MIN_SHM_GB" ]; then
  echo "[p41] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < ${MIN_SHM_GB}GiB（vLLM engram 共享表约 366 GiB）" | tee -a "$DRV" >&2
  exit 3
fi

# -----------------------------------------------------------------------------
# 5. 交给 1106 单机基线（拓扑 / 数据集 / 常驻 / 探针全部在它里面）
#    P0 开关是唯一新增的环境变量；其余逐项保持 1106 的默认。
# -----------------------------------------------------------------------------
export MY_ENV_DSV41_ENGRAM_SHARE_MODEL="$SHARE"
export MY_ENV_P3_ARM=engram
export MY_ENV_P3_STEPS="$STEPS"
export MY_ENV_P3_DATASET="$DATASET"
export MY_ENV_P3_RESIDENT=1
export MY_ENV_P3_PIN_ENGRAM=1
export MY_ENV_P3_REUSE_HOST_BUF=0
# 1103 的权重断言（top32=32）；1106 也会设，这里显式写一遍便于单看本脚本。
export MY_ENV_DSV41_EXPECT_EXPERTS=32
export MY_ENV_DSV41_REQUIRE_ENGRAM=1
# fadvise 守护指向真正在用的 top32 权重（1106 也会设，双保险）。
export MY_ENV_DSV41_FADVISE_DIR="$MODEL"

echo "[p41] MY_DEBUG_LOG share_env MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$MY_ENV_DSV41_ENGRAM_SHARE_MODEL arm=engram steps=$STEPS dataset=$DATASET model=$MODEL resident=1 pin=1 reuse_host_buf=0" | tee -a "$DRV"

echo "[p41] MY_DEBUG_LOG base_start base=$(basename "$BASE_PATH") time=$(date '+%F %T')" | tee -a "$DRV"
# 不用 tee：1105/1104 会起后台守护，它们继承 stdout 管道会让 tee 永远不返回
# （1106 单机脚本 12:11 那次的僵尸作业就是这个问题）。直接重定向即可。
bash "$BASE_PATH" >> "$DRV" 2>&1
RC=$?
echo "[p41] MY_DEBUG_LOG base_done rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# -----------------------------------------------------------------------------
# 6. 证据汇总（从基线自己的 cg 日志算 min MemAvailable / max Shmem）
#    判据：P0 生效时日志里应出现 "Engram table shared" 与
#    "[1109 P0] N of N engram table(s) reused"；min MemAvailable 应显著高于 130
#    的 153 GiB（对照档），且接近 130 的 675 GiB（共用档）。
# -----------------------------------------------------------------------------
RUN_LOG=$(grep -o 'p3s_start host=.* log=[^ ]*' "$DRV" 2>/dev/null | tail -1 | sed 's/.*log=//')
if [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ]; then
  CGLOG=${RUN_LOG%.log}.cg.log
  {
    echo "[p41] == evidence =="
    echo "[p41] run_log=$RUN_LOG"
    echo "[p41] p0_lines=$(grep -c '\[1109 P0\]' "$RUN_LOG" 2>/dev/null || true) shared_lines=$(grep -c 'Engram table shared' "$RUN_LOG" 2>/dev/null || true) loaded_lines=$(grep -c 'host-offload Engram tables loaded' "$RUN_LOG" 2>/dev/null || true)"
    grep -h '\[1109 P0\]' "$RUN_LOG" 2>/dev/null | sed -n '1,3p'
    if [ -f "$CGLOG" ]; then
      awk '
        /cg-probe/ && /MemAvail=/ {
          for (i=1;i<=NF;i++) {
            if ($i ~ /^MemAvail=/) { v=$i; sub(/GiB/,"",v); sub(/^MemAvail=/,"",v); if (min=="" || v+0<min) min=v+0 }
            if ($i ~ /^Shmem=/)    { v=$i; sub(/GiB/,"",v); sub(/^Shmem=/,"",v);   if (v+0>max) max=v+0 }
            if ($i ~ /^MemFree=/)  { v=$i; sub(/GiB/,"",v); sub(/^MemFree=/,"",v); if (minf=="" || v+0<minf) minf=v+0 }
          }
        }
        END { printf "[p41] cg_summary min_MemAvailable=%sGiB min_MemFree=%sGiB max_Shmem=%sGiB\n", (min==""?"NA":min), (minf==""?"NA":minf), (max==""?"NA":max) }
      ' "$CGLOG"
      echo "[p41] cg_log=$CGLOG"
    fi
  } | tee -a "$DRV"
else
  echo "[p41] WARN: 未能从 driver 日志解析到基线日志路径（RUN_LOG=$RUN_LOG），跳过证据汇总" | tee -a "$DRV"
fi

echo "[p41] MY_DEBUG_LOG p41_done host=$HOST share=$SHARE rc=$RC time=$(date '+%F %T') log=$DRV" | tee -a "$DRV"
exit "$RC"
