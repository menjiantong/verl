#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次创建（80.5.25.130）。成功后回填日志路径与结论。
# =============================================================================
# 1107 v2 DSV4.1 单机 16 卡：Engram P0（actor/ref 共用 frozen 表）A/B 单档
#
# 相对 v1（1107_run_dsv41_130_engram_share_ab_v1.sh）只加一个"页缓存看护"：
#   v1 的 on 档（SHARE=1）已经证明 P0 生效（第二次表加载 0.1 s、2 of 2 reused），
#   但 13:18 在 rollout 的 _gather_mm_embeddings 里挂在一个**小** pinned 分配上：
#     MemAvailable=702GiB 但 MemFree=60GiB，Cached=1099GiB（其中 Shmem=456GiB）
#     aclrtMallocHostWithCfg 207001 / rtsMallocHost out of memory
#   文件页缓存 ~643GiB 是读 checkpoint 留下的干净页，驱动 pinned 分配不会去回收它。
#   v2 在 `actor and ref model engine initialized` 之后对权重目录做 fadvise(DONTNEED)，
#   只回收 checkpoint 自己的干净页（全局 drop_caches 在本机约 20 GiB/min，太慢）；
#   训练期间 MemFree < 120GiB 时后台补 drop_caches（不阻塞训练）最多 2 次。
#   反向思考：这不是训练逻辑改动，等价于云道 relief 档的 posix_fadvise，
#   两个 arm 都会走同一套看护，A/B 仍然只差 SHARE 开关。
# =============================================================================
# 1107 DSV4.1 单机 16 卡：Engram P0（actor/ref 共用 frozen 表）A/B 单档
#
# 基线（规则 7.3/7.4：本脚本只写差异）：
#   my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh
#     → my_scripts/1105_run_dsv41_1node_engram_ab_v1.sh
#     → examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
#
# 本脚本相对 1106 基线只加两件事：
#   1) USE_KL=1 —— 这是 A/B 成立的前提。130 的 1106 跑（默认 USE_KL=0）日志里
#      `create worker group dict_keys(['actor_rollout'])`，同进程只有 1 份 engram 表
#      （`host-offload Engram tables loaded` 只出现 1 次）。云道死亡现场是
#      actor_rollout_ref：ref 先 build、actor 后 build，各分配一份表，
#      2 层 × 2 引擎 = 4 × ~259 GiB。P0 只有在 ref 存在时才有意义。
#   2) MY_ENV_DSV41_ENGRAM_SHARE_MODEL=0|1 —— P0 开关（默认 0，旧行为）。
#
# 两个 arm（同一份 top32-bf16、同一 dapo512、同一 3 步；只差 SHARE 开关）：
#   off  SHARE=0  旧行为：ref 与 actor 各分配一份表，预期 2 次 loaded
#   on   SHARE=1  P0：ref 分配并登记，actor 复用同一 storage，预期 1 次 loaded
#
# 验收（脚本末尾写 summary）：
#   off: `host-offload Engram tables loaded` = 2，无 `[1107 P0]`；
#   on : 同上 = 1，且有 `[1107 P0] 2 of 2 engram table(s) reused from the ref engine`；
#   on 的 MemAvailable 最低值应比 off 高 ~518 GiB（2 层 × ~259）；
#   on 的 startup 少一次表加载（130 上 ~150 s/次）；
#   两个 arm 都 3/3 步跑完，loss/grad_norm/pearson 同量级。
#
# 权重（用户指定 32 专家）：
#   /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
#
# 机器要求：1 机 × 16 卡，MemAvailable ≥ 2000 GiB（与 1106 基线同一门槛），NPU 空闲。
#
# 分支 / commit（规则 7.5）：
#   verl        dev_dsv41_rebase1030_cp_engram @ e18fe6ec（本地含 P0 与在途改动）
#   vllm-ascend from_9a38d36_1030           @ e3f220b05
#   FSDPTurbo   dev_dsv41_1027              @ 3ff111b（本地含 P0 与在途改动）
# =============================================================================
set -uo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

ARM=${MY_ENV_DSV41_SHARE_ARM:-off}
case "$ARM" in
  off) SHARE=0 ;;
  on)  SHARE=1 ;;
  *)
    echo "[share-ab] ABORT: MY_ENV_DSV41_SHARE_ARM=$ARM (expect off|on)" >&2
    exit 2
    ;;
esac
STEPS=${MY_ENV_DSV41_SHARE_STEPS:-3}

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
# [1109 matrix] 实验标签只影响日志名；未设置时保持原文件名。
TAG=${MY_ENV_DSV41_EXP_TAG:-}
DRV="logs/1107_share_ab_${ARM}${TAG:+_$TAG}_${TS}.driver.log"
echo "[share-ab] MY_DEBUG_LOG share_ab_start arm=$ARM share=$SHARE steps=$STEPS time=$(date '+%F %T') host=$(hostname) repo=$REPO" | tee -a "$DRV"

# 主机内存预检（规则 3.2）：两个 arm 用同一门槛，保证 A/B 起点一致。
# 1106 基线自己还会再查一次 2000 GiB（engram 档），这里只是提前给出可读的失败原因。
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
echo "[share-ab] MY_DEBUG_LOG share_ab_precheck MemAvailable=${AVAIL_GB}GiB need=2000GiB" | tee -a "$DRV"
if [ "$AVAIL_GB" -lt 2000 ]; then
  echo "[share-ab] ABORT: MemAvailable=${AVAIL_GB}GiB < 2000GiB" | tee -a "$DRV" >&2
  exit 3
fi

# ---------------- 与 1106 基线的唯一差异 ----------------
# [1107] 打开 ref：复现云道的 actor_rollout_ref（ref 先 build、actor 后 build）。
export USE_KL=1
# [1107 P0] 0=旧行为（各分配一份），1=actor 复用 ref 的物理表。
export MY_ENV_DSV41_ENGRAM_SHARE_MODEL=$SHARE
# ---------------- 其余沿用 1106 基线 ----------------
# [1109 matrix] 允许外层指定 engram|noengram；默认 engram，历史行为不变。
export MY_ENV_DSV41_AB_ARM=${MY_ENV_DSV41_AB_ARM:-engram}
export MY_ENV_DSV41_AB_STEPS=$STEPS
# 与 130 的 1106 跑同一数据集（dapo512），保持 step 时间/载荷可比。
export MY_ENV_DSV41_AB_DATASET=dapo512
export MY_ENV_DSV41_AB_RECOMPUTE_FULL=1
export MY_ENV_DSV41_AB_PROBE=1
export MY_ENV_DSV41_AB_PROBE_CALLS=64

echo "[share-ab] MY_DEBUG_LOG share_ab_knobs use_kl=1 share_model=$SHARE dataset=dapo512 recompute_full=1 probe=1 sleep_pin=1 free_cache_engine=False" | tee -a "$DRV"

# [1107 v2] 页缓存看护：只在模型初始化完成后做，避免影响 checkpoint 读取本身。
# 主手段是对权重目录 fadvise(DONTNEED)：只回收 checkpoint 自己的干净页，秒级；
# 训练期 MemFree 低于 120GiB 时，后台补 drop_caches（不阻塞训练）最多 2 次。
WEIGHTS=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
mem_free_gib() { awk '/^MemFree:/{print int($2/1024/1024)}' /proc/meminfo; }
cache_dropper() {
  local _stage=0 _drops=0 _last=0
  while :; do
    if [ "$_stage" = 0 ] && grep -aq "actor and ref model engine initialized" "$DRV" 2>/dev/null; then
      /usr/local/python3.11.15/bin/python3 my_scripts/1107_fadvise_weights_v1.py "$WEIGHTS" >> "$DRV" 2>&1
      _stage=1; _last=$(date +%s)
      echo "[share-ab] MY_DEBUG_LOG fadvise stage=after_init MemFree=$(mem_free_gib)GiB time=$(date '+%H:%M:%S')" >> "$DRV"
    elif [ "$_stage" = 1 ] && [ "$_drops" -lt 2 ] && [ $(( $(date +%s) - _last )) -ge 120 ] \
         && [ "$(mem_free_gib)" -lt 120 ]; then
      ( echo 1 > /proc/sys/vm/drop_caches 2>/dev/null ) &
      _drops=$((_drops+1)); _last=$(date +%s)
      echo "[share-ab] MY_DEBUG_LOG drop_caches stage=low_mem drop=$_drops MemFree=$(mem_free_gib)GiB time=$(date '+%H:%M:%S')" >> "$DRV"
    fi
    sleep 5
  done
}
cache_dropper &
DROP_PID=$!
trap 'kill "$DROP_PID" 2>/dev/null || true' EXIT

bash my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh >> "$DRV" 2>&1
RC=$?
kill "$DROP_PID" 2>/dev/null || true
echo "[share-ab] MY_DEBUG_LOG share_ab_done arm=$ARM rc=$RC time=$(date '+%F %T')" | tee -a "$DRV"

# ---------------- 结果汇总（规则 12.1/12.4） ----------------
SUM="logs/1107_share_ab_${ARM}${TAG:+_$TAG}_${TS}.summary.txt"
MEMLOG=$(grep -ao 'memlog=logs/[^ ]*\.mem\.log' "$DRV" | tail -1 | cut -d= -f2)
{
  echo "== 1107 P0 Engram share A/B summary"
  echo "arm=$ARM share_model=$SHARE steps=$STEPS rc=$RC time=$(date '+%F %T')"
  echo "driver_log=$DRV"
  echo
  echo "-- engram table loads"
  echo "loaded_lines=$(grep -acF 'host-offload Engram tables loaded' "$DRV")"
  grep -aF 'host-offload Engram tables loaded' "$DRV" | sed 's/\x1b\[[0-9;]*m//g'
  echo "p0_lines=$(grep -acF '[1107 P0]' "$DRV")"
  grep -aF '[1107 P0]' "$DRV" | sed 's/\x1b\[[0-9;]*m//g'
  echo
  echo "-- steps (global_step / update_actor / step / cpu_mem / pearson / grad_norm)"
  grep -a 'training/global_step:' "$DRV" | tail -3 | while IFS= read -r line; do
    echo "$line" | grep -oE 'training/global_step:[0-9]+|timing_s/update_actor:[0-9.]+|timing_s/step:[0-9.]+|actor/perf/cpu_memory_used_gb:[0-9.]+|training/rollout_actor_probs_pearson_corr:[0-9.]+|actor/grad_norm:[0-9.eE+-]+' | tr '\n' ' '
    echo
  done
  echo
  echo "-- host memory"
  if [ -n "${MEMLOG:-}" ] && [ -f "$MEMLOG" ]; then
    echo "memlog=$MEMLOG"
    awk '{for(i=1;i<=NF;i++){split($i,kv,"="); v=kv[2]+0;
            if(kv[1]=="MemAvailable" && (min=="" || v<min)) min=v;
            if(kv[1]=="Shmem" && v>sm) sm=v;
            if(kv[1]=="Cached" && v>cm) cm=v}}
         END{printf "min_MemAvailable=%dGiB max_Shmem=%dGiB max_Cached=%dGiB\n", min, sm, cm}' "$MEMLOG"
  else
    echo "memlog not found"
  fi
  echo
  echo "-- timing (startup)"
  grep -aE 'host-offload Engram tables loaded|actor and ref model engine initialized' "$DRV" | sed 's/\x1b\[[0-9;]*m//g' | tail -5
} > "$SUM" 2>&1
echo "[share-ab] MY_DEBUG_LOG share_ab_summary arm=$ARM summary=$SUM" | tee -a "$DRV"
cat "$SUM" >> "$DRV"

exit "$RC"
