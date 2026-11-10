#!/usr/bin/env bash
# =============================================================================
# 1105 DSV4.1 Engram RL profiling 启动器 v1
#
# 参考：my_scripts/pref_dsv41/_launch_profling_1023.sh（旧 noengram 版）
#   保留其 profiling 口径：global_profiler.tool=npu、指定 step、actor/ref/rollout
#   三角色、只采集 rank0、NPU level1 + npu/cpu/memory/shapes。
# 本脚本改动：
#   * 开启 engram：top32-sft116（带 engram 表），EP16 / GEN_TP8×DP2 / GEN_EP16，
#     vLLM DP 共享表（366GB），sleep 用 pageable 主机内存（与 v3/v4 闸位一致）。
#   * 跑 2 步，只采集第 2 步；保存到 $work_path/output/profiling/（rule 10.3）。
#   * 只采集一个 rank：actor/ref/rollout 均 all_ranks=False、ranks=[0]。
#
# 用法：bash my_scripts/pref_dsv41/1105_run_dsv41_profiling_engram_v1.sh [STEPS] [PROFILE_STEP] [CHIPS]
#   默认 2 2 16；CHIPS=16 时 EP16/GEN_DP2/GEN_EP16。
#   快速配置校验（不占 NPU）：CFG_CHECK=1 bash ...（Hydra --cfg job 打印合成配置）
#   只打印参数不执行：DRY_RUN=1 bash ...
#
# 机器要求：16×Ascend910，MemAvailable ≥2000 GiB；运行前 NPU 空闲。
# 日志：logs/1105_profile_engram_cp1_step${PROFILE_STEP}_${STEPS}steps_<TS>.log
# 脚本 commit：verl 1bfd4aab（首次提交）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-2}
PROFILE_STEP=${2:-2}
CHIPS=${3:-16}
ANALYSIS=${ANALYSIS:-True}   # 与参考脚本一致；只取原始 trace 时传 ANALYSIS=False

if [ "$PROFILE_STEP" -gt "$STEPS" ]; then
  echo "[prof-engram-v1] ABORT: PROFILE_STEP=$PROFILE_STEP > STEPS=$STEPS" >&2
  exit 2
fi

# [1105 engram] 带 engram 表的 SFT 权重；EP16 分片 + vLLM DP 共享表。
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116
export VLLM_ENGRAM_DP_SHARED_MEMORY=1

# [1105 sleep] engram 共享表 + FSDP offload 常驻后 rtMallocHost 会失败，用 pageable。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}

# [1105 transformers] 部分节点原生 5.10.4 缺 transformers.models.qwen4_exp；
# FSDPTurbo 导入链需要 5.21 叠加（125/130 原生 5.21，叠加同版本无行为差异）。
export MY_ENV_DSV41_TF_OVERLAY=${MY_ENV_DSV41_TF_OVERLAY:-/mnt/share/m00899630/env/py521}
if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
  export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
fi

# 16 卡 = TP8×DP2；EP 与 GEN_EP 对齐，触发 LOCAL_EXPERT_EXPORT。
export NDEVICES_PER_NODE=$CHIPS
export FSDP_SIZE=$CHIPS
EP_SIZE=8
[ "$CHIPS" = "16" ] && EP_SIZE=16
export EP_SIZE
export GEN_TP=8
export GEN_DP=$((CHIPS / 8))
export GEN_EP=$((GEN_TP * GEN_DP))

# 本次 profiling 不开 CP（CP 沿用 v3 默认的 CP=1 口径）。
export CP_SIZE=1
export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
export MY_ENV_DSV41_CP_GATHER_OUTPUT=0

# 主机内存预检（rule 3.2）：engram 表 + vLLM 共享表 + offload ≈ 2.0 TB。
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
echo "[prof-engram-v1] STEPS=$STEPS PROFILE_STEP=$PROFILE_STEP CHIPS=$CHIPS ANALYSIS=$ANALYSIS"
echo "[prof-engram-v1] MODEL_PATH=$MODEL_PATH EP=$EP_SIZE GEN_TP=$GEN_TP GEN_DP=$GEN_DP GEN_EP=$GEN_EP"
echo "[prof-engram-v1] MemAvailable=${AVAIL_GB} GiB"
if [ "$AVAIL_GB" -lt 2000 ]; then
  echo "[prof-engram-v1] ABORT: need >=2000 GiB available for engram RL" >&2
  exit 3
fi

# profiling 参数：只采集 rank0；actor/ref 走 level1 全量，rollout 离散采集。
PROF_ARGS=(
  global_profiler.tool=npu
  "global_profiler.steps=[${PROFILE_STEP}]"
  global_profiler.profile_continuous_steps=False
  "global_profiler.save_path=${SAVE_PATH:-/mnt/share/m00899630/workspace/dsv41-verl-rebase/output/profiling/engram_cp1_step${PROFILE_STEP}_$(date +%Y%m%d_%H%M%S)}"
  actor_rollout_ref.actor.profiler.enable=True
  actor_rollout_ref.actor.profiler.all_ranks=False
  "actor_rollout_ref.actor.profiler.ranks=[0]"
  "actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory,shapes]"
  actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
  "actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${ANALYSIS}"
  actor_rollout_ref.ref.profiler.enable=True
  actor_rollout_ref.ref.profiler.all_ranks=False
  "actor_rollout_ref.ref.profiler.ranks=[0]"
  "actor_rollout_ref.ref.profiler.tool_config.npu.contents=[npu,cpu,memory,shapes]"
  actor_rollout_ref.ref.profiler.tool_config.npu.level=level1
  "actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${ANALYSIS}"
  actor_rollout_ref.rollout.profiler.enable=True
  actor_rollout_ref.rollout.profiler.all_ranks=False
  "actor_rollout_ref.rollout.profiler.ranks=[0]"
  actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
  "actor_rollout_ref.rollout.profiler.tool_config.npu.contents=[npu,cpu,memory,shapes]"
  actor_rollout_ref.rollout.profiler.tool_config.npu.level=level1
  actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=False
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[prof-engram-v1] DRY_RUN=1"
  printf '  %s\n' "${PROF_ARGS[@]}"
  exit 0
fi

# 环境（PATH 切到 /usr/local/python3.11.15，含 ray；同时设置缓存目录）。
# 必须在 CFG_CHECK 之前 source：非登录 shell 的默认 python3 是 3.10.12，无 ray。
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

# 快速配置校验：Hydra 只合成配置并退出，不初始化 Ray/NPU。
if [ "${CFG_CHECK:-0}" = "1" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  mkdir -p logs
  CFG_LOG="logs/1105_prof_cfg_check_${TS}.yaml"
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
      trainer.total_training_steps="$STEPS" \
      "${PROF_ARGS[@]}" \
      --cfg job > "$CFG_LOG" 2>&1
  # 断言关键项已写入合成配置
  grep -q "save_path:" "$CFG_LOG" || { echo "[prof-engram-v1] CFG_CHECK missing save_path" >&2; exit 4; }
  grep -q "steps:" "$CFG_LOG" || { echo "[prof-engram-v1] CFG_CHECK missing steps" >&2; exit 4; }
  echo "MY_DEBUG_LOG prof_engram_cfg_check_ok=1 file=$CFG_LOG"
  grep -n -A3 "^global_profiler:" "$CFG_LOG" | head -8
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
SAVE_PATH=${SAVE_PATH:-/mnt/share/m00899630/workspace/dsv41-verl-rebase/output/profiling/engram_cp1_step${PROFILE_STEP}_${TS}}
mkdir -p "$SAVE_PATH" logs
LOG="logs/1105_profile_engram_cp1_step${PROFILE_STEP}_${STEPS}steps_${TS}.log"

# 把最终 save_path 回填到 PROF_ARGS（上面构造时可能是默认时间戳）。
for i in "${!PROF_ARGS[@]}"; do
  case "${PROF_ARGS[$i]}" in
    global_profiler.save_path=*) PROF_ARGS[$i]="global_profiler.save_path=${SAVE_PATH}" ;;
  esac
done

echo "[prof-engram-v1] save_path=$SAVE_PATH"
echo "[prof-engram-v1] log=$LOG"

ray stop --force >/dev/null 2>&1 || true
dsv41_ray_start

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    "${PROF_ARGS[@]}" \
    2>&1 | tee "$LOG"
