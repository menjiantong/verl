#!/usr/bin/env bash
# Baseline (pre-optimization) actor+ref profiling launcher.
# Reconstructed from the original 2026-10-20 run:
#   verl/logs/perf_profile_20261020_230432.outer.log:133
# Non-profiler args are the base-script defaults:
#   GSM8K, KL=1, bsz 8x8, resp 1024, micro=1, FFN recompute,
#   param/optimizer/offload_policy all True, rollout util=0.5.
# Usage:
#   bash _launch_baseline_prof.sh                 # 3 steps, profile step 2, DB analysis on
#   STEPS=2 PROFILE_STEP=1 bash _launch_baseline_prof.sh
#   ANALYSIS=False bash _launch_baseline_prof.sh  # raw trace only (no *.db)
set -e
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts

TS=$(date +%Y%m%d_%H%M%S)
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram}
STEPS=${STEPS:-3}
PROFILE_STEP=${PROFILE_STEP:-2}
ANALYSIS=${ANALYSIS:-True}
SAVE_PATH=${SAVE_PATH:-/mnt/share/m00899630/workspace/dsv41-verl-copy/profiles/run_baseline_profile_${TS}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-perf_baseline_profile_${TS}}
LOG=${LOG:-../logs/perf_baseline_profile_${TS}.outer.log}

PROF_ARGS="global_profiler.tool=npu"
PROF_ARGS="${PROF_ARGS} global_profiler.steps=[${PROFILE_STEP}]"
PROF_ARGS="${PROF_ARGS} global_profiler.profile_continuous_steps=False"
PROF_ARGS="${PROF_ARGS} global_profiler.save_path=${SAVE_PATH}"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.enable=True"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.all_ranks=False"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.ranks=[0]"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory]"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.tool_config.npu.level=level1"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${ANALYSIS}"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.enable=True"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.all_ranks=False"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.ranks=[0]"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.tool_config.npu.contents=[npu,cpu]"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.tool_config.npu.level=level1"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${ANALYSIS}"
PROF_ARGS="${PROF_ARGS} actor_rollout_ref.rollout.profiler.enable=False"
export TRAIN_EXTRA_ARGS="${PROF_ARGS}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[baseline-prof] DRY_RUN=1"
  echo "TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS}"
  exit 0
fi

mkdir -p "$SAVE_PATH" "$(dirname "$LOG")"
echo "[baseline-prof] steps=$STEPS profile_step=$PROFILE_STEP analysis=$ANALYSIS"
echo "[baseline-prof] save_path=$SAVE_PATH"
echo "[baseline-prof] log=$LOG"
 bash dsv41_grpo_1node_21.sh single "$STEPS"
