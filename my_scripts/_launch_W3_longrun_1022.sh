#!/usr/bin/env bash
# W3 long run, based on dsv41_grpo_2node_new.sh single. No perf optimizations.
# Usage: bash my_scripts/_launch_W3_longrun_1022.sh [STEPS]
set -eo pipefail

cd /mnt/share/m00899630/workspace/dsv41-verl/verl
BASE=my_scripts/dsv41_grpo_2node_new.sh
[ -f "$BASE" ] || { echo "[W3] missing $BASE"; exit 1; }
grep -Fq '${TRAIN_EXTRA_ARGS:-}' "$BASE" || { echo "[W3] $BASE missing TRAIN_EXTRA_ARGS"; exit 1; }

STEPS=${1:-500}
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/W3_longrun_${TS}.outer.log
RUN_LOG="${RUN_LOG:-logs/W3_long_run_baseline${STEPS}_$(date +%Y%m%d_%H%M%S).log}"


export MY_IP=80.5.25.119
export NIC=enp48s3u1u1

export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MAX_PROMPT_LEN=1152
export MAX_RESPONSE_LEN=512
export USE_KL=0
export ROLLOUT_GPU_MEM_UTIL=0.6
export VERL_DSV41_VISION=1   # freeze defaults to 1 in fsdp_turbo_dsv41_impl.py
export EXPERIMENT_NAME=GRPO-DSV41-countbenchqa-W3-longrun-${TS}

# Hydra extras: epoch cap, VLM load, agent/thinking, W3 sampler
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"

# No performance optimizations
unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP

mkdir -p logs
echo "[W3] ip=$MY_IP nic=$NIC steps=$STEPS"
echo "[W3] log=$LOG inner=$RUN_LOG"
# Foreground: keeps the process attached to this terminal and shows output.
# For background: nohup bash my_scripts/_launch_W3_longrun_1022.sh > /tmp/W3.out 2>&1 &
bash "$BASE" single "$STEPS" 2>&1 | tee "$LOG"
