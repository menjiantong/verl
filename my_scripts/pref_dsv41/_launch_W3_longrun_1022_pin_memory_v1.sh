#!/usr/bin/env bash
# W3 pin_memory + profiler 采集脚本 v1
# 基于用户原脚本 _launch_W3_longrun_1022_pin_memory.sh 修改，只修复阻塞问题：
#   1) MY_IP 原为 80.5.25.119，在 .120 上会让 Ray/HCCL 绑错 IP -> 改为 80.5.25.120
#   2) 原脚本两次 export TRAIN_EXTRA_ARGS，第二次覆盖了 PROF_ARGS -> 合并为一次 export
#   3) EXPERIMENT_NAME 原被第二次 export 覆盖 -> 只保留 pin/profile 命名
# 其余（pin、profiler、3 steps、profile step 2、save path）保持用户原意。
set -eo pipefail

cd /mnt/share/m00899630/workspace/dsv41-verl/verl
BASE=my_scripts/dsv41_grpo_2node_new.sh
[ -f "$BASE" ] || { echo "[W3-pin] missing $BASE"; exit 1; }
grep -Fq '${TRAIN_EXTRA_ARGS:-}' "$BASE" || { echo "[W3-pin] $BASE missing TRAIN_EXTRA_ARGS"; exit 1; }

STEPS=${STEPS:-3}
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/W3_pin_memory_${TS}.outer.log
RUN_LOG="${RUN_LOG:-logs/perf_test_enable_pin_memory_profile_${STEPS}_$(date +%Y%m%d_%H%M%S).log}"

PROFILE_STEP=${PROFILE_STEP:-2}
ANALYSIS=${ANALYSIS:-True}
# 用户原脚本的输出路径，保留；注意它在 dsv41-verl-copy 下，不在工作目录内
SAVE_PATH=${SAVE_PATH:-/mnt/share/m00899630/workspace/dsv41-verl-copy/profiles/run_enable_pin_memory_profile_${TS}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-perf_test_enable_pin_memory_profile_${TS}}

# NPU profiler 参数（actor + ref，只采 rank0，level1，含 npu/cpu/memory）
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

# pin 开关：base 用它决定是否注入 fsdp_plan.pin_memory=false；=1 表示保留 FSDPTurbo 默认 pinned
export VERL_DSV41_OFFLOAD_PIN_MEMORY=1
# .120 的业务 IP/网卡（enp48s3u1u1 已确认存在）
export MY_IP=80.5.25.120
export NIC=enp48s3u1u1

export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MAX_PROMPT_LEN=1152
export MAX_RESPONSE_LEN=512
export USE_KL=0
export ROLLOUT_GPU_MEM_UTIL=0.6
export VERL_DSV41_VISION=1   # freeze defaults to 1 in fsdp_turbo_dsv41_impl.py

# 关键修复：PROF_ARGS 与 W3 extras 合并，避免第二次 export 覆盖 profiler 参数
export TRAIN_EXTRA_ARGS="${PROF_ARGS} \
trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"

# No performance optimizations（与用户原脚本一致）
unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP

mkdir -p logs
echo "[W3-pin] ip=$MY_IP nic=$NIC steps=$STEPS profile_step=$PROFILE_STEP"
echo "[W3-pin] log=$LOG inner=$RUN_LOG save_path=$SAVE_PATH"
bash "$BASE" single "$STEPS" 2>&1 | tee "$LOG"
