#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash top32 bf16 noengram · CountBenchQA-lite VLM GRPO
# 单机 16 卡（0-15），vision 冻结、文本模型 RL。
#
#   bash dsv41_grpo_countbenchqa_1node.sh [STEPS]      # 建议 nohup 运行
#
# 与 dsv41_grpo_2node_new.sh 的差异（本任务专用，勿用于 GSM8K）：
#   * MODEL_PATH   = DeepSeek-V4.1-Flash-top32-bf16-noengram（image_token_id 129264）
#   * TRAIN/TEST   = countbenchqa_lite_{train,val}.parquet（450/50，图片内嵌 bytes）
#   * VERL_DSV41_VISION=1：训练模型构建 32 层 ViT+aligner（冻结）；rollout 由
#     vllm-ascend 的 AscendDeepseekV41ForCausalLM（VL wrapper）处理图片。
#   * default_agent_loop=deepseek_v41_vl_single_turn：紧凑 placeholder 给 vLLM，
#     展开后的 image span + token_types 给训练侧。
#   * USE_KL=0（与 countbench RL 基线一致，跳过 ref 模型以省内存）。
#
# 记录：wiki/worklog_dsv41_2nodes_countbenchqa_rl_1019.md
# =============================================================================
set -eo pipefail
STEPS=${1:-500}

MIRROR_ROOT=/mnt/share/m00899630/workspace/dsv41-verl
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:$MIRROR_ROOT/FSDPTurbo:$MIRROR_ROOT/verl:${PYTHONPATH:-}"
export PATH=/usr/local/python3.11.15/bin:$PATH
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true

MY_IP=${MY_IP:-$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^80\.5\.25\.')}
NIC=${NIC:-enp48s3u1u1}
RUN_LOG="logs/grpo_countbenchqa_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"
export RAY_AUTH_MODE=${RAY_AUTH_MODE:-disabled}
RAY_PORT=${RAY_PORT:-6390}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export NNODES=1 NDEVICES_PER_NODE=16 FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
export GEN_TP=8 GEN_DP=2 GEN_EP=16

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-countbenchqa-$(date +%m%d-%H%M)}
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/share/m00899630/dsv41/rl_tokenizer}
export TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet}
export TRAIN_BSZ=${TRAIN_BSZ:-8} SAMPLE_N=${SAMPLE_N:-8}
export LR=${LR:-5e-6} USE_KL=${USE_KL:-0} KL_COEF=${KL_COEF:-0.001}
export MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1152} MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}
# [2026-10-19 E01] 0.3 was the text-only proven value, but the VLM engine's
# profile run (vision tower + 1024-token dummy image) pushes non-KV memory
# above 0.3*total, so vLLM raised "No available memory for the cache blocks".
# 0.6 leaves ~39 GiB/rank for weights+KV; the actor is CPU-offloaded while the
# engine runs (colocated), so physical HBM stays within budget.
# [2026-10-20] 0.6 -> 0.5：每卡省 ~6.4 GiB HBM。若 VL profile 仍报
#   "No available memory for the cache blocks"，回调 0.55/0.6；纯文本模型 0.3 即可。
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
# 全模型每 rank 参数+梯度+Adam ~88 GiB > 64 GB，offload 不能关：
#   关掉会直接把 HBM 打爆；这里显式固定 True，避免从 example 继承的值被误改。
# 要把 host (~1.1 TB) 真正降下来，需要减少专家副本（EP_SIZE=16 或 EFSDP_SIZE=2），
#   而不是动这两个开关；EFSDP=2 会关闭 LOCAL_EXPERT_EXPORT，weight sync 会变慢。
export PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
export OFFLOAD_POLICY=${OFFLOAD_POLICY:-True}
# [2026-10-19 E05] This checkpoint is an unaligned base model: the official
# thinking prompt (with the Reasoning Effort preamble) degenerates into
# repetition loops in standalone vLLM, while chat mode answers simple counting
# questions (12-row greedy baseline in the worklog). Start RL in chat mode.
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export AGENT_LOOP=${AGENT_LOOP:-deepseek_v41_vl_single_turn}
# First-tranche debug evidence: prompt tail / response head+tail / reward per
# agent-loop worker process (4 samples each), see agent_loop_tq.py.
export VERL_AGENT_DUMP=${VERL_AGENT_DUMP:-$MIRROR_ROOT/logs/agent_dump_countbenchqa}

# Vision: build + load the tower in the trainer, freeze it, keep it out of FSDP.
export VERL_DSV41_VISION=${VERL_DSV41_VISION:-1}
export VERL_DSV41_VISION_FREEZE=${VERL_DSV41_VISION_FREEZE:-1}

# --- standard V4.1 run environment (mirrors dsv41_grpo_2node_new.sh) ---
export VA_V41_ATTN_EAGER=${VA_V41_ATTN_EAGER:-0} VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-0}
export VA_V41_OPROJ_FP32_PARTIAL=${VA_V41_OPROJ_FP32_PARTIAL:-0}
export VERL_DSV41_HC_BF16=${VERL_DSV41_HC_BF16:-0} VERL_DSV41_ALIGN_ATTN=${VERL_DSV41_ALIGN_ATTN:-0}
export VERL_DSV41_MOE_BF16=${VERL_DSV41_MOE_BF16:-0}
export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050 HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export VLLM_VERSION=0.30.0 HCCL_BUFFSIZE=1024 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=1000
export FSDP_TURBO_ATTN_RECOMPUTE=1 FSDP_TURBO_ATTN_CHUNK=64
export VERL_DSV41_DIRECT_SHARD_LOAD=1 VERL_DSV41_LOCAL_EXPERT_EXPORT=1 VERL_SYNC_PROFILE=1
export DSA_SCATTER_DROP_MARKERS=0
export VERL_DSV41_OFFLOAD_PIN_MEMORY=${VERL_DSV41_OFFLOAD_PIN_MEMORY:-0}
export VERL_DSV41_CAMEM_BACKUP_PIN=${VERL_DSV41_CAMEM_BACKUP_PIN:-0}
export VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=${VERL_DSV41_EMPTY_CACHE_AFTER_BUILD:-1}
export VERL_DSV41_EMPTY_CACHE_GRAD_CLIP=${VERL_DSV41_EMPTY_CACHE_GRAD_CLIP:-1}
export VERL_DSV41_NUMA_INTERLEAVE=${VERL_DSV41_NUMA_INTERLEAVE:-1}
export VERL_DSV41_ENGRAM_PIN_MEMORY=${VERL_DSV41_ENGRAM_PIN_MEMORY:-0}
export VERL_DSV41_OPT_STATE_DEBUG=${VERL_DSV41_OPT_STATE_DEBUG:-1}
export VERL_DSV41_OPT_STATE_DEBUG_DIR=${VERL_DSV41_OPT_STATE_DEBUG_DIR:-$MIRROR_ROOT/logs/opt_state_debug}
export VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=${VERL_DSV41_EMPTY_CACHE_BEFORE_OPT:-1}
export VERL_DSV41_DISABLE_THP=${VERL_DSV41_DISABLE_THP:-1}

echo "[single] host=$(hostname) ip=$MY_IP chips=$ASCEND_RT_VISIBLE_DEVICES steps=$STEPS"
echo "[single] model=$MODEL_PATH"
echo "[single] train=$TRAIN_FILE"
echo "[single] vision=$VERL_DSV41_VISION freeze=$VERL_DSV41_VISION_FREEZE agent=$AGENT_LOOP thinking=$ENABLE_THINKING"
echo "[single] bsz=${TRAIN_BSZ}x${SAMPLE_N} lr=$LR kl=$USE_KL prompt=$MAX_PROMPT_LEN response=$MAX_RESPONSE_LEN util=$ROLLOUT_GPU_MEM_UTIL offload=$PARAM_OFFLOAD/$OPTIMIZER_OFFLOAD/$OFFLOAD_POLICY"

for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
sleep 5
ray stop --force >/dev/null 2>&1 || true

cd "$MIRROR_ROOT/verl"
bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="${STEPS}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${OFFLOAD_POLICY} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true \
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
    actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    trainer.v1.sampler.sync_refill_failed_groups=True \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-null} \
    2>&1 | tee "${RUN_LOG}"
