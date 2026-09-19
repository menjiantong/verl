#!/usr/bin/env bash
# DeepSeek-V4.1 GRPO on Ascend NPU: verl + FSDPTurbo(fsdp_turbo_dsv41) + vllm/vllm-ascend
#
# Local baseline this script is aligned with (2026-09-17):
#   * training-side reference: FSDPTurbo/examples/deepseek_v41/run.sh + config2.yaml
#     (topology FSDP=8 / TP=1 / EP=8, apply_modules on `model.layers.{*}`, no recompute)
#   * rollout-side reference:  scripts/run_vllm2.sh
#     (vllm serve <8layer ckpt>, TP=8 + expert parallel, max-model-len 8192, eager,
#      additional-config {"enable_engram": false, "mc2_comm_alg": "fullmesh_v2", ...})
#
# Notes:
#  - actor/ref both run strategy=fsdp_turbo_dsv41: the DeepSeek-V4.1 training model is an
#    FSDPTurbo class, not a transformers AutoModel.
#  - Model structure comes from MODEL_PATH/config.json (8 layers, 384 experts, no engram);
#    the ~108B routed-expert weights are built on the meta device, then materialized from
#    the checkpoint shards by torch.distributed.checkpoint (rank 0 reads, everyone else
#    receives the broadcast).
#  - V4.1 recompute is intentionally not supported by the adapter -> no recompute plan.
#  - Pure text GRPO: padding path (use_remove_padding=False) because the reference model
#    consumes [batch, seq] + attention_mask.

set -xeuo pipefail

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-DeepSeekV41}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-8layer-8card}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
NNODES=${NNODES:-1}

# vLLM rollout (mirrors scripts/run_vllm2.sh): TP=8 with expert parallel.
GEN_TP=${GEN_TP:-8}
GEN_EP=${GEN_EP:-8}
# FSDPTurbo topology (mirrors FSDPTurbo/examples/deepseek_v41/config2.yaml).
FSDP_SIZE=${FSDP_SIZE:-8}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
EFSDP_SIZE=${EFSDP_SIZE:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

MODEL_PATH=${MODEL_PATH:-"/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random"}
# Released V4.1 checkpoints ship no chat template; this directory carries the same
# tokenizer files plus the single-turn template (see scripts/make_dsv41_rl_tokenizer.py).
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/share/m00899630/dsv41/rl_tokenizer"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"/mnt/share/m00899630/dsv41/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
# Sizes the RoPE tables and the per-layer attention scratch caches in the training model;
# must cover prompt+response (the checkpoint's own max_position_embeddings is 1M).
export VERL_DSV41_MAX_SEQ_LEN=${VERL_DSV41_MAX_SEQ_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}
SAMPLE_N=${SAMPLE_N:-5}
TRAIN_BSZ=${TRAIN_BSZ:-8}
VAL_BSZ=${VAL_BSZ:-8}

# Weight sync: stream only this rank's routed experts instead of all-gathering the packed
# expert tensor on every rank (see wiki/worklog_dsv41_rl.md G13). Colocate pairs trainer
# rank i with engine rank i and both sides shard the expert dimension contiguously by rank,
# so the other ranks' experts are dropped before the transfer. This is what makes the sync
# cheap when parameters are CPU-offloaded: measured 86.7s -> 6.3s on the 4-layer 32-expert
# checkpoint. Turn it on to get that speedup, but confirm correctness on the first step:
# training/rollout_probs_diff_mean must stay ~1e-2 and
# training/rollout_actor_probs_pearson_corr ~0.98 -- a rank/placement mismatch shows up
# there as O(1) / ~0.5.
LOCAL_EXPERT_EXPORT=${LOCAL_EXPERT_EXPORT:-${VERL_DSV41_LOCAL_EXPERT_EXPORT:-0}}
if [ "${LOCAL_EXPERT_EXPORT}" = "1" ] && [ "${GEN_EP}" != "${EP_SIZE}" ]; then
    echo "LOCAL_EXPERT_EXPORT=1 pairs trainer rank i with engine rank i, so it needs" \
         "GEN_EP == EP_SIZE (got ${GEN_EP} vs ${EP_SIZE}); disabling it." >&2
    LOCAL_EXPERT_EXPORT=0
fi
export VERL_DSV41_LOCAL_EXPERT_EXPORT=${LOCAL_EXPERT_EXPORT}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        # vllm-ascend's patches branch on this env var (see scripts/run_vllm2.sh): the
        # v41-private branch is written against the 0.27.1 API surface, and the installed
        # dev build (0.28.1rc1.dev...) is not a parseable version for their version check.
        export VLLM_VERSION=${VLLM_VERSION:-0.27.1}
        # EP MoE dispatch/combine (mc2) partitions its comm window from HCCL_BUFFSIZE:
        # too small and the operator tiling fails at vLLM engine warm-up
        # ("HCCL_BUFFSIZE_EP is too SMALL"). Value follows scripts/run_vllm2.sh.
        export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
        export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-10000}
        export ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT:-10000}
        export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}
        export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
        # Do NOT copy run_vllm2.sh's PYTORCH_NPU_ALLOC_CONF=expandable_segments:True here:
        # in colocated mode vLLM's sleep-mode memory pool rejects expandable segments
        # ("Expandable segments are not compatible with memory pool"), and verl manages the
        # setting itself (set_expandable_segments) around weight syncs.
        export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
        export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
        export HCCL_ASYNC_ERROR_HANDLING=0
        export HCCL_EXEC_TIMEOUT=3600
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

# This machine carries several checkouts of the same packages; pin the ones this
# integration was verified against (see wiki/worklog_dsv41_rl.md §5).
export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
# Tag the log with the checkpoint it ran against (layer counts differ between runs).
LOG_TAG=$(basename "${MODEL_PATH}")
mkdir -p logs

########################### shared turbo config values ###########################
ACTOR_TURBO="actor_rollout_ref.actor.fsdp_config.turbo_config"
REF_TURBO="actor_rollout_ref.ref.fsdp_config.turbo_config"

# Module names as seen by FSDPTurbo: the adapter wraps the backbone as `model`
# (DeepseekV41ForCausalLMAdapter.model = Transformer -> embed / layers.{N} / norm / head).
# Vision is disabled for text-only RL, so no `model.vision.*` entries.
FSDP_APPLY_MODULES='{model.embed:{},'\
'model.layers.\{*\}:{},'\
'model.norm:{},'\
'model.head:{}}'

HOOK_MODULES="['model.layers.{*}']"

# vLLM engine args that the V4.1 Ascend path needs (see scripts/run_vllm2.sh).
# Set as individual overrides: hydra cannot parse a JSON blob containing `{}`.
VLLM_ADDITIONAL_CONFIG_ROOT="+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BSZ}
    data.val_batch_size=${VAL_BSZ}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH}
    # The reference model consumes padded [batch, seq] + attention_mask; the packed
    # (remove-padding) path would require per-sequence position handling it does not have.
    actor_rollout_ref.model.use_remove_padding=False
    # V4.1 recompute is disabled inside the FSDPTurbo adapter.
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BSZ}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.strategy=fsdp_turbo_dsv41
    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${ACTOR_TURBO}.distributed.tensor_parallel_size=${TP_SIZE}"
    "${ACTOR_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${ACTOR_TURBO}.distributed.expert_fully_shard_parallel_size=${EFSDP_SIZE}"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=1"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.cast_forward_inputs=False"
    "${ACTOR_TURBO}.distributed.fsdp_plan.output_dtype=null"
    "+${ACTOR_TURBO}.distributed.ep_plan.apply_modules=['model.layers.{*}.ffn.experts']"
    "+${ACTOR_TURBO}.distributed.ep_plan.dispatcher=fused"
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.strategy=fsdp_turbo_dsv41
    "${REF_TURBO}.distributed.fully_shard_parallel_size=${FSDP_SIZE}"
    "${REF_TURBO}.distributed.tensor_parallel_size=${TP_SIZE}"
    "${REF_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${REF_TURBO}.distributed.expert_fully_shard_parallel_size=${EFSDP_SIZE}"
    "${REF_TURBO}.distributed.ulysses_parallel_size=1"
    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.cast_forward_inputs=False"
    "${REF_TURBO}.distributed.fsdp_plan.output_dtype=null"
    "+${REF_TURBO}.distributed.ep_plan.apply_modules=['model.layers.{*}.ffn.experts']"
    "+${REF_TURBO}.distributed.ep_plan.dispatcher=fused"
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
    actor_rollout_ref.rollout.n=${SAMPLE_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
    actor_rollout_ref.rollout.free_cache_engine=True
    # run_vllm2.sh serves the 8-layer checkpoint with --enforce-eager.
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    # Keep the transfer bucket small: the export materializes each fused expert tensor
    # (18.1GB bf16 for gate_up_proj) on every rank while the rollout engine already holds
    # ~32GB, so a large bucket is what tips the card over.
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.mc2_comm_alg=fullmesh_v2"
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.enable_engram=false"
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.engram_storage=int8"
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.ascend_compilation_config.enable_npugraph_ex=false"
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.ascend_compilation_config.enable_static_kernel=false"
    # RL weight updates need weight_nz_mode=0 (FRACTAL_NZ breaks synced-weight precision);
    # rl_config.enabled sets that (and disables expandable segments) on the Ascend side.
    "${VLLM_ADDITIONAL_CONFIG_ROOT}.rl_config.enabled=true"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.default_local_dir="${CKPTS_DIR}"
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "logs/${LOG_TAG}-${start_time}.log"
