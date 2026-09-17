#!/usr/bin/env bash
# Dependency: vllm(本地 pin a97dacb, ≈v0.28.x), vllm-ascend(本地定制分支, DeepSeek-V4.1 eager),
#            transformers(@cc7ab9be 附近), FSDPTurbo(deepseek_v41 BF16 训练适配)
#
# DeepSeek-V4.1 纯文本 GRPO，verl + FSDPTurbo(fsdp_turbo_dsv41) + vllm/vllm-ascend
#
# 说明:
#  - 训练后端: actor/ref 均 strategy=fsdp_turbo_dsv41 (DeepSeek-V4.1 专用引擎子类)
#  - 模型结构目前由 FSDPTurbo 内置 config.json (4-layer demo) 决定, model.path 只提供 tokenizer
#  - 先屏蔽 V4.1 新结构 (Engram/CSA2/稀疏索引), 纯文本 GRPO 起步
#  - 8 卡默认拓扑: FSDP=4, TP=2, EP=1 (后续全层/MoE 再调)

set -xeuo pipefail

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-DeepSeekV41}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-8card}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-2}                  # rollout vLLM tensor parallel
FSDP_SIZE=${FSDP_SIZE:-4}            # FSDPTurbo fully_shard_parallel_size
TP_SIZE=${TP_SIZE:-2}                # FSDPTurbo tensor_parallel_size
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${RAY_DATA_HOME}/models/DeepSeek-V4.1-Flash"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-4}
tp_size=${TP_SIZE:-2}

case "${DEVICE}" in
    gpu)
        ;;
    npu)
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

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

########################### shared turbo config values ###########################
ACTOR_TURBO="actor_rollout_ref.actor.fsdp_config.turbo_config"
REF_TURBO="actor_rollout_ref.ref.fsdp_config.turbo_config"

# DeepSeek-V4.1 FSDPTurbo backbone module names (DeepseekV41ForCausalLMAdapter:
#   self.model = backbone; backbone has embed / layers.{N} / norm / head / vision)
# NOTE: 前缀按 FSDPTurbo(...).model 实际 module 名可能需要补 `model.` 前缀, 真机首次
#       启动后按日志里 module 名修正.
FSDP_APPLY_MODULES='{model.model.embed:{},'\
'model.model.layers.\{*\}:{},'\
'model.model.vision.blocks.\{*\}:{},'\
'model.model.norm:{},'\
'model.model.head:{}}'

HOOK_MODULES="['model.model.layers.{*}']"
RECOMPUTE_PLAN="['model.model.layers.{*}']"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=8
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${TOKENIZER_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.strategy=fsdp_turbo_dsv41
    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${ACTOR_TURBO}.distributed.tensor_parallel_size=${tp_size}"
    "${ACTOR_TURBO}.distributed.expert_parallel_size=1"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=1"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${ACTOR_TURBO}.memory.recompute=True"
    "+${ACTOR_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
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
    "${REF_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${REF_TURBO}.distributed.tensor_parallel_size=${tp_size}"
    "${REF_TURBO}.distributed.expert_parallel_size=1"
    "${REF_TURBO}.distributed.ulysses_parallel_size=1"
    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${REF_TURBO}.memory.recompute=True"
    "+${REF_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,12,16,24,32,48,56,64]"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.resume_from_path=checkpoints/
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=15
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/deepseek_v41-8card-${start_time}.log
