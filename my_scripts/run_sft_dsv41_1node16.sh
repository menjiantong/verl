#!/usr/bin/env bash
# SFT for DeepSeek-V4.1-Flash top32 on ONE 16-chip node (GSM8K warm-start).
#
#   bash scripts/run_sft_dsv41_1node16.sh 1        # 1-step smoke (default)
#   bash scripts/run_sft_dsv41_1node16.sh 934      # ~1 epoch of main/train (7473 rows, gbs 16)
#
# Engine wiring copies the PROVEN GRPO actor config (run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
# lines 209-229): strategy fsdp_turbo_dsv41, FSDP16 / EP8 / TP1, fused EP dispatcher,
# ffn-scoped recompute, offload_policy (param+optimizer on host), no torch.compile.
# Differences vs GRPO: no rollout/no vLLM (Engram host table exists ONCE per node -- this is
# why the single-node layout that died for RL fits here), and sft_loss drives the update.
#
# The model's data contract (dsv41 adapter): use_remove_padding=False + use_dynamic_bsz=False
# and padded batches; pad_to_length_bucket=1024 (engine default) keeps shapes static.
#
# Host budget (top32): engram 366 GB + optimizer (cpu) ~428 GB + params/grads staging ~214 GB
#  -> ~1.1 TB of 2 TB. HBM: params+grads ~13 GB/card + short-sequence activations.
set -eo pipefail

STEPS=${1:-1}
NIC=${NIC:-enp48s3u1u1}
MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/share/m00899630/dsv41/rl_tokenizer}
TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/sft_data/gsm8k_main_train_sft.parquet}
CKPT_DIR=${CKPT_DIR:-/workspace-verl/ckpt_sft_dsv41}
MICRO_BSZ=${MICRO_BSZ:-1}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
GLOBAL_BSZ=${GLOBAL_BSZ:-16}
MAX_LEN=${MAX_LEN:-1024}
LR=${LR:-1e-5}
SAVE_FREQ=${SAVE_FREQ:-${STEPS}}
EXPERIMENT=${EXPERIMENT:-gsm8k-top32}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"
MY_IP=$(python3 - "$NIC" <<'PY'
import fcntl, socket, struct, sys
nic = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", nic.encode()[:15]))[20:24]))
PY
)
export HCCL_IF_IP=${HCCL_IF_IP:-$MY_IP}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NIC}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$NIC}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=0
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export RAY_memory_usage_threshold=${RAY_MEMORY_THRESHOLD:-0.99}
export RAY_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-1000}

# Attention chunk/recompute (bound the sparse-attention gather; proven defaults).
export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}
# Engram/compressor state buffers are sized [max_batch_size, ...] (ModelArgs default 4).
export VERL_DSV41_MAX_BATCH_SIZE=${VERL_DSV41_MAX_BATCH_SIZE:-$MICRO_BSZ}
export VERL_DSV41_MAX_SEQ_LEN=${VERL_DSV41_MAX_SEQ_LEN:-512}
# Per-rank direct shard load (seconds instead of the broadcast).
export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}

FSDP_APPLY_MODULES='{model.embed:{},model.layers.\{*\}:{},model.norm:{},model.head:{}}'

cd /workspace-verl/verl
mkdir -p logs
RUN_LOG="logs/sft_dsv41_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"
echo "[sft] steps=${STEPS} gbs=${GLOBAL_BSZ} micro=${MICRO_BSZ} lr=${LR} log=${RUN_LOG}"

# Fresh local ray for the SFT driver (no cross-machine cluster involved).
ray stop --force >/dev/null 2>&1 || true

python3 -m verl.trainer.sft_trainer_ray \
  trainer.n_gpus_per_node=16 \
  trainer.nnodes=1 \
  trainer.total_training_steps=${STEPS} \
  trainer.total_epochs=1 \
  trainer.logger="['console']" \
  trainer.project_name=dsv41-sft \
  trainer.experiment_name=${EXPERIMENT} \
  trainer.default_local_dir=${CKPT_DIR} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.max_ckpt_to_keep=${MAX_KEEP:-1} \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  model.path=${MODEL_PATH} \
  model.tokenizer_path=${TOKENIZER_PATH} \
  model.use_remove_padding=False \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=False \
  data.train_files=${TRAIN_FILE} \
  data.val_files=null \
  data.messages_key=messages \
  data.max_length=${MAX_LEN} \
  data.truncation=error \
  data.use_dynamic_bsz=False \
  data.micro_batch_size_per_gpu=${MICRO_BSZ} \
  data.train_batch_size=${GLOBAL_BSZ} \
  data.num_workers=4 \
  optim.lr=${LR} \
  optim.lr_warmup_steps_ratio=0.0 \
  checkpoint.save_contents="['model']" \
  engine.strategy=fsdp_turbo_dsv41 \
  engine.param_offload=${PARAM_OFFLOAD} \
  engine.optimizer_offload=${OPTIMIZER_OFFLOAD} \
  engine.offload_policy=${PARAM_OFFLOAD} \
  engine.reshard_after_forward=True \
  engine.entropy_checkpointing=False \
  engine.entropy_from_logits_with_chunking=False \
  engine.use_torch_compile=False \
  engine.pad_to_length_bucket=${PAD_BUCKET:-512} \
  engine.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=${FWD_PREFETCH:-1} \
  engine.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=${BWD_PREFETCH:-1} \
  engine.turbo_config.distributed.fully_shard_parallel_size=16 \
  engine.turbo_config.distributed.tensor_parallel_size=1 \
  engine.turbo_config.distributed.expert_parallel_size=8 \
  engine.turbo_config.distributed.expert_fully_shard_parallel_size=1 \
  engine.turbo_config.distributed.ulysses_parallel_size=1 \
  "+engine.turbo_config.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}" \
  "+engine.turbo_config.distributed.fsdp_plan.hook_modules=['model.layers.{*}']" \
  "+engine.turbo_config.distributed.fsdp_plan.cast_forward_inputs=False" \
  "engine.turbo_config.distributed.fsdp_plan.output_dtype=null" \
  "+engine.turbo_config.distributed.ep_plan.apply_modules=['model.layers.{*}.ffn.experts']" \
  "+engine.turbo_config.distributed.ep_plan.dispatcher=fused" \
  +engine.turbo_config.memory.recompute=true \
  '+engine.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
  2>&1 | tee "${RUN_LOG}"
