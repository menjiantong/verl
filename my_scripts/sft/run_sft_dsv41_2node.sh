#!/usr/bin/env bash
# SFT (GSM8K warm-start) for DeepSeek-V4.1-Flash top32 on TWO nodes x 16 chips:
# one Ray cluster, FSDP32 / EP8, no parameter/optimizer offload by default.
#
#   master (131):  bash scripts/run_sft_dsv41_2node.sh master [STEPS]
#   slave  (128):  bash scripts/run_sft_dsv41_2node.sh slave <MASTER_IP>     # via ssh
#
# Why 2 nodes: per-card state on FSDP32 = params 3.25 GB + grads 3.25 GB + AdamW fp32
# moments ~13 GB + activations -- ~41 GB of 61.3 GiB, which sidesteps the single-node
# no-offload wall (57.4 GB) and the single-node offload path's +5 GB/worker/step host
# leak. Host per node stays ~1.3 TB (engram 787 GB incl. the two node-local EP-group
# copies + states), comfortably under the 0.995 Ray kill line.
set -eo pipefail

MODE=${1:?usage: $0 master [STEPS] | $0 slave <MASTER_IP>}
NIC=${NIC:-enp48s3u1u1}
RAY_PORT=${RAY_PORT:-6379}
MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/share/m00899630/dsv41/rl_tokenizer}
TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/sft_data/gsm8k_main_train_sft.parquet}
CKPT_DIR=${CKPT_DIR:-/workspace-verl/ckpt_sft_dsv41}
MICRO_BSZ=${MICRO_BSZ:-1}
GLOBAL_BSZ=${GLOBAL_BSZ:-64}
MAX_LEN=${MAX_LEN:-1024}
LR=${LR:-1e-5}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
EXPERIMENT=${EXPERIMENT:-gsm8k-top32-2node}

detect_ip() {
    python3 - "$1" <<'PY'
import fcntl, socket, struct, sys
nic = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", nic.encode()[:15]))[20:24]))
PY
}

pick_devices() {
    local free=() i
    for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        [ -e "/dev/davinci$i" ] || continue
        if ! npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep -q "Process id"; then
            free+=("$i"); [ "${#free[@]}" -eq 16 ] && break
        fi
    done
    if [ "${#free[@]}" -ne 16 ]; then echo "[sft2] ERROR: only ${#free[@]} free chips (need 16)" >&2; exit 1; fi
    local IFS=,
    echo "${free[*]}"
}

set_env() {
    local my_ip=$1
    export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NIC}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$NIC}
    export HCCL_IF_IP=$my_ip
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_ASYNC_ERROR_HANDLING=0
    export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
    export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export RAY_memory_usage_threshold=${RAY_MEMORY_THRESHOLD:-0.995}
    export RAY_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-1000}
    export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
    export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}
    export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}
    export VERL_DSV41_MAX_BATCH_SIZE=${VERL_DSV41_MAX_BATCH_SIZE:-$MICRO_BSZ}
    export VERL_DSV41_MAX_SEQ_LEN=${VERL_DSV41_MAX_SEQ_LEN:-512}
}

SFT_ARGS_COMMON=(
  trainer.n_gpus_per_node=16
  trainer.nnodes=2
  trainer.total_epochs=1
  trainer.logger="['console']"
  trainer.project_name=dsv41-sft
  trainer.experiment_name=${EXPERIMENT}
  trainer.default_local_dir=${CKPT_DIR}
  trainer.save_freq=${SAVE_FREQ:-200}
  trainer.max_ckpt_to_keep=${MAX_KEEP:-1}
  trainer.test_freq=-1
  trainer.resume_mode=disable
  model.path=${MODEL_PATH}
  model.tokenizer_path=${TOKENIZER_PATH}
  model.use_remove_padding=False
  model.trust_remote_code=True
  model.enable_gradient_checkpointing=False
  data.train_files=${TRAIN_FILE}
  data.val_files=null
  data.messages_key=messages
  data.max_length=${MAX_LEN}
  data.truncation=error
  data.use_dynamic_bsz=False
  data.micro_batch_size_per_gpu=${MICRO_BSZ}
  data.train_batch_size=${GLOBAL_BSZ}
  data.num_workers=4
  optim.lr=${LR}
  checkpoint.save_contents="['model']"
  engine.strategy=fsdp_turbo_dsv41
  engine.param_offload=${PARAM_OFFLOAD}
  engine.optimizer_offload=${OPTIMIZER_OFFLOAD}
  engine.offload_policy=${PARAM_OFFLOAD}
  engine.reshard_after_forward=True
  engine.entropy_checkpointing=False
  engine.entropy_from_logits_with_chunking=False
  engine.use_torch_compile=False
  engine.pad_to_length_bucket=${PAD_BUCKET:-512}
  engine.turbo_config.distributed.fully_shard_parallel_size=32
  engine.turbo_config.distributed.tensor_parallel_size=1
  engine.turbo_config.distributed.expert_parallel_size=8
  engine.turbo_config.distributed.expert_fully_shard_parallel_size=2
  engine.turbo_config.distributed.ulysses_parallel_size=1
  "+engine.turbo_config.distributed.fsdp_plan.apply_modules={model.embed:{},model.layers.\{*\}:{},model.norm:{},model.head:{}}"
  "+engine.turbo_config.distributed.fsdp_plan.hook_modules=['model.layers.{*}']"
  "+engine.turbo_config.distributed.fsdp_plan.cast_forward_inputs=False"
  "engine.turbo_config.distributed.fsdp_plan.output_dtype=null"
  "+engine.turbo_config.distributed.ep_plan.apply_modules=['model.layers.{*}.ffn.experts']"
  "+engine.turbo_config.distributed.ep_plan.dispatcher=fused"
  +engine.turbo_config.memory.recompute=true
  '+engine.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]'
)

if [ "$MODE" = "master" ]; then
    STEPS=${2:-467}
    MY_IP=${MASTER_IP:-$(detect_ip "$NIC")}
    DEV=$(pick_devices)
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$DEV}
    set_env "$MY_IP"
    echo "[sft2] master ip=$MY_IP devices=$ASCEND_RT_VISIBLE_DEVICES"

    for pattern in "VLLM::" "ray::" "vllm serve" "sft_trainer_ray"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    sleep 5
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
    export RAY_ADDRESS="$MY_IP:$RAY_PORT"

    echo "[sft2] waiting for the slave (run on 128): bash scripts/run_sft_dsv41_2node.sh slave $MY_IP"
    for i in $(seq 1 120); do
        active=$(ray status 2>/dev/null | sed -n '/^Active:/,/^Pending:/p' | grep -c "node_" || true)
        if [ "${active:-0}" -ge 2 ]; then
            npu_total=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ NPU' | tail -1 | cut -d/ -f2 | cut -d. -f1)
            echo "[sft2] slave present, NPU total=${npu_total}"
            [ "${npu_total:-0}" -ge 32 ] || { echo "[sft2] ERROR: expected 32 NPU"; exit 1; }
            break
        fi
        [ "$i" = "120" ] && { echo "[sft2] ERROR: slave did not join in 20 min"; exit 1; }
        sleep 10
    done

    cd /workspace-verl/verl
    mkdir -p logs
    RUN_LOG="logs/sft_dsv41_2node_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"
    echo "[sft2] launching SFT: steps=${STEPS} log=${RUN_LOG}"
    python3 -m verl.trainer.sft_trainer_ray \
      "${SFT_ARGS_COMMON[@]}" \
      trainer.total_training_steps=${STEPS} \
      2>&1 | tee "${RUN_LOG}"
    exit $?
fi

if [ "$MODE" = "slave" ]; then
    MASTER_IP=${2:?usage: $0 slave <MASTER_IP>}
    SLAVE_IP=${SLAVE_IP:-$(detect_ip "$NIC")}
    DEV=$(pick_devices)
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$DEV}
    set_env "$SLAVE_IP"
    echo "[sft2] slave ip=$SLAVE_IP devices=$ASCEND_RT_VISIBLE_DEVICES"

    for pattern in "VLLM::" "ray::" "vllm serve" "sft_trainer_ray"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    sleep 5
    ray stop --force >/dev/null 2>&1 || true
    ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$SLAVE_IP" --disable-usage-stats
    echo "[sft2] slave joined"
    exit 0
fi

echo "unknown mode: $MODE" >&2
