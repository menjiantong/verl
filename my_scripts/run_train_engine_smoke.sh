#!/usr/bin/env bash
# Run the trainer-side-only smoke (train_engine_smoke.py) on 8 NPUs.
#
#   bash run_train_engine_smoke.sh                       # top64 checkpoint, production topology
#   bash run_train_engine_smoke.sh --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real \
#        --seq-len 256 --batch-size 1                    # cheap harness check (no Engram layers)
#
# Env mirrors examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh; the host does
# NOT resolve node-29-131, so the rendezvous and the socket interfaces are pinned explicitly.
set -eo pipefail

export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}
export VLLM_VERSION=${VLLM_VERSION:-0.30.0}
export VERL_DSV41_MAX_SEQ_LEN=${VERL_DSV41_MAX_SEQ_LEN:-2048}
# Match the GRPO script: local expert export (each rank ships only its own expert shard).
# Without it the export check takes the all-gather path, which all-gathers CPU-offloaded
# DTensor shards and fails with "No backend type associated with device type cpu" -- that is
# the *export* path production never uses, not a training problem.
export VERL_DSV41_LOCAL_EXPERT_EXPORT=${VERL_DSV41_LOCAL_EXPERT_EXPORT:-1}

# Pick the first 8 free chips, preferring identity 0-7 (same logic as run_grpo_2node.sh).
# 2026-10-09 09:33: another container on this host grabbed chips 0-7 with vLLM workers
# (~57 GB/card) mid-launch, and the smoke died with a misleading "NPU out of memory ...
# 61.27 GiB total, 5.25 GiB already allocated" inside model build. Override by exporting
# ASCEND_RT_VISIBLE_DEVICES yourself.
pick_devices() {
    local free=() i
    for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        [ -e "/dev/davinci$i" ] || continue
        if ! npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep -q "Process id"; then
            free+=("$i")
            [ "${#free[@]}" -eq 8 ] && break
        fi
    done
    if [ "${#free[@]}" -ne 8 ]; then
        echo "[smoke] ERROR: only ${#free[@]} free chips on this node (need 8); holders:" >&2
        for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            out=$(npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep "Process id" || true)
            [ -n "$out" ] && echo "  NPU $i: $out" >&2
        done
        exit 1
    fi
    local IFS=,
    echo "${free[*]}"
}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(pick_devices)}
export MASTER_ADDR=127.0.0.1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_IF_IP=${HCCL_IF_IP:-80.5.25.131}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=0
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}

NPROC=${NPROC:-8}
LOG_DIR=${LOG_DIR:-/workspace-verl/verl/logs}
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/train_engine_smoke_$(date +%Y%m%d_%H%M%S).log"

echo "[smoke] devices=${ASCEND_RT_VISIBLE_DEVICES} nproc=${NPROC}"
echo "[smoke] log=${RUN_LOG}"

torchrun --nproc_per_node="${NPROC}" --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29591}" \
  /workspace-verl/scripts/train_engine_smoke.py "$@" 2>&1 | tee "${RUN_LOG}"