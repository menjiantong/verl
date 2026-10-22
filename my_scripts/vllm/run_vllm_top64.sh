#!/usr/bin/env bash
# Standalone inference bring-up for DeepSeek-V4.1-Flash-top64 on THIS node
# (node-29-131 / 80.5.25.131, 16 NPUs).
#
# Adapted from the reference command the user gave on 2026-09-30, with three changes:
#   * paths:      /workspace/vllm-ascend        -> /workspace-verl/vllm-ascend-v41-private
#                 /workspace/cut_small/logs     -> /workspace-verl/verl/logs
#   * node IP:    80.5.25.124                   -> 80.5.25.131 (HCCL_IF_IP, and this is the
#                 interface enp48s3u1u1 carries; hostname node-29-131 does not resolve here,
#                 so the socket interfaces are pinned explicitly)
#   * VLLM_VERSION=0.30.0 is exported: vllm-ascend's `vllm_version_is()` compares
#                 Version(vllm.__version__).public against the target, and the rebuilt vllm
#                 reports 0.30.1.dev0+gced6857af.d20260930.empty -> public "0.30.1.dev0", which
#                 would NOT compare equal to "0.30.0" and would select the wrong code paths.
#   * LD_PRELOAD of libjemalloc.so.2 is dropped: this container has no
#                 /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 (a copy exists on the share at
#                 /mnt/share/z00841464/small_valid_large_sft/libjemalloc.so.2 if wanted).
#
# Usage: bash run_vllm_top64.sh            (16 NPUs, DP=2 x TP=8)
#        ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GEN_DP=1 bash run_vllm_top64.sh   # 8 NPUs
set -eo pipefail

export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/workspace-verl/vllm-ascend-v41-private/vllm_ascend:${LD_LIBRARY_PATH:-}
export VLLM_VERSION=${VLLM_VERSION:-0.30.0}

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-16666-16766}

GEN_DP=${GEN_DP:-2}
GEN_TP=${GEN_TP:-8}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HCCL_IF_IP=${HCCL_IF_IP:-80.5.25.131}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top64-bf16}
PORT=${PORT:-20007}
LOG_DIR=${LOG_DIR:-/workspace-verl/verl/logs}
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/vllm_top64_$(date +%Y%m%d_%H%M%S).log"

ulimit -l unlimited || true
ulimit -c 0 || true

echo "[serve] model=${MODEL_PATH}"
echo "[serve] DP=${GEN_DP} TP=${GEN_TP} devices=${ASCEND_RT_VISIBLE_DEVICES} port=${PORT}"
echo "[serve] log=${RUN_LOG}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --data-parallel-size "${GEN_DP}" \
  --data-parallel-size-local "${GEN_DP}" \
  --tensor-parallel-size "${GEN_TP}" \
  --enable-expert-parallel \
  --served-model-name auto \
  --trust-remote-code \
  --tokenizer-mode deepseek_v41 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --block-size 128 \
  --enforce-eager \
  --mm-processor-cache-gb 0 \
  --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":64}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --enable-return-routed-experts \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  2>&1 | tee "${RUN_LOG}"