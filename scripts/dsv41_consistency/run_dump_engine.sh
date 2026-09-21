#!/usr/bin/env bash
# Environment wrapper for the engine-side consistency dumps.
#
# Mirrors the NPU branch of examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
# (same vLLM/vllm-ascend knobs as scripts/run_vllm2.sh): VLLM_VERSION override for the
# 0.27.1-era patches, HCCL_BUFFSIZE for the mc2 MoE all-to-all window, single-threaded
# OMP, and the v41 model code from the local source trees.
set -xeuo pipefail

export VLLM_VERSION=${VLLM_VERSION:-0.27.1}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-10000}
export ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT:-10000}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600

REPO=${REPO:-/workspace-verl/verl}
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${REPO}:${PYTHONPATH:-}

cd "${REPO}"
export PYTHONPATH=/workspace-verl/verl/scripts/dsv41_consistency:${PYTHONPATH}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

exec python3 scripts/dsv41_consistency/dump_engine_stages.py "$@"
