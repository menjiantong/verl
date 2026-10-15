#!/usr/bin/env bash
# Trainer-side stage dump on the top32 checkpoint (Engram host-offload load path).
# 2026-10-13: the harness now runs the production materialization (dump_trainer_stages.py
# §"[2026-10-13 top32]") so top32's Engram tables load via _load_host_engram_tables.
set -xeuo pipefail
export PATH=/usr/local/python3.11.15/bin:$PATH
export VLLM_VERSION=${VLLM_VERSION:-0.30.0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
cd /workspace-verl/verl
torchrun --nproc_per_node=8 --master_port=59522 scripts/dsv41_consistency/dump_trainer_stages.py \
  --model-path ${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116} \
  --out-dir ${OUT_DIR:-/mnt/share/m00899630/dsv41/dump/cons_v4_gsm8k/trainer_top32} \
  --lengths ${LENGTHS:-64} \
  ${EXTRA_ARGS:-}
