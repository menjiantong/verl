#!/usr/bin/env bash
# =============================================================================
# vLLM serve for the **epoch3** GSM8K SFT model (top32, noengram), machine 128.
#
#   bash my_scripts/run_vllm_top32_epoch123.sh
#
# Based on the verified epoch-eval config:
#   - 16 chips, DP2 x TP8, expert parallel
#   - exported model: /mnt/share/m00899630/weights/
#       DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3
#   - NO --engram-config / --enable-return-routed-experts:
#     the exported model config has engram_layer_ids=[], and an explicit
#     EngramConfig is rejected for models without n-gram layers.
#
# To serve epoch1/epoch2 instead, override MODEL_PATH:
#   MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1 \
#     bash my_scripts/run_vllm_top32_epoch123.sh
#
# After the server is healthy (curl http://127.0.0.1:20007/health), evaluate:
#   python3 my_scripts/eval_gsm8k_serve.py --port 20007 --n 50 --max-tokens 1024 \
#     --test-file /mnt/share/m00899630/dsv41/sft_data/gsm8k_main_test_50qa.parquet \
#     --out /mnt/share/m00899630/dsv41/sft_data/eval_3ep_epoch3_50.jsonl
# =============================================================================
set -eo pipefail

MIRROR_ROOT=/mnt/share/m00899630/workspace/dsv41-verl
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$MIRROR_ROOT/vllm-ascend/vllm_ascend:${LD_LIBRARY_PATH:-}"

# CANN / ATB environment (vLLM engine worker needs libatb.so).
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3}
PORT=${PORT:-20007}
MY_IP=${MY_IP:-80.5.25.128}
NIC=${NIC:-enp48s3u1u1}
TAG=$(basename "$MODEL_PATH")
RUN_LOG=${RUN_LOG:-logs/vllm_${TAG}_$(date +%Y%m%d_%H%M%S).log}
mkdir -p logs

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_IF_IP=$MY_IP
export GLOO_SOCKET_IFNAME=$NIC
export TP_SOCKET_IFNAME=$NIC
export HCCL_SOCKET_IFNAME=$NIC
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_RPC_TIMEOUT=1800000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true
ulimit -l unlimited || true
ulimit -c 0 || true

echo "[vllm] model=$MODEL_PATH port=$PORT ip=$MY_IP nic=$NIC log=$RUN_LOG"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name auto \
  --trust-remote-code \
  --tokenizer-mode deepseek_v41 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --block-size 128 \
  --enforce-eager \
  --mm-processor-cache-gb 0 \
  --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":64}' \
  --additional-config '{"enable_cpu_binding":true}' \
  2>&1 | tee "$RUN_LOG"
