#!/usr/bin/env bash
set -e
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PATH=/usr/local/python3.11.15/bin:$PATH
R=/mnt/share/m00899630/workspace/dsv41-verl-copy
export PYTHONPATH="$R/vllm:$R/vllm-ascend:$R/FSDPTurbo:$R/verl:${PYTHONPATH:-}"
export VLLM_VERSION=0.30.0 VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_BUFFSIZE=1024 HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE=65000-65050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export HCCL_SOCKET_IFNAME=enp48s3u1u1 HCCL_IF_IP=80.5.21.113
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=10
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
cd "$R/vllm"
nohup vllm serve /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram \
  --tokenizer /mnt/share/m00899630/dsv41/rl_tokenizer \
  --served-model-name ds \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --port 8010 \
  --additional-config '{"mc2_comm_alg":"fullmesh_v2","ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  > "$R/logs/vllm_serve_infer_113.log" 2>&1 &
echo "serve pid $!"
