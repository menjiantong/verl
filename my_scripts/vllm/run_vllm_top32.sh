model_path="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16"
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10


vllm serve "$model_path"\
  --host 0.0.0.0 \
  --port 20007 \
  --tensor-parallel-size 8 \
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
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
