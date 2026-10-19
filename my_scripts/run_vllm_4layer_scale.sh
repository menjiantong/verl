model_path="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled"
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
export VLLM_VERSION=0.30.1

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port 8000 \
    --load-format safetensors \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --served-model-name deepseek-v41-engram \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --block-size 128 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --additional-config '{"mc2_comm_alg":"fullmesh_v2","enable_engram":false,"engram_storage":"int8","ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/m00899630/profiling","torch_profiler_with_stack":false,"torch_profiler_with_memory":false}'
