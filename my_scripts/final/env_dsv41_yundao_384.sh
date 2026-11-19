#!/usr/bin/env bash
# DSV4.1-Flash 单机 16 卡公共环境：final/ 下的启动脚本 source 本文件。
# 代码基线：FSDPTurbo 1852e93 / vllm-ascend b0901f79 / vllm fd46ca5a / verl 见启动说明。
# 只做环境与 Ray 启动，不含任何监控/分支；训练参数由各启动脚本给出。
set -eo pipefail


export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-sft116-$(date +%m%d-%H%M)}

# ---- 模型与数据（countbenchqa_lite，32 专家权重） ----
export MODEL_PATH=${MODEL_PATH:-/efs_rl/weights/DeepSeek-V4.1-Flash-bf16}   
export TRAIN_FILE=${TRAIN_FILE:-/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet}
export TEST_FILE=${TEST_FILE:-/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/efs_rl/m00899630/dsv41/rl_tokenizer}
export CKPTS_DIR=${CKPTS_DIR:-"/efs_rl/m00899630/dsv41/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
export LR=${LR:-5e-6} USE_KL=${USE_KL:-1} KL_COEF=${KL_COEF:-0.001}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
export MY_ENV_CACHE_ROOT=/efs_rl/m00899630/cache

# ---- 并行：训练 FSDP16/EP16/CP4，推理 TP8×DP2/EP16 ----
export NDEVICES_PER_NODE=16 NNODES=${NNODES:-1} 
export FSDP_SIZE=${FSDP_SIZE:-64} EP_SIZE=${EP_SIZE:-64} 
export GEN_TP=${GEN_TP:-8} GEN_DP=${GEN_DP:-8} GEN_EP=${GEN_EP:-64} CP_SIZE=${CP_SIZE:-4}
export MY_ENV_DSV41_CP_PAD_MULTIPLE=8 MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1 MY_ENV_DSV41_CP_GATHER_OUTPUT=1

# ---- KL / engram / sleep ----
export USE_KL=1 MY_ENV_DSV41_ENGRAM_SHARE_MODEL=1 VLLM_ENGRAM_DP_SHARED_MEMORY=1 FREE_CACHE_ENGINE=False
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-1}

# ---- 训练器开关（已验证） ----
export MY_ENV_DSV41_MODEL_INIT=zero MY_ENV_DSV41_ENGRAM_INIT=zero MY_ENV_DSV41_DISABLE_THP=1
export MY_ENV_VLLM_VIT_ATTN_OP=fa MY_ENV_TRAIN_FUSED_VISION_ATTN=1
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-${STEPS:-200}}
export VERL_DSV41_OFFLOAD_PIN_MEMORY=0 VERL_DSV41_DIRECT_SHARD_LOAD=1 VERL_DSV41_LOCAL_EXPERT_EXPORT=1
export VERL_DSV41_VISION=1 VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=1 VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1
export FSDP_TURBO_ATTN_RECOMPUTE=1 FSDP_TURBO_ATTN_CHUNK=64

# ---- CANN / Python ----
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PATH=/usr/local/python3.11.15/bin:$PATH


# ---- 缓存/临时目录（根盘常满，全部落共享盘） ----
export MY_ENV_CACHE_ROOT=${MY_ENV_CACHE_ROOT:-/mnt/share/m00899630/cache}
export HF_HOME=$MY_ENV_CACHE_ROOT/huggingface HF_DATASETS_CACHE=$MY_ENV_CACHE_ROOT/huggingface/datasets
export HF_HUB_CACHE=$MY_ENV_CACHE_ROOT/huggingface/hub TRANSFORMERS_CACHE=$MY_ENV_CACHE_ROOT/huggingface/transformers
export TORCH_EXTENSIONS_DIR=$MY_ENV_CACHE_ROOT/torch_extensions XDG_CACHE_HOME=$MY_ENV_CACHE_ROOT/xdg
export TRITON_CACHE_DIR=$MY_ENV_CACHE_ROOT/triton TORCHINDUCTOR_CACHE_DIR=$MY_ENV_CACHE_ROOT/torchinductor
export VLLM_CACHE_ROOT=$MY_ENV_CACHE_ROOT/vllm
export MY_ENV_LOCAL_TMP=/dev/shm/m00899630 TMPDIR=${TMPDIR:-/mnt/share/m00899630/tmp} RAY_TMPDIR=/dev/shm/m00899630/ray_tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$RAY_TMPDIR" "$TORCH_EXTENSIONS_DIR" \
         "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

# ---- 业务网卡（不用 172.x 容器网） ----
export MY_IP=${MY_IP:-$(hostname -I | tr ' ' '\n' | grep -m1 '^7\.' || true)}
export NIC=${NIC:-$(python3 -c 'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(i for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address==ip))' 2>/dev/null || echo enp48s3u1u1)}
export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60150 HCCL_NPU_SOCKET_PORT_RANGE=61000-61150

# ---- Ray（dashboard 必须关，否则 raylet 自检失败自杀） ----
export RAY_PORT=${RAY_PORT:-6390} RAY_AUTH_MODE=disabled
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=1000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ---- vLLM / 引擎 ----
export VLLM_VERSION=0.30.0 HCCL_BUFFSIZE=1024 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1
