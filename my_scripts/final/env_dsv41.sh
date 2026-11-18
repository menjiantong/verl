#!/usr/bin/env bash
# DSV4.1-Flash 单机 16 卡公共环境：final/ 下的启动脚本 source 本文件。
# 代码基线：FSDPTurbo 1852e93 / vllm-ascend b0901f79 / vllm fd46ca5a / verl 见启动说明。
# 只做环境与 Ray 启动，不含任何监控/分支；训练参数由各启动脚本给出。
set -eo pipefail
R=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$R/verl"

# ---- 模型与数据（countbenchqa_lite，32 专家权重） ----
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
export TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet}

# ---- 并行：训练 FSDP16/EP16/CP4，推理 TP8×DP2/EP16 ----
export NDEVICES_PER_NODE=16 NNODES=1 FSDP_SIZE=16 EP_SIZE=16 GEN_TP=8 GEN_DP=2 GEN_EP=16 CP_SIZE=4
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
# [1117 B075] 灵枢 80.5.27.x 的 CANN toolkit 是 B075，其 libnnopbase.so 丢
# npu_sparse_flash_mla 的 view stride（73856）→ DSV4.1 rollout logprob 全 NaN
# （mjt_workplace/dsv41_nan_rootcause_b075_libnnopbase_20261117.md）。受影响节点把
# MY_ENV_DSV41_CANN_OVERLAY 指向 B243 整包解压目录（/root/cann_b243/cann-9.1.0），
# 整车改用该构建；未设置时行为与原来完全一致。ATB 是独立组件，始终用系统版本。
# 默认 auto：仅在检测到 B075 且 overlay 已解包时启用；显式设路径可强制，设 0 可关闭。
if [ "${MY_ENV_DSV41_CANN_OVERLAY:-auto}" = "auto" ]; then
  _dsv41_toolkit_info=/usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info
  _dsv41_inner=$(grep -o 'innerversion=[^[:space:]]*' "$_dsv41_toolkit_info" 2>/dev/null | head -1 || true)
  if grep -q "innerversion=V100R001C11B075" "$_dsv41_toolkit_info" 2>/dev/null; then
    if [ -f /root/cann_b243/cann-9.1.0/set_env.sh ]; then
      export MY_ENV_DSV41_CANN_OVERLAY=/root/cann_b243/cann-9.1.0
      echo "MY_DEBUG_LOG env_dsv41 cann_overlay=auto->B243 ${_dsv41_inner:-inner=unknown}"
    else
      # Running on B075 anyway produces NaN rollout logprobs (libnnopbase view-stride).
      echo "ERROR: env_dsv41 detected B075 but /root/cann_b243/cann-9.1.0/set_env.sh is missing;" \
           "unpack cann_910_b243_124.tar there first (see dsv41_nan_rootcause_b075_libnnopbase_20261117.md)" >&2
      unset MY_ENV_DSV41_CANN_OVERLAY
    fi
  else
    unset MY_ENV_DSV41_CANN_OVERLAY
    echo "MY_DEBUG_LOG env_dsv41 cann_overlay=none ${_dsv41_inner:-inner=unknown}"
  fi
fi
if [ -n "${MY_ENV_DSV41_CANN_OVERLAY:-}" ] && [ ! -f "${MY_ENV_DSV41_CANN_OVERLAY}/set_env.sh" ]; then
  echo "ERROR: MY_ENV_DSV41_CANN_OVERLAY=${MY_ENV_DSV41_CANN_OVERLAY} has no set_env.sh" >&2
  exit 1
fi
if [ -n "${MY_ENV_DSV41_CANN_OVERLAY:-}" ]; then
  # overlay 包里的 set_env.sh 把 version_dirpath 写死为 /usr/local/Ascend/cann-9.1.0，
  # 直接 source 会指回旧 B075，因此按它的逻辑显式导出（架构固定 aarch64）。
  _ov="${MY_ENV_DSV41_CANN_OVERLAY}"
  export ASCEND_TOOLKIT_HOME="$_ov" ASCEND_HOME_PATH="$_ov" ASCEND_OPP_PATH="$_ov/opp"
  export ASCEND_AICPU_PATH="$_ov" TOOLCHAIN_HOME="$_ov/toolkit"
  export PATH="$_ov/bin:$_ov/tools/ccec_compiler/bin:$_ov/tools/profiler/bin:$PATH"
  _ov_py="$_ov/python/site-packages:$_ov/opp/built-in/op_impl/ai_core/tbe"
  [ -n "${PYTHONPATH:-}" ] && _ov_py="$_ov_py:$PYTHONPATH"
  export PYTHONPATH="$_ov_py"
  _ov_ld="$_ov/lib64:$_ov/lib64/plugin/opskernel:$_ov/lib64/plugin/nnengine:$_ov/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
  [ -d "$_ov/devlib" ] && _ov_ld="$_ov_ld:$_ov/devlib"
  [ -n "${LD_LIBRARY_PATH:-}" ] && _ov_ld="$_ov_ld:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$_ov_ld"
  mkdir -p "$_ov/var/log" 2>/dev/null || true
  [ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
  echo "MY_DEBUG_LOG env_dsv41 cann_overlay=$_ov"
else
  for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
              /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
  done
fi
export PATH=/usr/local/python3.11.15/bin:$PATH
export MY_ENV_COMPAT_ROOT=/mnt/share/m00899630/envcompat
export PYTHONPATH="$MY_ENV_COMPAT_ROOT:$R/vllm:$R/vllm-ascend:$R/FSDPTurbo:$R/verl:${PYTHONPATH:-}"
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true
# 80.5.25.x 镜像缺 transformers.models.qwen4_exp 时前置 5.21.0 副本（27.x 自带，条件不成立）
TF_DIR=$(python3 -c 'import os, transformers; print(os.path.dirname(transformers.__file__))' 2>/dev/null || true)
if [ -n "$TF_DIR" ] && [ ! -d "$TF_DIR/models/qwen4_exp" ]; then
  export PYTHONPATH="/mnt/share/m00899630/env/py521:${PYTHONPATH:-}"
fi

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
export MY_IP=${MY_IP:-$(hostname -I | tr ' ' '\n' | grep -m1 '^80\.5\.' || true)}
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

# [1117 4node-mixed] 多机编排时 Ray 由启动器统一起（默认 0 = 单机行为逐字不变）
if [ "${MY_ENV_DSV41_RAY_SKIP_START:-0}" = "1" ]; then
  echo "MY_DEBUG_LOG env_dsv41 ray_skip_start=1"
else
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats --include-dashboard=false
fi
