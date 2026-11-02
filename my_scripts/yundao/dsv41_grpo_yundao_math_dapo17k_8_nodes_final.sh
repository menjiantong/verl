#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1 · Math DAPO 17k · 云道 8 机 128 卡 · 全量 384 专家
#
# 单文件自包含版本：所有训练、保存、并行、显存和引擎开关都写在下面，
# 不再依赖 1030_ray_dsv41_env.sh 或启动命令行 export。
#
# 依赖修复：云道镜像自带 transformers 5.10.4，缺 FSDPTurbo 需要的
# transformers.models.qwen4_exp；脚本会把验证容器（80.5.27.182）的
# transformers 5.21.0 + tokenizers 0.23.3 覆盖包放到 PYTHONPATH 最前面。
#
# 云道“启动命令”只填这一行：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# 注意：不要给本脚本加 `set -u`。CANN 的 set_env.sh 会读取未定义的 $1。
# =============================================================================
set -eo pipefail

# =============================================================================
# 一、固定配置（全部写死，改这里即可）
# =============================================================================

# ------------------------------ 路径 ----------------------------------------
MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl
MODEL_PATH=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16-noengram
TOKENIZER_PATH=/efs_rl/m00899630/dsv41/rl_tokenizer
TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
TEST_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet

PROJECT_NAME=GRPO-DeepSeekV41
RUN_STAMP=$(date +%m%d-%H%M%S)
EXPERIMENT_NAME="GRPO-DSV41-MathDAPO17k-8node-${RUN_STAMP}"
CKPTS_DIR="/efs_rl/m00899630/dsv41/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"

WORKING_DIR="${MIRROR_ROOT}/verl"
EXAMPLE="${WORKING_DIR}/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh"
RUNTIME_ENV="${WORKING_DIR}/verl/trainer/runtime_env.yaml"
LOG_BASE="${WORKING_DIR}/logs"

# 依赖覆盖包：从 32 专家验证容器 80.5.27.182 打包
#   transformers 5.21.0（含 models/qwen4_exp）
#   tokenizers  0.23.3
#   transformers-5.21.0.dist-info / tokenizers-0.23.3.dist-info
DEPS_DIR=/efs_rl/m00899630/dsv41/deps/transformers521
DEPS_TAR="${DEPS_DIR}/transformers521_tokenizers023_meta.tar.gz"

# ------------------------------ 数据/序列 ------------------------------------
MAX_PROMPT_LEN=384
MAX_RESPONSE_LEN=1536
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
TRAIN_BSZ=16
VAL_BSZ=8
SAMPLE_N=8

# ------------------------------ 优化 ----------------------------------------
LR=5e-6
USE_KL=1
KL_COEF=0.001
TRAIN_EPOCHS=1000
SYNC_REFILL=True

# ------------------------------ 并行 ----------------------------------------
# 8 机 × 16 卡 = 128 rank；384 专家 / EP128 = 3 专家/rank。
NNODES=8
NDEVICES_PER_NODE=16
FSDP_SIZE=128
TP_SIZE=1
EP_SIZE=128
EFSDP_SIZE=1
GEN_TP=8
GEN_DP=16
GEN_EP=128

# ------------------------------ rollout / offload ---------------------------
INFER_BACKEND=vllm
ROLLOUT_GPU_MEM_UTIL=0.5
FREE_CACHE_ENGINE=True
BATCH_INVARIANT=true
LOAD_FORMAT=auto
AGENT_LOOP=deepseek_v41_vl_single_turn
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
OFFLOAD_POLICY=True
REF_PARAM_OFFLOAD=${PARAM_OFFLOAD}

# ------------------------------ 保存 ----------------------------------------
# 110 步保存一次；500 步末步也会保存一份。
# 当前 /efs_rl 约 4.8T，每份全模型约 1.2T，keep=3 约 3.6T，安全。
# 若把共享盘清到 >6.5T，可改成 5，保留 110/220/330/440/500 五份。
STEPS=500
SAVE_FREQ=110
MAX_CKPT_KEEP=3
SAVE_CONTENTS="['model']"
LOAD_CONTENTS="['model']"
ROLLOUT_IS=null

# ------------------------------ recompute -----------------------------------
# [2026-11-01 OOM 修复] 必须是 ffn + attn，和单机 rebase 长跑（logs/longrun500_20261101_073304.log）一致。
# 证据：
#   1) 10/31 8 机 ffn-only 时 actor/perf/max_memory_allocated_gb=50.27、reserved=52.05，
#      已经贴着 61.27GiB 的上限（PyTorch 可用约 52GiB，其余是 CANN/HCCL 预留）；
#   2) 11/01 8 机 rebase（本次 OOM）在第一个 update 的 routed expert SwiGLU 处
#      allocate 646MiB 失败：已分配 51.60GiB / 可用 151MiB；
#   3) 本模型 40 层、64 heads × head_dim 512，单条 1920 token 的 Q / attn 输出
#      各约 126MiB/层，attn 重计算可回收约 10GiB，正好留出安全余量。
# 代价：attention 前向重算一次，单步约 +2%~3%。
RECOMPUTE_PLAN='["model.layers.{*}.ffn","model.layers.{*}.attn"]'

# ------------------------------ 网络覆盖（一般留空） ------------------------
# 云道容器默认取 hostname -I 的第一个地址（172.16.x / eth0）；
# 物理机如需 80.5.x，可把 MY_IP_OVERRIDE 写成具体 IP。
MY_IP_OVERRIDE=""
NIC_OVERRIDE=""

# =============================================================================
# 二、训练/显存/引擎环境变量（全部显式导出）
# =============================================================================

export MIRROR_ROOT MODEL_PATH TOKENIZER_PATH TRAIN_FILE TEST_FILE
export PROJECT_NAME EXPERIMENT_NAME CKPTS_DIR
export WORKING_DIR EXAMPLE RUNTIME_ENV LOG_BASE
export VERL_LOG_DIR="$LOG_BASE"
export DEVICE=npu
export VERL_DSV41_MAX_SEQ_LEN="$MAX_MODEL_LEN"
export VERL_DSV41_ALIGN_NOBI=0
export MAX_PROMPT_LEN MAX_RESPONSE_LEN MAX_MODEL_LEN
export TRAIN_BSZ VAL_BSZ SAMPLE_N
export LR USE_KL KL_COEF TRAIN_EPOCHS SYNC_REFILL
export NNODES NDEVICES_PER_NODE
export FSDP_SIZE TP_SIZE EP_SIZE EFSDP_SIZE
export GEN_TP GEN_DP GEN_EP
export INFER_BACKEND ROLLOUT_GPU_MEM_UTIL FREE_CACHE_ENGINE
export BATCH_INVARIANT LOAD_FORMAT AGENT_LOOP
export PARAM_OFFLOAD OPTIMIZER_OFFLOAD OFFLOAD_POLICY REF_PARAM_OFFLOAD

# 训练/一致性
export VERL_DSV41_VISION=1
export MY_ENV_TRAIN_FUSED_VISION_ATTN=1
export VERL_DSV41_OFFLOAD_PIN_MEMORY=0
export VERL_DSV41_CAMEM_BACKUP_PIN=0
export VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=1
export VERL_DSV41_EMPTY_CACHE_GRAD_CLIP=1
export VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1
export VERL_DSV41_NUMA_INTERLEAVE=1
export VERL_DSV41_ENGRAM_PIN_MEMORY=0
export VERL_DSV41_DIRECT_SHARD_LOAD=1
export VERL_DSV41_LOCAL_EXPERT_EXPORT=1
export VERL_SYNC_PROFILE=1
export DSA_SCATTER_DROP_MARKERS=0
export VERL_DSV41_OPT_STATE_DEBUG=1
export VERL_DSV41_OPT_STATE_DEBUG_DIR="${MIRROR_ROOT}/logs/opt_state_debug"
export VERL_DSV41_DISABLE_THP=1
export FSDP_TURBO_ATTN_RECOMPUTE=1
export FSDP_TURBO_ATTN_CHUNK=64
export VA_V41_ATTN_EAGER=0
export VA_V41_OPROJ_FP32_AR=0
export VA_V41_OPROJ_FP32_PARTIAL=0
export VERL_DSV41_HC_BF16=0
export VERL_DSV41_ALIGN_ATTN=0
export VERL_DSV41_MOE_BF16=0
export ALIGN_NOBI=0
export LOCAL_EXPERT_EXPORT=1
export VERL_DSV41_ENGINE_LEGACY=0
export VLLM_ENGRAM_DP_SHARED_MEMORY=0

# vLLM / Ascend 引擎
export VLLM_VERSION=0.30.0
export HCCL_BUFFSIZE=1024
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60150
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61150

# Ray（云道容器沿用 10/31 验证过的 6379）
export RAY_PORT=6379
export RAY_AUTH_MODE=disabled
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=1000

# =============================================================================
# 三、CANN / Python 路径 / checkpoint 修复守卫
# =============================================================================
for _env in /usr/local/Ascend/cann/set_env.sh \
            /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh \
            /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done

export PATH=/usr/local/python3.11.15/bin:$PATH

# 依赖修复模块：保证 qwen4_exp 存在，并让覆盖包优先于镜像自带 transformers。
ensure_dsv41_dependencies() {
  if [ -d "$DEPS_DIR/transformers/models/qwen4_exp" ] && \
     [ -d "$DEPS_DIR/tokenizers" ] && \
     [ -d "$DEPS_DIR/transformers-5.21.0.dist-info" ] && \
     [ -d "$DEPS_DIR/tokenizers-0.23.3.dist-info" ]; then
    return 0
  fi
  if [ ! -f "$DEPS_TAR" ]; then
    echo "ERROR: dependency overlay missing: $DEPS_DIR" >&2
    echo "       expected transformers 5.21.0 + tokenizers 0.23.3 with qwen4_exp" >&2
    exit 1
  fi
  echo "[deps] extracting $DEPS_TAR -> $DEPS_DIR"
  mkdir -p "$DEPS_DIR"
  tar xzf "$DEPS_TAR" -C "$DEPS_DIR"
  if [ ! -d "$DEPS_DIR/transformers/models/qwen4_exp" ]; then
    echo "ERROR: qwen4_exp still missing after extracting ${DEPS_TAR}" >&2
    exit 1
  fi
}
ensure_dsv41_dependencies

export PYTHONPATH="${DEPS_DIR}:${MIRROR_ROOT}/vllm:${MIRROR_ROOT}/vllm-ascend:${MIRROR_ROOT}/FSDPTurbo:${MIRROR_ROOT}/verl:${PYTHONPATH:-}"
if ! python3 -c 'import transformers, tokenizers, transformers.models.qwen4_exp; assert transformers.__version__ == "5.21.0", transformers.__version__; assert tokenizers.__version__ == "0.23.3", tokenizers.__version__; print("[deps] transformers", transformers.__version__, "tokenizers", tokenizers.__version__)'; then
  echo "ERROR: dependency overlay is not effective; check PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true

grep -q "2026-11-01 ckpt-fix" \
  "${MIRROR_ROOT}/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py" || {
  echo "ERROR: verl checkpoint fix not found under ${MIRROR_ROOT}/verl" >&2
  exit 1
}

for _p in "$MODEL_PATH" "$TOKENIZER_PATH" "$TRAIN_FILE" "$TEST_FILE" "$EXAMPLE" "$RUNTIME_ENV"; do
  if [ ! -e "$_p" ]; then
    echo "ERROR: required path missing: $_p" >&2
    exit 1
  fi
done

# =============================================================================
# 四、云道运行时信息（节点角色 / 业务网卡）
# =============================================================================
NODE_RANK="${VC_TASK_INDEX:-0}"
if [ "${MA_NUM_HOSTS:-0}" -ne "$NNODES" ]; then
  echo "ERROR: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}, expected ${NNODES}" >&2
  exit 1
fi
if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
  echo "ERROR: VC_TASK_INDEX=${NODE_RANK} out of range [0, ${NNODES})" >&2
  exit 1
fi
if [ -z "${VC_WORKER_HOSTS:-}" ]; then
  echo "ERROR: VC_WORKER_HOSTS is empty" >&2
  exit 1
fi
HEAD_HOST="${VC_WORKER_HOSTS%%,*}"
export NODE_RANK HEAD_HOST

if [ -n "$MY_IP_OVERRIDE" ]; then
  MY_IP="$MY_IP_OVERRIDE"
else
  MY_IP=$(hostname -I | awk '{print $1}')
fi
if [ -z "$MY_IP" ]; then
  echo "ERROR: cannot determine MY_IP" >&2
  exit 1
fi

if [ -n "$NIC_OVERRIDE" ]; then
  NIC="$NIC_OVERRIDE"
else
  NIC=$(MY_IP="$MY_IP" python3 -c \
    'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(i for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address==ip))' \
    2>/dev/null || echo eth0)
fi
if [ -z "$NIC" ]; then
  echo "ERROR: cannot determine NIC for MY_IP=${MY_IP}" >&2
  exit 1
fi

export MY_IP NIC
export GLOO_SOCKET_IFNAME="$NIC"
export HCCL_SOCKET_IFNAME="$NIC"
export HCCL_IF_IP="$MY_IP"

echo "[=====info=====] ModelArts: MA_NUM_HOSTS=${MA_NUM_HOSTS} VC_TASK_INDEX=${NODE_RANK} HEAD_HOST=${HEAD_HOST}"
echo "[=====info=====] Net: MY_IP=${MY_IP} NIC=${NIC} GLOO=${GLOO_SOCKET_IFNAME} HCCL=${HCCL_SOCKET_IFNAME}"
echo "[=====info=====] Deps: ${DEPS_DIR} (transformers 5.21.0, tokenizers 0.23.3)"
echo "[=====info=====] Topology: world=$((NNODES * NDEVICES_PER_NODE)) FSDP=${FSDP_SIZE} EP=${EP_SIZE} EFSDP=${EFSDP_SIZE}"
echo "[=====info=====] Rollout: TP=${GEN_TP} DP=${GEN_DP} EP=${GEN_EP}"
echo "[=====info=====] Data: prompt=${MAX_PROMPT_LEN} response=${MAX_RESPONSE_LEN} max_model_len=${MAX_MODEL_LEN}"
echo "[=====info=====] Save: every ${SAVE_FREQ} steps, keep=${MAX_CKPT_KEEP}, steps=${STEPS}"
echo "[=====info=====] CKPTS_DIR=${CKPTS_DIR}"

AVAIL_T=$(df -Pk /efs_rl | awk 'NR==2 {printf "%.1f", $4/1024/1024/1024}')
NEED_T=$(awk -v k="$MAX_CKPT_KEEP" 'BEGIN{printf "%.1f", k*1.2}')
echo "[=====info=====] Disk: /efs_rl avail=${AVAIL_T}T, ${MAX_CKPT_KEEP} checkpoints ~${NEED_T}T"
if awk -v a="$AVAIL_T" -v n="$NEED_T" 'BEGIN{exit !(a < n + 0.5)}'; then
  echo "[WARN] /efs_rl free space may be insufficient for ${MAX_CKPT_KEEP} checkpoints" >&2
fi

# =============================================================================
# 五、启动 Ray
# =============================================================================
if [ "$NODE_RANK" -eq 0 ]; then
  echo "[=====info=====] start Ray head on ${HEAD_HOST}:${RAY_PORT}"
  ray start --head --port="$RAY_PORT" --dashboard-host=0.0.0.0 \
    --node-ip-address="$HEAD_HOST" --disable-usage-stats
  sleep 10
else
  echo "[=====info=====] start Ray worker, join ${HEAD_HOST}:${RAY_PORT}"
  ray start --address="${HEAD_HOST}:${RAY_PORT}" --disable-usage-stats
  sleep 5
fi
ray status

# =============================================================================
# 六、rank0 启动训练；其他节点保活
# =============================================================================
if [ "$NODE_RANK" -eq 0 ]; then
  echo "[=====info=====] code: verl=$(git -C "${MIRROR_ROOT}/verl" rev-parse --short HEAD 2>/dev/null) vllm=$(git -C "${MIRROR_ROOT}/vllm" rev-parse --short HEAD 2>/dev/null) vllm-ascend=$(git -C "${MIRROR_ROOT}/vllm-ascend" rev-parse --short HEAD 2>/dev/null) FSDPTurbo=$(git -C "${MIRROR_ROOT}/FSDPTurbo" rev-parse --short HEAD 2>/dev/null)"

  RUN_TAG="dsv41-8node-math-dapo17k-p384-rebase-s${SAVE_FREQ}-k${MAX_CKPT_KEEP}"
  RUN_ID="${RUN_TAG}-$(date +%Y%m%d_%H%M%S)-$$"
  RUN_DIR="${LOG_BASE}/${RUN_ID}"
  mkdir -p "$RUN_DIR"
  cp "$EXAMPLE" "${RUN_DIR}/"
  cp "$(readlink -f "$0")" "${RUN_DIR}/launch_script.sh.bak"
  RUN_LOG="${RUN_DIR}/train-math-dapo17k.log"
  export ASCEND_PROCESS_LOG_PATH="${RUN_DIR}/plog"

  cd "$WORKING_DIR"

  EXTRA_ARGS=(
    "trainer.total_epochs=${TRAIN_EPOCHS}"
    "actor_rollout_ref.rollout.load_format=${LOAD_FORMAT}"
    "actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP}"
    "trainer.v1.sampler.sync_refill_failed_groups=${SYNC_REFILL}"
  )

  bash "$EXAMPLE" \
    "trainer.total_training_steps=${STEPS}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}" \
    "actor_rollout_ref.actor.checkpoint.load_contents=${LOAD_CONTENTS}" \
    "++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true" \
    "++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=${RECOMPUTE_PLAN}" \
    "algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_LOG"
else
  echo "[=====info=====] worker node: keep container alive"
  sleep infinity
fi
