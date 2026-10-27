#!/usr/bin/env bash
set -eo pipefail

MIRROR_ROOT=/efs_rl/m00899630/workspace/dsv41-verl/
# 昇腾环境（ssh/非登录 shell 下必须；登录 shell 里已有则不重复生效也无害）。
# nnal/atb 的 set_env 提供 libatb.so —— vLLM 引擎 worker 需要，缺它报
# "OSError: libatb.so: cannot open shared object file"。
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done

export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:$MIRROR_ROOT/FSDPTurbo:$MIRROR_ROOT/verl:${PYTHONPATH:-}"
export PATH=/usr/local/python3.11.15/bin:$PATH
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true
# 本机业务网卡 IP：默认从 hostname -I 里取 80.5.25.x 段（131 机=80.5.25.131，119 机=80.5.25.119），可用 MY_IP 覆盖


export MY_IP=$(hostname -I | awk '{print $1}')
export NIC="$(python3 -c 'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(iface for iface,addrs in psutil.net_if_addrs().items() for addr in addrs if addr.family==socket.AF_INET and addr.address==ip))')"

export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME

RAY_PORT=${RAY_PORT:-6390}                                  # 避外部 GCS（F-77）
# 本 ray 版本默认开 token 认证：跨机 `ray start --address` 会因 InvalidAuthToken 失败
# （即使 ~/.ray/auth_token 与 head 完全一致）。按报错提示显式退出认证（内网集群）。
export RAY_AUTH_MODE=${RAY_AUTH_MODE:-disabled}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# -------------------------------------------- 模型配置 --------------------------------------------------
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-sft116-$(date +%m%d-%H%M)}
export MODEL_PATH=${MODEL_PATH:-/efs_rl/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram}   
export TRAIN_FILE=${TRAIN_FILE:-/efs_rl/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet}
export TEST_FILE=${TEST_FILE:-/efs_rl/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/efs_rl/m00899630/dsv41/rl_tokenizer}
export CKPTS_DIR=${CKPTS_DIR:-"/efs_rl/m00899630/dsv41/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
export TRAIN_BSZ=${TRAIN_BSZ:-8} SAMPLE_N=${SAMPLE_N:-8}
export LR=${LR:-5e-6} USE_KL=${USE_KL:-1} KL_COEF=${KL_COEF:-0.001}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
# -------------------------------------------- 模型配置 --------------------------------------------------


# ---------- 内存/offload 策略（均可用环境变量覆盖） ----------
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
export PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
export OFFLOAD_POLICY=${OFFLOAD_POLICY:-True}

export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-${PARAM_OFFLOAD}}

# 训推一致性开关（默认全关）。⚠️ E39：这些是 **worker 进程读 env** 的开关，而 worker 继承本机
# raylet 的环境（≠ driver）——只写在主机命令行上会导致从机 worker 拿不到！两机要一致必须
# 双方命令行都传（或改这里的默认值，脚本双方都执行，最稳）。任务 55 起：两机都在命令行显式传。
export VA_V41_ATTN_EAGER=${VA_V41_ATTN_EAGER:-0} VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-0}
export VA_V41_OPROJ_FP32_PARTIAL=${VA_V41_OPROJ_FP32_PARTIAL:-0}
export VERL_DSV41_HC_BF16=${VERL_DSV41_HC_BF16:-0} VERL_DSV41_ALIGN_ATTN=${VERL_DSV41_ALIGN_ATTN:-0}
export VERL_DSV41_MOE_BF16=${VERL_DSV41_MOE_BF16:-0}

# [2026-10-26] DeepSeek-V4.1 是 VL 模型，countbenchqa_lite 的 prompt 自带 <image>；
# 训练侧 FSDPTurbo 的 vision tower 由 VERL_DSV41_VISION 控制，默认 0（不建 tower），
# 而 rollout 侧 load_format=auto 会带图。两边不一致时 compute_log_prob 报
# "Image inputs require an enabled DeepSeek vision tower."。
# 参考 _launch_W3_longrun_1022.sh:30 的 VERL_DSV41_VISION=1；freeze 默认 1。
export VERL_DSV41_VISION=${VERL_DSV41_VISION:-1}

export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0
# [2026-10-25 全量模型] 16 rank/机 + FSDP128/EP64 的 comm 数比 top32 单机多，端口段从 50 放宽到
# 150；engine server actor 在 vllm_async_server.py 里固定用 60200-60350/61200-61350，互不重叠
# （10-19 v9 端口冲突证据见 worklog_dsv41_16nodes_1019.md 01:40）。
export HCCL_HOST_SOCKET_PORT_RANGE=${MY_ENV_HCCL_HOST_RANGE:-60000-60150} HCCL_NPU_SOCKET_PORT_RANGE=${MY_ENV_HCCL_NPU_RANGE:-61000-61150}
export VLLM_VERSION=0.30.0 HCCL_BUFFSIZE=1024 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=1000
export FSDP_TURBO_ATTN_RECOMPUTE=1 FSDP_TURBO_ATTN_CHUNK=64
export VERL_DSV41_DIRECT_SHARD_LOAD=1 VERL_DSV41_LOCAL_EXPERT_EXPORT=1 VERL_SYNC_PROFILE=1
export DSA_SCATTER_DROP_MARKERS=0
# 全模型（含 engram）双机跑：FSDP 梯度/参数 offload 关闭 pinned host 分配——本驱动在共享多 NUMA
# 节点上对超大 pinned 分配会失败（aclrtMallocHostWithCfg 207001，见 worklog_dsv41_119_bringup.md
# E10-E13）；pageable 同步拷贝替代，语义不变（代码默认 1=保留历史行为，仅本启动器默认 0）。
export VERL_DSV41_OFFLOAD_PIN_MEMORY=${VERL_DSV41_OFFLOAD_PIN_MEMORY:-0}
# [2026-10-17 E39] Ray worker 继承的是"本机 raylet"的环境，不是 driver 的环境：从机 shell 里
# 只跑 `... slave <IP>` 时拿不到主机的命令行 env。历史上主机传 OFFLOAD_PIN=0/CAMEM_BACKUP_PIN=0/
# EMPTY_CACHE_*=1/HC_BF16=1 而从机没有 → 两机行为分裂（从机 pinned 慢路径/E34 慢睡、精度不一致）。
# 这些标准开关一律改为脚本内默认（master/slave 双方都会执行本脚本，天然同源）。
export VERL_DSV41_CAMEM_BACKUP_PIN=${VERL_DSV41_CAMEM_BACKUP_PIN:-0}
export VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=${VERL_DSV41_EMPTY_CACHE_AFTER_BUILD:-1}
export VERL_DSV41_EMPTY_CACHE_GRAD_CLIP=${VERL_DSV41_EMPTY_CACHE_GRAD_CLIP:-1}
export VERL_DSV41_NUMA_INTERLEAVE=${VERL_DSV41_NUMA_INTERLEAVE:-1}
# 训练侧 engram 表默认不 pinned（省 ~366GB/节点 驱动 pinned 压力；见 M17/E10-E14）。
export VERL_DSV41_ENGRAM_PIN_MEMORY=${VERL_DSV41_ENGRAM_PIN_MEMORY:-0}
export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}
export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}
# M29/M30（E38：task54 首步 Adam 惰性建状态卡死 ~3h 的排查与规避）：
# M29 预建优化器状态 + 每参数进度/看门狗日志（数值口径与 torch 惰性初始化一致）；
# M30 step 前 empty_cache+gc。两机同源（本脚本双方都会执行）保证 env 一致。
export VERL_DSV41_OPT_STATE_DEBUG=${VERL_DSV41_OPT_STATE_DEBUG:-1}
export VERL_DSV41_OPT_STATE_DEBUG_DIR=${VERL_DSV41_OPT_STATE_DEBUG_DIR:-$MIRROR_ROOT/logs/opt_state_debug}
export VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=${VERL_DSV41_EMPTY_CACHE_BEFORE_OPT:-1}
# M32（E41：宿主机 THP=always + 碎片化 → 大分配缺页触发分钟级 compaction stall）：
# 进程级关闭 THP（prctl PR_SET_THP_DISABLE，见 sitecustomize.py），只影响本作业进程。
export VERL_DSV41_DISABLE_THP=${VERL_DSV41_DISABLE_THP:-1}


# -------------------------------------------- 并行策略 --------------------------------------------------
RAY_PORT=6379

# ModelArts injected envs:

export NDEVICES_PER_NODE=16
export NNODES="${MA_NUM_HOSTS:-1}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
export HEAD_HOST="${VC_WORKER_HOSTS%%,*}"

# 打印云道注入的环境变量，便于排障；只读，不改变启动逻辑。
echo "[=====info=====] ModelArts env: MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>} VC_WORKER_NUM=${VC_WORKER_NUM:-<unset>} VC_TASK_INDEX=${VC_TASK_INDEX:-<unset>}"
echo "[=====info=====] ModelArts env: MA_NUM_GPUS=${MA_NUM_GPUS:-<unset>} MA_CURRENT_IP=${MA_CURRENT_IP:-<unset>} MA_TASK_NAME=${MA_TASK_NAME:-<unset>} MA_CURRENT_INSTANCE_NAME=${MA_CURRENT_INSTANCE_NAME:-<unset>}"
echo "[=====info=====] ModelArts env: MA_HOME=${MA_HOME:-<unset>} MA_JOB_DIR=${MA_JOB_DIR:-<unset>} MA_MOUNT_PATH=${MA_MOUNT_PATH:-<unset>} MA_LOG_DIR=${MA_LOG_DIR:-<unset>}"
echo "[=====info=====] ModelArts env: MA_SCRIPT_INTERPRETER=${MA_SCRIPT_INTERPRETER:-<unset>} MA_HANG_DETECT_TIME=${MA_HANG_DETECT_TIME:-<unset>} MA_SKIP_IMAGE_DETECT=${MA_SKIP_IMAGE_DETECT:-<unset>}"
echo "[=====info=====] ModelArts env: VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-<unset>}"
echo "[=====info=====] Net env: MY_IP=${MY_IP} NIC=${NIC} HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-<unset>} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"

export EXPERIMENT_NAME=GRPO-DSV41-Full-NoEngram-8node-$(date +%m%d-%H%M)
export FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"
# -------------------------------------------- 并行策略 --------------------------------------------------



EXAMPLE=examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh


echo "[=====info=====] NNODES=${NNODES} NODE_RANK=${NODE_RANK} HEAD_HOST=${HEAD_HOST} NDEVICES_PER_NODE=${NDEVICES_PER_NODE} FSDP_SIZE=${FSDP_SIZE} EP_SIZE=${EP_SIZE} world=$((NNODES * NDEVICES_PER_NODE))"

if [[ "${NNODES}" -eq 1 ]]; then
  echo "[=====info=====] single node: start ray head locally"
  ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
else
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    echo "[=====info=====] start ray head on ${HEAD_HOST}"
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0 --node-ip-address="$HEAD_HOST"
    sleep 10
  else
    echo "[=====info=====] start ray worker, connect to ${HEAD_HOST}:${RAY_PORT}"
    ray start --address="${HEAD_HOST}:${RAY_PORT}"
    sleep 5
  fi
fi

ray status
LOG_BASE="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs"
RUN_TAG="dsv41-8node-1node"
STEPS=500

# IMPORTANT: only run the training entry on rank0, others just sleep
if [[ "${NODE_RANK}" -eq 0 ]]; then
  echo "[=====info=====] launch training on head"
  RUN_ID="${RUN_TAG}-$(date +%Y%m%d_%H%M%S)-$$"
  RUN_DIR="${LOG_BASE}/${RUN_ID}"
  mkdir -p "${RUN_DIR}"
  cp "$MIRROR_ROOT/verl/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh" "${RUN_DIR}/"
  cp "$0" "${RUN_DIR}/ray_start.sh.bak"
  RUN_LOG="${RUN_DIR}/train-20k-1node.log"
  export ASCEND_PROCESS_LOG_PATH="${RUN_DIR}/plog"
  cd "$MIRROR_ROOT/verl"
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
      trainer.total_training_steps="${STEPS}" \
      +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true \
      '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
      algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-null} \
      ${TRAIN_EXTRA_ARGS:-} \
      2>&1 | tee "${RUN_LOG}"
else
  echo "[=====info=====] worker node: keep container alive"
  sleep infinity
fi






# export TRANSFORMERS_VERBOSITY=error
# export VLLM_ASCEND_TASK_QUEUE_ENABLE=1
# export VLLM_VERSION=0.23.0

# export VLLM_SLEEP_LEVEL=2
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='true'
# export CUDA_DEVICE_MAX_CONNECTIONS="1"
# export VLLM_USE_V1="1"
# export HCCL_CONNECT_TIMEOUT="7200"
# export HCCL_EXEC_TIMEOUT="7200"
# export RAY_DEDUP_LOGS="1"
# # export VLLM_ASCEND_ENABLE_NZ="0"
# # export VLLM_ALL2ALL_BACKEND="flashinfer_all2allv"
# export HCCL_OP_EXPANSION_MODE="AIV"
# export SWE_REX_LOG_STREAM_LEVEL="ERROR"
# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# export HYDRA_FULL_ERROR=1

# export RAY_memory_usage_threshold=0.99
# export RAY_DEBUG=0
# export RAY_DEBUG_POST_MORTEM=0
# export ASCEND_LAUNCH_BLOCKING=0
# export ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
# #TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
# export TASK_QUEUE_ENABLE=1
# export HCCL_ASYNC_ERROR_HANDLING=0
# export GLOO_CONNECT_TIMEOUT=7200
# export HCCL_IF_BASE_PORT=50000
# #export HCCL_HOST_SOCKET_PORT_RANGE=auto
# # EI0020 NPU socket bind 冲突（vLLM DP/EP 单卡多进程下 auto 会撞端口），固定端口段
# #export HCCL_NPU_SOCKET_PORT_RANGE=61000-61200 
# export HCCL_BUFFSIZE=150
# export HCCL_ENTRY_LOG_ENABLE=1
# export HCCL_DEBUG=INFO
# export CPU_AFFINITY_CONF=1
# export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
# export PYTHONUNBUFFERED=1

# ## VLLM AND CUSTOM
# export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

# # export ASCEND_PROCESS_LOG_PATH="/efs_rl/l00870118/run_glm52_pp8/process_logs_32/$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S')"

# # git -C /verl apply /efs_rl/l00870118/run_glm52_pp8/0001-apply-vllm-patch.patch

# RAY_PORT=6379

# # ModelArts injected envs:
# NNODES="${MA_NUM_HOSTS:-1}"
# NODE_RANK="${VC_TASK_INDEX:-0}"

# # Pick the first host as head
# HEAD_HOST="${VC_WORKER_HOSTS%%,*}"


# echo "[info] NNODES=${NNODES} NODE_RANK=${NODE_RANK} HEAD_HOST=${HEAD_HOST}"

# if [[ "${NNODES}" -eq 1 ]]; then
#   echo "[info] single node: start ray head locally"
#   ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
# else
#   if [[ "${NODE_RANK}" -eq 0 ]]; then
#     echo "[info] start ray head on ${HEAD_HOST}"
#     ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0 --node-ip-address="$HEAD_HOST"
#     sleep 10
#   else
#     echo "[info] start ray worker, connect to ${HEAD_HOST}:${RAY_PORT}"
#     ray start --address="${HEAD_HOST}:${RAY_PORT}"
#     sleep 5
#   fi
# fi

# ray status
# LOG_BASE="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs"
# RUN_TAG="dsv41-8node-1node"

# # IMPORTANT: only run the training entry on rank0, others just sleep
# if [[ "${NODE_RANK}" -eq 0 ]]; then
#   echo "[info] launch training on head"
#   RUN_ID="${RUN_TAG}-$(date +%Y%m%d_%H%M%S)-$$"
#   RUN_DIR="${LOG_BASE}/${RUN_ID}"
#   mkdir -p "${RUN_DIR}"
#   cp /efs_rl/y00943488/v4-a3/sh/train_deepseek_v4_grpo_mindspeed_vllm-20k-1node-xc-0.sh "${RUN_DIR}/"
#   cp "$0" "${RUN_DIR}/ray_start.sh.bak"
#   LOG_PATH="${RUN_DIR}/train-20k-1node.log"
#   export ASCEND_PROCESS_LOG_PATH="${RUN_DIR}/plog"
#   cd /efs_rl/y00943488/v4-a3/verl
#   bash /efs_rl/y00943488/v4-a3/sh/train_deepseek_v4_grpo_mindspeed_vllm-20k-1node-xc-0.sh  2>&1 | tee ${LOG_PATH}
# else
#   echo "[info] worker node: keep container alive"
#   sleep infinity
# fi
