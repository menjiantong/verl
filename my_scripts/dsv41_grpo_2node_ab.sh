#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1 top32 GRPO · 双机一键（master/slave 合并版）
#
#   master： bash dsv41_grpo_2node.sh master [STEPS]      # 建议 nohup 运行
#   slave ： bash dsv41_grpo_2node.sh slave <MASTER_IP>   # 等 master 打出 waiting 后执行
#   single： bash dsv41_grpo_2node.sh single [STEPS]      # 单机 16 卡（0-15），无 slave/无等待
#
# =============================================================================
set -eo pipefail
MODE=${1:?usage: $0 master [STEPS] | $0 slave <MASTER_IP> | $0 single [STEPS]}

MIRROR_ROOT=/mnt/share/m00899630/workspace/dsv41-verl
# 昇腾环境（ssh/非登录 shell 下必须；登录 shell 里已有则不重复生效也无害）。
# nnal/atb 的 set_env 提供 libatb.so —— vLLM 引擎 worker 需要，缺它报
# "OSError: libatb.so: cannot open shared object file"。
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:$MIRROR_ROOT/FSDPTurbo:$MIRROR_ROOT/verl:${PYTHONPATH:-}"
export PATH=/usr/local/python3.11.15/bin:$PATH
# ssh 非交互 shell 的 ulimit -n 默认 1024，ray 双机 grpc 连接会 "Too many open files"
# （worker.cc:239 / socket() failed）。抬高到硬上限（root 硬限 1048576）。
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true
# 本机业务网卡 IP：默认从 hostname -I 里取 80.5.25.x 段（131 机=80.5.25.131，119 机=80.5.25.119），可用 MY_IP 覆盖
MY_IP=${MY_IP:-$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^80\.5\.25\.')}
NIC=${NIC:-enp48s3u1u1}
RUN_LOG="logs/grpo_top32_sft_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"


RAY_PORT=${RAY_PORT:-6390}                                  # 避外部 GCS（F-77）
# 本 ray 版本默认开 token 认证：跨机 `ray start --address` 会因 InvalidAuthToken 失败
# （即使 ~/.ray/auth_token 与 head 完全一致）。按报错提示显式退出认证（内网集群）。
export RAY_AUTH_MODE=${RAY_AUTH_MODE:-disabled}
if [ "$MODE" = "single" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
else
  # [2026-10-17 E40] CPN 与可见卡必须一致，否则 FSDP_SIZE/GEN_DP 算错：
  # task55 首启只给默认 CPN=8 → 每机 8 卡 + GEN_DP=1，引擎侧
  # `dp_shared_memory requires data_parallel_size > 1`（VllmConfig 校验）直接炸。
  # 规则：显式给了可见卡就反推每机卡数；否则 CPN 显式给定时按它展开可见卡，缺省 8。
  if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    if [ "${CPN:-8}" = "16" ]; then
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    else
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi
  fi
  CPN=${CPN:-$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-sft116-$(date +%m%d-%H%M)}
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer}   
export TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_train_rl.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_test_rl.parquet}
export TRAIN_BSZ=${TRAIN_BSZ:-8} SAMPLE_N=${SAMPLE_N:-8}
export LR=${LR:-5e-6} USE_KL=${USE_KL:-1} KL_COEF=${KL_COEF:-0.001}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024} ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
# 训推一致性开关（默认全关）。⚠️ E39：这些是 **worker 进程读 env** 的开关，而 worker 继承本机
# raylet 的环境（≠ driver）——只写在主机命令行上会导致从机 worker 拿不到！两机要一致必须
# 双方命令行都传（或改这里的默认值，脚本双方都执行，最稳）。任务 55 起：两机都在命令行显式传。
export VA_V41_ATTN_EAGER=${VA_V41_ATTN_EAGER:-0} VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-0}
export VA_V41_OPROJ_FP32_PARTIAL=${VA_V41_OPROJ_FP32_PARTIAL:-0}
export VERL_DSV41_HC_BF16=${VERL_DSV41_HC_BF16:-0} VERL_DSV41_ALIGN_ATTN=${VERL_DSV41_ALIGN_ATTN:-0}
export VERL_DSV41_MOE_BF16=${VERL_DSV41_MOE_BF16:-0}

export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050 HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
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
# ===== [3.2 训推一致性 A/B] 实验开关（每次实验只改一个；本脚本是 dsv41_grpo_2node_new.sh 的副本，
# 原脚本保持不变，便于对照）=====
# AB_ENFORCE_EAGER=1  → actor_rollout_ref.rollout.enforce_eager=True（引擎禁用 graph/cudagraph）
# AB_NO_ASCEND_COMPILE=1 → 显式 additional_config.ascend_compilation_config.{enable_npugraph_ex,enable_static_kernel}=false
#   （注意：当前 cudagraph_mode=FULL_AND_PIECEWISE 时平台代码已自动置 false，此项预期为空操作，用于核对）
# AB_ENGRAM_SHARED=0  → VLLM_ENGRAM_DP_SHARED_MEMORY=0（引擎侧 engram 表不共享，逐副本独立）
# AB_RUN_TAG=xxx      → 仅用于日志/RUN_LOG 命名，便于对照记录
export AB_ENFORCE_EAGER=${AB_ENFORCE_EAGER:-0}
export AB_NO_ASCEND_COMPILE=${AB_NO_ASCEND_COMPILE:-0}
export AB_ENGRAM_SHARED=${AB_ENGRAM_SHARED:-1}
export AB_RUN_TAG=${AB_RUN_TAG:-base}
# [3.2 A/B] 每个实验独立日志目录：logs/ab/<tag>/（主日志 + tee 日志 + nohup）
export VERL_LOG_DIR=${VERL_LOG_DIR:-logs/ab/${AB_RUN_TAG}}
mkdir -p "${VERL_LOG_DIR}"
RUN_LOG="${VERL_LOG_DIR}/grpo_top32_sft_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"
if [ "${AB_ENGRAM_SHARED}" = "0" ]; then export VLLM_ENGRAM_DP_SHARED_MEMORY=0; fi

# export VERL_AGENT_DUMP=${VERL_AGENT_DUMP:-/tmp/agent_dump.txt}
# export VERL_DSV41_DUMP_BATCH=${VERL_DSV41_DUMP_BATCH:-}
# export VERL_DSV41_ALIGN_ROUTER=${VERL_DSV41_ALIGN_ROUTER:-0}
# export VERL_DSV41_ALIGN_MOE=${VERL_DSV41_ALIGN_MOE:-0}
# export VERL_DSV41_ALIGN_HEAD=${VERL_DSV41_ALIGN_HEAD:-0}

if [ "$MODE" = "single" ]; then
  # 单机 = 16 卡一条 node，拓扑与双机 2×8 完全相同（FSDP16/EP8），集合通信全走机内
  export NNODES=1 NDEVICES_PER_NODE=16 FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
else
  # 双机配置（2026-10-16 用户指定）：CPN=每机卡数（8 → 16 卡双机 FSDP16；16 → 32 卡双机 FSDP32）。
  # 引擎按 §八 配方落地：每机 TP8×本地DP(CPN/8)+engram dp_shared_memory；全局 DP=2*CPN/8 个引擎
  # 进程组，EP=TP×DP（rollout 配置 assert 要求）。训练侧 EFSDP2；EFSDP2 下 LOCAL_EXPERT_EXPORT
  # 由 example 脚本守卫自动关闭（该路径只对"单专家维分片"正确），走通用导出（语义正确、较慢）。
  CPN=${CPN:-8}
  # [2026-10-17 E42] 支持任意机数：NNODES 默认为 2（历史双机行为），3 机（48 卡）用 NNODES=3。
  # 世界大小 = NNODES*CPN；EFSDP=2/EP=8 时专家组 16 卡，其余维为 dp_replicate（每机仍持有一份
  # 16 路专家分片，故每机训练内存不随 NNODES 变小）；引擎仍是"每机本地 TP8×DP2"一个副本。
  # ⚠️ bash 陷阱（E45）：同一行 export 里 FSDP_SIZE=$((NNODES * CPN)) 展开时 NNODES 尚未生效（=空→0）
  # → fully_shard_parallel_size=0 → FSDPTurbo get_last_mesh_dim 除零。必须分两行。
  export NNODES=${NNODES:-2} NDEVICES_PER_NODE=${CPN} EP_SIZE=8 EFSDP_SIZE=${EFSDP_SIZE:-2} TP_SIZE=1
  export FSDP_SIZE=$((NNODES * CPN))
  export GEN_TP=${GEN_TP:-8}
  # 引擎 DP = 每机本地 DP（CPN/TP）：每节点一个"§八 式 单机 DP2×TP8"引擎（表共享、start_rank=0、全本地），
  # 双机 = 2 个节点局部副本（verl 级 replica，每站一个 API server）。不要用"跨机 DP4 单引擎"
  # （vLLM hybrid-LB 会拒绝远端 headless：E22）。
  export GEN_DP=${GEN_DP:-$((CPN / GEN_TP))}
  export GEN_EP=${GEN_EP:-$((GEN_TP * GEN_DP))}
  export VLLM_ENGRAM_DP_SHARED_MEMORY=${VLLM_ENGRAM_DP_SHARED_MEMORY:-1}
fi


EXAMPLE=examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh

if [ "$MODE" = "master" ]; then
  STEPS=${2:-50}
  python3 -c "import torch_npu, vllm_ascend" 2>/dev/null || { echo "ERROR: torch_npu/vllm_ascend 导入失败"; exit 1; }
  [ $((TRAIN_BSZ * SAMPLE_N % (NNODES * CPN))) -eq 0 ] || { echo "ERROR: TRAIN_BSZ*SAMPLE_N % (NNODES*CPN=$((NNODES * CPN))) != 0"; exit 1; }
  echo "[master] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | nodes=$NNODES steps=$STEPS lr=$LR kl=$USE_KL bsz=${TRAIN_BSZ}x${SAMPLE_N}"
  for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
  sleep 5; ray stop --force >/dev/null 2>&1 || true
  ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  export RAY_ADDRESS="$MY_IP:$RAY_PORT"
  echo "[master] waiting for $((NNODES - 1)) slave(s) →  到从机执行:  bash $(basename "$0") join $MY_IP  （有外部租户的机器用 join，不做 pkill）"
  for i in $(seq 1 180); do
    active=$(ray status 2>/dev/null | sed -n '/^Active:/,/^Pending:/p' | grep -c "node_" || true)
    [ "${active:-0}" -ge "$NNODES" ] && { echo "[master] all $NNODES nodes joined"; break; }
    [ $((i % 5)) -eq 0 ] && echo "[master] waiting... $((i * 20))s (active=${active:-0}/$NNODES)"
    [ "$i" = 180 ] && { echo "ERROR: slaves 12 分钟未全部加入"; exit 1; }
    sleep 20
  done
  cd "$MIRROR_ROOT/verl"
  echo "[master] launching GRPO: steps=$STEPS"
  AB_ARGS=()
  if [ "${AB_ENFORCE_EAGER}" = "1" ]; then AB_ARGS+=("actor_rollout_ref.rollout.enforce_eager=True"); fi
  if [ "${AB_NO_ASCEND_COMPILE}" = "1" ]; then
    AB_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.ascend_compilation_config.enable_npugraph_ex=false")
    AB_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.ascend_compilation_config.enable_static_kernel=false")
  fi
  echo "[AB] tag=${AB_RUN_TAG} enforce_eager=${AB_ENFORCE_EAGER} no_ascend_compile=${AB_NO_ASCEND_COMPILE} engram_shared=${AB_ENGRAM_SHARED}"
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
      trainer.total_training_steps="${STEPS}" \
      "${AB_ARGS[@]}" \
      data.max_response_length=${MAX_RESPONSE_LEN:-1024} \
      +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true \
      '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
      algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-null} \
      2>&1 | tee "${RUN_LOG}"
elif [ "$MODE" = "single" ]; then
  STEPS=${2:-50}
  echo "[single] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | steps=$STEPS lr=$LR kl=$USE_KL bsz=${TRAIN_BSZ}x${SAMPLE_N}"
  for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
  sleep 5; ray stop --force >/dev/null 2>&1 || true
  cd "$MIRROR_ROOT/verl"
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
      trainer.total_training_steps="${STEPS}" \
      data.max_response_length=${MAX_RESPONSE_LEN:-1024} \
      +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true \
      '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
      algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-null} \
      2>&1 | tee "${RUN_LOG}"
elif [ "$MODE" = "join" ]; then
  # [2026-10-17 E42] 安全加入模式：有外部租户的机器（124/126 等）用它——只 ray start，绝不 pkill。
  # 仍需在本 shell 导出标准开关，raylet 才能把正确 env 传给本机 worker。
  MASTER_IP=${2:?usage: $0 join <MASTER_IP>}
  echo "[join] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | join $MASTER_IP:$RAY_PORT（无 pkill）"
  ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  echo "[join] joined:"; ray status --address="$MASTER_IP:$RAY_PORT" 2>&1 | tail -8
  echo "[join] ready — 保持容器运行"
else
  MASTER_IP=${2:?usage: $0 slave <MASTER_IP>}
  echo "[slave] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | join $MASTER_IP:$RAY_PORT"
  for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
  sleep 5; ray stop --force >/dev/null 2>&1 || true
  ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  echo "[slave] joined:"; ray status --address="$MASTER_IP:$RAY_PORT" 2>&1 | tail -8
  echo "[slave] ready — 保持容器运行"
fi
