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
export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:$MIRROR_ROOT/FSDPTurbo:$MIRROR_ROOT/verl:${PYTHONPATH:-}"
export PATH=/usr/local/python3.11.15/bin:$PATH
MY_IP=80.5.25.131
NIC=${NIC:-enp48s3u1u1}
RUN_LOG="logs/grpo_top32_sft_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"


RAY_PORT=${RAY_PORT:-6390}                                  # 避外部 GCS（F-77）
if [ "$MODE" = "single" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
else
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-sft116-$(date +%m%d-%H%M)}
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer}   
export TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_train_rl.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_test_rl.parquet}
export TRAIN_BSZ=${TRAIN_BSZ:-8} SAMPLE_N=${SAMPLE_N:-8}
export LR=${LR:-5e-6} USE_KL=${USE_KL:-1} KL_COEF=${KL_COEF:-0.001}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024} ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
# 训推一致性开关（默认全关；两机需一致——slave 同样经本脚本导出，天然一致）
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
export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}
export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}

# export VERL_AGENT_DUMP=${VERL_AGENT_DUMP:-/tmp/agent_dump.txt}
# export VERL_DSV41_DUMP_BATCH=${VERL_DSV41_DUMP_BATCH:-}
# export VERL_DSV41_ALIGN_ROUTER=${VERL_DSV41_ALIGN_ROUTER:-0}
# export VERL_DSV41_ALIGN_MOE=${VERL_DSV41_ALIGN_MOE:-0}
# export VERL_DSV41_ALIGN_HEAD=${VERL_DSV41_ALIGN_HEAD:-0}

if [ "$MODE" = "single" ]; then
  # 单机 = 16 卡一条 node，拓扑与双机 2×8 完全相同（FSDP16/EP8），集合通信全走机内
  export NNODES=1 NDEVICES_PER_NODE=16 FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
else
  export NNODES=2 FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
fi


EXAMPLE=examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh

if [ "$MODE" = "master" ]; then
  STEPS=${2:-50}
  python3 -c "import torch_npu, vllm_ascend" 2>/dev/null || { echo "ERROR: torch_npu/vllm_ascend 导入失败"; exit 1; }
  [ $((TRAIN_BSZ * SAMPLE_N % 16)) -eq 0 ] || { echo "ERROR: TRAIN_BSZ*SAMPLE_N %16 != 0"; exit 1; }
  echo "[master] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | steps=$STEPS lr=$LR kl=$USE_KL bsz=${TRAIN_BSZ}x${SAMPLE_N}"
  for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
  sleep 5; ray stop --force >/dev/null 2>&1 || true
  ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  export RAY_ADDRESS="$MY_IP:$RAY_PORT"
  echo "[master] waiting for slave →  到从机执行:  bash $(basename "$0") slave $MY_IP"
  for i in $(seq 1 180); do
    active=$(ray status 2>/dev/null | sed -n '/^Active:/,/^Pending:/p' | grep -c "node_" || true)
    [ "${active:-0}" -ge 2 ] && { echo "[master] slave joined"; break; }
    [ $((i % 5)) -eq 0 ] && echo "[master] waiting... $((i * 20))s"
    [ "$i" = 180 ] && { echo "ERROR: slave 12 分钟未加入"; exit 1; }
    sleep 20
  done
  cd "$MIRROR_ROOT/verl"
  echo "[master] launching GRPO: steps=$STEPS"
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
      trainer.total_training_steps="${STEPS}" \
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
else
  MASTER_IP=${2:?usage: $0 slave <MASTER_IP>}
  echo "[slave] $MY_IP | chips=$ASCEND_RT_VISIBLE_DEVICES | join $MASTER_IP:$RAY_PORT"
  for p in "VLLM::" "ray::" "vllm serve"; do pkill -9 -f "$p" 2>/dev/null || true; done
  sleep 5; ray stop --force >/dev/null 2>&1 || true
  ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  echo "[slave] joined:"; ray status --address="$MASTER_IP:$RAY_PORT" 2>&1 | tail -8
  echo "[slave] ready — 保持容器运行"
fi
