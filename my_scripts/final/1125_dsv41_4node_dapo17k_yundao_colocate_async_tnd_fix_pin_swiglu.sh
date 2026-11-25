#!/usr/bin/env bash
# =============================================================================
# [1123 4node-TND v1_2] DSV4.1 4机×16卡 DAPO-17k：
# 4机 colocate_async 基线 + TND + Swiglu优化 + fix ref pin memory
#
# 基线
#   1123_dsv41_4node_dapo17k_yundao_colocate_async_tnd.sh
# 相对基线 A 的全部差异：
#   1) verl/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh 
#      增加 "+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.pin_memory=false"
#   2) VLLM_ASCEND_CLIPPED_SWIGLU_FUSED=1，代码在vllm-ascend
#   3) 日志级别更改：
#      export VERL_LOGGING_LEVEL=INFO
#      export VLLM_LOGGING_LEVEL=INFO
#      export PYTHONUNBUFFERED=1
#      actor_rollout_ref.rollout.disable_log_stats=False
# =============================================================================
set -eo pipefail

here=$(cd "$(dirname "$0")" && pwd)

# ---- 云道 rank 信息：必须在 source env 之前（env_dsv41_yundao.sh 里 NNODES 写死为 1）----
export NNODES=${MA_NUM_HOSTS:-1}
export NODE_RANK=${VC_TASK_INDEX:-0}
# 作业容器只有 172.16.x/eth0（证据：1110 8 机 driver 日志 node0-7 Net 行），devserver 的
# 首个地址才是 7.150.x。env_dsv41_yundao_930.sh 的 MY_IP 探测只认 7.x，在作业容器里会得到
# 空值。这里先按首地址定 MY_IP，两种环境都对：作业容器 172.16.x、裸机 devserver 7.150.x。
export MY_IP=${MY_IP:-$(hostname -I | awk '{print $1}')}
# VC_WORKER_HOSTS 首项是平台给的 worker-0 主机名（形如 ma-job-...-worker-0...），不是 IP，
# 直接原样用作 Ray head 地址（1110 8 机就是这样起的）。
HEAD_HOST=${VC_WORKER_HOSTS%%,*}
if [ "$NNODES" -gt 1 ] && [ -z "$HEAD_HOST" ]; then
  echo "MY_DEBUG_LOG [yundao] no_head nnodes=$NNODES (VC_WORKER_HOSTS unset)" >&2
  exit 2
fi

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_ASCEND_CLIPPED_SWIGLU_FUSED=${VLLM_ASCEND_CLIPPED_SWIGLU_FUSED:-1}



# ---- sleep L2（必须在 ray start 之前导出：worker 进程继承 raylet 环境） ----
export MY_ENV_DSV41_SLEEP_LEVEL=${MY_ENV_DSV41_SLEEP_LEVEL:-2}
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
# vLLM 日志级别：constants_ppo.py 的 get_ppo_ray_runtime_env() 会把“启动 shell 已设置”的
# 同名变量从 runtime_env 中剔除（constants_ppo.py:125-127），因此这里导出即生效（不被
# 硬编码的 WARN 覆盖）。默认 INFO：采集 vLLM 初始化 15 min 的分段（权重加载/编译/图捕获）。
# 长跑嫌吵可 export VLLM_LOGGING_LEVEL=WARN 覆盖。
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
# swapopt 旋钮；step 内换入批次占“空闲显存”的比例。
# v1_1 用 0.03 在首个 optimizer step 断言：运行时树 ping-pong 需同时准入 2 个 141.56M
# 元素的融合专家参数（283.1M 元素）⇒ step 时刻空闲 ≥1.1325 GB/f；0.5 ⇒ ≥2.27 GB。
# 已核对 f 只作准入阈值与分块预算（后者恒被 4M 封顶），调大不增加实际驻留（峰值 1.13 GB）。
export MY_ENV_DSV41_SWAP_MEM_FRACTION=${MY_ENV_DSV41_SWAP_MEM_FRACTION:-0.5}
export MY_ENV_DSV41_SWAP_PARAM_PREFETCH=${MY_ENV_DSV41_SWAP_PARAM_PREFETCH:-0}
# 主机矩用 pageable（见基线文件头）；=1 回退到 pinned
export MY_ENV_DSV41_SWAP_HOST_PINNED=${MY_ENV_DSV41_SWAP_HOST_PINNED:-0}

# ---- 步数必须在 source 之前定：env_dsv41_yundao.sh 会把它默认成 200 ----
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-${STEPS:-100}}

# ---- 云道公共环境（9.3 镜像；不起 Ray，Ray 由本脚本按 ModelArts rank 编排） ----
source "$here/env_dsv41_yundao_930.sh"

# env_dsv41_yundao.sh 把 NNODES 写死为 1、拓扑写死为单机 16 卡；恢复云道 rank。
export NNODES=${MA_NUM_HOSTS:-1}
export NODE_RANK=${VC_TASK_INDEX:-0}
if [ "$NNODES" -gt 1 ] && [ -z "$HEAD_HOST" ]; then
  echo "MY_DEBUG_LOG [yundao] no_head nnodes=$NNODES (VC_WORKER_HOSTS unset)" >&2
  exit 2
fi

# env_dsv41_yundao_930.sh 用 3.11.18+psutil 预探测 NIC；探测失败会静默回退到
# enp48s3u1u1（9.3 镜像已知问题，见 env_dsv41_yundao_930.sh 头注 4）。起 Ray 前
# 显式校验，避免 Gloo/HCCL 绑到不存在的网卡后才报错。
if [ -z "${NIC:-}" ] || [ ! -e "/sys/class/net/$NIC" ]; then
  echo "MY_DEBUG_LOG [yundao] bad_nic nic=${NIC:-<empty>} ip=$MY_IP" >&2
  exit 3
fi

cd /efs_rl/m00899630/workspace/dsv41-verl-refactor/verl

# ---- 云道 env 不含 PYTHONPATH，镜像里也没有安装 vllm/verl：显式加 4 个仓库根 ----
# 顺序与基线一致；TND 代码在 refactor/verl（96fa2208）与 refactor/FSDPTurbo（996893b）。
export PYTHONPATH=/efs_rl/m00899630/workspace/dsv41-verl-refactor/vllm:/efs_rl/m00899630/workspace/dsv41-verl-refactor/vllm-ascend:/efs_rl/m00899630/workspace/dsv41-verl-refactor/FSDPTurbo:/efs_rl/m00899630/workspace/dsv41-verl-refactor/verl:${PYTHONPATH:-}

# ---- 与 4 机基线完全一致的形状/图/KV/micro-batch ----
export MAX_PROMPT_LEN=2048 MAX_RESPONSE_LEN=6144
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=${MY_ENV_DSV41_KV_CACHE_BYTES:-1073741824}
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

# ---- 4 机拓扑（64 die）；CP=1 是 TND 的硬约束（见头注 3） ----
export NDEVICES_PER_NODE=16
export FSDP_SIZE=64 EP_SIZE=64 EFSDP_SIZE=1
export GEN_TP=8 GEN_DP=8 GEN_EP=64
export CP_SIZE=1

# ---- TND（基线 B 已验证开关） ----
export MY_ENV_DSV41_RL_SFA_LAYOUT=TND
export MY_ENV_DSV41_RL_TND_PAD_MULTIPLE=2
# 1=每个微批打印 packed 布局；默认 0（=基线 B 100 步交付值，避免长跑刷屏/同步 flush）。
# 冒烟时显式打开：MY_ENV_DSV41_STEPS=3 MY_ENV_DSV41_RL_TND_DEBUG=1 bash ...
export MY_ENV_DSV41_RL_TND_DEBUG=${MY_ENV_DSV41_RL_TND_DEBUG:-0}
export MY_ENV_TRAIN_SPARSE_FLASH_ATTN=1
# A2/A3(910) 用 3；A5 才用 0（FSDPTurbo/fsdp_turbo/ops/npu/sparse_attention.py:41-43）
export FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV=3
# torch_npu 的 get_device_capability() 默认返回 None；dynamo 的 TMA 探测会与 (9,0) 比较而抛
# TypeError。8.0 < 9.0 ⇒ 不启用 CUDA TMA 分支（Ascend 无 TMA），仅影响编译特性判定。
export TORCH_NPU_DEVICE_CAPABILITY=8.0

# ---- 全量模型 + DAPO-17k；KL 与 4 机基线一致（USE_KL=1） ----
export USE_KL=1
export MODEL_PATH=${MODEL_PATH_OVERRIDE:-/efs_rl/weights/DeepSeek-V4.1-Flash-bf16}
export TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
export TEST_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-DSV41-4node-dapo17k-yundao-tnd}

# ---- 缓存隔离（同基线：多机 64 进程并发编译不能共享同一个目录） ----
export TRITON_CACHE_DIR=/tmp/m00899630_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/m00899630_cache/torchinductor
export TORCH_EXTENSIONS_DIR=$MY_ENV_CACHE_ROOT/torch_extensions/$(hostname)
export VLLM_CACHE_ROOT=$MY_ENV_CACHE_ROOT/vllm/$(hostname)
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

# Ray object store 钉小（训练状态占主机内存；默认 200 GB 太大）
RAY_OBJ_STORE=${RAY_OBJECT_STORE_BYTES:-8000000000}

# ---- 透传参数守卫（"$@" 在入口脚本参数之后生效，以下覆盖会让 TND 必崩/必挂） ----
# 1) AdamWSwapSafe 还不能 load_state_dict，save 出来的 checkpoint 续训必崩 -> save_freq 固定 -1；
# 2) TND 要求 packed 输入：use_remove_padding=False 会让首个 forward 报 packed shape 错；
# 3) TND 每 rank 形状不同：engine 内层 compile=True 复现 autotune 长尾 -> EI0006（基线 B §33）；
# 4) TND 与 kv_allgather(CP) 互斥：>1 会在 fsdp_turbo_config.py:1024 报错。
for _a in "$@"; do
  case "$_a" in
    trainer.save_freq=*|++trainer.save_freq=*)
      case "$_a" in
        trainer.save_freq=-1|++trainer.save_freq=-1) ;;
        *) echo "ERROR: trainer.save_freq must stay -1 with AdamWSwapSafe (got $_a)" >&2; exit 1 ;;
      esac ;;
    *use_remove_padding=False|*use_remove_padding=false)
      echo "ERROR: TND requires actor_rollout_ref.model.use_remove_padding=True (got $_a)" >&2; exit 1 ;;
    *fsdp_config.use_torch_compile=True|*fsdp_config.use_torch_compile=true)
      echo "ERROR: TND requires fsdp_config.use_torch_compile=False (got $_a)" >&2; exit 1 ;;
    *kv_allgather_parallel_size=*)
      case "$_a" in
        *kv_allgather_parallel_size=1) ;;
        *) echo "ERROR: TND requires kv_allgather_parallel_size=1 (got $_a)" >&2; exit 1 ;;
      esac ;;
    # 5) CP=1 ⇒ DP=64，每 rank 只有 1 条样本：任何 per-gpu micro batch > 1 都会在
    #    engine/utils.py:125 断言（v1 的 ref mb=2 即此）。三分支都必须为 1。
    *log_prob_micro_batch_size_per_gpu=*|*ppo_micro_batch_size_per_gpu=*)
      case "$_a" in
        *micro_batch_size_per_gpu=1) ;;
        *) echo "ERROR: CP=1 gives 1 sample/rank; micro batch size must be 1 (got $_a)" >&2; exit 1 ;;
      esac ;;
  esac
done

# ---- 非 rank0：join 后保持容器存活（云道方式：只有 rank0 跑训练） ----
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" -ne 0 ]; then
  echo "MY_DEBUG_LOG [yundao] join rank=$NODE_RANK/$NNODES host=$(hostname) ip=$MY_IP -> $HEAD_HOST:$RAY_PORT"
  # 先清掉上一轮残留的 raylet：否则要么端口冲突返回非 0，要么复用旧 raylet，
  # worker 会继承旧环境（无 CANN overlay / pinned=1 等）。
  ray stop --force >/dev/null 2>&1 || true
  ok=0
  for _ in $(seq 1 60); do
    # 注意：--include-dashboard 只能配 --head；worker 分支传了会 PANIC
    # "`--include-dashboard` should only be specified when starting head node"（ray 2.60）。
    if ray start --address="${HEAD_HOST}:${RAY_PORT}" --node-ip-address="$MY_IP" \
         --disable-usage-stats --object-store-memory="$RAY_OBJ_STORE"; then
      ok=1
      break
    fi
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "MY_DEBUG_LOG [yundao] join_failed head=$HEAD_HOST:$RAY_PORT" >&2; exit 5; }
  # pipefail 下 status 抖动不能把 worker 容器带走（保持 sleep infinity 等 rank0）。
  ray status --address="${HEAD_HOST}:${RAY_PORT}" 2>&1 | tail -6 || true
  echo "MY_DEBUG_LOG [yundao] worker_ready rank=$NODE_RANK/$NNODES"
  sleep infinity
  exit 0
fi

# ---- rank0（含单机裸跑）：起 Ray head，等齐后再跑训练 ----
echo "MY_DEBUG_LOG [yundao] master rank=$NODE_RANK/$NNODES host=$(hostname) ip=$MY_IP nic=$NIC head=$HEAD_HOST nodes=$NNODES fsdp=$FSDP_SIZE ep=$EP_SIZE gen=${GEN_TP}x${GEN_DP} cp=$CP_SIZE kl=$USE_KL steps=$MY_ENV_DSV41_STEPS sleep=${MY_ENV_DSV41_SLEEP_LEVEL} sfa=${MY_ENV_DSV41_RL_SFA_LAYOUT}"
for f in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TEST_FILE"; do
  [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

# rank0 必须是平台列表首项：否则多机时会出现两个 head（平台按 VC_WORKER_HOSTS 顺序发 rank）。
# 主机名可能同时解析出 IPv6，解析结果不等于 MY_IP 不代表错，这里只告警不退出；
# head 直接用 HEAD_HOST 绑定（与 1110 8 机 driver 日志里的做法逐字一致）。
if [ "$NNODES" -gt 1 ]; then
  _head_addr=$(getent hosts "$HEAD_HOST" 2>/dev/null | awk 'NR==1{print $1}' || true)
  case "$_head_addr" in
    ""|"$MY_IP"|"$HEAD_HOST") ;;
    *) echo "MY_DEBUG_LOG [yundao] head_host_resolve head=$HEAD_HOST(${_head_addr}) my_ip=$MY_IP (continue)" ;;
  esac
fi

_head_bind=${HEAD_HOST:-$MY_IP}
ray stop --force >/dev/null 2>&1 || true
ray start --head --port="$RAY_PORT" --node-ip-address="$_head_bind" \
          --disable-usage-stats --include-dashboard=false --object-store-memory="$RAY_OBJ_STORE"
export RAY_ADDRESS="$_head_bind:$RAY_PORT"

if [ "$NNODES" -gt 1 ]; then
  want=$((NNODES * NDEVICES_PER_NODE))
  got=0
  for _ in $(seq 1 120); do
    got=$(ray status 2>/dev/null | awk '/NPU/{split($1,a,"/"); s+=a[2]} END{print int(s)}') || true
    [ "${got:-0}" -ge "$want" ] && break
    sleep 5
  done
  [ "${got:-0}" -ge "$want" ] || {
    echo "MY_DEBUG_LOG [yundao] cluster_not_ready npu=${got:-0}/$want" >&2
    exit 4
  }
  echo "MY_DEBUG_LOG [yundao] cluster_ready npu=${got}/${want}"
fi
ray status | tail -20

export VERL_LOGGING_LEVEL=INFO
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

# mb：CP=1 ⇒ DP=64；train_batch_size 8 × rollout.n 8 = 64 条/步 ⇒ 每 rank 1 条。
#   ref 的 infer 默认非动态（use_dynamic_bsz=False），engine/utils.py:125 要求
#   len(data) % mb == 0 ⇒ v1_1 把 ref mb 2→1（v1 失败根因，见头注）。
#   actor 更新：mini_batch_size=8×8=64，64 % dp(64)==0，per-gpu mini/micro 均 1；
#   rollout/actor log_prob：infer mb=1。三条路径都满足每 rank 1 条。
# 注意：CP=1 后单 rank 承载整条样本（≤8192 token），[tokens,vocab] fp32 logits 与
# 重算层输入约为 CP=4 的 4×；长 response 步的 HBM 峰值按冒烟判据监控。
# 启动提速：v1 的 filter 在 rank0 单进程跑了 11:54（1,791,700 行全通过），
#   v1_1 关掉 data.filter_overlong_prompts（下方参数）；本数据集无需过滤。
#   若改 TRAIN_FILE，先离线核对 prompt 长度，或重新打开 filter 并调大 workers。
#   val 同样不再过滤；本脚本 test_freq=-1 / val_before_train=False，若启用验证需重新核对。
# TND 相关覆盖（相对基线 A 新增/删除）：
#   - use_remove_padding=True：TND 要求 packed [1,total]（基线 A 入口脚本默认 False）；
#   - actor/ref fsdp_config.use_torch_compile=False：内层键才生效（§33）；
#   - 不再给 ref 设 kv_allgather_parallel_size=4：TND 要求为 1（默认即 1）。
exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  actor_rollout_ref.model.use_remove_padding=True \
  data.filter_overlong_prompts=False \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  ++actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.dsv41_swap_optim \
  actor_rollout_ref.actor.optim.optimizer=AdamWSwapSafe \
  trainer.save_freq=-1 \
  trainer.v1.trainer_mode=colocate_async \
  trainer.v1.colocate_async.num_warmup_batches=1 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  "$@"
