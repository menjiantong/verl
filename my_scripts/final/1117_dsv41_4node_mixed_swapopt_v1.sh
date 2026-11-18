#!/usr/bin/env bash
# =============================================================================
# [4node-mixed] 2×灵枢(80.5.27.17 head / 80.5.27.18) + 2×苏州(80.5.25.119/120)
#   全参 DeepSeek-V4.1-Flash-bf16（含 engram）× DAPO math 1k→2k × 64 卡
#   = sleep mode=2（训练期释放 vLLM 驻卡） + 参数/梯度驻卡、Adam 矩留主机（swapopt）
#
# 基线脚本（均未修改）：
#   - my_scripts/final/1116_dsv41_4node_1k2k_dapo_v1.sh   4 机拓扑/Ray 编排/DAPO/全量
#   - my_scripts/final/1117_dsv41_sleep2_single.sh        sleep L2 三个开关（单机 2/2 步验证）
#   - 云道 1117_dsv41_sleep2_swapopt_*                     五个优化器覆盖（单机 2/2 步验证）
# 代码依赖：verl @ dev_dsv41_rebase1030_cp_engram（+ verl/utils/dsv41_swap_optim.py）
#           vllm-ascend @ 16c3159dd（sleep L2 Engram UVA 修复）
#           FSDPTurbo @ 1852e93 + swap_optimizer.py torch.accelerator 兼容补丁
#
# 本脚本相对 1116 4 机脚本只加两类（拓扑/形状/图/KV/mb 完全一致）：
#   1) sleep L2：
#        actor_rollout_ref.rollout.free_cache_engine=True
#        MY_ENV_DSV41_SLEEP_LEVEL=2（默认；level-2 丢弃 weights+kv 池，无 CPU 备份）
#        MY_ENV_VLLM_SLEEP_PIN_MEMORY=0（仅约束 level-1 回退路径）
#      → 训练阶段每卡释放 ~14–16 GiB（单机实测），供布局 B 的常驻参数/梯度使用。
#   2) 布局 B（主机内存 12 B/param → 4 B/param）：
#        actor.fsdp_config.offload_policy=False / param_offload=False / optimizer_offload=False
#        actor.optim.optimizer_impl=verl.utils.dsv41_swap_optim / optimizer=AdamWSwapSafe
#      → params/grads 常驻 NPU（33 GiB/卡），Adam 一二阶矩（bf16 zeros_like）在主机，
#        step() 内按空闲显存分批 H2D→fused_adamw→D2H。
#      主机矩默认 pageable（MY_ENV_DSV41_SWAP_HOST_PINNED=0）：2026-11-17 smoke_v1 中
#        pinned 路径在本 fleet 不可靠——基座 halMemAlloc 对 512 MiB aclrtMallocHost 返回
#        207001（driver devmm OOM），当时节点 MemFree 仍有 ~90 GB，64 rank 仅 ~7 个把
#        32.1 GiB 矩分配完。pageable 只是把 H2D/D2H 变成同步拷贝，不触驱动 host-pin。
#      动机（公式）：全参 566.28B/64 rank = 8.85B/rank
#        默认布局 Adam 矩 fp32 8 B/param = 70.8 GB/rank ≈ 66 GiB/rank → 1.05 TiB/节点，
#        叠加 params/grads 主机副本 0.53 TiB + engram 92 GiB + 引擎 shm 370 GiB ≈ 2.0 TiB，
#        2454 GiB 机器必触顶（历史证据：优化器状态创建 direct reclaim，单张量 516 s）。
#        布局 B 矩 bf16 4 B/param = 35.4 GB/rank ≈ 33 GiB/rank → 528 GiB/节点，总 ~1.0 TiB。
#
# 用法：
#   master（80.5.27.17）：
#     STEPS=2 bash my_scripts/final/1117_dsv41_4node_mixed_swapopt_v1.sh
#   join（80.5.27.18 / 80.5.25.119 / 80.5.25.120）：
#     bash my_scripts/final/1117_dsv41_4node_mixed_swapopt_v1.sh join 80.5.27.17
# =============================================================================
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)

MODE=master
case "${1:-}" in
  master|join) MODE=$1; shift ;;
esac

# ---- sleep L2（必须在 ray start 之前导出：worker 进程继承 raylet 环境） ----
export MY_ENV_DSV41_SLEEP_LEVEL=${MY_ENV_DSV41_SLEEP_LEVEL:-2}
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
# swapopt 旋钮（默认值）；step 内换入批次占“空闲显存”的比例
export MY_ENV_DSV41_SWAP_MEM_FRACTION=${MY_ENV_DSV41_SWAP_MEM_FRACTION:-0.25}
export MY_ENV_DSV41_SWAP_PARAM_PREFETCH=${MY_ENV_DSV41_SWAP_PARAM_PREFETCH:-0}
# 主机矩用 pageable（见文件头）；=1 回退到 pinned（单机/小规模已验证的旧行为）
export MY_ENV_DSV41_SWAP_HOST_PINNED=${MY_ENV_DSV41_SWAP_HOST_PINNED:-0}

# ---- 步数必须在 source 之前定：env_dsv41.sh 会把它默认成 200，之后这里的 ${STEPS:-2} 就成了死代码 ----
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-${STEPS:-2}}

# ---- 多机：Ray 由本脚本编排；让 env_dsv41.sh 跳过单机 ray head（默认 0，行为不变） ----
export MY_ENV_DSV41_RAY_SKIP_START=1
source "$here/env_dsv41.sh"

# ---- 与 1116 4 机脚本完全一致的形状/图/KV/micro-batch ----
export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=2048
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"

# ---- 4 机拓扑（env_dsv41.sh 的单机默认在上面 source 时被覆盖） ----
export NNODES=4 NDEVICES_PER_NODE=16
export FSDP_SIZE=64 EP_SIZE=64 EFSDP_SIZE=1
export GEN_TP=8 GEN_DP=8 GEN_EP=64
export CP_SIZE=4

# ---- 全量模型 + DAPO；KL 关闭（否则 ref worker 再持一份 engram+优化器） ----
export USE_KL=0
export MODEL_PATH=${MODEL_PATH_OVERRIDE:-/mnt/share/weight/DeepSeek-V4.1-Flash-bf16}
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DSV41-4node-mixed-swapopt-dapo}

# ---- 缓存隔离（同 1116：4 机 64 进程并发编译不能共享 NFS 目录；/dev/shm 是 noexec） ----
export TRITON_CACHE_DIR=/tmp/m00899630_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/m00899630_cache/torchinductor
export TORCH_EXTENSIONS_DIR=$MY_ENV_CACHE_ROOT/torch_extensions/$(hostname)
export VLLM_CACHE_ROOT=$MY_ENV_CACHE_ROOT/vllm/$(hostname)
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

MASTER_IP=${MASTER_IP:-80.5.27.17}
# Ray object store 钉小（训练状态占主机内存；默认 200 GB 太大）
RAY_OBJ_STORE=${RAY_OBJECT_STORE_BYTES:-8000000000}

if [ "$MODE" = "join" ]; then
  [ $# -ge 1 ] && MASTER_IP=$1
  echo "MY_DEBUG_LOG [join] host=$(hostname) ip=$MY_IP -> $MASTER_IP:$RAY_PORT"
  # 先清掉上一轮残留的 raylet：否则要么端口冲突返回非 0，要么复用旧 raylet，
  # worker 会继承旧环境（无 CANN overlay → B075 NaN；或 pinned=1 → 207001）。
  ray stop --force >/dev/null 2>&1 || true
  ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$MY_IP" \
            --disable-usage-stats --object-store-memory="$RAY_OBJ_STORE"
  ray status --address="$MASTER_IP:$RAY_PORT" 2>&1 | tail -6
  echo "MY_DEBUG_LOG [join] ready host=$(hostname)"
  exit 0
fi

# ---- master ----
echo "MY_DEBUG_LOG [master] host=$(hostname) ip=$MY_IP nic=$NIC nodes=$NNODES fsdp=$FSDP_SIZE ep=$EP_SIZE gen=${GEN_TP}x${GEN_DP} kl=$USE_KL steps=$MY_ENV_DSV41_STEPS sleep=${MY_ENV_DSV41_SLEEP_LEVEL}"
for f in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TEST_FILE"; do
  [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

ray stop --force >/dev/null 2>&1 || true
ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" \
          --disable-usage-stats --include-dashboard=false --object-store-memory="$RAY_OBJ_STORE"
export RAY_ADDRESS="$MY_IP:$RAY_PORT"

echo "MY_DEBUG_LOG [master] waiting for $((NNODES - 1)) join node(s):"
echo "         bash $(basename "$0") join $MY_IP"
for i in $(seq 1 120); do
  active=$(ray status 2>/dev/null | sed -n '/^Active:/,/^Pending:/p' | grep -c "node_" || true)
  [ "${active:-0}" -ge "$NNODES" ] && { echo "MY_DEBUG_LOG [master] all $NNODES nodes joined"; break; }
  [ $((i % 3)) -eq 0 ] && echo "MY_DEBUG_LOG [master] waiting $((i * 10))s (active=${active:-0}/$NNODES)"
  [ "$i" = 120 ] && { echo "ERROR: 20min 内未全部 join"; exit 1; }
  sleep 10
done
ray status | tail -20

# AdamWSwapSafe 还不能 load_state_dict，save 出来的 checkpoint 续训必崩；固定关闭保存，
# 并拒绝任何把 save_freq 覆盖成其它值的透传参数（"$@" 在入口脚本参数之后生效）。
for _a in "$@"; do
  case "$_a" in
    trainer.save_freq=*|++trainer.save_freq=*)
      case "$_a" in
        trainer.save_freq=-1|++trainer.save_freq=-1) ;;
        *) echo "ERROR: trainer.save_freq must stay -1 with AdamWSwapSafe (got $_a)" >&2; exit 1 ;;
      esac ;;
  esac
done

# mb 与 1116 run7 一致：actor mb1 / logprob mb2（全量 64 卡下 [mb×seq, vocab] fp32 logits
# 缓冲是最大单次分配；mb2 时 2.12 GiB，run7 实测两步峰值 59.6 GiB/卡）
exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.dsv41_swap_optim \
  actor_rollout_ref.actor.optim.optimizer=AdamWSwapSafe \
  trainer.save_freq=-1 \
  "$@"
