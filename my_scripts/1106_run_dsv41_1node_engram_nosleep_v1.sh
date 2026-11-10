#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-06 由 1105 版派生（80.5.27.120），首次实跑。
# [1106 派生原因] 1105 版在 engram 配置下卡在 vLLM camem.sleep（rtMallocHost→devmm_svm_ioctl，
#   py-spy 已取证），拿不到 e2e 的 update_actor 数字。本版关掉 free_cache_engine，
#   既不 sleep 也不释放 KV，把 vLLM 常驻在训练旁边跑 —— 用来回答
#   "云道 E1 的 update_actor 慢，是 vLLM 常驻造成的，还是 EP128 规模造成的"。
#   显存账：vLLM 0.6×64=38.4 GB + 训练峰值 19.4 GB = 57.8 GB < 64 GB，可共存。
# =============================================================================
# 1105 DSV4.1 单机 16 卡：Engram A/B（带表 / 不带表）+ fetch 分段探针
#
# 基线（rule 7.3：本脚本只写差异）：
#   my_scripts/pref_dsv41/1105_run_dsv41_profiling_engram_v1.sh（去掉 profiling）
#     → examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
#
# 目的：在**同一台机器、同一份代码、同一批量**下，只改一个变量——模型是否带
#   engram 表——测 update_actor 的差异，并在带表那一组里用分段探针把 update_actor
#   的内部时间拆开（all_to_all / D2H / CPU gather / H2D）。
#
# 两个 arm（权重是同一份 268 分片，只差 config.json 的 engram_layer_ids）：
#   engram    /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
#             engram_layer_ids=[1,14]，两层表 [384006168,256] / [384016682,256] bf16
#   noengram  /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
#             engram_layer_ids=[]，其余 268 个 safetensors 全部软链到上面那份
#
# 环境变量（rule 6.1：全部 MY_ENV_ 前缀，全部有默认值）：
#   MY_ENV_DSV41_AB_ARM=engram|noengram    默认 engram
#   MY_ENV_DSV41_AB_STEPS=3                训练步数（step1 含 AdamW 状态创建，看 step2）
#   MY_ENV_DSV41_AB_PROBE=1                1=打开 engram fetch 分段探针
#   MY_ENV_DSV41_AB_PROBE_SYNC=0           1=逐段 device 同步（更准但改变时序）
#   MY_ENV_DSV41_AB_PROBE_CALLS=64         每进程每类最多打印次数
#   MY_ENV_DSV41_AB_MEMLOG_S=15            主机内存采样间隔（秒），0=关
#
# 用法：
#   MY_ENV_DSV41_AB_ARM=engram bash my_scripts/1105_run_dsv41_1node_engram_ab_v1.sh
#   MY_ENV_DSV41_AB_ARM=noengram bash my_scripts/1105_run_dsv41_1node_engram_ab_v1.sh
#
# 机器要求：16×Ascend910，engram arm 需 MemAvailable ≥ 2000 GiB，NPU 空闲。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

ARM=${MY_ENV_DSV41_AB_ARM:-engram}
STEPS=${MY_ENV_DSV41_AB_STEPS:-3}
MEMLOG_S=${MY_ENV_DSV41_AB_MEMLOG_S:-15}

case "$ARM" in
  engram)
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
    # 节点内 DP2 共享一份 TP 分片表（8 rank × 45.8 GiB = 366 GB），否则两份表撑爆内存
    export VLLM_ENGRAM_DP_SHARED_MEMORY=1
    MIN_MEM_GB=2000
    ;;
  noengram)
    export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
    export VLLM_ENGRAM_DP_SHARED_MEMORY=0
    MIN_MEM_GB=1200
    ;;
  *)
    echo "[ab] ABORT: MY_ENV_DSV41_AB_ARM=$ARM, expect engram|noengram" >&2
    exit 2
    ;;
esac

# [1106 数据集] 云道用 dapo_math_17k（探针 uniq=33,583），本地 countbenchqa_lite
# 只有 uniq=1,424（4.2%），engram 的 all_to_all 载荷与 CPU gather 规模差 24 倍。
# 用 MY_ENV_ 前缀开关映射到 base 脚本已有的 TRAIN_FILE/TEST_FILE（rule 6.1）。
if [ "${MY_ENV_DSV41_AB_DATASET:-countbenchqa}" = "dapo512" ]; then
  export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet
  export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
  echo "[ab] MY_DEBUG_LOG dataset=dapo512 train=$TRAIN_FILE"
fi

# 探针：模块级开关，必须在 import FSDPTurbo 之前进入环境（Ray worker 继承 driver 环境）
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=${MY_ENV_DSV41_AB_PROBE:-1}
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=${MY_ENV_DSV41_AB_PROBE_SYNC:-0}
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=${MY_ENV_DSV41_AB_PROBE_CALLS:-64}

# [1105 sleep] sleep 备份内存必须是 **pinned**（默认），不要设成 0。
# 证据：已知跑通的 my_scripts/1102_run_dsv41_engram_rl_validate_v3.sh 里
#   `grep -c SLEEP_PIN_MEMORY` = 0（即用默认 pinned）；
# 而 2026-11-05 14:57 本次首跑抄了 pref_dsv41/1105 的 `=0`（pageable），
# 日志停在 `[sleep-debug] replica.sleep: drained; sleeping 1 server(s)` 后再无输出
# 80 分钟，16 个 VLLM::Worker 各 3~5 核空转 —— 与
# wiki/worklog_20261103_dsv41_engram_rl.md §2.2 记录的 pageable 路径卡在
# `camem.py:230 unmap_and_release` 完全一致。改回 pinned 后与 v3 闸位同一口径。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_DSV41_AB_SLEEP_PIN:-1}
# [1106] 关掉 cache engine 释放：vllm_async_server.py:898 的闸门 `node_rank != 0 or
#   not free_cache_engine` 会直接返回，既不做 sleep 也不做 release_kv_cache，
#   彻底绕开 camem.py 里的 rtMallocHost 卡死路径。
export FREE_CACHE_ENGINE=${MY_ENV_DSV41_AB_FREE_CACHE_ENGINE:-False}
# 部分节点原生 transformers 5.10.4 缺 qwen4_exp，叠加 5.21
export MY_ENV_DSV41_TF_OVERLAY=${MY_ENV_DSV41_TF_OVERLAY:-/mnt/share/m00899630/env/py521}

# 16 卡：FSDP16 / EP16；rollout TP8×DP2、GEN_EP16（与 EP 对齐触发 LOCAL_EXPERT_EXPORT）
export NDEVICES_PER_NODE=16
export NNODES=1
export FSDP_SIZE=16
export EP_SIZE=16
export GEN_TP=8
export GEN_DP=2
export GEN_EP=16
# [1109 CP] 保留外层传入的 CP_SIZE（默认 1，行为与 1106 基线一致）；
# CP=2 由 my_scripts/1109_run_dsv41_cp2_engram_share_v1.sh 显式传入。
export CP_SIZE=${CP_SIZE:-1}

# 主机内存预检（rule 3.2）
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
echo "[ab] MY_DEBUG_LOG ab_precheck arm=$ARM steps=$STEPS model=$MODEL_PATH dp_shared=$VLLM_ENGRAM_DP_SHARED_MEMORY MemAvailable=${AVAIL_GB}GiB need=${MIN_MEM_GB}GiB probe=$MY_ENV_DSV41_ENGRAM_FETCH_PROBE sync=$MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC free_cache_engine=$FREE_CACHE_ENGINE"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[ab] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" >&2
  exit 3
fi

# 环境（CANN / PYTHONPATH / 缓存目录 / 网卡 / Ray 端口）
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
  export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1106_nosleep_1node_${ARM}_${TS}.log"
MEMLOG="logs/1106_nosleep_1node_${ARM}_${TS}.mem.log"

# 主机内存采样（rule 12.3）：每 MEMLOG_S 秒一行，字段固定便于 awk
mem_sampler() {
  while :; do
    awk -v ts="$(date '+%H:%M:%S')" '
      /^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^MemAvailable:/{a=$2}
      /^Cached:/{c=$2} /^Shmem:/{s=$2}
      END{printf "%s MemTotal=%dGiB MemFree=%dGiB MemAvailable=%dGiB Cached=%dGiB Shmem=%dGiB\n",
                 ts, t/1048576, f/1048576, a/1048576, c/1048576, s/1048576}' /proc/meminfo
    sleep "$MEMLOG_S"
  done
}
if [ "$MEMLOG_S" -gt 0 ]; then
  mem_sampler >> "$MEMLOG" 2>&1 &
  MEM_PID=$!
  trap 'kill '"$MEM_PID"' 2>/dev/null || true' EXIT
fi

echo "[ab] MY_DEBUG_LOG ab_start arm=$ARM log=$LOG memlog=$MEMLOG time=$(date '+%Y-%m-%d %H:%M:%S') ip=$MY_IP nic=$NIC"

# [1105 fix] `ray stop --force` 在本容器里不一定杀得掉旧 raylet：实测 14:44 起的
# raylet 在 stop 之后仍存活，其环境里的 MY_ENV_VLLM_SLEEP_PIN_MEMORY=0 会被新 actor
# 继承，sleep 又走回 pageable 卡死路径。这里显式硬停，保证新 raylet 带上本次环境。
if [ "${MY_ENV_DSV41_AB_HARD_RAY_STOP:-1}" = "1" ]; then
  ray stop --force >/dev/null 2>&1 || true
  for _pat in "ray/core/src/ray/rayle[t]" "gcs_serve[r]" "plasma_store_serve[r]"; do
    for _p in $(pgrep -f "$_pat" 2>/dev/null || true); do
      kill -9 "$_p" 2>/dev/null || true
    done
  done
  sleep 3
fi
dsv41_ray_start

# sleep 卡死看门狗（rule 8.2 唯一标识符）：观察到 sleep 开始 600 s 仍未返回就抓栈并告警，
# 避免再出现"日志 80 分钟无输出、只能干等"的情况。只是观测，不改任何训练行为。
sleep_watchdog() {
  _started=""; _t0=0
  while :; do
    if [ -z "$_started" ] && grep -aq "replica.sleep: drained; sleeping" "$LOG" 2>/dev/null; then
      _started=1; _t0=$(date +%s)
      echo "[ab] MY_DEBUG_LOG sleep_watch_start time=$(date '+%H:%M:%S')" >> "$LOG"
    fi
    if [ -n "$_started" ]; then
      if grep -aq "engine sleep call returned" "$LOG" 2>/dev/null; then
        echo "[ab] MY_DEBUG_LOG sleep_watch_ok elapsed_s=$(( $(date +%s) - _t0 ))" >> "$LOG"
        _started=""
      elif [ $(( $(date +%s) - _t0 )) -ge 600 ]; then
        echo "[ab] MY_DEBUG_LOG sleep_watch_hang elapsed_s=$(( $(date +%s) - _t0 ))" >> "$LOG"
        _p=$(pgrep -f "VLLM::Worke[r]" 2>/dev/null | head -1 || true)
        if [ -n "${_p:-}" ]; then
          timeout 60 py-spy dump --native --pid "$_p" >> "$LOG" 2>&1 || true
        fi
        _started=""
      fi
    fi
    sleep 15
  done
}
sleep_watchdog &
WATCHDOG_PID=$!
# 覆盖前面的 mem_sampler trap，同时回收两个后台任务
trap 'kill "$WATCHDOG_PID" "${MEM_PID:-}" 2>/dev/null || true' EXIT

# [1106 对齐] 云道 E1/E2 用的是整层重算（日志首行 [recompute-full]，
# override 为 ++...memory.recompute_plan=["model.layers.{*}"]）。
# 本地默认的 .ffn/.attn 子模块重算不含 engram（engram 挂在 Block 上，
# layers.py:316），会让本地测不到 E1 里那条路径。这里对齐成整层。
EXTRA=()
if [ "${MY_ENV_DSV41_AB_RECOMPUTE_FULL:-1}" = "1" ]; then
  EXTRA+=('++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]')
  echo "[ab] MY_DEBUG_LOG recompute_full=1 (aligned with yundao E1)"
fi

bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG"

echo "[ab] MY_DEBUG_LOG ab_done arm=$ARM rc=$? time=$(date '+%Y-%m-%d %H:%M:%S') log=$LOG"
