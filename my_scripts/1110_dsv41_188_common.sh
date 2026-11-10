#!/usr/bin/env bash
# =============================================================================
# 1110 DSV4.1 单机 16 卡（80.5.27.188）CP=4 + Engram 公共启动库
#
# 基线（规则 7.3/7.4，本库只写与基线的差异）：
#   my_scripts/1109_exp_B3_engram_cp4_3step.sh（CP=4 口径）
#     -> 1107_run_dsv41_130_engram_share_ab_v2.sh（USE_KL=1 + SHARE=1）
#     -> 1106_run_dsv41_1node_engram_nosleep_v1.sh（ray / 内存采样 / 预检）
#     -> examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
#
# 特性开关（逗号分隔，默认空=基线；feature 脚本只设置 MY_ENV_DSV41_FEATURES）：
#   thpoff  MY_ENV_DSV41_DISABLE_THP=1     训练/vLLM 进程关 THP，规避低水位节点直接压缩
#   spread  MY_ENV_DSV41_ENGRAM_SPREAD=1   engram 大表按 rank 钉 NUMA 节点做 first-touch
#   pinproc MY_ENV_DSV41_PIN_PROCESS=1     整个 worker 进程钉 NUMA 节点（摊开全部主机分配）
#
# 已否决：sleep level 2（FREE_CACHE_ENGINE=True + SLEEP_LEVEL=2）。vllm-ascend 的
# level-2 会丢弃全部 CaMem 分配（offload_tags=()），而唤醒只恢复 weights/kv_cache 两个
# tag，engram/DSA 元数据保持未映射；首个 rollout 在 engram_gate 触发 507057
# (rtEventSynchronize suspect remote error)，引擎随即 EngineDeadError。结论见 worklog。
#
# 用法（由 1110_run_dsv41_188_*.sh 调用）：
#   MY_ENV_DSV41_FEATURES=spread MY_ENV_DSV41_EXP_TAG=spread \
#     bash 1110_dsv41_188_common.sh [额外 hydra 参数...]
#
# 机器要求：16×Ascend910 64 GB，MemAvailable ≥ 2000 GiB（engram 档），NPU 空闲。
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

FEATURES=${MY_ENV_DSV41_FEATURES:-}
STEPS=${MY_ENV_DSV41_STEPS:-3}
TAG=${MY_ENV_DSV41_EXP_TAG:-base}
MEMLOG_S=${MY_ENV_DSV41_MEMLOG_S:-15}
EXTRA_ARGS=("$@")

has_feature() { case ",${FEATURES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# ---------------- engram 档固定项（与 1109 B3 同口径） ----------------
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
# 节点内 DP2 共享一份 TP 分片表（8 rank × 45.8 GiB = 366 GiB），否则两份表撑爆内存。
export VLLM_ENGRAM_DP_SHARED_MEMORY=1
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet
# ref + actor 共用 frozen engram 表（P0）；KL=1 才有 ref。
export USE_KL=1
export MY_ENV_DSV41_ENGRAM_SHARE_MODEL=1
# CP=4：PAD 必须为 8（length%4==0 且 length/4%2==0）。
export CP_SIZE=4
export MY_ENV_DSV41_CP_PAD_MULTIPLE=8
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
# 16 卡：FSDP16 / EP16；rollout TP8×DP2、GEN_EP16（与 EP 对齐触发 LOCAL_EXPERT_EXPORT）。
export NDEVICES_PER_NODE=16
export NNODES=1
export FSDP_SIZE=16
export EP_SIZE=16
export GEN_TP=8
export GEN_DP=2
export GEN_EP=16
# sleep 备份默认 pinned：pageable 路径在 camem.unmap_and_release 卡死过（见 1106 注释）。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=1
export MY_ENV_DSV41_TF_OVERLAY=${MY_ENV_DSV41_TF_OVERLAY:-/mnt/share/m00899630/env/py521}
# 整层重算，与云道 E1 / 1109 B3 对齐。
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=64
export MY_ENV_DSV41_AB_RECOMPUTE_FULL=1

# ---------------- 特性开关 ----------------
export FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-False}
if has_feature thpoff; then
  export MY_ENV_DSV41_DISABLE_THP=1
fi
if has_feature spread; then
  export MY_ENV_DSV41_ENGRAM_SPREAD=1
fi
if has_feature pinproc; then
  # 进程级 first-touch：整个 worker 的主机分配都按 rank 钉节点。
  # 推理侧：6 个满容量节点（排除半容量 node3）。
  export MY_ENV_DSV41_PIN_PROCESS=1
  export MY_ENV_DSV41_NUMA_NODES=${MY_ENV_DSV41_NUMA_NODES:-0,2,4,5,6,7}
  # 训练侧：按容量加权到 7 个节点（node3 只放 2 个 rank，node6/7 各 3 个）。
  # 16 rank 的落点：N0×2 N2×2 N3×2 N4×2 N5×2 N6×3 N7×3。
  export MY_ENV_DSV41_NUMA_RANK_MAP=${MY_ENV_DSV41_NUMA_RANK_MAP:-0,2,3,4,5,6,7,0,2,3,4,5,6,7,6,7}
fi
# sleep2 已否决（见文件头）：不提供开关，避免误用后引擎 507057 崩溃。

# ---------------- 环境（CANN / PYTHONPATH / 缓存 / 网卡 / Ray 端口） ----------------
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh
if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
  export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/1110_188_${TAG}_${TS}.log"

# 主机内存采样（规则 12.3）：全局水位 + 各 NUMA 节点 free + THP 压缩计数。
mem_sampler() {
  while :; do
    awk -v ts="$(date '+%H:%M:%S')" '
      /^MemFree:/{f=$2} /^MemAvailable:/{a=$2} /^Cached:/{c=$2} /^Shmem:/{s=$2}
      /^compact_stall /{cs=$2} /^compact_fail /{cf=$2} /^numa_miss /{nm=$2}
      /^thp_fault_alloc /{ta=$2}
      END{printf "%s MemFree=%dGiB Avail=%dGiB Cached=%dGiB Shmem=%dGiB compact_stall=%.0f compact_fail=%.0f numa_miss=%.0f thp_fault=%.0f",
                 ts, f/1048576, a/1048576, c/1048576, s/1048576, cs, cf, nm, ta}' /proc/meminfo /proc/vmstat
    for node in 0 2 3 4 5 6 7; do
      # nodeN/meminfo 行格式 "Node N MemFree: <kB> kB"，数值在 $4。
      awk -v n="$node" '/MemFree:/{printf " N%s=%dGiB", n, $4/1048576}' \
        "/sys/devices/system/node/node${node}/meminfo" 2>/dev/null || true
    done
    echo
    sleep "$MEMLOG_S"
  done
}
if [ "$MEMLOG_S" -gt 0 ]; then
  mem_sampler >> "$LOG" 2>&1 &
  MEM_PID=$!
  trap 'kill "${MEM_PID:-}" 2>/dev/null || true' EXIT
fi

# ---------------- 预检 + Ray ----------------
# 清理被强杀的上一次运行遗留的 psm_* engram 共享内存表（本次运行的产物，非用户数据）。
# 只删“没有任何进程映射”的文件；正在使用的表保持不动。否则每次异常退出会在
# /dev/shm 遗留 366 GiB，两三次后预检（MemAvailable≥2000 GiB）就无法通过。
MAPPED_PSM=$(for p in /proc/[0-9]*/maps; do grep -ho "psm_[0-9a-f]*" "$p" 2>/dev/null; done | sort -u || true)
for psm in /dev/shm/psm_*; do
  [ -e "$psm" ] || continue
  name=$(basename "$psm")
  if ! printf '%s\n' "$MAPPED_PSM" | grep -qx "$name"; then rm -f "$psm"; fi
done
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
echo "[1110] MY_DEBUG_LOG start tag=$TAG features=${FEATURES:-none} steps=$STEPS cp=$CP_SIZE share=1 use_kl=1 free_cache_engine=$FREE_CACHE_ENGINE thp_off=${MY_ENV_DSV41_DISABLE_THP:-0} spread=${MY_ENV_DSV41_ENGRAM_SPREAD:-0} sleep_level=${MY_ENV_DSV41_SLEEP_LEVEL:-default} MemAvailable=${AVAIL_GB}GiB log=$LOG" | tee -a "$LOG"
if [ "$AVAIL_GB" -lt 2000 ]; then
  echo "[1110] ABORT: MemAvailable=${AVAIL_GB}GiB < 2000GiB" | tee -a "$LOG" >&2
  exit 3
fi

# 硬停旧 raylet，保证新 actor 继承本次环境（旧 raylet 曾把 pageable sleep 开关带进来）。
ray stop --force >/dev/null 2>&1 || true
for pat in "ray/core/src/ray/rayle[t]" "gcs_serve[r]" "plasma_store_serve[r]"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do kill -9 "$pid" 2>/dev/null || true; done
done
sleep 3
dsv41_ray_start

# ---------------- 启动 ----------------
EXTRA=()
if [ "${MY_ENV_DSV41_AB_RECOMPUTE_FULL:-1}" = "1" ]; then
  EXTRA+=('++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]')
fi

set +e
bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps="$STEPS" \
    "${EXTRA[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[1110] MY_DEBUG_LOG done tag=$TAG rc=$RC time=$(date '+%F %T') log=$LOG" | tee -a "$LOG"
exit "$RC"
