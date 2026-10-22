#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash top32 bf16 noengram · CountBenchQA-lite VLM GRPO
# FINAL consistency long-run config (W2 / W2b), single node 16 cards.
#
#   bash my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh [STEPS]
#
# Final config:
#   engine : TP8 x DP1 x EP8  (16 chips / TP8 => 2 engine replicas, each EP8)
#   trainer: FSDP16 / EP8 / EFSDP1 / TP1
#   data   : countbenchqa_lite, chat mode (enable_thinking=false)
#   batch  : TRAIN_BSZ=8 x SAMPLE_N=8, prompt 1152, response 512, LR 5e-6, KL0
#   rollout: gpu_memory_utilization=0.6, logprobs_mode=processed_logprobs
#
# Diagnostic switches are intentionally OFF:
#   - VA_V41_PREFILL_LOGPROBS      (prefill rescore; verified but no measurable gain)
#   - VERL_DISABLE_ACTOR_AUTOCAST  (autocast off; 512 response OOMs at step4)
#   - VERL_DSV41_PATCH_PROBE_DIR   (file probes; temporary, reverted)
#
# Why EP8: engine EP8 matches trainer EP8, and 2xTP8/EP8 is the proven stable
# topology.  The old 1xTP8/DP2/EP16 config was both less consistent at short
# responses and hung in vLLM-Ascend sleep/weight-sync.
# See wiki/worklog_consistenty_analysis_running_smaller_20261020.md.
#
# Usage:
#   bash my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh              # 500 steps, detached
#   STEPS=10 bash my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh
#   DETACH=0 bash my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh     # foreground
#   MY_IP=80.5.21.104 NIC=enp48s3u1u1 bash ...
#   DRY_RUN=1 bash ...                                               # print config only
#
# Logs:
#   logs/cons_loc_<timestamp>.nohup.log              (outer nohup)
#   logs/grpo_countbenchqa_steps<N>_<timestamp>.log  (inner tee)
# =============================================================================
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="$ROOT/my_scripts/dsv41_grpo_countbenchqa_1node.sh"
[ -f "$BASE" ] || { echo "[cons-loc] ERROR: missing $BASE"; exit 1; }

STEPS="${STEPS:-${1:-500}}"
DETACH="${DETACH:-1}"
FORCE="${FORCE:-0}"

# --- node identity (cluster 2: 80.5.21.101-108, host network) ---
if [ -z "${MY_IP:-}" ]; then
  MY_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^80\.5\.21\.' || true)"
fi
if [ -z "${MY_IP:-}" ]; then
  echo "[cons-loc] ERROR: cannot detect an 80.5.21.x IP; set MY_IP=80.5.21.10x"
  exit 1
fi

# The container has no `ip` binary; inspect /sys/class/net instead.
nic_exists() { [ -d "/sys/class/net/$1" ]; }
if [ -z "${NIC:-}" ]; then
  for cand in enp48s3u1u1 enp162s0f0; do
    if nic_exists "$cand"; then NIC="$cand"; break; fi
  done
fi
NIC="${NIC:-enp48s3u1u1}"
if ! nic_exists "$NIC"; then
  echo "[cons-loc] WARN: NIC $NIC not found; picking first enp*/eth*"
  NIC="$(ls /sys/class/net | grep -m1 '^enp' || ls /sys/class/net | grep -m1 '^eth' || true)"
fi
[ -n "${NIC:-}" ] || { echo "[cons-loc] ERROR: no usable NIC"; exit 1; }

# --- refuse to stack on an existing training job ---
if pgrep -f 'main[_]ppo' >/dev/null 2>&1; then
  if [ "$FORCE" != "1" ]; then
    echo "[cons-loc] ERROR: main_ppo already running; stop it or set FORCE=1"
    pgrep -af 'main[_]ppo' | head -3
    exit 1
  fi
fi

# --- final production config ---
export MY_IP NIC
export GEN_TP=8 GEN_DP=1 GEN_EP=8
export FSDP_SIZE=16 EP_SIZE=8 EFSDP_SIZE=1 TP_SIZE=1
export NNODES=1 NDEVICES_PER_NODE=16
export TRAIN_BSZ=8 SAMPLE_N=8
export MAX_PROMPT_LEN=1152 MAX_RESPONSE_LEN=512
export LR=5e-6 USE_KL=0 KL_COEF=0.001
export ROLLOUT_GPU_MEM_UTIL=0.6
export ENABLE_THINKING=false
export AGENT_LOOP=deepseek_v41_vl_single_turn
export LOGPROBS_MODE=processed_logprobs
export PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True OFFLOAD_POLICY=True
export VERL_DSV41_VISION=1 VERL_DSV41_VISION_FREEZE=1
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-GRPO-DSV41-countbenchqa-cons-loc-$(date +%Y%m%d_%H%M)}"

# Diagnostic switches must be OFF for the production run.
unset VA_V41_PREFILL_LOGPROBS VERL_DISABLE_ACTOR_AUTOCAST VERL_DSV41_PATCH_PROBE_DIR
unset VA_V41_ATTN_EAGER VA_V41_OPROJ_FP32_AR VA_V41_OPROJ_FP32_PARTIAL
unset VERL_DSV41_HC_BF16 VERL_DSV41_MOE_BF16 VERL_DSV41_DUMP_BATCH

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/cons_loc_${TS}.nohup.log"
mkdir -p logs

echo "[cons-loc] host=$(hostname) ip=$MY_IP nic=$NIC steps=$STEPS"
echo "[cons-loc] engine=TP${GEN_TP}xDP${GEN_DP}xEP${GEN_EP} (2 replicas on 16 chips)"
echo "[cons-loc] trainer=FSDP${FSDP_SIZE}/EP${EP_SIZE}/EFSDP${EFSDP_SIZE}/TP${TP_SIZE} bsz=${TRAIN_BSZ}x${SAMPLE_N}"
echo "[cons-loc] prompt=${MAX_PROMPT_LEN} response=${MAX_RESPONSE_LEN} lr=$LR kl=$USE_KL thinking=$ENABLE_THINKING util=$ROLLOUT_GPU_MEM_UTIL"
echo "[cons-loc] experiment=$EXPERIMENT_NAME"
echo "[cons-loc] log=$LOG"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[cons-loc] DRY_RUN=1, not starting"
  exit 0
fi

if [ "$DETACH" = "1" ]; then
  nohup bash "$BASE" "$STEPS" > "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "[cons-loc] started pid=$pid; tail -f $LOG"
else
  exec bash "$BASE" "$STEPS"
fi
