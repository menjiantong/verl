#!/usr/bin/env bash
# E9 repro harness: run the *rollout engine alone*, with the exact argv + env the GRPO recipe
# hands to vLLM, and send one completion request. No trainer, no Ray, no DCP load -> reaches the
# first real prefill in ~10 min instead of ~40.
#
# Why: the 10-08 03:09 GRPO run (4th attempt) got further than any run ever had -- the weight
# sync completed, the engine came up, generation started, and then ALL EIGHT engine workers died
# inside the same device kernel (dumped by CANN as ScatterNdUpdateSk_38f6c0f4..., a vector-core
# "MTE accesses an invalid GM address"). See wiki/worklog_dsv41_top64_grpo.md §5.7.8 (E9).
#
# This script answers one question with one cheap run: is that crash reachable in the engine
# *itself* (dummy weights + the RL config + a real request), or does it need the trainer's synced
# weights? If it reproduces here, the loop is minutes; if it does not, the sync content is
# implicated and the next GRPO run goes in instrumented (see DIAGNOSTICS below).
#
# Usage:
#   bash scripts/run_vllm_dsv41_rl_dummy.sh                 # batch-invariant ON  (= production)
#   BATCH_INVARIANT=false bash scripts/run_vllm_dsv41_rl_dummy.sh   # A/B the rl_config flag
#   LOAD_FORMAT=auto bash scripts/run_vllm_dsv41_rl_dummy.sh        # real weights instead of dummy
#
# DIAGNOSTICS baked in (all override-able):
#   * VLLM_LOGGING_LEVEL=INFO  -- verl's engine args are logged at INFO, and the Engram
#     "table offloaded to registered host memory" line is INFO too. The GRPO run's log only
#     carried WARNING+, which is why we could not tell whether the engine offloaded its tables.
#   * ASCEND_LAUNCH_BLOCKING=1 + TASK_QUEUE_ENABLE=0 -- device launches become synchronous, so
#     the *Python* traceback names the op that faulted instead of the later op (RmsNorm) that
#     merely noticed the device was already dead.
#   * VLLM_DEBUG_DUMP_PATH is intentionally NOT set: the aicore exception dump already lands in
#     verl/extra-info/data-dump/<rank>/ when the runtime decides to dump.
set -eo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
PORT=${PORT:-20007}
LOG_DIR=${LOG_DIR:-/workspace-verl/verl/logs}
LOAD_FORMAT=${LOAD_FORMAT:-dummy}
BATCH_INVARIANT=${BATCH_INVARIANT:-true}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
GEN_TP=${GEN_TP:-8}
DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-900}
# E9 A/B knobs: each isolates one difference from the 09-30 serve run that worked.
#   WORKER_EXT   worker extension class (probe wrapper by default)
#   BLOCK_SIZE   vLLM cache block size; the 09-30 serve passed 128, the GRPO recipe passes none
#   RL_ENABLED / BATCH_INVARIANT  the two rl_config flags verl sends
#   DO_SLEEP_WAKE=1  sleep(level)/wake_up over HTTP before the first request, as verl does
WORKER_EXT=${WORKER_EXT:-dsa_scatter_probe.ProbeWorkerExtension}
BLOCK_SIZE=${BLOCK_SIZE:-}
RL_ENABLED=${RL_ENABLED:-true}
BATCH_INVARIANT=${BATCH_INVARIANT:-true}
DO_SLEEP_WAKE=${DO_SLEEP_WAKE:-0}
SLEEP_LEVEL=${SLEEP_LEVEL:-1}
# E9 serve-mirror knobs. These four are the ways the plain `vllm serve`
# (scripts/run_vllm_top32.sh) differs from the GRPO argv in a way that can plausibly
# change *where a bad write lands* rather than whether it is issued:
#   SLEEP_MODE=1      --enable-sleep-mode          (GRPO yes, plain serve no)
#   EXPANDABLE=1      PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#                     (plain serve yes, GRPO no -- run_deepseek_v41_grpo_fsdp_turbo_npu.sh:113
#                      says the sleep-mode memory pool rejects it)
#   PREFIX_CACHING=1  keep vLLM's default prefix caching (serve yes, GRPO no)
#   MEM_UTIL / MAX_NUM_BATCHED / MAX_MODEL_LEN  serve's 0.90 / 8192 / 8192 vs GRPO's 0.6 / 2048 / 2048
SLEEP_MODE=${SLEEP_MODE:-1}
EXPANDABLE=${EXPANDABLE:-0}
PREFIX_CACHING=${PREFIX_CACHING:-0}
MEM_UTIL=${MEM_UTIL:-0.6}
MAX_NUM_BATCHED=${MAX_NUM_BATCHED:-2048}
MAX_SEQS=${MAX_SEQS:-256}

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/vllm_rl_dummy_$(date +%Y%m%d_%H%M%S).log"

# ---- env: mirrored from the GRPO run's `+ export` list (logs/grpo_top64_steps1_20261008_030936.log)
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}
export VLLM_VERSION=0.30.0
export HCCL_BUFFSIZE=1024
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export ASCEND_RT_VISIBLE_DEVICES=${DEVICES}
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_IF_IP=${HCCL_IF_IP:-80.5.25.131}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
# ---- diagnostics
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
# E9 probe: bounds-check every npu_scatter_nd_update_sk write and raise with numbers instead
# of letting the device fault asynchronously. See scripts/dsa_scatter_probe.py.
export DSA_SCATTER_PROBE=${DSA_SCATTER_PROBE:-1}
export PYTHONPATH=/workspace-verl/scripts:${PYTHONPATH}
# Deliberately NOT set (the GRPO recipe does not set them either): ACL_OP_INIT_MODE,
# USE_MULTI_BLOCK_POOL, HCCL_OP_EXPANSION_MODE, LD_PRELOAD. PYTORCH_NPU_ALLOC_CONF is off
# unless EXPANDABLE=1, which mirrors the plain-serve script.
if [ "${EXPANDABLE}" = "1" ]; then
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
fi

ulimit -l unlimited || true
ulimit -c 0 || true

echo "[repro] model=${MODEL_PATH} load_format=${LOAD_FORMAT} batch_invariant=${BATCH_INVARIANT}"
echo "[repro] TP=${GEN_TP} devices=${DEVICES} port=${PORT} launch_blocking=${ASCEND_LAUNCH_BLOCKING} task_queue=${TASK_QUEUE_ENABLE}"
echo "[repro] log=${RUN_LOG}"

ADDITIONAL_CONFIG=$(cat <<JSON
{"mc2_comm_alg": "fullmesh_v2", "ascend_compilation_config": {"enable_npugraph_ex": false, "enable_static_kernel": false}, "rl_config": {"enabled": ${RL_ENABLED}, "enable_batch_invariant": ${BATCH_INVARIANT}}}
JSON
)

EXTRA_ARGS=()
if [ -n "${BLOCK_SIZE}" ]; then
    EXTRA_ARGS+=(--block-size "${BLOCK_SIZE}")
fi
if [ "${SLEEP_MODE}" = "1" ]; then
    EXTRA_ARGS+=(--enable-sleep-mode)
fi
if [ "${PREFIX_CACHING}" != "1" ]; then
    EXTRA_ARGS+=(--no-enable-prefix-caching)
fi
echo "[repro] worker_ext=${WORKER_EXT} block_size=${BLOCK_SIZE:-<default>} rl_enabled=${RL_ENABLED} sleep_wake=${DO_SLEEP_WAKE}"
echo "[repro] serve-mirror: sleep_mode=${SLEEP_MODE} expandable=${EXPANDABLE} prefix_caching=${PREFIX_CACHING} mem_util=${MEM_UTIL} max_num_batched=${MAX_NUM_BATCHED}"

vllm serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --load-format "${LOAD_FORMAT}" \
  --distributed-executor-backend mp \
  --worker-extension-cls "${WORKER_EXT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --enable-chunked-prefill \
  --max-num-batched-tokens "${MAX_NUM_BATCHED}" \
  --logprobs-mode processed_logprobs \
  --enforce-eager \
  --gpu-memory-utilization "${MEM_UTIL}" \
  --disable-log-stats \
  --tensor-parallel-size "${GEN_TP}" \
  --enable-expert-parallel \
  --seed 42 \
  --tokenizer-mode deepseek_v41 \
  --served-model-name auto \
  --override-generation-config '{"temperature": 1.0, "top_k": -1, "top_p": 1, "repetition_penalty": 1.0, "max_new_tokens": 1024}' \
  --hf-overrides '{}' \
  --scheduling-policy fcfs \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --additional-config "${ADDITIONAL_CONFIG}" \
  "${EXTRA_ARGS[@]}" \
  >"${RUN_LOG}" 2>&1 &
SERVER_PID=$!
echo "[repro] server pid=${SERVER_PID}"

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    pkill -9 -f "VLLM::" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the OpenAI server to answer on /health (it is served only once the engine is ready).
DEADLINE=$(( $(date +%s) + VLLM_ENGINE_READY_TIMEOUT_S ))
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[repro] SERVER PROCESS DIED during startup; tail of log:"
        tail -40 "${RUN_LOG}"
        exit 1
    fi
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[repro] server did not become healthy within ${VLLM_ENGINE_READY_TIMEOUT_S}s"
        tail -40 "${RUN_LOG}"
        exit 1
    fi
    sleep 5
done
echo "[repro] server healthy at $(date +%H:%M:%S); sending requests"

if [ "${DO_SLEEP_WAKE}" = "1" ]; then
    # Same shape as verl's init: sleep the replicas (discard KV, offload weights) then wake up.
    echo "[repro] POST /sleep?level=${SLEEP_LEVEL}"
    curl -sS -X POST "http://127.0.0.1:${PORT}/sleep?level=${SLEEP_LEVEL}" | head -c 200; echo
    sleep 10
    echo "[repro] POST /wake_up"
    curl -sS -X POST "http://127.0.0.1:${PORT}/wake_up" | head -c 200; echo
    sleep 10
fi

send() {
    local label="$1" prompt="$2" max_tokens="$3"
    echo "[repro] >>> ${label} (max_tokens=${max_tokens})"
    curl -sS --max-time "${REQUEST_TIMEOUT}" "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c '
import json, sys
print(json.dumps({"model": "auto", "prompt": sys.argv[2], "max_tokens": int(sys.argv[3]),
                  "temperature": 1.0, "n": 1}))' _ "${prompt}" "${max_tokens}")" \
        | head -c 600
    echo
    echo "[repro] <<< ${label} done at $(date +%H:%M:%S)"
}

# Three shapes: a trivial decode, a longer prefill (crossing a chunk boundary), and a batch.
send "short-3tok"  "The capital of France is" 8
send "long-prefill" "$(python3 -c 'print("math problem: what is 2 + 2? " * 60)')" 16
send "n4-batch"     "Solve for x: 3x + 7 = 16" 32

echo "[repro] ALL REQUESTS RETURNED WITHOUT A DEVICE FAULT -> the engine alone is fine;"
echo "[repro] the trainer-sync path is what differs from this run. Copy of log: ${RUN_LOG}"
grep -c "MTE accesses an invalid GM address\|vector core exception" "${RUN_LOG}" || true
