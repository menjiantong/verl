#!/usr/bin/env bash
# DeepSeek-V4.1 top32 GRPO on ONE machine with 16 NPUs -- the same 16-rank topology the
# two-node script builds (FSDP=16 / EP=8 / TP=8 x2 replicas), with all collectives intra-node.
#
#   bash scripts/run_grpo_1node16.sh [STEPS]        # default 50
#
# Why: the two-node run deadlocked in the model-load broadcast (rank0's CPU state dict,
# gloo data plane, never completed across machines -- see wiki/worklog_dsv41_top64_grpo.md,
# 10-08 16:58 hang). The training-side math of "16 chips" is identical here; only the
# cross-machine glue is removed. Use this to validate the FSDP16 topology + the update
# memory question on one machine, at intra-node speeds.
#
# Requirements: 16 FREE chips on this machine (check with npu-smi; a chip held by another
# container aborts the precheck below), /mnt/share mounted, >= ~1.6 TB free host RAM.
# Host totals are rank-count invariant (engram 366 GB + optimizer ~428 GB + params/grads
# ~214 GB + transients), so the memory budget matches the single-node runs' accounting.
#
# Known-unverified seam: vllm-ascend's multi-node serving logic keys "replica per node" on
# RAY_LOCAL_WORLD_SIZE; on one node with 16 ranks local_world_size=16, so the engine builds
# 2 x TP8 replicas on the same node through the same-machine path. Rank math is identical,
# but this exact configuration has not run before -- if the engine init errors about node/
# replica mapping, that is the place to look (and the finding belongs in the worklog).
set -eo pipefail

STEPS=${1:-50}
NIC=${NIC:-enp48s3u1u1}
MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-top32-1node16}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

detect_ip() {
    python3 - "$1" <<'PY'
import fcntl, socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", sys.argv[1].encode()[:15]))[20:24]))
PY
}
MY_IP=${MY_IP:-$(detect_ip "$NIC")}

export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NIC}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$NIC}
export HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=0
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

echo "[1node16] NIC $NIC -> $MY_IP | devices=$ASCEND_RT_VISIBLE_DEVICES"
echo "[1node16] NPU holders (any holder = that chip is not ours; 16 free chips are required):"
for i in 0 1 2 3 4 5 6 7; do
    out=$(npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep "Process id" || true)
    if [ -n "$out" ]; then echo "  NPU $i: $out"; fi
done
python3 - <<'PY'
import sys, torch_npu
bad = []
for d in range(16):
    try:
        free, total = torch_npu.npu.mem_get_info(d)
        if free / 2**30 < 55:
            bad.append((d, round(free / 2**30, 1)))
    except Exception:
        bad.append((d, "unstartable"))
if bad:
    print(f"[1node16] ERROR: chips not usable: {bad}. Free them (or use another machine) first.")
    sys.exit(1)
print("[1node16] all 16 chips have >=55 GiB free")
PY
free_gb=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.0f", a/1048576}' /proc/meminfo)
echo "[1node16] host available: ${free_gb} GB (want >= ~1600)"
if [ "${free_gb:-0}" -lt 1600 ]; then echo "[1node16] ERROR: host RAM too low (leftovers? sweep first)"; exit 1; fi
ls "$MODEL_PATH/config.json" >/dev/null || { echo "[1node16] ERROR: $MODEL_PATH not reachable (/mnt/share mounted?)"; exit 1; }
python3 -c "import torch_npu, vllm_ascend" 2>/dev/null || { echo "[1node16] ERROR: imports failed (PYTHONPATH/copy?)"; exit 1; }
echo "[1node16] precheck ok"

# Single-node flow: main_ppo starts its own local Ray with the exported env; clear any stale
# local cluster first so the NPU resources are re-detected with the env above.
ray stop --force >/dev/null 2>&1 || true

cd /workspace-verl
# NDEVICES_PER_NODE=16 + NNODES=1 -> n_gpus = 16, exactly like two nodes of 8.
# SAMPLE_N=8 keeps train_batch_size x n = 64 divisible by 16 (verl/utils/config.py:123).
# FSDP=16 shards the trainable params across all 16 chips (this is the memory lever the
# two-node run was built around); EP stays 8 == GEN_EP so LOCAL_EXPERT_EXPORT stays on.
echo "[1node16] launching GRPO: steps=$STEPS (log under verl/logs/grpo_top64_steps${STEPS}_*.log)"
NNODES=1 NDEVICES_PER_NODE=16 \
FSDP_SIZE=${FSDP_SIZE:-16} EP_SIZE=${EP_SIZE:-8} SAMPLE_N=${SAMPLE_N:-8} \
DSA_SCATTER_DROP_MARKERS=0 MODEL_PATH="$MODEL_PATH" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
bash scripts/run_grpo_top64.sh "$STEPS"