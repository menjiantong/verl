#!/usr/bin/env bash
# DeepSeek-V4.1 top32 GRPO on TWO nodes (8 NPU each), colocated engine, Ray across nodes.
#
#   master (this node, runs the driver):   bash scripts/run_grpo_2node.sh master [STEPS]
#   slave  (second node, joins the Ray):   bash scripts/run_grpo_2node.sh slave <MASTER_IP> [SLAVE_IP]
#
# What it wires (all of it is load-bearing; see wiki/worklog_dsv41_top64_grpo.md):
#   * Each node pins its OWN HCCL_IF_IP to the IP of NIC enp48s3u1u1 (auto-detected; the
#     hostname does not resolve to a usable address on these nodes, and a single pinned
#     80.5.25.131 from the single-node wrapper is wrong for the slave). MASTER_ADDR for the
#     torch rendezvous is derived by verl itself from the first placement group's node IP;
#     the driver's own MASTER_ADDR env is irrelevant to the workers.
#   * Ray head on the master (port 6379, --node-ip-address pinned), slave joins with
#     --address. Both raylets are started with the full env (ASCEND_RT_VISIBLE_DEVICES,
#     RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1, PYTHONPATH pin) because Ray
#     detects the NPU resources and spawns workers from the raylet's environment.
#   * Topology defaults: NNODES=2, FSDP_SIZE=16, EP_SIZE=8, GEN_TP=GEN_EP=8.
#     - FSDP=16 shards the trainable params across both nodes: this is the knob that lets
#       the 40-layer update fit (per-rank param materialization, ~0.9 GiB/layer, is what
#       filled a single 61 GB card; the same per-rank math applies to the activation set).
#     - EP must stay 8 (== GEN_EP) or LOCAL_EXPERT_EXPORT disables itself and the export
#       falls into the all-gather path that dies on CPU-offloaded DTensors ("No backend
#       type associated with device type cpu"); 16 ranks / EP 8 = two node-local expert
#       groups, which is the layout the single-node runs validated.
#     - Engine: 16 ranks / TP 8 = two node-local DP replicas (vllm-ascend's node-local DP
#       groups; the rollout adapter maps trainer rank i to the co-located replica). Do NOT
#       set GEN_TP=16: the Engram validation requires TP in {1,2,4,8}.
#     FSDP_SIZE=8 falls back to "two identical 8-rank replicas" (validated topology, but
#     per-rank memory identical to single-node, i.e. the old wall stays) -- use it only to
#     bisect a launch problem.
#   * ASCEND_RT_VISIBLE_DEVICES is per node and now PICKED DYNAMICALLY: the first 8 free
#     chips, preferring identity 0-7. On 2026-10-08 node-29-119 had chip 2 held by another
#     container, so the validated non-contiguous list was 0,1,3,4,5,6,7,8 -- pick_devices()
#     reproduces that automatically. Values are the Ray NPU ids; worker.py maps them back
#     to logical ids (worklog E10). Override by exporting the env var yourself.
#   * DSA_SCATTER_DROP_MARKERS=0 (kernel-side marker fix, verified end-to-end by the
#     14:01 run's 5-minute rollout) and the FSDPTurbo attention chunk+recompute fixes
#     are in-tree already; no extra env needed.
#
# 2026-10-09 updates (why a re-run now looks different):
#   * The Engram host-table tiling check (_load_host_engram_tables in
#     verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py) is REPLICA-AWARE now. FSDP16/EP8
#     builds two node-local EP groups, i.e. two copies of every row range, which the old
#     check read as "rows are not tiled across ranks at 0" and aborted right after the
#     state-dict broadcast. BOTH NODES MUST CARRY THE FIX: /workspace-verl is per-node
#     local disk (last full copy to node-29-119 was 10-08 13:31), so re-sync the code to
#     the slave before launching. precheck() below greps for the fix on each node.
#   * The 2026-10-08 16:58 two-node attempt deadlocked in the model-load cross-machine
#     gloo broadcast and never completed -- still UNRESOLVED. On a single node the same
#     broadcast finishes in ~7 min (2026-10-09 1node16 run: "[fsdp_turbo_dsv41] parameters
#     materialized in 398.9s"). If materialize stalls (no `materialized in Xs` line, host
#     mem flat, AICore 0%), that is this hang: py-spy a ray::WorkerDict on BOTH nodes
#     (and keep worker-*.out from both) before killing anything.
#   * The 06:20 run died before the driver could attach: run_grpo_top64.sh's leftover sweep
#     (`pkill -9 -f "ray::"`) ran AFTER this launcher built the cluster, matched ray's own
#     live agents (setproctitle names them ray::DashboardAgent / ray::RuntimeEnvAgent), and
#     the head raylet fate-shared with them ("The raylet exited immediately because one Ray
#     agent failed") -> driver "Failed to connect to Ray cluster". Fixed: the wrapper now
#     skips the sweep when RAY_ADDRESS is set, and this launcher sweeps *before* `ray start`
#     on each node. The wait loop also refuses to launch unless the joined cluster
#     advertises 16 NPU -- a half-registered slave (0 NPU, 06:20 case) is not a cluster.
#   * Per-node host memory on this topology: trainer engram 366 GiB + params ~53 GB +
#     engine engram 366 GiB + transients; AdamW (~210 GB/node) only appears at step 1.
#     Expected peak <= ~1.7 TB/node -- well under the 0.99 kill line (2.43 TB).
#
# Before the first 2-node run, on BOTH nodes: NPU holders must be ours (npu-smi proc-mem),
# /mnt/share must be mounted, and this container's /workspace-verl must be the same copy.
set -eo pipefail

# File-descriptor headroom: the image's soft limit is 1024 and the HEAD node's raylet
# exhausts it within minutes (grpc keeps opening channels; "File descriptor limit reached.
# Retrying." floods raylet.err and then no actor can be spawned). 2026-10-12 on 119/121:
# the run sat right after "Connected to Ray cluster" with the driver blocked in
# ray.get(placement group) (main_ppo.py:93) while the head raylet's fd table showed
# exactly 1024/1024 (the slave raylet was at 58/1024 -- head-only problem). The hard
# limit is 1073741816, so raising the soft limit is enough; raylet/workers/the trainer
# all inherit it from here.
ulimit -n 65536 2>/dev/null || true

MODE=${1:?usage: $0 master [STEPS] | $0 slave <MASTER_IP> [SLAVE_IP]}

NIC=${NIC:-enp48s3u1u1}
RAY_PORT=${RAY_PORT:-6379}
MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DeepSeekV41-top32-2node}
ENGINE_IMPL=${ENGINE_IMPL:-/workspace-verl/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py}

detect_ip() {
    python3 - "$1" <<'PY'
import fcntl, socket, struct, sys
nic = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", nic.encode()[:15]))[20:24]))
PY
}

pick_devices() {
    # The first 8 free chips on this node, preferring identity 0-7. A chip is "free" when
    # npu-smi proc-mem lists no Process id for it. /dev/davinci<i> keeps 8-chip machines
    # from probing past chip 7. Values are the Ray NPU accelerator ids (the env entries);
    # worker.py maps value -> logical position, so a non-contiguous list is fine.
    local free=() i
    for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        [ -e "/dev/davinci$i" ] || continue
        if ! npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep -q "Process id"; then
            free+=("$i")
            [ "${#free[@]}" -eq 8 ] && break
        fi
    done
    if [ "${#free[@]}" -ne 8 ]; then
        echo "[2node] ERROR: only ${#free[@]} free chips on this node (need 8); holders:" >&2
        for i in 0 1 2 3 4 5 6 7; do
            out=$(npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep "Process id" || true)
            [ -n "$out" ] && echo "  NPU $i: $out" >&2
        done
        exit 1
    fi
    local IFS=,
    echo "${free[*]}"
}

# ------------------------------------------------------------------ shared environment
set_env() {
    local my_ip=$1
    export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NIC}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$NIC}
    export HCCL_IF_IP=$my_ip
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
    export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
    export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_ASYNC_ERROR_HANDLING=0
    export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
    export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export RAY_memory_usage_threshold=${RAY_MEMORY_THRESHOLD:-0.99}
    export RAY_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-1000}
    # 2026-10-13: back on the PRE-REBASE stack (FSDPTurbo 7714bef / vllm-ascend b2da8fb47 /
    # vllm ced6857af): vllm_version_is() reads this env FIRST and the old vllm-ascend gates
    # ("0.30.0") need the old-line answer. The REBASED stack (vllm-ascend 0568da9db+, vllm
    # 8cc9aa5a) must instead leave this UNSET (the 0.30.0 pin flipped patch_step3p5 onto a
    # symbol vllm main removed). Flip this line when swapping stacks.
    export VLLM_VERSION=${VLLM_VERSION:-0.30.0}
    # 2026-10-09 evening, REVERTED to the memory-safe attention defaults: the 12:06 two-node
    # run OOM'd in step 1's actor update with the chunk+recompute pair disabled --
    # torch.gather in ops/cpu/sparse_attention.py:_indexed_sparse_attention_queries asked for
    # 7.2-7.3 GiB with ~48-51 GiB already allocated on a 61 GiB card, on both nodes, several
    # ranks at once (12:45). The earlier "only the single-node 8-card path needs this" reading
    # of the E11 caution was wrong -- FSDP16 halves the per-rank update tokens here but the
    # un-chunked gather is still far over the card. Restore the old behaviour only with
    # shrunken update shapes:
    #   FSDP_TURBO_ATTN_RECOMPUTE=0 FSDP_TURBO_ATTN_CHUNK=100000000 \
    #       bash scripts/run_grpo_2node.sh master ...
    # 2026-10-09 13:55: chunk 256 -> 64. With 256 the 12:55 run's actor update still OOM'd,
    # but by a hair: rank 2 asked for 1.54 GiB with 1.33 GiB free (53.81/61.27 already
    # allocated; the co-located vLLM worker holds ~17.4 GiB of that). 64 cuts the failing
    # per-chunk allocation to ~0.4 GiB, clearing the 0.2 GiB gap with margin. Raises to
    # 128/256 are safe once the run is proven; both nodes must use the same value (raylet
    # env per node).
    export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
    export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}
    # 2026-10-09: every rank slice-reads its own checkpoint shards (verl engine
    # fsdp_turbo_dsv41_impl.py) instead of the rank0-read + per-tensor Gloo broadcast,
    # which measured ~30 MB/s across machines and stretched the load toward ~1.5 h.
    # Set 0 to fall back to the broadcast path; the code also falls back automatically
    # (with a logged reason) when a DTensor layout cannot be planned.
    export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}
export VERL_AGENT_DUMP=${VERL_AGENT_DUMP:-/tmp/agent_dump.txt}
export VERL_DSV41_DUMP_BATCH=${VERL_DSV41_DUMP_BATCH:-}
    # [2026-10-13] opt-in fp32 all-reduce for the v4.1 wo_b (consistency experiment)
    export VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-0}
export VERL_DSV41_ALIGN_ATTN=${VERL_DSV41_ALIGN_ATTN:-0}
export VERL_DSV41_ALIGN_ROUTER=${VERL_DSV41_ALIGN_ROUTER:-0}
export VERL_DSV41_ALIGN_MOE=${VERL_DSV41_ALIGN_MOE:-0}
export VERL_DSV41_ALIGN_HEAD=${VERL_DSV41_ALIGN_HEAD:-0}
    # 2026-10-09 evening: weight sync took timing_s/update_weights=876.4s in the 10th run
    # (the 920s-class full-export path of worklog_dsv41_rl.md G13: every rank all-gathers
    # every expert block, ~8x redundant). VERL_DSV41_LOCAL_EXPERT_EXPORT streams only this
    # rank's own expert shard instead -- G13 scaled32: export 80s->0.048s, update_weights
    # 86.7s->6.5s with the consistency triplet unchanged-or-better. Opt-in in G13 because
    # the full-size (384-expert) replay was still owed; the 10th run's step:1 gives that
    # replay baseline now (diff_mean 0.0550 / pearson 0.9598 / kl 0.291), so enable it and
    # compare step:1 of the next run against those three numbers (a pairing mistake is
    # loud: they jump to O(1) / ~0.5).
    export VERL_DSV41_LOCAL_EXPERT_EXPORT=${VERL_DSV41_LOCAL_EXPERT_EXPORT:-1}
    # Prints "SYNC-PROFILE sender: N tensors, X GiB, B buckets | export ...s, flush ...s"
    # at the end of each weight sync (bucketed_weight_transfer.py:244) -- the direct
    # witness of whether the export shortcut above took effect.
    export VERL_SYNC_PROFILE=${VERL_SYNC_PROFILE:-1}
}

precheck() {
    echo "[2node] NIC $NIC -> $1 | ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    echo "[2node] npu holders among the picked chips (foreign PIDs mean another container):"
    local d out
    for d in ${ASCEND_RT_VISIBLE_DEVICES//,/ }; do
        out=$(npu-smi info -t proc-mem -i "$d" 2>/dev/null | grep "Process id" || true)
        if [ -n "$out" ]; then echo "  NPU $d: $out"; fi
    done
    ls "$MODEL_PATH/config.json" >/dev/null || { echo "[2node] ERROR: $MODEL_PATH not reachable (is /mnt/share mounted?)"; exit 1; }
    python3 -c "import torch_npu, vllm_ascend" 2>/dev/null || { echo "[2node] ERROR: torch_npu/vllm_ascend import failed (check PYTHONPATH/copy)"; exit 1; }
    if ! grep -q "copies_per_range" "$ENGINE_IMPL" 2>/dev/null; then
        echo "[2node] WARNING: $ENGINE_IMPL lacks the 2026-10-09 replica-aware Engram tiling"
        echo "[2node]          fix. This node will abort with 'rows are not tiled across ranks"
        echo "[2node]          at 0' once the engram tables load. Sync the code to this node."
    fi
    echo "[2node] precheck ok"
}

# ------------------------------------------------------------------ master
if [ "$MODE" = "master" ]; then
    STEPS=${2:-1}
    MY_IP=${MASTER_IP:-$(detect_ip "$NIC")}
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(pick_devices)}
    set_env "$MY_IP"
    precheck "$MY_IP"

    echo "[2node] restarting Ray head on $MY_IP:$RAY_PORT"
    # Sweep leftovers BEFORE the cluster exists -- the only safe point for `pkill -f "ray::"`
    # (it matches ray's live agents; see the 2026-10-09 note in the header).
    for pattern in "VLLM::" "ray::" "vllm serve"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    sleep 5
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
    export RAY_ADDRESS="$MY_IP:$RAY_PORT"

    echo "[2node] waiting for the slave node (run on the slave:"
    echo "[2node]   bash scripts/run_grpo_2node.sh slave $MY_IP)"
    WAIT_ITERS=${WAIT_ITERS:-180}
    for i in $(seq 1 "$WAIT_ITERS"); do
        active=$(ray status 2>/dev/null | sed -n '/^Active:/,/^Pending:/p' | grep -c "node_" || true)
        if [ "${active:-0}" -ge 2 ]; then
            npu_total=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ NPU' | tail -1 | cut -d/ -f2 | cut -d. -f1)
            if [ "${npu_total:-0}" -lt 16 ]; then
                echo "[2node] ERROR: 2 nodes joined but the cluster shows only ${npu_total} NPU (expected 16)."
                echo "[2node]        The slave's raylet must be started with ASCEND_RT_VISIBLE_DEVICES exported"
                echo "[2node]        (this script's slave mode does that). Check npu-smi on the slave, then rerun."
                exit 1
            fi
            echo "[2node] slave present ($active active nodes, ${npu_total} NPU total)"
            break
        fi
        if [ $((i % 5)) -eq 0 ]; then echo "[2node]   nodes=$active/2 after $((i * 20))s"; fi
        if [ "$i" = "$WAIT_ITERS" ]; then echo "[2node] ERROR: slave did not join within $((WAIT_ITERS / 3)) min"; exit 1; fi
        sleep 20
    done

    cd /workspace-verl
    # SAMPLE_N=8: the trainer asserts train_batch_size x rollout.n % (n_gpus_per_node * nnodes) == 0
    # (verl/utils/config.py:123). Single node: 8x5=40 % 8 == 0. Two nodes: 40 % 16 != 0, so the
    # group size goes to 8 -> 8x8=64 % 16 == 0 (4 samples per rank). Keep the product a multiple
    # of 16 if either knob changes.
    echo "[2node] launching GRPO: steps=$STEPS log=verl/logs/grpo_top64_steps${STEPS}_<stamp>.log"
    NNODES=2 \
    FSDP_SIZE=${FSDP_SIZE:-16} EP_SIZE=${EP_SIZE:-8} EFSDP_SIZE=${EFSDP_SIZE:-2} SAMPLE_N=${SAMPLE_N:-8} \
    HCCL_IF_IP="$MY_IP" ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    DSA_SCATTER_DROP_MARKERS=0 MODEL_PATH="$MODEL_PATH" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    bash scripts/run_grpo_top64.sh "$STEPS"
    exit $?
fi

# ------------------------------------------------------------------ slave
if [ "$MODE" = "slave" ]; then
    MASTER_IP=${2:?usage: $0 slave <MASTER_IP> [SLAVE_IP]}
    SLAVE_IP=${3:-$(detect_ip "$NIC")}
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(pick_devices)}
    set_env "$SLAVE_IP"
    precheck "$SLAVE_IP"

    echo "[2node] joining Ray at $MASTER_IP:$RAY_PORT as $SLAVE_IP"
    # Same pre-cluster sweep as the master: safe here, fatal once the cluster is up.
    for pattern in "VLLM::" "ray::" "vllm serve"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    sleep 5
    ray stop --force >/dev/null 2>&1 || true
    ray start --address="$MASTER_IP:$RAY_PORT" --node-ip-address="$SLAVE_IP" --disable-usage-stats
    echo "[2node] joined. Cluster status:"
    ray status --address="$MASTER_IP:$RAY_PORT" 2>&1 | tail -12
    echo "[2node] slave ready. Keep the container running; stop Ray with 'ray stop --force'."
    exit 0
fi

echo "unknown mode: $MODE (expected master|slave)" >&2
