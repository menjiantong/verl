#!/usr/bin/env bash
# GRPO on DeepSeek-V4.1-Flash-top64 (8 NPUs, vLLM rollout colocated with the FSDP-Turbo trainer).
#
#   bash run_grpo_top64.sh 1     # smoke: one training step (default)
#   bash run_grpo_top64.sh 15    # the ~15-step run
#
# What this wrapper adds on top of examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh:
#   * MASTER_ADDR/GLOO_SOCKET_IFNAME: this host does not resolve its own hostname (node-29-131 is
#     absent from /etc/hosts), and any process group built from the hostname dies before PyTorch
#     ever starts. Pinning both sidesteps name resolution entirely.
#   * HCCL_IF_IP / socket interface pinned to the NIC that carries 80.5.25.131 (enp48s3u1u1).
#   * An absolute log path under /workspace-verl/verl/logs so the run's log survives a container
#     restart (the 03:03 run's log was lost with the old container).
#   * total_training_steps is taken from $1, defaulting to a 1-step smoke.
set -eo pipefail

# See run_grpo_2node.sh: the 1024 soft fd limit starves the raylet on this image; raise it
# for the single-node flow too (the 2-node flow inherits it from run_grpo_2node.sh).
ulimit -n 65536 2>/dev/null || true

STEPS=${1:-1}

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_IF_IP=${HCCL_IF_IP:-80.5.25.131}

# Host memory on this node is the binding constraint for the full top64 model:
#   trainer (loaded, measured)          ~1.34 TB   (Engram tables 393 GB + CPU-offloaded experts 181 GB
#                                                   + AdamW state ~395 GB + load transients not yet returned)
#   rollout engine init (transient)     ~+1.0 TB   (its own Engram tables 393 GB + load/profile staging)
#   peak observed                       2.35 TB    -> over Ray's default kill line (0.95 * 2454 = 2.33 TB)
# so the run dies with "12 worker(s) were killed due to the node running low on memory" *before*
# step 1, even though the sustained level afterwards looks like ~1.8 TB. Raising the threshold to
# 0.99 (2.43 TB) lets the transient through while still leaving a physical safety band; without it
# Ray kills at 2.33 TB. Set RAY_MEMORY_THRESHOLD=0.95 to reproduce the default behaviour.
export RAY_memory_usage_threshold=${RAY_MEMORY_THRESHOLD:-0.99}
export RAY_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-1000}

# 2026-10-09 evening, REVERTED to the memory-safe attention defaults: the 12:06 two-node run
# OOM'd in step 1's actor update *with the chunk+recompute pair disabled* -- torch.gather in
# ops/cpu/sparse_attention.py:_indexed_sparse_attention_queries asked for 7.2-7.3 GiB with
# ~48-51 GiB already allocated on a 61 GiB card, on both nodes, several ranks at once (12:45).
# The pair is what bounds that gather (per-chunk int64 index + selected KV) and drops the
# ~5 GiB/layer backward retention. Restore only if the update shapes shrink enough that the
# un-chunked gather fits:
#   FSDP_TURBO_ATTN_RECOMPUTE=0 FSDP_TURBO_ATTN_CHUNK=100000000 bash scripts/run_grpo_top64.sh ...
# In the 2-node flow these must be exported by run_grpo_2node.sh's set_env() BEFORE `ray start`
# (the raylet env is what workers inherit); this copy covers the single-node flow.
# 2026-10-09 13:55: chunk 256 -> 64 (same reason as run_grpo_2node.sh: the 12:55 update OOM'd
# by 0.21 GiB at chunk 256; ~0.4 GiB/chunk at 64 clears it).
export FSDP_TURBO_ATTN_RECOMPUTE=${FSDP_TURBO_ATTN_RECOMPUTE:-1}
export FSDP_TURBO_ATTN_CHUNK=${FSDP_TURBO_ATTN_CHUNK:-64}

# 2026-10-09: every rank slice-reads its own checkpoint shards (verl engine
# fsdp_turbo_dsv41_impl.py) instead of the rank0-read + per-tensor Gloo broadcast, which
# measured ~30 MB/s across machines and stretched the load toward ~1.5 h. Set 0 to fall
# back; the code also falls back automatically (logged reason) when a DTensor layout
# cannot be planned. In the 2-node flow this must be in run_grpo_2node.sh's set_env()
# so the workers inherit it from the raylet; this copy covers the single-node flow.
export VERL_DSV41_DIRECT_SHARD_LOAD=${VERL_DSV41_DIRECT_SHARD_LOAD:-1}

cd /workspace-verl/verl
mkdir -p logs
RUN_LOG="logs/grpo_top64_steps${STEPS}_$(date +%Y%m%d_%H%M%S).log"

# Leftovers from a killed run keep their NPU *device-registered host memory* alive, and that
# memory is invisible to `ps` RSS: after the 06:00 run died, `free` still showed ~700 GB used
# with no process holding it. The next run then hit 2422/2454 GB and Ray's OOM killer killed 13
# workers. Sweep them before starting, and show the headroom so the peak is readable.
#
# NEVER sweep while attaching to a pre-started cluster (RAY_ADDRESS set -- the two-node flow):
# ray titles its agents and workers "ray::DashboardAgent" / "ray::RuntimeEnvAgent" / "ray::WorkerDict"
# with setproctitle (argv is rewritten, so `pkill -f "ray::"` matches them). Killing the live
# agents makes the raylet fate-share and exit -- "The raylet exited immediately because one Ray
# agent failed" -- after which the driver dies with "Failed to connect to Ray cluster". Verified
# live on 2026-10-09 06:20: this exact sequence killed the two-node run's head raylet one second
# after this wrapper started. The 2-node launcher sweeps *before* `ray start` instead
# (run_grpo_2node.sh); override with GRPO_FORCE_SWEEP=1 / GRPO_SKIP_SWEEP=1 if ever needed.
if [ -n "${RAY_ADDRESS:-}" ] && [ "${GRPO_FORCE_SWEEP:-0}" != "1" ]; then
    echo "[grpo] RAY_ADDRESS=$RAY_ADDRESS set (attaching to an existing cluster) -- skipping leftover sweep"
elif [ "${GRPO_SKIP_SWEEP:-0}" = "1" ]; then
    echo "[grpo] GRPO_SKIP_SWEEP=1 -- skipping leftover sweep"
else
    for pattern in "VLLM::" "ray::" "vllm serve"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    sleep 5
fi
echo "[grpo] steps=${STEPS} log=${RUN_LOG}"
echo "[grpo] memory before start:"; free -g | head -2

# 2026-10-09 21:50: 512 -> 2048. The 10th run's step:1 showed response_length
# mean=min=max=512.0 with clip_ratio=1.0: the checkpoint is a thinking model whose
# generation prompt opens a <think> block (rl_tokenizer chat template:
# "...<｜Assistant｜><think>"), so 512 tokens are consumed inside the think block and the
# required "Answer:" last line never appears -> math_dapo -1 everywhere -> GRPO
# advantages 0 -> grad_norm 0 (no learning signal). Memory is what forced 512 earlier;
# it is bought back with TRAIN_BSZ 8 -> 2 instead (peak ~= 5GB + 21.4MB/rank-token was
# measured across both shapes: 2613 tok -> 60.8GB on 2026-10-09; 2048+141 tok x 1 seq/rank
# -> ~52GB). Keep train_batch_size x rollout.n a multiple of 16 (2x8=16).
# 2026-10-10 00:10: ffn-scoped recompute. The update plateau is ~45GB of per-layer
# saved-for-backward (probe: +1.1GB/layer, never released during backward) and is
# nearly independent of the response length, so memory must come off that plateau.
# Scoping to `.ffn` keeps the attention's own custom-Function checkpoint OUT of the
# recompute region -- the whole-layer recompute crashed with "A different number of
# tensors was saved during the original forward and recomputation" (the attention's
# chunk path switches on torch.is_grad_enabled(), which differs under replay).
# Single-node smoke at seq=1165: torch peak 54.93 -> 50.21 GiB, grad_norm bit-identical.
bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${STEPS}" \
  data.max_response_length=${MAX_RESPONSE_LEN:-1024} \
  +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=true \
  '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]' \
  algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-null} \
  2>&1 | tee "${RUN_LOG}"