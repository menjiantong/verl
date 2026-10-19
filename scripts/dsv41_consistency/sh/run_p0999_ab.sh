#!/bin/bash
# Pearson-0.999 A/B driver (2026-09-27).
#
# One GRPO run on the real 4-layer slice per invocation (2 steps, ~15 min). All runs share the
# same data order (`data.shuffle=False`) and the weights never move (reward is all -1 -> advantage
# 0), so the only run-to-run variation is which responses were sampled. Measured spread of the
# metric across the three historical runs at this config is +-0.0005.
#
#   TAG=<name> [ALIGN=attn,router,moe | ALIGN=all] bash run_p0999_ab.sh
#
# ALIGN maps to the env gates added to FSDPTurbo:
#   attn   -> VERL_DSV41_ALIGN_ATTN=1    (attention scores/probabilities stay fp32)
#   router -> VERL_DSV41_ALIGN_ROUTER=1  (MoE router reads the pre-cast fp32 norm output)
#   moe    -> VERL_DSV41_ALIGN_MOE=1     (routing weights applied by the combine, engine order)
set -xeuo pipefail
cd /workspace-verl/verl

TAG=${TAG:-baseline}
ALIGN=${ALIGN:-}

for one in ${ALIGN//,/ }; do
    case "$one" in
        attn)   export VERL_DSV41_ALIGN_ATTN=1 ;;
        router) export VERL_DSV41_ALIGN_ROUTER=1 ;;
        moe)    export VERL_DSV41_ALIGN_MOE=1 ;;
        head)   export VERL_DSV41_ALIGN_HEAD=1 ;;
        attn,router,moe,head|all)
                export VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_ALIGN_ROUTER=1 \
                       VERL_DSV41_ALIGN_MOE=1 VERL_DSV41_ALIGN_HEAD=1 ;;
        attn,router,moe|am)
                export VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_ALIGN_ROUTER=1 \
                       VERL_DSV41_ALIGN_MOE=1 ;;
        "")     ;;
        *)      echo "unknown ALIGN knob: $one" >&2; exit 1 ;;
    esac
done

# Extra hydra overrides for this run, e.g.
#   EXTRA_OVERRIDES="+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enable_batch_invariant=false"
EXTRA=()
for one in ${EXTRA_OVERRIDES:-}; do
    [ -n "$one" ] && EXTRA+=("$one")
done

echo "[p0999] NOBI=${VERL_DSV41_ALIGN_NOBI:-0} TAG=${TAG} ALIGN=${ALIGN:-none} gates: ATTN=${VERL_DSV41_ALIGN_ATTN:-0} " \
     "ROUTER=${VERL_DSV41_ALIGN_ROUTER:-0} MOE=${VERL_DSV41_ALIGN_MOE:-0} HEAD=${VERL_DSV41_ALIGN_HEAD:-0}"

# Harness choices for this series (2026-09-27, other tenants hold 20-37 GB on every card):
#   * ROLLOUT_GPU_MEM_UTIL 0.4 instead of the script's 0.6 -> 24.5 GiB fits where 36.8 GiB does not.
#     The KV pool is not the binding constraint here (4 layers) and it changes no arithmetic.
#   * SAMPLE_N=2 instead of 4: the actor backward needs ~40.4 GB on the busiest rank and only
#     ~39.5 GB is free with the other tenant resident; halving the sequences removes >1 GB of
#     activations, which is the whole deficit. Same 8 prompts per step as the historical runs.
#   * STEPS=2: the weights never move (reward is all -1 -> advantage 0), so extra steps only
#     resample the same prompts.
# NOTE: keep every comment in this file OUT of a backslash-continued assignment chain -- a `#`
# there ends the logical line and silently drops the assignments before it (this bit us once).
TAG="${TAG}" \
EXPERIMENT_NAME="GRPO-DSV41-p0999-${TAG}" \
STEPS="${STEPS:-2}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}" \
SAMPLE_N="${SAMPLE_N:-2}" \
VERL_DSV41_DUMP_BATCH="${VERL_DSV41_DUMP_BATCH:-/tmp/p0999/batch_${TAG}}" \
LOCAL_EXPERT_EXPORT=1 \
bash scripts/dsv41_consistency/sh/run_real4_grpo.sh "${EXTRA[@]}" 2>&1 | tee "logs/p0999-${TAG}.log"