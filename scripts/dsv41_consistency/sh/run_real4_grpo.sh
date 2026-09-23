#!/bin/bash
# GRPO end-to-end on the **real** 4-layer slice (384 experts, /mnt/share/m00899630/weights/
# DeepSeek-V4.1-Flash-4layer-real) with the fp32 router-correction-bias fix in place (2026-09-22).
#
# Why this run (2026-09-23): the fix's own regression was the forward-only harness (§5.2/§5.3) plus a
# 1-step smoke on the *random* scaled32 fixture (§5.5) -- neither is a real-weight production path.
# This run watches the production metrics on real weights for 10 steps, focused on
# `training/rollout_actor_probs_pearson_corr` (and the two companions `rollout_probs_diff_mean` /
# `rollout_corr/kl`):
#   * expected (healthy): diff_mean ~1e-2, pearson ~0.9x, kl ~1e-1 -- the band this fixture family
#     shows at 32 experts is 0.967-0.979 pearson / 0.010-0.012 diff_mean / 0.10-0.12 kl
#     (worklog_dsv41_rl.md G13);
#   * a rank/placement or dtype mismatch shows up as an order-of-magnitude cliff (diff_mean O(1),
#     pearson ~0.5) -- see the same note in run_fix_grpo_smoke.sh.
# Per-step tensors behind those metrics are dumped (VERL_DSV41_DUMP_BATCH, metrics.py:68 hook) so the
# metric can be decomposed per token offline.
#
# Cost note: the default weight sync exports the full model on every rank (verl/wiki/worklog_dsv41_rl.md
# G13 measured 86.7s for the 14 GB scaled32 model, dominated by CPU-offloaded all-gathers), so ~108 GB
# of this checkpoint will make each step's `update_weights` the dominant term -- measured live on
# 2026-09-23 (logs/DeepSeek-V4.1-Flash-4layer-real-20260923_104558.log): update_weights 1032.7s of a
# 1230.1s step, sender 4702 tensors / 104.96 GiB, export ~900-1000s, flush only ~13s.
# That first default-path step came back healthy (diff_mean 0.0047 / pearson 0.995), and the fast path
# is now validated on THIS checkpoint too (`..._093816.log`: 670 tensors / 16.37 GiB, export 0.9s,
# update_weights 18.5s, pearson 0.988 at half batch) -- so run real4 iterations with
# `LOCAL_EXPERT_EXPORT=1` (5.7x step speedup; a wrong rank pairing cannot hide: the metrics would
# cliff to O(1)/~0.5). See wiki/worklog_dsv41_real4_prod_consistency.md §2.
set -xeuo pipefail
cd /workspace-verl/verl

# This box's hostname still resolves to a stale IP (/etc/hosts), and gloo picks its interface by
# hostname -- every hostname-based process group dies with "Unable to find interface for
# [141.61.29.117]" (E1'/E9 in the worklogs). `enp48s3u1u1` carries the machine's real address.
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}

# Debug-only: dump the rollout/actor logprob tensors per step for the per-token analysis.
export VERL_DSV41_DUMP_BATCH=${VERL_DSV41_DUMP_BATCH:-/tmp/dsv41_batch_real4}
mkdir -p "${VERL_DSV41_DUMP_BATCH}"
# Debug-only: print the weight-sync phase timings (export / flush) per step.
export VERL_SYNC_PROFILE=${VERL_SYNC_PROFILE:-1}

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real} \
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512} \
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-128} \
SAMPLE_N=${SAMPLE_N:-4} \
TRAIN_BSZ=${TRAIN_BSZ:-8} \
VAL_BSZ=${VAL_BSZ:-8} \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DSV41-real4-biasfix} \
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps=${STEPS:-10}
