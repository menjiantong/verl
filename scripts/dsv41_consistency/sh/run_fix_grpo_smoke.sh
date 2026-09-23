#!/bin/bash
# GRPO bring-up smoke for the fp32 router-correction-bias fix (plan §12.2.5, "risk" item).
#
# The harness (run_fix_regression.sh) covers build + load + forward. This covers the rest of the
# production path with the new buffer in place: FSDP wrap -> optimizer -> weight sync to
# vLLM-Ascend (the buffer now travels in `state_dict()`, and its name must still hit the
# engine's `.ffn.gate.bias` -> `.ffn.gate.e_score_correction_bias` mapping) -> rollout ->
# train step. One step is enough; the driver caps it with `trainer.total_training_steps=1`.
#
# Uses the 4-layer 32-expert scaled fixture (14 GB) so the whole thing fits comfortably in the
# 8 shared cards. Watch `training/rollout_probs_diff_mean` (~1e-2) and
# `training/rollout_actor_probs_pearson_corr` (~0.98) on the first step: an O(1) / ~0.5 reading
# means the sync landed on the wrong tensors (see wiki/worklog_dsv41_rl.md G13).
set -xeuo pipefail
cd /workspace-verl/verl

# This box's hostname still resolves to a stale IP (`/etc/hosts`), and gloo picks its interface
# by hostname -- every hostname-based process group dies with "Unable to find interface for
# [141.61.29.117]". `enp48s3u1u1` carries the machine's real address (80.5.25.117). Same
# workaround as the staged dumps (see worklog_dsv41_real_weights_4layer.md E1').
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32} \
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-256} \
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-64} \
SAMPLE_N=${SAMPLE_N:-2} \
TRAIN_BSZ=${TRAIN_BSZ:-4} \
VAL_BSZ=${VAL_BSZ:-4} \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-DSV41-gate-bias-fp32-smoke} \
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps=1
