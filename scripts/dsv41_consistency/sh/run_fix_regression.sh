#!/bin/bash
# Post-fix regression for the fp32 router-correction-bias fix (`Gate.bias` -> fp32 buffer).
#
# Runs both halves of the consistency harness on the real 4-layer slice with the *fixed*
# trainer, i.e. without `VERL_DSV41_KEEP_FP32_PARAMS`:
#
#   1. trainer smoke   -> dump/trainer_real4_fix/   (does the load still work? is the bias fp32?)
#   2. input probe     -> dump/probe_input_real4_fix/  (the σ / forced-input floor table)
#
# then prints the CPU-only provenance + identification check
# (`verify_gate_bias_fp32.py`) and the summarizer command to compare against the pre-fix
# numbers in wiki/worklog_dsv41_real_weights_4layer.md §6.3.
#
# 8 NPUs; ~10 min each. Ports and chip range are the ones the earlier runs used.
set -xeuo pipefail
export VLLM_VERSION=0.27.1
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
cd /workspace-verl/verl

MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real}
DUMP_ROOT=${DUMP_ROOT:-/mnt/share/m00899630/dsv41/dump}
ENGINE_DIR=${ENGINE_DIR:-$DUMP_ROOT/engine_real4}
LOG_DIR=${LOG_DIR:-/tmp}
STEPS=${STEPS:-smoke,probe}

if [[ "$STEPS" == *smoke* ]]; then
  echo "=== trainer smoke (len 64,200) $(date +%H:%M:%S) ==="
  ASCEND_RT_VISIBLE_DEVICES=${SMOKE_DEVICES:-0,1,2,3,4,5,6,7} \
  torchrun --nproc_per_node=8 --master_port=${SMOKE_PORT:-59531} \
    scripts/dsv41_consistency/dump_trainer_stages.py \
    --model-path "$MODEL_PATH" \
    --out-dir "$DUMP_ROOT/trainer_real4_fix" \
    --out-tag _fix --lengths 64,200 \
    > "$LOG_DIR/dump_trainer_real4_fix.log" 2>&1
  echo "=== trainer smoke done (exit $?) $(date +%H:%M:%S) ==="
fi

if [[ "$STEPS" == *probe* ]]; then
  echo "=== input probe (4 variants x len 64,200) $(date +%H:%M:%S) ==="
  MODEL_PATH="$MODEL_PATH" ENGINE_DIR="$ENGINE_DIR" OUT_DIR="$DUMP_ROOT/probe_input_real4_fix" \
    MASTER_PORT=${PROBE_PORT:-59541} ASCEND_RT_VISIBLE_DEVICES=${PROBE_DEVICES:-2,3,4,5,6,7,8,9} \
    bash scripts/dsv41_consistency/sh/run_probe_inputs.sh \
    > "$LOG_DIR/probe_input_real4_fix.log" 2>&1
  echo "=== input probe done (exit $?) $(date +%H:%M:%S) ==="
fi

echo "=== summary + CPU-only verification ==="
python3 scripts/dsv41_consistency/summarize_probe.py --lengths 64,200 \
  --probe-dir "$DUMP_ROOT/probe_input_real4_fix" --engine-dir "$ENGINE_DIR" \
  2>&1 | tee "$LOG_DIR/summarize_probe_real4_fix.txt"
python3 scripts/dsv41_consistency/verify_gate_bias_fp32.py \
  --model-path "$MODEL_PATH" \
  --trainer-dir "$DUMP_ROOT/trainer_real4_fix" \
  --probe-dir "$DUMP_ROOT/probe_input_real4_fix" \
  --lengths 64,200 2>&1 | tee "$LOG_DIR/verify_gate_bias_fp32.txt"
echo "=== all done $(date +%H:%M:%S) ==="
