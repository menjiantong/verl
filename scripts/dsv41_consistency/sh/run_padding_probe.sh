#!/bin/bash
# Does left-padding the trainer's input change its logprobs? (wiki/work_log/pearson_0999/)
#
# The engine always sees `prompt + response` at positions 0..N-1 with no padding. verl's padded
# batch path left-pads the prompt to `prompt_length` with `padding_side="left"` (agent_loop.py),
# and FSDPTurboDSV41EngineWithLMHead.prepare_model_inputs *drops* position_ids, so the V4.1
# reference forward -- which indexes its RoPE tables by absolute position -- evaluates the same
# tokens at RoPE positions `pad + i`. This run measures that effect directly: identical tokens,
# unpadded vs left-padded, same model instance.
set -xeuo pipefail
export VLLM_VERSION=0.27.1
export HCCL_CONNECT_TIMEOUT=1500
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
cd /workspace-verl/verl
torchrun --nproc_per_node=8 --master_port=${MASTER_PORT:-59577} scripts/dsv41_consistency/dump_trainer_stages.py \
  --model-path ${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real} \
  --out-dir ${OUT_DIR:-/mnt/share/m00899630/dsv41/dump/padding_probe} \
  --lengths ${LENGTHS:-200,64} \
  --logprob-probe \
  --padding-probe ${PAD:-440}