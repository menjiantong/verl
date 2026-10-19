set -xeuo pipefail
cd /workspace-verl/verl
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32
export VERL_SYNC_PROFILE=1
export VERL_DSV41_LOCAL_EXPERT_EXPORT=1
export DEVICE=npu
export MAX_RESPONSE_LEN=128
export SAMPLE_N=2
export TRAIN_BSZ=8
export VAL_BSZ=8
export VERL_DSV41_DUMP_BATCH=/mnt/share/m00899630/dsv41/dump/batch_bi
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh 2>&1 | tee /tmp/grpo_bi.log
