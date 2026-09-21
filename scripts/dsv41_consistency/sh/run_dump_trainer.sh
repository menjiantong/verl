set -xeuo pipefail
export VLLM_VERSION=0.27.1
export HCCL_CONNECT_TIMEOUT=1500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
cd /workspace-verl/verl
torchrun --nproc_per_node=8 --master_port=59521 scripts/dsv41_consistency/dump_trainer_stages.py \
  --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32 \
  --out-dir /mnt/share/m00899630/dsv41/dump/trainer
