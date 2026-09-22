set -xeuo pipefail
export VLLM_VERSION=0.27.1
export HCCL_CONNECT_TIMEOUT=1500
# chips 2..9 (NPU 1..4) were free when the run started; chips 0/1 have other tenants' jobs
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7,8,9}
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl:${PYTHONPATH:-}
cd /workspace-verl/verl
torchrun --nproc_per_node=8 --master_port=${MASTER_PORT:-59561} scripts/dsv41_consistency/probe_module_inputs.py \
  --model-path ${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32} \
  --engine-dir ${ENGINE_DIR:-/mnt/share/m00899630/dsv41/dump/engine} \
  --out-dir ${OUT_DIR:-/mnt/share/m00899630/dsv41/dump/probe_input} \
  --lengths ${LENGTHS:-64,200} \
  --variants ${VARIANTS:-baseline,in.all.ffn,in.all.attn,in.all}
