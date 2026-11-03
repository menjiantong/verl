#!/usr/bin/env bash
# 1103 Engram hash kernel 单卡复现启动器 v1
# 用法：bash my_scripts/1103_run_engram_hash_kernel_repro_v1.sh [full|nocache|known] [NPU]
set -eo pipefail
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$REPO/verl"
VARIANT=${1:-full}
NPU=${2:-0}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=$NPU
export ASCEND_LAUNCH_BLOCKING=1
export VLLM_USE_V2_MODEL_RUNNER=0
unset OMP_NUM_THREADS OMP_PROC_BIND VLLM_VERSION
export MY_ENV_DSV41_ENGRAM_MODEL=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/1103_engram_hash_repro_v3_${VARIANT}_${TS}.log"
echo "[engram-hash-repro] variant=$VARIANT npu=$NPU log=$LOG"
python3 my_scripts/1103_repro_engram_hash_kernel_v3.py "$VARIANT" "${STAGE:-8}" 2>&1 | tee "$LOG"
