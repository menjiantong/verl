#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 Engram HostUVA 探针启动器 v1
#
# 目的：在单卡上验证 vllm-ascend 的 HostUvaBuffer + Triton gather 可用，
#       不加载 494 GB 检查点，是 Engram 单推理/RL 的第一道闸。
# 用法：bash my_scripts/1103_run_dsv41_engram_uva_probe_v1.sh [NPU_ID]
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$REPO/verl"

NPU=${1:-7}
# 环境变量全部在脚本内设置（rule 7.1）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=$NPU
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset OMP_NUM_THREADS OMP_PROC_BIND VLLM_VERSION

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/1103_engram_uva_probe_${TS}.log"
echo "[engram-uva-probe] NPU=$NPU log=$LOG"
python3 my_scripts/1103_probe_ascend_engram_uva_v1.py 2>&1 | tee "$LOG"
