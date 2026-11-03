#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 Engram 单推理验证启动器 v1
#
# 闸位：
#   dummy : 只验证 HostUvaBuffer 分配/注册 + 前向链路（不读 393 GB engram 表）
#   real  : 真实加载完整检查点（494 GB，NFS 读盘较慢）
#
# 用法：bash my_scripts/1103_run_dsv41_engram_infer_validate_v1.sh [dummy|real] [START_NPU] [DEBUG] [BLOCKING] [base|sft]
#
# 验证通过的 commit（2026-11-02）：vllm-ascend 69de6ca8c（dummy/real、base/sft 均通过）
#   dummy: logs/1103_engram_infer_dummy_20261102_150922.log（vectorcore=0）
#   real : logs/1103_engram_infer_real_20261102_151109.log（493.40 GiB / 2:51）
#   sft  : logs/1103_engram_infer_real_sft_20261102_151908.log（46 shards / 1:20）
#   base=top32-bf16 原始基座；sft=top32-sft116（SFT 权重，engram 表 shard 软链到基座）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$REPO/verl"

MODE=${1:-dummy}
START=${2:-0}
DEBUG=${3:-0}
BLOCKING=${4:-0}
MODEL_TAG=${5:-base}
if [ "$MODE" != "dummy" ] && [ "$MODE" != "real" ]; then
    echo "usage: $0 [dummy|real] [START_NPU]" >&2
    exit 2
fi

# 环境变量全部在脚本内设置（rule 7.1）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, "$START" $((START+7)))
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset OMP_NUM_THREADS OMP_PROC_BIND VLLM_VERSION

# 完整检查点（含 engram_layer_ids=[1,14] 的 bf16 表）
# 第 5 参数选模型：base=原始基座；sft=top32-sft116（SFT 权重 + engram 表）
case "$MODEL_TAG" in
  base) ENG_MODEL=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16 ;;
  sft)  ENG_MODEL=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116 ;;
  *)    echo "unknown MODEL_TAG '$MODEL_TAG' (expect base|sft)" >&2; exit 2 ;;
esac
export MY_ENV_DSV41_ENGRAM_MODEL=$ENG_MODEL
export MY_ENV_DSV41_ENGRAM_LOAD_FORMAT=$MODE
# 1 = 打开 vllm-ascend engram UVA 的调试日志（设备/ID 范围）
export MY_ENV_DSV41_ENGRAM_DEBUG=$DEBUG
# 1 = 打开 ASCEND_LAUNCH_BLOCKING，异步错误定位到真实 kernel（仅调试）
export ASCEND_LAUNCH_BLOCKING=$BLOCKING

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/1103_engram_infer_${MODE}_${MODEL_TAG}_${TS}.log"
echo "[engram-infer] mode=$MODE npus=$ASCEND_RT_VISIBLE_DEVICES log=$LOG"
python3 my_scripts/1103_dsv41_engram_infer_validate_v1.py 2>&1 | tee "$LOG"
