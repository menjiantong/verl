#!/bin/bash
# 1103 vLLM 生成自检启动器 v1
# 验证基线 commit：vllm 2f0dec710f / vllm-ascend 69de6ca8c / verl 979dfa29
# 目的：隔离 verl，验证 vLLM 栈在 noengram 基座上的 EOS / 文本 / logprob 是否正常
# 已知结果（2026-11-03）：
#   254 容器：sample 档 128×token0、logprob 全 nan；seq 档第 2 个请求坏（隔一个坏一次）
#             logs/1103_quickgen_v1_DeepSeek-V4.1-Flash-top32-bf16-noengram_20261102_200506.log
#   125 节点：seq 档三个请求全部正常
#             logs/1103_quickgen_v1_125_seq_20261102_201000.log
# 环境变量：MY_ENV_DSV41_CASE=all|greedy_nologp|sample_nologp|sample_logp|sample_t07_logp|seq
#           MY_ENV_DSV41_BATCH_INVARIANT=1/0  MY_ENV_DSV41_SAMPLER_DEBUG=1/0
set -eo pipefail
# 可选：用旧栈做 A/B（默认 rebase 栈）
export MY_ENV_DSV41_VLLM_REPO=${MY_ENV_DSV41_VLLM_REPO:-/mnt/share/m00899630/workspace/dsv41-verl-rebase}
REPO=$MY_ENV_DSV41_VLLM_REPO
cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl || exit 1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
export VLLM_USE_V2_MODEL_RUNNER=0
unset OMP_NUM_THREADS OMP_PROC_BIND VLLM_VERSION

export MY_ENV_DSV41_MODEL=${1:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram}
export MY_ENV_DSV41_TOKENIZER=/mnt/share/m00899630/dsv41/rl_tokenizer
export MY_ENV_DSV41_VAL_PARQUET=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MY_ENV_DSV41_MAXTOK=512
export MY_ENV_DSV41_USE_IMAGE=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TASK_QUEUE_ENABLE=2
export HCCL_IF_BASE_PORT=64000
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/1103_quickgen_v1_$(basename "$MY_ENV_DSV41_MODEL")_${TS}.log
mkdir -p logs
echo "[quickgen] log=$LOG model=$MY_ENV_DSV41_MODEL" | tee "$LOG"
/usr/local/python3.11.15/bin/python3 my_scripts/1103_quick_vllm_gen_check_v1.py 2>&1 | tee -a "$LOG"
echo "[quickgen] rc=$?" | tee -a "$LOG"
