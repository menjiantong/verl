#!/bin/bash
# [1030 验证] 清理所有调试代码后的单推理复验（与训练共用 V1 model runner + spawn）。
# 复用调试阶段的单推理脚本 1030_debug_single_inference_v1.py（仅作为验证输入，不再含调试开关）。
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
# 单推理只用 8 卡（TP=8），避免占用训练所需的 16 卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# V1 model runner + spawn：规避 OMP fork 线程池断言
unset OMP_NUM_THREADS OMP_PROC_BIND VLLM_VERSION
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd $REPO/verl
exec python3 -c "exec(open(\"my_scripts/1030_debug_single_inference_v1.py\").read())"
