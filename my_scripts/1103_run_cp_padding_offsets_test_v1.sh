#!/usr/bin/env bash
# 1103 CP 补齐切片单测启动器 v1（纯 CPU）
# 用法：bash my_scripts/1103_run_cp_padding_offsets_test_v1.sh
set -eo pipefail
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$REPO/verl"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH=$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:$REPO/verl:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=""
python3 my_scripts/1103_test_cp_padding_offsets_v1.py
