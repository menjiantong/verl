#!/usr/bin/env bash
# =============================================================================
# 1103 CP 补齐下 response_from_nested 切片单测启动器 v1（纯 CPU）
#
# 用法：bash my_scripts/1103_run_cp_response_from_nested_test_v1.sh
# 期望（修复前）：cp_resp_test_003 ok=0；修复后 ok=1
# 验证 commit：verl c064ca65（脚本基线）；修复 commit verl 8c1518c8
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

# ssh 非登录 shell 需要 CANN 环境才能 import torch_npu（torch 自动加载后端）
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PATH=/usr/local/python3.11.15/bin:$PATH
export PYTHONPATH="$REPO:$REPO/vllm:$REPO/vllm-ascend:$REPO/FSDPTurbo:${PYTHONPATH:-}"

# 单测固定用 4（CP=2 的 pad 倍数），与启动器一致
export MY_ENV_DSV41_CP_PAD_MULTIPLE=${MY_ENV_DSV41_CP_PAD_MULTIPLE:-4}

python3 my_scripts/1103_test_cp_response_from_nested_v1.py
