#!/usr/bin/env bash
# v1 (2026-10-23)：按用户要求把 MAX_RESPONSE_LEN 从 2048 降到 1024。
# 原因（evidence）：2048 在 16 卡单机 step1 actor update 触发 NPU OOM
#   NPU2: 50.40 GiB active / 137.87 MiB free / 50.91 GiB reserved（总 61.27 GiB）；
#   vLLM sleep(level=1) 后仍占 ~10 GiB persistent。512 response 时训练峰值 26.4 GiB，
#   1024 的 seq=1152+1024=2176，按 seq 线性外推峰值约 34-36 GiB，训练侧可用 ~51 GiB，余量充足。
# 其余配置（countbenchqa2 打分、enable_thinking=true、500 step、120 单机、无性能优化）完全继承 base。
# 用法：bash my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking_v1.sh [STEPS]
set -eo pipefail
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
export MY_ENV_MAX_RESPONSE_LEN=1024
exec bash my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking.sh "$@"
