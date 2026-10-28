#!/usr/bin/env bash
# v2 (2026-10-23)：v1（1024 response）在 120 启动时撞上外部租户（HBM 58GB/卡），
# trainer 建模型阶段 OOM；现改为在空闲节点 118 上重跑，response 仍为 1024。
# 其余完全继承 _launch_W3_longrun_1023_change_score_and_len_thinking.sh（countbenchqa2、
# enable_thinking=true、500 step、无性能优化）。
# 用法：bash my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking_v2.sh [STEPS]
set -eo pipefail
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
export MY_IP=80.5.25.118
export MY_ENV_MAX_RESPONSE_LEN=1024
exec bash my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking.sh "$@"
