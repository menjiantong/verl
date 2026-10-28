#!/usr/bin/env bash
# 任务二 v1 (2026-10-23)：验证 response length 是否为 diff_mean 恶化的原因。
# 基线 _launch_W3_longrun_1022.sh：512 response、countbenchqa、enable_thinking=false。
# 本次仅把 MAX_RESPONSE_LEN 限制为 50（用户允许 20/50，取 50 给答案留余量），
# 在空闲节点 122 上跑 150 step（基线 diff_mean 恶化窗口为 step50-90）。
# 预期：若短 response 下 diff_mean 不再升到 0.03+，则支持“与 response len 相关”的假设。
# 用法：bash my_scripts/_launch_W3_longrun_1022_maxresp50_v1.sh
set -eo pipefail
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
export MY_IP=80.5.25.122
export MY_ENV_MAX_RESPONSE_LEN=50
exec bash my_scripts/_launch_W3_longrun_1022.sh 150
