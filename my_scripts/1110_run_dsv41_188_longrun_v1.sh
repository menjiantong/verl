#!/usr/bin/env bash
# [1110] 188 单机 16 卡 CP=4 + Engram：长跑档
# 特性：thpoff（唯一 A/B 验证能消除 144 s 尾的配置；update_actor 97.7/86.2/76.6）。
# pinproc 单档与容量加权档均在 step3 出现 132 s 尾（节点仍被抽干），不用于长跑。
# 步数默认 200，可用 MY_ENV_DSV41_STEPS 覆盖。
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_FEATURES="thpoff"
export MY_ENV_DSV41_EXP_TAG="longrun"
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-200}
exec bash "$DIR/1110_dsv41_188_common.sh" "$@"
