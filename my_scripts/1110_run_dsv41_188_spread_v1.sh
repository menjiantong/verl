#!/usr/bin/env bash
# [1110] 188 单机 16 卡 CP=4 + Engram：spread 档
# 特性：spread
# 基线/公共配置见同目录 1110_dsv41_188_common.sh（规则 7.3/7.4）。
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_FEATURES="spread"
export MY_ENV_DSV41_EXP_TAG="spread"
exec bash "$DIR/1110_dsv41_188_common.sh" "$@"
