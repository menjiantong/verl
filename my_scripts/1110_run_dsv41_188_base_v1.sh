#!/usr/bin/env bash
# [1110] 188 单机 16 卡 CP=4 + Engram：base 档
# 特性：无（基线：THP on、表不摊开、FREE_CACHE_ENGINE=False）
# 基线/公共配置见同目录 1110_dsv41_188_common.sh（规则 7.3/7.4）。
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_FEATURES=""
export MY_ENV_DSV41_EXP_TAG="base"
exec bash "$DIR/1110_dsv41_188_common.sh" "$@"
