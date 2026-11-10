#!/usr/bin/env bash
# [1110] 188 单机 16 卡 CP=4 + Engram：组合档（进程级摊开 + THP off 兜底）
# 若 pinproc 单档已验证足够，长跑优先用 pinproc 单档；本档是“最稳”组合。
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_FEATURES="thpoff,pinproc"
export MY_ENV_DSV41_EXP_TAG="all"
exec bash "$DIR/1110_dsv41_188_common.sh" "$@"
