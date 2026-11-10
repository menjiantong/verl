#!/usr/bin/env bash
# [1110] 188 单机 16 卡 CP=4 + Engram：进程级 first-touch 摊开（pinproc）
# 特性：MY_ENV_DSV41_PIN_PROCESS=1，训练/vLLM worker 整个进程钉到 rank 对应的
#      NUMA 节点（默认 0,2,4,5,6,7，排除半容量 node3），摊开全部主机分配；
#      THP 保持 on，用于验证“不关 THP 也能避免慢路径”的根治目标。
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_FEATURES="pinproc"
export MY_ENV_DSV41_EXP_TAG="pinproc"
exec bash "$DIR/1110_dsv41_188_common.sh" "$@"
