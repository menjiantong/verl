#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次实跑
# =============================================================================
# 1106 P4-R1 修复 arm：engram + 复用 pinned 主机缓冲（去掉每次 fetch 的 rtMallocHost）
#
# 用法（云道「启动命令」只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1106_run_dsv41_yundao_8node_engram_reusebuf_p1_v1.sh
#
# 预期（可证伪）：若 M-LOCK 成立，[engram-probe] 的 buf= 从 alloc 变 reuse，
#   t_buf_alloc_ms 从秒级回落到 <1 ms，update_actor 应显著下降；
#   若 update_actor 不变，则 pinned 分配不是主因，M-LOCK 被否。
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_P3_ARM=engram
export MY_ENV_P3_REUSE_HOST_BUF=1
exec bash "$HERE/1106_run_dsv41_yundao_8node_engram_p1_v1.sh" "$@"
