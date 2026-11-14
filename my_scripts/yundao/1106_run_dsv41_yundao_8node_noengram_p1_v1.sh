#!/usr/bin/env bash
# [启动结果] ⏳ 待验证：2026-11-06 首次实跑
# =============================================================================
# 1106 P3 对照 arm：noengram（去掉 engram 表，其余与 engram arm 完全相同）
#
# 用法（云道「启动命令」只填这一行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1106_run_dsv41_yundao_8node_noengram_p1_v1.sh
#
# 说明：本文件只是把 arm 写死成 noengram 后转调主脚本，四路探针、日志命名、
# 步数、cgroup 采样、抓栈看门狗全部与 engram arm 一致 —— 保证 A/B 只差模型配置。
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_P3_ARM=noengram
exec bash "$HERE/1106_run_dsv41_yundao_8node_engram_p1_v1.sh" "$@"
