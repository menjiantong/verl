#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-06 第三 arm（engram + 重算复用），在本地 A/B 之后串行跑。
# =============================================================================
# 目的：隔离"整层重算期间那次 engram fetch 的阻塞同步（两次 .to('cpu') 会把 NPU
#   stream 排空）"是不是 update_actor 变慢的元凶。
#
# 做法：只改一个环境变量 MY_ENV_DSV41_ENGRAM_REUSE_RECOMPUTE=1。
#   打开后 _TorchHostFetch.forward 在重算调用时按内容命中缓存（numel+shape+torch.equal），
#   直接返回上一次的 out，完全跳过 unique / all_to_all / D2H / H2D / CPU gather。
#   数值等价：表 frozen（fsdp_turbo_dsv41_impl.py:1332,1378），同一 row_ids 结果逐位相同。
#
# 判读：
#   - reuse arm 的 update_actor 明显低于 engram 基线 ⇒ 阻塞同步是元凶；
#   - 基本不变 ⇒ 元凶不在 fetch 本身（转向 EP 通信器交错 / 主机侧停顿）。
#
# 基线：my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh（本脚本只加一个 env）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

echo "[reuse] MY_DEBUG_LOG reuse_arm_start time=$(date '+%Y-%m-%d %H:%M:%S')"

MY_ENV_DSV41_AB_ARM=engram \
MY_ENV_DSV41_AB_STEPS=${MY_ENV_DSV41_AB_STEPS:-3} \
MY_ENV_DSV41_AB_PROBE_CALLS=0 \
MY_ENV_DSV41_ENGRAM_REUSE_RECOMPUTE=1 \
  bash my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh

echo "[reuse] MY_DEBUG_LOG reuse_arm_done time=$(date '+%Y-%m-%d %H:%M:%S')"
