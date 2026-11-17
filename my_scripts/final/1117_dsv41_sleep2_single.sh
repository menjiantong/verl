#!/usr/bin/env bash
# =============================================================================
# [final] 单机 16 卡 sleep mode=2 交付脚本 v4（2026-11-17）
#
# 结果：成功（vllm-ascend 16c3159dd / 分支 from_9a38d36_1030）
# 成功日志：verl/logs/1117_sleep2fix_227.log、1117_sleep2fix_228.log
# 记录：mjt_workplace/task/sleep2_l2fix_20261117/
#       worklog_20261117_sleep2_l2fix.md、report_sleep2_l2fix_20261117.md
#
# 基线：1116_dsv41_1k2k_graph_mb2_v1.sh（已验证，逐字不变）
# 本脚本只加 3 个开关：
#   1) actor_rollout_ref.rollout.free_cache_engine=True
#        rollout 结束后 sleep_replicas()，真正进入 sleep
#   2) MY_ENV_DSV41_SLEEP_LEVEL=2（默认）
#        vLLM level-2：丢弃整个 CAMEM weights/kv 池（无 CPU 备份），
#        update_weights / generate 前 wake_up 重新映射
#   3) MY_ENV_VLLM_SLEEP_PIN_MEMORY=0
#        仅约束 level-1 回退路径（本镜像 pin 分配 10.98 GiB 会失败）
#
# 代码依赖（vllm-ascend 16c3159dd）：
#   fix(engram): keep host-UVA address tables across level-2 sleep
#   Engram HostUvaBuffer/SharedUvaBuffer.ptrs 注册为 non-persistent buffer，
#   由 worker._sleep_saved_buffers 在 sleep/wake 自动备份/还原；缺这步则
#   wake 后地址表是未初始化 HBM → Engram gather 报 MTE invalid GM。
#
# 验证（80.5.27.227 / 80.5.27.228 单机并行，各 2 步，开 MY_ENV_DSV41_SLEEP2_TRACE）：
#   3 个 sleep/wake 周期零 MTE；释放 13.98–15.87 GiB/卡；
#   每卡峰值 42.13 / 43.04 GiB；step2 409.53 / 413.02 s；
#   update_actor 每 token 1.3407 ms（L1 基线 1.3386）→ sleep 无可测开销。
#   注：TRACE 打点来自未提交的调试补丁（仅日志、默认行为与 HEAD 一致），
#       归档在 mjt_workplace/task/sleep2_l2fix_20261117/debug_patches/。
#
# 用法：
#   STEPS=2 bash my_scripts/final/1117_dsv41_sleep2_single_v4.sh
#   对照关 sleep：STEPS=2 bash my_scripts/final/1116_dsv41_1k2k_graph_mb2_v1.sh
# =============================================================================
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_SLEEP_LEVEL=${MY_ENV_DSV41_SLEEP_LEVEL:-2}
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
exec bash "$here/1116_dsv41_1k2k_graph_mb2_v1.sh" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  "$@"
