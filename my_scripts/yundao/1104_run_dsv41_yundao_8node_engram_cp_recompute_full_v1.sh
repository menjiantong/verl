#!/usr/bin/env bash
# [验证结果] ✅ 成功：11/05 两轮 8 机跑通；整层重算把 max_memory_reserved 从 50.22 压到 19.07-19.20 GiB。
# =============================================================================
# 1104 DSV4.1 云道 8 机 Engram+CP：整层重计算（方案 B）启动器 v1
#
# 基线（rule 7.3/7.4：本脚本只写差异，其余全部沿用基线）：
#   my_scripts/yundao/1103_run_dsv41_yundao_8node_engram_cp_v1.sh
#   → 8 机 × 16 卡 = 128 rank，FSDP128/EP128/EFSDP1，rollout TP8×DP16(GEN_EP=128)，
#     CP=2（KV-AllGather），engram 表节点内 DP 共享，prompt 384 / response 1536。
#
# 与 1103 的唯一差异：重计算粒度从“子模块级”换成“整层”。
#   旧：recompute_plan=["model.layers.{*}.ffn","model.layers.{*}.attn"]
#   新：recompute_plan=["model.layers.{*}"]

# -----------------------------------------------------------------------------
# 为什么换成整层（2026-11-04 11/03 23:51 运行 OOM 分析结论）
#
# 1) 旧 plan 只 checkpoint 两个兄弟子模块，Block.forward 里下面这些张量全部留在卡上：
#      - fused_mhc_pre_sinkhorn_premix 里 eager h_pre 子图的 flat_x = x.flatten(-2).float()
#        [b,s,4*5120] fp32 = 80 KiB/token，每个 Block 的 attn / ffn 各一份 → 160 KiB/token/层
#      - 两个子层输出 h_out bf16 [b,s,5120]                        → 20 KiB/token/层
#      - ffn 的 residual（attn hc_post 输出）[b,s,4,5120] bf16     → 40 KiB/token/层
#    合计约 220 KiB/token/层，40 层、peak rank T≈1827 时 ≈15.3 GiB。
# 2) 换成整层后这些全部进 checkpoint，常驻只剩 Block 输入 40 KiB/token/层（≈2.4 GiB）。
#    预计 update 峰值 reserved 从 52.2 GiB 降到 ~37 GiB（设备 61.28 GiB）。
# 3) 代价：hc_mixes（[T,20480]×[24,20480] 小矩阵乘）+ sinkhorn 20 次小算子重放一次，
#    估算单步 +1%~3%（对比 1dae0be 实测 attn 重算 +2.33%）。
#
# 安全性：
#   - model.layers.{*} 在 str_match 里编译成 ^model\.layers\.(\d+)$，只匹配 40 个 Block，
#     不会顺带匹配 .ffn / .attn 子模块，因此不存在嵌套双重 checkpoint。
#   - CP 状态 ContextParallelAttentionState 是 NamedTuple，runtime.py 注释明确写了它
#     “跨激活 checkpoint 边界显式传递”，整层 checkpoint 是它的设计用例。
#   - evaluate 数值前先在 16p/4 层单训 CP 配置（FSDPTurbo/examples/deepseek_v41/
#     config_16p_cp_validate.yaml，commit 3ff111b）上比对 grad_norm 相对差。
#
# 回滚：把 MY_ENV_DSV41_RECOMPUTE_PLAN 去掉，或直接跑 1103 脚本即可（基线默认值不变）。
#
# 用法（云道“启动命令”只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_cp_engram_recompute_full_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1103_run_dsv41_yundao_8node_engram_cp_v1.sh"

if [ ! -f "$BASE" ]; then
  echo "[recompute-full] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 整层重计算：替换而不是追加（见文件头说明）。
export MY_ENV_DSV41_RECOMPUTE_PLAN='["model.layers.{*}"]'

echo "[recompute-full] MY_DEBUG_LOG recompute_plan env=${MY_ENV_DSV41_RECOMPUTE_PLAN} base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

# 其余（Ray 启动 / ModelArts 节点角色 / Engram / CP / 训练参数）全部交给 1103。
exec bash "$BASE"
