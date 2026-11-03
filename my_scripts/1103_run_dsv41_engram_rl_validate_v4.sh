#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 Engram RL 验证启动器 v4
#
# v3 -> v4 演进：
#   * 80.5.25 fleet 节点 transformers 版本不一致：125 是 5.21.0（含
#     transformers.models.qwen4_exp），127 是 5.10.4（无该子模块）。
#     FSDPTurbo/fsdp_turbo/models/__init__.py 无条件导入
#     Qwen4ExpForConditionalGeneration，其 modeling_qwen4_exp.py:62 又导入
#     transformers.models.qwen4_exp，因此 127 上 HFModelConfig 实例化即
#     ModuleNotFoundError（日志 1103_launch_engram_cp2_20261102_211242.log）。
#   * 修复（不改系统 site-packages）：从 125 复制已验证的 transformers 5.21.0
#     + tokenizers 0.23.3（连同 dist-info，importlib.metadata 的版本检查需要）
#     到 /mnt/share/m00899630/env/py521，运行时用 PYTHONPATH 叠加。
#     已确认 $REPO/{vllm,vllm-ascend,FSDPTurbo,verl}/transformers 均不存在，
#     不会 shadow；叠加后 127 上 import fsdp_turbo.models / deepseek_v41 通过。
#   * v4 只是 v3 的包装，不改变任何训练/并行语义；125 原生 5.21 下等价。
#
# 用法：bash my_scripts/1103_run_dsv41_engram_rl_validate_v4.sh [STEPS] [CP_SIZE] [CHIPS] [SHARED] [MODEL]
#   默认 1 1 16 1 sft；CP+Engram 集成闸位用 2 2 16 1 sft
#
# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c
# 修复 commit（CP response_from_nested 切片，v1 trainer）：verl 8c1518c8
#   基线环境：80.5.25.125（transformers 5.21.0，CP+Engram 2 步闸位验证后回填本行）
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

# [1103 v4] sleep 备份 weights 用 pageable 主机内存：121 上 rtMallocHost 在
# engram 共享表 + FSDP offload 常驻后失败（aclrtMallocHostWithCfg 207001）。
# pageable 绕过驱动 pinned 分配器；需要回退 pinned 时显式传 1。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}

# [1103 v4] transformers 5.21 叠加：仅当本机缺 qwen4_exp 时需要；
# 叠加版本与 125 原生一致，统一开启不会改变 125 行为。
export MY_ENV_DSV41_TF_OVERLAY=/mnt/share/m00899630/env/py521
if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
    export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
else
    echo "[engram-rl-v4] WARN: overlay not found at $MY_ENV_DSV41_TF_OVERLAY" >&2
fi

exec bash my_scripts/1102_run_dsv41_engram_rl_validate_v3.sh "$@"
