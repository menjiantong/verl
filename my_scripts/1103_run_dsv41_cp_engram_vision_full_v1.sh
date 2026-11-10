#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 CP + Engram + 视觉算子 全修复集成验证启动器 v1
#
# 依赖的核心修复 commit：
#   FSDPTurbo   CP（KV-AllGather）        d84d257 / 03c3c1d / 8703d04
#   verl        CP 响应切片（v1 trainer）  8c1518c8
#   verl        环境/视觉开关              9575baaa（MY_ENV_VLLM_VIT_ATTN_OP=fa 默认）
#   vllm        Engram NPU 主机注册        fd46ca5ada（aclrtHostRegister + UVA 视图）
#   vllm        构建适配 torch 2.10        cb6d4ae5b9
#   vllm-ascend 视觉注意力同算子            e3f220b05（fa → npu_fusion_attention）
#   vllm-ascend sleep 主机内存门控          633c48011（pageable）
#
# 功能：countbenchqa（带图）上跑 CP=2 + Engram（EP16/DP2 共享表）+ 视觉 FA，
#       覆盖“CP 数值、Engram 表/共享内存/sleep、视觉训推算子一致”三条链路；
#       默认 2 step，即全功能集成闸位。
#
# 用法：bash my_scripts/1103_run_dsv41_cp_engram_vision_full_v1.sh [STEPS] [CP_SIZE] [CHIPS]
#   默认 2 2 16；模型固定 top32-sft116（带 engram 表的 SFT 权重）。
#   机器：80.5.25.121（16×Ascend910，内存 ≥2000 GiB 可用）。
#
# 脚本 commit：verl 7a781f15（首次提交）；验证的核心修复 commit 见上表
# =============================================================================
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"

STEPS=${1:-2}
CP_SIZE=${2:-2}
CHIPS=${3:-16}

# [1103 full-fix] 视觉：推理侧视觉注意力用训练同一个 FlashAttentionScore。
# FIA(TND) 与 FA 单算子相对误差约 5e-5，经 32 层 ViT 放大后视觉特征约 0.8%；
# fa 路径需要 enforce_eager=True（run 脚本已设置）。
export MY_ENV_VLLM_VIT_ATTN_OP=${MY_ENV_VLLM_VIT_ATTN_OP:-fa}
# 训练侧融合视觉注意力（1030 env 默认 1，这里显式写出便于核对）
export MY_ENV_TRAIN_FUSED_VISION_ATTN=${MY_ENV_TRAIN_FUSED_VISION_ATTN:-1}
# sleep 备份 weights 用 pageable 主机内存：121 上 engram 共享表 + FSDP offload
# 常驻约 1.6 TB 后，rtMallocHost 会报 aclrtMallocHostWithCfg 207001。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=${MY_ENV_VLLM_SLEEP_PIN_MEMORY:-0}
# transformers 5.21 叠加：原生 5.10.4 节点缺 transformers.models.qwen4_exp，
# FSDPTurbo 的模型导入链需要；125/121 叠加同版本无行为差异。
export MY_ENV_DSV41_TF_OVERLAY=${MY_ENV_DSV41_TF_OVERLAY:-/mnt/share/m00899630/env/py521}
if [ -d "$MY_ENV_DSV41_TF_OVERLAY/transformers/models/qwen4_exp" ]; then
    export PYTHONPATH="$MY_ENV_DSV41_TF_OVERLAY:${PYTHONPATH:-}"
fi

# 日志标识（rule 8）：每行唯一 key，便于 grep 定位本次运行对应的修复版本。
echo "MY_DEBUG_LOG full_fix_commit_fsdpturbo_cp=d84d257,03c3c1d,8703d04"
echo "MY_DEBUG_LOG full_fix_commit_verl_cp=8c1518c8"
echo "MY_DEBUG_LOG full_fix_commit_verl_env=9575baaa"
echo "MY_DEBUG_LOG full_fix_commit_vllm_engram=fd46ca5ada"
echo "MY_DEBUG_LOG full_fix_commit_vllm_build=cb6d4ae5b9"
echo "MY_DEBUG_LOG full_fix_commit_vllm_ascend_vision=e3f220b05"
echo "MY_DEBUG_LOG full_fix_commit_vllm_ascend_sleep=633c48011"
echo "MY_DEBUG_LOG full_fix_config steps=${STEPS} cp=${CP_SIZE} chips=${CHIPS} model=top32-sft116 vit_attn=${MY_ENV_VLLM_VIT_ATTN_OP}"

exec bash my_scripts/1103_run_dsv41_engram_rl_validate_v4.sh "$STEPS" "$CP_SIZE" "$CHIPS" 1 sft
