#!/usr/bin/env bash
# [1031 算子对比] 164 单机 16 卡 1-step：同时 dump 训练/推理两侧算子输入输出。
# 用法: bash my_scripts/diag/1031_launch_164_opdump.sh [base|fused]
#   base  = 训练视觉 attention 用 SDPA（现有生产路径）
#   fused = 训练视觉 attention 用 torch_npu 融合 FlashAttentionScore
# 产物:
#   logs/opdump_164_<tag>_<ts>/        逐算子 .pt + index_*.jsonl
#   logs/dsv41_opdump_164_<tag>_<ts>.log
# 说明: 只落 rank0；1 prompt × 16 response，16 卡每卡 1 条序列。
set -eo pipefail

REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO"
source my_scripts/1030_ray_dsv41_env.sh

TAG=${1:-base}
TS=$(date +%Y%m%d_%H%M%S)
DUMP_DIR="$REPO/logs/opdump_164_${TAG}_${TS}"
CKPT_DIR="$REPO/logs/diag_ckpt_164_${TAG}_${TS}"
LOG="$REPO/logs/dsv41_opdump_164_${TAG}_${TS}.log"

# 1-step 诊断 batch：与 126 上的基线一致，保证可逐算子对齐。
export TRAIN_BSZ=1 SAMPLE_N=16 VAL_BSZ=1
export CKPTS_DIR="$CKPT_DIR"

# 训练侧融合开关：base=全关；fused=仅视觉；all=视觉+其余五项；
# sfa/mhc/comp/swiglu/rope/idx 只开对应单项，用于归因。
case "$TAG" in
  base)  export MY_ENV_TRAIN_FUSED_VISION_ATTN=0 ;;
  fused) export MY_ENV_TRAIN_FUSED_VISION_ATTN=1 ;;
  all)
    export MY_ENV_TRAIN_FUSED_VISION_ATTN=1
    export MY_ENV_TRAIN_SPARSE_FLASH_ATTN=1
    export MY_ENV_TRAIN_FUSED_MHC=1
    export MY_ENV_TRAIN_FUSED_COMPRESSOR=1
    export MY_ENV_TRAIN_FUSED_CLIPPED_SWIGLU=1
    export MY_ENV_TRAIN_TRITON_INPLACE_ROPE=1
    export MY_ENV_TRAIN_FUSED_LIGHTNING_INDEXER=1
    ;;
  sfa)    export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_SPARSE_FLASH_ATTN=1 ;;
  mhc)    export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_FUSED_MHC=1 ;;
  comp)   export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_FUSED_COMPRESSOR=1 ;;
  swiglu) export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_FUSED_CLIPPED_SWIGLU=1 ;;
  rope)   export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_TRITON_INPLACE_ROPE=1 ;;
  idx)    export MY_ENV_TRAIN_FUSED_VISION_ATTN=0; export MY_ENV_TRAIN_FUSED_LIGHTNING_INDEXER=1 ;;
  *) echo "unknown tag: $TAG" >&2; exit 2 ;;
esac

# 逐算子 dump：只落 rank0；类过滤覆盖 train/infer 两侧及融合 router/LM head。
export MY_ENV_OP_DUMP=1
export MY_ENV_OP_DUMP_DIR="$DUMP_DIR"
export MY_ENV_OP_DUMP_MAX_FORWARD=4
export MY_ENV_OP_DUMP_MAX_NUMEL=$((64 * 1024 * 1024))
export MY_ENV_OP_DUMP_MAX_CALLS=600
export MY_ENV_OP_DUMP_MAX_FILES=4000
export MY_ENV_OP_DUMP_MAX_SECONDS=120
export MY_ENV_OP_DUMP_ALL_RANKS=0
export MY_ENV_OP_DUMP_EXPERTS=0
export MY_ENV_OP_DUMP_CLASSES="Block,Attention,MoE,Gate,ParallelHead,ParallelEmbedding,DeepseekV41DecoderLayer,DeepseekV41Attention,DeepseekV41MoE,AscendDeepseekV41LLMForCausalLM,VocabParallelEmbedding,RouterFused,LMHeadGEMM"
# sitecustomize 必须在 PYTHONPATH 中；Ray head 启动后 worker 会继承该环境。
export PYTHONPATH="$REPO/my_scripts/diag:${PYTHONPATH:-}"

echo "[MY_DEBUG_LOG opdump164] tag=$TAG dump=$DUMP_DIR ckpt=$CKPT_DIR log=$LOG"
dsv41_ray_start
bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
    trainer.total_training_steps=1 \
    2>&1 | tee "$LOG"
