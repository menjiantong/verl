#!/usr/bin/env bash
# =============================================================================
# 2026-10-25 全量 566B（去 engram）GRPO 公共环境
#
# 被 1025_W3_longrun_full_noengram_{8,16}node_v1.sh 在 master/join 两端 source。
# 为什么单独成文件：ray worker 继承的是“本机 raylet”的环境，不是 driver 的（E39）。
# 8/16 机必须保证两端 MODEL_PATH / offload / 一致性开关完全相同，否则训练与引擎行为分裂。
#
# 权重：/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-bf16-noengram
#   - 275 个文件 = 274 个软链 + 1 个 config.json（config 由原始 config 派生：
#     engram_layer_ids=[]、engram_num_embeddings=[]，n_routed_experts=384 不变）
#   - 原始 763.21B（engram 196.93B）→ 去 engram 566.28B，bf16 约 1.13TB
#   - VERL_DSV41_DIRECT_SHARD_LOAD=1 只读模型参数，196GB 的 engram 分片不会被读
# =============================================================================
set -eo pipefail

VERL_ROOT=/mnt/share/m00899630/workspace/dsv41-verl/verl
cd "$VERL_ROOT" || exit 1

export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-bf16-noengram
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet
export MAX_PROMPT_LEN=1152
export MAX_RESPONSE_LEN=${MY_ENV_MAX_RESPONSE_LEN:-512}
export USE_KL=0
# 全量模型每卡引擎权重约 20.9GB（EP64，见 worklog）：0.6*61.27=36.8GB 预算，KV 约 15GB。
export ROLLOUT_GPU_MEM_UTIL=${MY_ENV_ROLLOUT_GPU_MEM_UTIL:-0.6}
export VERL_DSV41_VISION=1
# 无 engram：关闭多机 engram 主机表共享，避免向引擎传无用的 engram_config。
export VLLM_ENGRAM_DP_SHARED_MEMORY=0

# 与 W3 基线一致：单轮 VL agent、不开 thinking、失败组同步补采样。
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"

# 全量模型必须全 offload：每个训练 rank 只持 566.28B/128=4.35B 专家参数，
# bf16 param+grad + fp32 master/m/v ≈ 70GB/rank；16 rank/节点 ≈ 1.13TB host，
# 2.4TB 内存可容纳。HBM 只放当前层 all-gather/激活与 vLLM sleep 后的常驻元数据。
export PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True OFFLOAD_POLICY=True

# 与 W3 基线一致：避免 allocator/AdamW swap 干扰 offload 路径。
unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP
