#!/usr/bin/env bash
# [验证结果] ✅ 成功：CP=1，11/05 14:29（engram）与 18:09（noengram）两轮 8 机均跑通 2 步。
# =============================================================================
# 1105 DSV4.1 云道 8 机：Engram 开、CP 关（CP=1）v1
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_memprobe_v1.sh
#     → noval_v2（跳过 step0 验证、TEST_FREQ=-1、STEPS=2）
#       → recompute_full_v1（方案 B：recompute_plan=["model.layers.{*}"])
#         → 1103_..._engram_cp_v1.sh（Engram + CP，engram 表节点内 DP 共享）
#           → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#   三路探针（host 5 s / npu 30 s / proc 60 s）也保持不变。
#
# 与基线唯一的差异：
#   MY_ENV_DSV41_CP_SIZE=1   → CP（KV-AllGather）关闭
#   1103 会按 CP_SIZE 自动把三个 CP 内部开关设成关的形态（见下方“CP 相关”）。
#
# 目的：把 CP 这个变量摘掉，单独看 Engram 在 8 机上的表现。
# 提醒（rule 4.1，先说清楚预期）：CP 主要吃的是显存（每层全序列 KV/indexer 缓冲），
#   对主机内存几乎没有贡献。11/04 15:00 那次的失败点是主机 MemFree 掉到 7–31 GB
#   （cgroup 1.65–1.69 TB），所以这一轮如果目的仍是“绕开主机内存那堵墙”，大概率
#   还会撞上；它的价值在于把 CP 从因果链里摘干净。
#
# -----------------------------------------------------------------------------
# Engram 相关配置逐条（值 + 作用 + 出处；全部由 1103 设置，本脚本不重复写）
# -----------------------------------------------------------------------------
# 1) 权重
#    MY_ENV_DSV41_MODEL_PATH=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16      (1103:63)
#      → 全尺寸 384 专家 + 带 engram 表的基座。1103 的 python 预检会断言：
#        config.json 的 engram_layer_ids=[1,14] 非空、n_routed_experts==384、
#        index 里恰好有 2 个 layers.{1,14}.engram.embed.weight（1103:107-127）。
#        换成 noengram 权重会直接 ABORT。
#
# 2) 推理侧共享表
#    MY_ENV_DSV41_ENGRAM_DP_SHARED=1                                       (1103:78)
#      → 基线 dsv41_..._8_nodes_final.sh:83 读成 VLLM_ENGRAM_DP_SHARED_MEMORY
#      → example run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh:207-224 追加：
#          +...vllm.engram_config.cpu_offload=true
#          +...vllm.engram_config.dp_shared_memory=true
#      作用：节点内**同一 TP 分片**的多个 DP 副本共用一份物理页
#      （vllm-ascend/.../engram/embedding.py “One physical copy of the mapped range”）。
#      8 机 GEN_TP=8 / GEN_DP=16 → 每节点 2 个副本 → 节点内 DP 组大小 = 2（日志会打
#      “Engram node-local DP group size: 2 (TP=8)”）→ 每节点物理 366 GiB（2 表 × 8 分片
#      × 22.89 GiB）。如果复制比是 1（每节点 1 副本），它会自动退化成每副本一份。
#
# 3) 权重加载方式
#    MY_ENV_DSV41_LOAD_FORMAT=safetensors                                 (1103:77)
#      → 基线 :92/:380 → actor_rollout_ref.rollout.load_format
#      → example:213-221 在 dp_shared_memory=1 时还会追加 safetensors +
#        enable_multithread_load=true / num_threads=64 / safetensors_load_strategy=lazy
#      作用：必须真读 checkpoint。verl 默认 dummy 会让 vllm-ascend 跳过 engram 表读取，
#      表是空的但形状对，rollout 结果会静默错；safetensors 是 vLLM 对 engram 校验
#      接受的格式（auto/safetensors/pt）。
#
# 4) 主机内存的 pin 策略
#    MY_ENV_VLLM_SLEEP_PIN_MEMORY=0                                        (1103:102)
#      → camem 补丁读取；sleep 时 27.5 GiB/worker 的权重备份用 pageable 而不是 pinned，
#        绕开 rtMallocHost（11/03 单机 worklog §2.2 记录：主机映射大时该路径会自旋）。
#    基线 :171 VERL_DSV41_ENGRAM_PIN_MEMORY=0
#      → 训练侧 engram host 表也用 pageable。
#
# 5) 训练侧 engram 表
#    由 FSDPTurbo 按 EP 行切：EP_SIZE=128 → 每 rank 2.9 GiB（1/128 张表），冻结、
#    不建 AdamW 状态，每节点 16 × 2.9 ≈ 46 GiB。日志会打
#    “host-offload Engram tables loaded: 2 table(s), 2.9 GiB per rank (frozen)”。
#
# 6) 专家导出
#    LOCAL_EXPERT_EXPORT=1（基线 :188）+ VERL_DSV41_LOCAL_EXPERT_EXPORT=1（:173）
#      → example:87-94：只有 GEN_EP == EP_SIZE 且 EFSDP_SIZE == 1 时才保持 1。
#      本机 EP_SIZE=128、GEN_EP=TP8×DP16=128、EFSDP=1 → 保持 1，权重同步走本地专家
#      切片，不会每步 all-gather 全部专家。
#
# 7) 日志级别
#    MY_ENV_VLLM_LOGGING_LEVEL=INFO                                        (1103:83)
#      → 不放开到 INFO 就看不到 engram 拓扑与共享表那几行，无法确认共享是否生效。
#
# -----------------------------------------------------------------------------
# CP 相关（本脚本只把 CP_SIZE 设成 1，其余由 1103:90-101 按 CP_SIZE 自动派生）
# -----------------------------------------------------------------------------
#   MY_ENV_DSV41_CP_SIZE=1        → example:283-284 追加
#                                   +...distributed.kv_allgather_parallel_size=1
#   MY_ENV_DSV41_CP_PAD_MULTIPLE=1   CP 关闭时长度不需要 4 的倍数
#   MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
#   MY_ENV_DSV41_CP_GATHER_OUTPUT=0
#
# 用法（云道“启动命令”只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_engram_cp1_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1104_run_dsv41_yundao_8node_engram_cp_memprobe_v1.sh"

if [ ! -f "$BASE" ]; then
  echo "[engram-cp1] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 唯一差异：关 CP。
export MY_ENV_DSV41_CP_SIZE=1

echo "[engram-cp1] MY_DEBUG_LOG cp=1（KV-AllGather 关）engram 保持 1103 默认：model=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16 dp_shared=1 load_format=safetensors sleep_pin=0 base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
