#!/usr/bin/env bash
# [验证结果] ✅ 成功：11/05 18:09 跑通 2 步；step2 update_actor 49.06 s vs engram 版 1428.9 s（快 29 倍），确证 engram 是性能劣化主因。
# =============================================================================
# 1106 DSV4.1 云道 8 机：**去掉 Engram** 的对照组（CP=1 + fadvise 守护）
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
#     → 1105_..._cp1_v1.sh（CP=1）
#       → 1104_..._cp_memprobe_v1.sh（三路探针 host 5s / npu 30s / proc 60s）
#         → 1104_..._cp_recompute_full_noval_v2.sh（跳过 step0 验证、TEST_FREQ=-1、STEPS=2）
#           → 1104_..._cp_recompute_full_v1.sh（方案 B：recompute_plan=["model.layers.{*}"])
#             → 1103_..._engram_cp_v1.sh（Engram，CP 内部开关按 CP_SIZE 派生）
#               → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# -----------------------------------------------------------------------------
# 一、这个脚本要回答的问题
# -----------------------------------------------------------------------------
#   11/05 14:29 那次（带 Engram）跑通了 2 步，但 update_actor 慢到 1429–1601 s；
#   11/01 那次（不带 Engram）同样 8 机、同样 train_batch_size=16、rollout.n=8，
#   update_actor 只有 67–80 s。**同拓扑、同批量，慢 15–20 倍**。
#   本次把 Engram 这个变量单独摘掉，其余一切不动，看 update_actor 会不会回来。
#
#   归一化数据（每 token 吞吐，排除"这次 token 更多"的干扰）：
#     11/01   update_actor 153084 tok / 70 s  = 2187 tok/s
#     11/05   update_actor 202467 tok / 1429 s =  142 tok/s   ← 慢 15.4 倍
#     11/01   gen          153084 tok / 512 s  =  299 tok/s
#     11/05   gen          202467 tok / 585 s  =  346 tok/s   ← 生成侧反而更快
#
# -----------------------------------------------------------------------------
# 二、与 1105 fadvise_v3 的唯一差异（3 个 export + 1 个门槛）
# -----------------------------------------------------------------------------
#   1) MY_ENV_DSV41_MODEL_OVERRIDE=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16-noengram
#      → 1103 的 ENGRAM_MODEL 从带表基座换成 noengram 基座。
#      注意：这个目录是指向同一份 1.4 TB 权重的**软链农场**，268 个
#      safetensors 分片全部软链到 DeepSeek-V4.1-Flash-bf16/，index.json 与 engram 版
#      **md5 完全相同**（41327409365e13b68099e8af2140afed），只有 config.json 不同：
#        engram_layer_ids: [1,14]  →  []
#        engram_num_embeddings: [384006168,384016682] → []
#      也就是说：**读盘字节完全一致，只是模型不再构建 engram 那两个层**。
#      这正是 11/01 长跑用的权重，A/B 的对照条件是干净的。
#   2) MY_ENV_DSV41_REQUIRE_ENGRAM=0
#      → 跳过 1103 的"engram_layer_ids 必须非空"断言和"/dev/shm ≥ 450 GiB"预检
#        （那两个门槛只对 engram 共享表有意义）。
#      384 专家的断言仍然保留，确保对照的模型规模一致。
#   3) MY_ENV_DSV41_ENGRAM_DP_SHARED=0
#      → 不向 vLLM 传 engram_config.dp_shared_memory / cpu_offload。
#   4) MY_ENV_DSV41_MIN_MEM_GB=1200
#      → 无 engram 时单节点主机占用约 1.27 TB（11/01 实测 cpu_memory_used=1269 GiB），
#        不再需要 1600 GiB 的门槛，但仍然拦住明显不干净、没有足够空闲内存的节点。
#
#   其余全部不变：fadvise 守护、memprobe 三路探针、整层重算、CP=1、
#   sleep pin=0、batch/lr/步数（STEPS=2）都一样。
#
# -----------------------------------------------------------------------------
# 三、判读规则（只看 step2，step1 里含 AdamW 状态创建，会污染数据）
# -----------------------------------------------------------------------------
#   A) update_actor 从 ~1429 s → 100 s 以内
#      → **Engram 就是元凶**（训练侧 host-offload 表 / 366 GiB tmpfs / 主机内存压力）。
#        下一步在 engram 版本上做 P1-②③：表放 NPU、sleep 备份去重。
#   B) update_actor 仍在 1000 s 以上
#      → 与 Engram 无关，回到「整层重算」和「主机内存总量」两个变量。
#        下一步做 P1-①（recompute 回退 ffn+attn）。
#   C) 中间值（200–1000 s）→ 部分相关，按 P0 打点结果再拆。
#
#   同时要看的三个辅助量（同一份 memprobe 日志）：
#     shmem         366 GiB → 0        （engram 共享表消失了）
#     cpu_memory_used 1593 GiB → ~1270 GiB（少掉 engram 训练表 + 共享表）
#     Camem sleep   每小时 3 次 → 数量不变，但宿主机压力小很多
#
# 用法（云道"启动命令"只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1106_run_dsv41_yundao_8node_noengram_cp1_fadvise_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_noengram_cp1_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 回滚：跑 1105_..._cp1_fadvise_v3.sh（本脚本只改 4 个环境变量，不动代码）。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh"

if [ ! -f "$BASE" ]; then
  echo "[noengram] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 1) 换权重：指向只有 config.json 不同的 noengram 软链农场
export MY_ENV_DSV41_MODEL_OVERRIDE=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16-noengram
# 2) 跳过 engram 相关预检（engram 表断言、/dev/shm ≥ 450 GiB）
export MY_ENV_DSV41_REQUIRE_ENGRAM=0
# 3) 关掉 vLLM 侧的 engram DP 共享表开关
export MY_ENV_DSV41_ENGRAM_DP_SHARED=0
# 4) 主机内存门槛按无 engram 的预算（~1.27 TB）设定
export MY_ENV_DSV41_MIN_MEM_GB=1200
# 日志/实验标签，便于在 logs/ 与 wandb 目录下区分本组
export MY_ENV_DSV41_EXPERIMENT_TAG="noengram-cp1"
export MY_ENV_DSV41_RUN_TAG="noengram-cp1-math-dapo17k"

echo "[noengram] MY_DEBUG_LOG 对照组启动 model=${MY_ENV_DSV41_MODEL_OVERRIDE} require_engram=0 dp_shared=0 min_mem=${MY_ENV_DSV41_MIN_MEM_GB}GiB base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
