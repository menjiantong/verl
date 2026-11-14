#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-07 E1-8机分段探针脚本，尚未实跑。
# =============================================================================
# 1107 DSV4.1 云道 8 机：Engram fetch 分段探针（E1-8机，CP=1 + fadvise 守护）
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh
#     → 1105_..._cp1_v1.sh（CP=1）
#       → 1104_..._cp_memprobe_v1.sh（三路探针 host 5s / npu 30s / proc 60s）
#         → 1104_..._cp_recompute_full_noval_v2.sh（跳过 step0 验证、TEST_FREQ=-1、STEPS=2）
#           → 1104_..._cp_recompute_full_v1.sh（方案 B：recompute_plan=["model.layers.{*}"])
#             → 1103_..._engram_cp_v1.sh（Engram）
#               → 基线 dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#
# -----------------------------------------------------------------------------
# 一、E1 要回答的问题
# -----------------------------------------------------------------------------
#   已经定性：去掉 Engram 后 step2 update_actor 从 1428.9 s 掉到 49.06 s（29×），
#   每 token 慢 15.4×。但**微观机制没有取证** —— 1429 s 里究竟是谁在等：
#     a) 3~4 次 all_to_all_single（EP128 跨机集合通信）
#     b) 2~3 次 D2H 同步 + numpy().tolist()（每次都要等 NPU 排空）
#     c) CPU 在 2.9 GiB pinned host 表上做 index_select（主机内存带宽/NUMA 惩罚）
#     d) H2D
#   本脚本在 `_TorchHostFetch.forward/backward` 里逐段打点，把 1429 s 拆到段。
#
# -----------------------------------------------------------------------------
# 二、代码侧改动（已在 11/05 20:00 落到工作目录，带备份）
# -----------------------------------------------------------------------------
#   FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py
#     备份：host_offload_engram.py.bak.20261105_2000
#     新增（全部默认关闭，不设环境变量时对训练零影响）：
#       MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1        打开探针
#       MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=1   每段前 torch.npu.synchronize（时间更准，但会串行化）
#       MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=N  每进程每类最多打印 N 次，0=不限
#   打印格式（每 rank 每层每 micro-batch 一行）：
#     [engram-probe] kind=fwd rank=R layer=L rows=.. uniq=.. send=.. recv=.. ep=..
#       pinned=True local_rows=.. t_uniq_ms=.. t_a2a_cnt_ms=.. t_d2h_list_ms=..
#       t_a2a_ids_ms=.. t_cpu_gather_ms=.. t_h2d_ms=.. t_a2a_val_ms=.. total_ms=..
#     [engram-probe] kind=bwd rank=R layer=L grad_rows=.. send=.. recv=.. ep=..
#       t_prep_ms=.. t_a2a_grad_ms=.. t_accum_ms=.. total_ms=..
#
#   判读（8 机）：
#     若 t_a2a_* 三段合计 > 50%  → 跨机集合通信是主因（EP128 的 all_to_all）
#     若 t_cpu_gather_ms 明显    → 主机侧 index_select / 内存带宽是主因
#     若 t_d2h_list_ms 明显      → 同步排空 + tolist 是主因
#     与单机脚本（117）对照：单机没有跨机 a2a，同一段若仍慢 → 与网络无关
#
# -----------------------------------------------------------------------------
# 三、本脚本与 1105 fadvise_v3 的唯一差异（3 个 export）
# -----------------------------------------------------------------------------
#   PROBE=1 / PROBE_SYNC=0 / PROBE_CALLS=0
#   SYNC=0：保持默认异步，不改变训练时序，先拿"不扰动"的分段数据
#           （all_to_all / .to("cpu") / tolist 本身就是阻塞点，时间基本准）。
#   CALLS=0：不限次数。每 rank 每步 2 层 × (1 fwd + 1 bwd) = 4 行，
#           STEPS=2 → 每 rank 8 行，128 rank ≈ 1024 行，量很小。
#   其余全部不变：CP=1、整层重算、fadvise 守护、memprobe、STEPS=2。
#
# 用法（云道"启动命令"只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1107_run_dsv41_yundao_8node_engram_probe_v1.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_probe_8node_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 回滚：跑 1105_..._cp1_fadvise_v3.sh（探针默认关闭，代码可原地保留）。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1105_run_dsv41_yundao_8node_engram_cp1_fadvise_v3.sh"

if [ ! -f "$BASE" ]; then
  echo "[probe8] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 1) 打开 engram fetch 分段探针
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1
# 2) 不做逐段同步：不改变训练时序，先拿"零扰动"数据
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=0
# 3) 打印次数不限（整轮只有 ~1024 行）
export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=0

# 实验标签，便于在 logs/ 与 wandb 目录下区分本组
export MY_ENV_DSV41_EXPERIMENT_TAG="probe8-cp1"
export MY_ENV_DSV41_RUN_TAG="probe8-cp1-math-dapo17k"

echo "[probe8] MY_DEBUG_LOG E1-8机 分段探针启动 probe=1 sync=0 calls=0 base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
