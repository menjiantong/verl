#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 云道 8 机 全尺寸（384 专家）Engram + CP 验证启动器 v1
#
# 基线（rule 7.3/7.4：本脚本只写差异，其余全部沿用基线）：
#   my_scripts/yundao/dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
#   8 机 × 16 卡 = 128 rank，FSDP128/EP128，rollout TP8×DP16（GEN_EP=128），
#   Math DAPO-17k 纯文本，prompt 384 / response 1536，ffn+attn 重算。
#
# 与基线的 5 处差异：
#   1. 权重换成带 Engram 表的全尺寸基座 /efs_rl/weights/DeepSeek-V4.1-Flash-bf16：
#      text_config.engram_layer_ids=[1,14]，两层表各 [384006168,256] /
#      [384016682,256] bf16 = 196.6 GB，合计 393.2 GB。
#   2. CP（KV-AllGather）= 2：example 第 284 行透传
#      +distributed.kv_allgather_parallel_size=2；配套
#      MY_ENV_DSV41_CP_PAD_MULTIPLE=4（本模型 max compress_ratio=2，cp=2 时
#      稠密总长必须是 4 的倍数，transformer_impl._cp_pad_jagged 逐样本补齐）、
#      KEEP_TRAIN_MODE=1、GATHER_OUTPUT=1。
#   3. vLLM 侧 Engram 表节点内 DP 共享：VLLM_ENGRAM_DP_SHARED_MEMORY=1。
#      8 机 GEN_DP=16 → 每节点 2 个 replica，upstream _engram_dp_shard_size
#      在 8 节点 × 每节点 2 replica 下返回 gcd(16,2)=2，构成节点内共享组，
#      每节点只保留一份 TP 分片表（393 GB / 8 卡 = 49 GB 每卡，落在 /dev/shm）。
#   4. sleep 备份用 pageable 主机内存：MY_ENV_VLLM_SLEEP_PIN_MEMORY=0
#      （engram 共享表常驻后 rtMallocHost 会报 207001）。
#   5. 验证短跑：STEPS=10、SAVE_FREQ=-1（全模型含 engram 单份 ckpt 约 1.4 TB，
#      验证阶段不落盘；需要存权重时把 SAVE_FREQ 改成正数）。
#
# 前置代码修复（已在工作目录提交，见 worklog）：
#   vllm-ascend 5a1ce4bfa  hash kernel 负地址钳 0（源自 25.x 69de6ca8c）
#   vllm-ascend 95fbfc99f  camem sleep 备份 pageable 开关（源自 25.x 633c48011）
#   已有：FSDPTurbo CP d84d257/03c3c1d/8703d04、verl CP 响应切片（padding.py
#   content_ends）、vllm engram 主机注册 fd46ca5ada、verl 环境开关 9575baaa。
#
# 用法：云道“启动命令”只填这一行（8 机作业，每节点都会执行本脚本）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1103_run_dsv41_yundao_8node_engram_cp_v1.sh
#
# 可用环境变量覆盖（不设置则用下面的默认值）：
#   MY_ENV_DSV41_STEPS        训练步数，默认 10
#   MY_ENV_DSV41_SAVE_FREQ    保存频率，默认 -1（不存）
#   MY_ENV_DSV41_CP_SIZE      CP 大小，默认 2（1=关）
#   MY_ENV_DSV41_MIN_MEM_GB   单节点最低 MemAvailable，默认 1600
# =============================================================================
set -eo pipefail

BASE=/efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh
ENGRAM_MODEL=/efs_rl/weights/DeepSeek-V4.1-Flash-bf16

STEPS=${MY_ENV_DSV41_STEPS:-10}
SAVE_FREQ=${MY_ENV_DSV41_SAVE_FREQ:--1}
CP_SIZE=${MY_ENV_DSV41_CP_SIZE:-2}
MIN_MEM_GB=${MY_ENV_DSV41_MIN_MEM_GB:-1600}

# -----------------------------------------------------------------------------
# 1. 传给基线的差异（rule 7.1：全部写在本脚本内，不在命令行 export）
# -----------------------------------------------------------------------------
export MY_ENV_DSV41_MODEL_PATH="$ENGRAM_MODEL"
export MY_ENV_DSV41_EXPERIMENT_TAG="engram-cp${CP_SIZE}"
export MY_ENV_DSV41_STEPS="$STEPS"
export MY_ENV_DSV41_SAVE_FREQ="$SAVE_FREQ"
# Engram DP 共享表要求真正从 checkpoint 读权重；example 在
# VLLM_ENGRAM_DP_SHARED_MEMORY=1 时会追加 load_format=safetensors 与
# 多线程 lazy 读盘（example 第 207-225 行），这里显式对齐，避免基线
# 末尾的 load_format=auto 覆盖掉它。
export MY_ENV_DSV41_LOAD_FORMAT=safetensors
export MY_ENV_DSV41_ENGRAM_DP_SHARED=1

# CP：example 读 CP_SIZE；三个 MY_ENV_ 由 verl 内部消费。
export MY_ENV_DSV41_CP_SIZE="$CP_SIZE"
if [ "$CP_SIZE" -gt 1 ]; then
  export MY_ENV_DSV41_CP_PAD_MULTIPLE=4
  export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1
  export MY_ENV_DSV41_CP_GATHER_OUTPUT=1
else
  export MY_ENV_DSV41_CP_PAD_MULTIPLE=1
  export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=0
  export MY_ENV_DSV41_CP_GATHER_OUTPUT=0
fi

# sleep 备份用 pageable 主机内存（camem 补丁读取该变量）。
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=0

# -----------------------------------------------------------------------------
# 2. 权重预检：必须是带 engram 表的全尺寸基座（rule 4.1：结论有证据）
# -----------------------------------------------------------------------------
python3 - "$ENGRAM_MODEL" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cfg = json.loads((root / "config.json").read_text())
text = cfg.get("text_config", cfg)
ids = text.get("engram_layer_ids") or []
nexp = text.get("n_routed_experts")
assert ids, f"{root}: engram_layer_ids empty, not an engram checkpoint"
assert nexp == 384, f"{root}: n_routed_experts={nexp}, need the full 384-expert checkpoint"
w = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
tables = [k for k in w if "engram" in k and k.endswith("embed.weight")]
assert len(tables) == len(ids), f"{root}: engram tables {tables} vs ids {ids}"
print(f"[engram-cp] MY_DEBUG_LOG model_check_ok root={root} layers={text.get('num_hidden_layers')} "
      f"experts={nexp} engram_layers={ids} engram_tables={len(tables)}")
PY

# -----------------------------------------------------------------------------
# 3. 主机内存 / /dev/shm 预检（rule 3.2；每个节点都会执行）
#   预算（单节点）：基线 noengram 8 机实测 cpu_memory_used ≈1049 GB（10/31 日志）
#   + vLLM 共享 engram 表 393 GB + 训练侧 engram 表 EP128 分片 49 GB ≈ 1.49 TB，
#   节点总量 2.01 TB，故要求 MemAvailable ≥ 1600 GB。
# -----------------------------------------------------------------------------
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
SHM_AVAIL_GB=$(df -Pk /dev/shm | awk 'NR==2{print int($4/1024/1024)}')
echo "[engram-cp] MY_DEBUG_LOG host_mem_check MemAvailable=${AVAIL_GB}GiB shm_avail=${SHM_AVAIL_GB}GiB need_mem=${MIN_MEM_GB}GiB"
if [ "$AVAIL_GB" -lt "$MIN_MEM_GB" ]; then
  echo "[engram-cp] ABORT: MemAvailable=${AVAIL_GB}GiB < ${MIN_MEM_GB}GiB" >&2
  exit 3
fi
# 单份共享表 393 GB，留 15% 余量。
if [ "$SHM_AVAIL_GB" -lt 450 ]; then
  echo "[engram-cp] ABORT: /dev/shm avail=${SHM_AVAIL_GB}GiB < 450GiB (engram shared table needs ~393GiB)" >&2
  exit 3
fi

# 存 checkpoint 时才有磁盘门槛：全模型含 engram 单份约 1.4 TB。
if [ "$SAVE_FREQ" -ge 0 ]; then
  FREE_T=$(df -Pk /efs_rl | awk 'NR==2{printf "%.1f", $4/1024/1024/1024}')
  echo "[engram-cp] MY_DEBUG_LOG ckpt_space_check free=${FREE_T}T need=1.5T"
  awk -v f="$FREE_T" 'BEGIN{exit !(f < 1.5)}' && {
    echo "[engram-cp] ABORT: /efs_rl free ${FREE_T}T < 1.5T for one engram checkpoint" >&2
    exit 3
  }
fi

echo "[engram-cp] MY_DEBUG_LOG engram_cp_config steps=${STEPS} save_freq=${SAVE_FREQ} cp=${CP_SIZE} model=${ENGRAM_MODEL} dp_shared=1 load_format=safetensors"

# -----------------------------------------------------------------------------
# 4. 交给基线脚本（Ray 启动 / ModelArts 节点角色 / 训练参数全部在基线里）
# -----------------------------------------------------------------------------
exec bash "$BASE"
