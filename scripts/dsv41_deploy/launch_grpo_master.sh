#!/usr/bin/env bash
# 一键启动：GRPO (RL) 双机 —— master 侧（司机节点）。
# 用法（在 master 节点）:
#     bash launch_grpo_master.sh [STEPS]        # 默认 5 步；冒烟用 1
# 之后在 slave 节点执行:
#     bash launch_grpo_slave.sh <MASTER_IP>
# 所有环境变量均可从外部覆盖（见 README.md §2 速查表）。日志默认 /tmp/grpo_2node_master_<stamp>.log。
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKDIR=${WORKDIR:-/workspace-verl}
STEPS=${1:-${STEPS:-5}}

ulimit -n 65536 2>/dev/null || true          # F-64：fd 软限 1024 会让 head raylet 卡死
export PATH=/usr/local/python3.11.15/bin:$PATH
# CANN 自举（F-51：非 login shell 下缺 LD_LIBRARY_PATH → import/npu-smi 全挂）
if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  for f in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/cann-9.1.0/set_env.sh; do
    [ -f "$f" ] && source "$f" >/dev/null 2>&1 || true
  done
fi

# ---- 数据 / 权重（默认 = 已验证的 SFT 热启动权重 + GSM8K RL 数据）----
export TRAIN_FILE=${TRAIN_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_train_rl.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_test_rl.parquet}
# 注意：不要用原始 top32-bf16 权重跑 RL（never-EOS，F-31）；用 SFT 热启动后的 sft116。
export MODEL_PATH=${MODEL_PATH:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116}

# ---- RL 侧关键开关（proven 配置，改动前先读失败库）----
export EFSDP_SIZE=${EFSDP_SIZE:-1}            # ⚠ RL 必须 =1（EFSDP=2 会静默损坏权重同步，F-30）。
                                              #   run_grpo_2node.sh 内的历史默认是 2，本启动器显式覆盖。
export ROLLOUT_IS=${ROLLOUT_IS:-token}        # TIS（截断重要性采样），threshold 2.0；置 null 关闭。
export VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-1}   # 引擎 o-proj 段 fp32 all-reduce（一致性优化，建议保持 1）。

# 需要固定卡位时（如避开邻居占用的 die）自行 export ASCEND_RT_VISIBLE_DEVICES=0,1,...；留空则自动挑前 8 张空闲卡。
LOG=${LOG:-/tmp/grpo_2node_master_$(date +%Y%m%d_%H%M%S).log}

cd "$WORKDIR"
[ -f "$SCRIPT_DIR/run_grpo_2node.sh" ] || { echo "缺 run_grpo_2node.sh（应与本脚本同目录）"; exit 1; }

nohup bash "$SCRIPT_DIR/run_grpo_2node.sh" master "$STEPS" > "$LOG" 2>&1 &
echo "[launch] GRPO master pid=$! steps=$STEPS"
echo "[launch] log: $LOG   （tail -f 观察；出现 '[2node] waiting for the slave' 后再去从机）"
