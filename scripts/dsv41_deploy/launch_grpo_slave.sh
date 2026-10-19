#!/usr/bin/env bash
# 一键启动：GRPO (RL) 双机 —— slave 侧（第二台机器）。
# 用法（在 slave 节点）:
#     bash launch_grpo_slave.sh <MASTER_IP>       # 例如 80.5.25.119
# 时机：必须在 master 日志出现 "[2node] waiting for the slave" 之后执行。
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKDIR=${WORKDIR:-/workspace-verl}
MASTER_IP=${1:?用法: bash launch_grpo_slave.sh <MASTER_IP>}

ulimit -n 65536 2>/dev/null || true
export PATH=/usr/local/python3.11.15/bin:$PATH
# CANN 自举（F-51：非 login shell 下缺 LD_LIBRARY_PATH → import/npu-smi 全挂）
if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  for f in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/cann-9.1.0/set_env.sh; do
    [ -f "$f" ] && source "$f" >/dev/null 2>&1 || true
  done
fi

# ⚠ 需要与 master 一致的环境开关必须在 slave 侧也导出（worker 继承本机 raylet 环境）：
export VA_V41_OPROJ_FP32_AR=${VA_V41_OPROJ_FP32_AR:-1}

# 需要固定卡位时自行 export ASCEND_RT_VISIBLE_DEVICES=...；留空则自动挑前 8 张空闲卡。
LOG=${LOG:-/tmp/grpo_2node_slave_$(date +%Y%m%d_%H%M%S).log}

cd "$WORKDIR"
[ -f "$SCRIPT_DIR/run_grpo_2node.sh" ] || { echo "缺 run_grpo_2node.sh（应与本脚本同目录）"; exit 1; }

nohup bash "$SCRIPT_DIR/run_grpo_2node.sh" slave "$MASTER_IP" > "$LOG" 2>&1 &
echo "[launch] GRPO slave pid=$! master=$MASTER_IP"
echo "[launch] log: $LOG   （出现 'slave ready' 即就绪；若加入失败见失败库 F-63）"
