#!/usr/bin/env bash
# =============================================================================
# 1102 DSV4.1 CP「单训练」验证 v1
#
# 目的：绕开 verl 的 RL 数据管线（vLLM / Ray / transfer_queue），只验证
#       FSDPTurbo 自己训练路径下 CP 的前向+反向是否正确、是否等价、是否更快。
#
# 用法：
#   bash my_scripts/1102_run_dsv41_cp_train_validate_v1.sh cp1
#   bash my_scripts/1102_run_dsv41_cp_train_validate_v1.sh cp2
#
# 两个闸位共用同一份配置，唯一差异是 distributed.kv_allgather_parallel_size。
# cp2 的配置由 cp1 现场 sed 生成，避免两份文件漂移。
# =============================================================================
# 验证通过的 commit（2026-11-02）：
#   FSDPTurbo 3ff111b  单训练 CP 验证配置（4 层、随机初始化、关 fused 算子）
#   FSDPTurbo 8703d04  backbone gather 环境变量开关（默认关，单训练用局部输出）
#   verl 54fa7a47      cp1/cp2 闸位启动器
#   verl ce992efa      CP>1 时导出 MY_ENV_DSV41_CP_GATHER_OUTPUT=1
# 证据：logs/1102_cp_train_cp1_*.log / cp2_*.log（均 RC=0，step1 loss 相对误差 2.0e-5）
set -eo pipefail

GATE=${1:?usage: $0 <cp1|cp2>}
case "$GATE" in
  cp1) CP=1 ;;
  cp2) CP=2 ;;
  *) echo "unknown gate $GATE (expect cp1|cp2)" >&2; exit 2 ;;
esac

MIRROR=/mnt/share/m00899630/workspace/dsv41-verl-rebase
REPO=$MIRROR/verl
EX=$MIRROR/FSDPTurbo/examples/deepseek_v41
cd "$REPO"
# 只取 CANN/PYTHONPATH/缓存目录等环境，本脚本不启动 Ray（被 source 时该脚本只导出变量）
# shellcheck disable=SC1091
source my_scripts/1030_ray_dsv41_env.sh

BASE_CFG=$EX/config_16p_cp_validate.yaml
CFG=$EX/config_16p_cp_validate_${GATE}.yaml
sed "s/^  kv_allgather_parallel_size: .*/  kv_allgather_parallel_size: ${CP}/" "$BASE_CFG" > "$CFG"
echo "[cp-train] gate=$GATE cp=$CP cfg=$CFG"

OUT=$MIRROR/verl/logs
mkdir -p "$OUT"
LOG=$OUT/1102_cp_train_${GATE}_$(date +%Y%m%d_%H%M%S).log
echo "[cp-train] log=$LOG"

cd "$EX"
export PYTHONPATH="$MIRROR/FSDPTurbo:$MIRROR/vllm:$MIRROR/vllm-ascend:$MIRROR/verl:${PYTHONPATH:-}"
torchrun \
    --nnodes=1 \
    --nproc_per_node=16 \
    --master_addr=127.0.0.1 \
    --master_port=59501 \
    train.py --config "$CFG" 2>&1 | tee "$LOG"
