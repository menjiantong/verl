#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-06 单机 16 卡，抓 update_actor 窗口的 Python 栈。
# =============================================================================
# 目的：本地 engram 让 update_actor 贵 19–22%（≈16 s/步），但静态分析已排除
#   fetch（复用补丁无效）、FSDP all-gather（显存只差 0.59 GB）、内存压力、
#   梯度裁剪、优化器状态。剩下只能采样看栈。
#
# 做法：跑 engram arm；轮询日志里的 `use_reentrant` 警告（torch.utils.checkpoint 第一次
#   被调用 = update_actor 开始），然后对最大的 WorkerDict python 进程连续 py-spy dump。
#
# 环境变量（rule 6.1）：
#   MY_ENV_DSV41_PROF_DUMPS=12    抓栈次数
#   MY_ENV_DSV41_PROF_INT=2       每次间隔（秒）
#   MY_ENV_DSV41_PROF_DATASET=dapo512|countbenchqa
# =============================================================================
set -u
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO" || exit 1

DUMPS=${MY_ENV_DSV41_PROF_DUMPS:-12}
INT=${MY_ENV_DSV41_PROF_INT:-2}
DS=${MY_ENV_DSV41_PROF_DATASET:-dapo512}
TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/1106_prof_update_actor_${TS}

echo "[prof] MY_DEBUG_LOG prof_start dumps=$DUMPS int=$INT dataset=$DS out=$OUT"

# 先起训练（后台）
MY_ENV_DSV41_AB_ARM=engram \
MY_ENV_DSV41_AB_PROBE_CALLS=0 \
MY_ENV_DSV41_AB_DATASET="$DS" \
  bash my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh > "${OUT}_train.log" 2>&1 &
TRAIN_PID=$!

# 等 update_actor 开始：checkpoint 的 use_reentrant 警告
LOG=""
for i in $(seq 1 400); do
  LOG=$(ls -t logs/1106_nosleep_1node_engram_2*.log 2>/dev/null | grep -v mem | head -1)
  if [ -n "$LOG" ] && grep -aq "use_reentrant" "$LOG" 2>/dev/null; then
    echo "[prof] MY_DEBUG_LOG prof_window_open time=$(date +%T) log=$LOG"
    break
  fi
  sleep 5
done

# 选 RSS 最大的 Ray actor worker（训练在 WorkerDict 里，driver 是 main_ppo，要排除）
ps -eo pid,rss,args --sort=-rss | grep -aE "ray::|WorkerDict" | grep -av grep | head -5 > "${OUT}_candidates.txt"
PID=$(head -1 "${OUT}_candidates.txt" | awk '{print $1}')
echo "[prof] MY_DEBUG_LOG prof_target pid=${PID:-none} time=$(date +%T)"
cat "${OUT}_candidates.txt" | cut -c1-160 >> "${OUT}_dumps.txt"

if [ -n "${PID:-}" ]; then
  for k in $(seq 1 "$DUMPS"); do
    echo "===== dump $k $(date +%T) =====" >> "${OUT}_dumps.txt"
    timeout 30 py-spy dump --pid "$PID" >> "${OUT}_dumps.txt" 2>&1 || echo "dump_failed" >> "${OUT}_dumps.txt"
    sleep "$INT"
  done
fi

wait $TRAIN_PID
echo "[prof] MY_DEBUG_LOG prof_done rc=$? time=$(date +%T) out=$OUT"
