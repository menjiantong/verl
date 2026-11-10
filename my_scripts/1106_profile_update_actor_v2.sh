#!/usr/bin/env bash
# [验证结果] ⏳ 待验证：2026-11-06 单机 16 卡，抓 update_actor 窗口的 Python 栈（v2）。
# =============================================================================
# v1 的三个错误（已修）：
#   1. 没 source 1030_ray_dsv41_env.sh → 找不到 py-spy（在 /usr/local/python3.11.15/bin）
#   2. 用 `ls -t ... | head -1` 会拿到**上一轮**的日志（新日志还没生成），于是立刻开始 dump
#   3. 选进程按 rayr:: 最大 RSS，抓到了 DashboardAgent
#
# 目的：本地 engram 让 update_actor 贵 19–22%（≈16 s/步），但 fetch 复用补丁无收益、
#   FSDP all-gather / 内存压力 / 梯度裁剪 / 优化器状态都已排除，只能采样看栈。
# =============================================================================
set -u
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
cd "$REPO" || exit 1
# [prof2 fix] 不 source 环境脚本（在 setsid 脱离的上下文里会让脚本静默退出），只补 PATH。
export PATH=/usr/local/python3.11.15/bin:$PATH

DUMPS=${MY_ENV_DSV41_PROF_DUMPS:-15}
INT=${MY_ENV_DSV41_PROF_INT:-2}
DS=${MY_ENV_DSV41_PROF_DATASET:-dapo512}
TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/1106_prof2_${TS}
START_EPOCH=$(date +%s)

echo "[prof2] MY_DEBUG_LOG prof2_start dumps=$DUMPS int=$INT dataset=$DS out=$OUT pyspy=$(command -v py-spy)"

MY_ENV_DSV41_AB_ARM=engram \
MY_ENV_DSV41_AB_PROBE_CALLS=0 \
MY_ENV_DSV41_AB_DATASET="$DS" \
  bash my_scripts/1106_run_dsv41_1node_engram_nosleep_v1.sh > "${OUT}_train.log" 2>&1 &
TRAIN_PID=$!

# 只认**本次启动之后创建**的日志（v1 的 bug 就出在这里）
LOG=""
for i in $(seq 1 400); do
  for f in $(ls -t logs/1106_nosleep_1node_engram_2*.log 2>/dev/null | grep -v mem); do
    if [ "$(stat -c %Y "$f")" -ge "$START_EPOCH" ]; then LOG="$f"; break; fi
  done
  [ -n "$LOG" ] && break
  sleep 3
done
echo "[prof2] MY_DEBUG_LOG prof2_log log=${LOG:-none} time=$(date +%T)"

# 等 update_actor 开始（checkpoint 第一次调用）
for i in $(seq 1 400); do
  if [ -n "$LOG" ] && grep -aq "use_reentrant" "$LOG" 2>/dev/null; then
    echo "[prof2] MY_DEBUG_LOG prof2_window_open time=$(date +%T)"
    break
  fi
  sleep 3
done

# 训练 worker 是 ray::WorkerDict，取 RSS 最大的那个
ps -eo pid,rss,args --sort=-rss | grep -a "ray::WorkerDict" | grep -av grep | head -5 > "${OUT}_candidates.txt"
PID=$(head -1 "${OUT}_candidates.txt" | awk '{print $1}')
echo "[prof2] MY_DEBUG_LOG prof2_target pid=${PID:-none} time=$(date +%T)"
cat "${OUT}_candidates.txt" | cut -c1-150 >> "${OUT}_dumps.txt"

if [ -n "${PID:-}" ]; then
  for k in $(seq 1 "$DUMPS"); do
    echo "===== dump $k $(date +%T) =====" >> "${OUT}_dumps.txt"
    timeout 30 py-spy dump --pid "$PID" >> "${OUT}_dumps.txt" 2>&1 || echo "dump_failed" >> "${OUT}_dumps.txt"
    sleep "$INT"
  done
fi

wait $TRAIN_PID
echo "[prof2] MY_DEBUG_LOG prof2_done rc=$? time=$(date +%T) out=$OUT"
