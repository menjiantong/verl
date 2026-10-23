#!/usr/bin/env bash
# 监控 80.5.25.119 上的 DSV4.1 GRPO 长跑。
#
# 设计思路（工作准则 3/4/7）：
#   1. 119 的 SSH 50002 当前拒绝连接，无法在容器内执行 ps/npu-smi；
#      但 /mnt/share 是共享 NFS，outer log 仍在更新，所以用
#      “日志 mtime + 是否出现新 step” 判断训练进程是否存活。
#   2. 只读日志，不杀进程、不重启；重启由上层 agent 在确认是“报错”后执行。
#   3. 所有可调参数集中在脚本内（准则 7.1），新增变量统一 MY_ENV_ 前缀（准则 6.1）。
#
# 退出码：0=ALIVE/INIT，2=STALLED（可能被杀或挂起），3=ERROR（日志有异常），4=NO_LOG。
set -euo pipefail

MY_ENV_WORK_DIR=/mnt/share/m00899630/workspace/dsv41-verl/verl
MY_ENV_LOG_GLOB='logs/W2_longrun_*.outer.log'
MY_ENV_STATE_FILE="$MY_ENV_WORK_DIR/logs/monitor_119_longrun.state"
MY_ENV_STATUS_FILE="$MY_ENV_WORK_DIR/logs/monitor_119_longrun.status"
# 20 分钟没有新 step 视为停滞。当前实测单 step 453-631 s，20 min 大于 2 个 step。
MY_ENV_STALL_SEC=1200
MY_ENV_PYTHON=/usr/local/python3.11.15/bin/python3

cd "$MY_ENV_WORK_DIR"
LOG=$(ls -t $MY_ENV_LOG_GLOB 2>/dev/null | head -1 || true)
if [ -z "${LOG:-}" ]; then
  echo "[monitor] NO_LOG: no $MY_ENV_LOG_GLOB under $MY_ENV_WORK_DIR"
  exit 4
fi

"$MY_ENV_PYTHON" - "$LOG" "$MY_ENV_STATE_FILE" "$MY_ENV_STATUS_FILE" "$MY_ENV_STALL_SEC" <<'PY'
import os, re, sys, time

log, state_path, status_path, stall = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# 日志里有 ANSI 颜色码，先去掉再匹配
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
# 真正的错误模式；Killed/OutOfMemory/HCCL error 都要算
# \berror\b 避免把 HCCL_ASYNC_ERROR_HANDLING 这种配置名误判成 HCCL 错误
err_re = re.compile(r'Traceback|RuntimeError|OutOfMemory|Killed|Config_Error|HCCL[^\n]*\berror\b|Exception|NPU error|CUDA error', re.I)
# 已知无害：缺失可选引擎、triton 弃用告警、warnings.warn、HCCL 配置名、atexit 退出噪声
benign = re.compile(r'UserWarning|ModuleNotFoundError|torchtitan engine is not available|veomni engine is not available|nemo_automodel|megatron engine is not available|warnings\.warn|triton/language|HCCL_ASYNC_ERROR_HANDLING|Exception ignored')

steps = []
errs = 0
last_err = ''
for raw in open(log, errors='ignore'):
    line = ansi.sub('', raw)
    if 'training/global_step:' in line and ' - ' in line:
        d = {}
        for kv in line.split(' - '):
            if ':' in kv:
                k, v = kv.strip().split(':', 1)
                d[k] = v
        steps.append(d)
    if err_re.search(line) and not benign.search(line):
        errs += 1
        last_err = line.strip()[:300]

last = steps[-1] if steps else {}
def g(k):
    return last.get(k, 'na')

# 读取上一次检查的 step，用于区分“正在跑”和“卡住”
prev_step = -1
if os.path.exists(state_path):
    for ln in open(state_path):
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            if k == 'last_step':
                prev_step = int(v)

step = int(float(g('training/global_step'))) if g('training/global_step') != 'na' else -1
mtime = os.stat(log).st_mtime
now = time.time()
# 119 与 120 有约 3 分钟时钟偏差，mtime 可能在未来；负数按 0 处理
age = max(0, int(now - mtime))

if errs:
    status = 'ERROR'
elif step < 0:
    status = 'INIT' if age <= stall else 'STALLED'
elif step > prev_step:
    status = 'ALIVE'
elif age > stall:
    status = 'STALLED'
else:
    status = 'ALIVE'

ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
print(f"[monitor] {ts} UTC status={status}")
print(f"[monitor] log={log} age={age}s steps={len(steps)} last_step={step} prev_step={prev_step}")
if steps:
    print("[monitor] reward={} diff_mean={} pearson={} hbm_alloc_gb={} hbm_reserved_gb={} cpu_mem_gb={} step_s={} resp_len={}".format(
        g('critic/rewards/mean'), g('training/rollout_probs_diff_mean'),
        g('training/rollout_actor_probs_pearson_corr'),
        g('actor/perf/max_memory_allocated_gb'), g('actor/perf/max_memory_reserved_gb'),
        g('actor/perf/cpu_memory_used_gb'), g('timing_s/step'), g('response_length/mean')))
print(f"[monitor] real_errors={errs}" + (f" last={last_err}" if last_err else ''))

with open(state_path, 'w') as f:
    f.write(f"last_step={step}\nlast_check={int(now)}\n")
with open(status_path, 'a') as f:
    f.write(f"{ts} status={status} step={step} age={age}s errors={errs}\n")

sys.exit({'ALIVE': 0, 'INIT': 0, 'STALLED': 2, 'ERROR': 3}.get(status, 1))
PY
