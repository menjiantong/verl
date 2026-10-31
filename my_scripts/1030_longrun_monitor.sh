#!/bin/bash
# [1030 长跑监控] 每 10 分钟采样一次：最近一步指标、NPU HBM、主机内存，追加到 wiki 下的 CSV。
# 规则 3.2：长跑必须记录显存/内存并与预估对比；本脚本只做采样，不做任何训练干预。
# 用法: setsid -f bash my_scripts/1030_longrun_monitor.sh
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
OUT=$REPO/wiki/longrun_monitor_20261031_dsv41_500step.csv
INTERVAL=${MY_ENV_MONITOR_INTERVAL:-600}
[ -f "$OUT" ] || echo "time,log,step,probs_diff,pearson,train_ppl,reward,step_time_s,hbm_max_mib,host_used_gib,host_avail_gib,main_ppo_alive" > "$OUT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
while true; do
    cd "$REPO" || exit 1
    LOG=$(ls -t logs/dsv41_grpo_longrun_500steps_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        line=$(grep "training/rollout_probs_diff_mean" "$LOG" 2>/dev/null | tail -1)
        step=$(printf '%s' "$line" | grep -o 'training/global_step:[0-9]*' | tail -1 | cut -d: -f2)
        get() { printf '%s' "$line" | grep -o "$1:[0-9.eE+-]*" | tail -1 | cut -d: -f2; }
        probs=$(get 'training/rollout_probs_diff_mean')
        pearson=$(get 'training/rollout_actor_probs_pearson_corr')
        ppl=$(get 'rollout_corr/training_ppl')
        reward=$(get 'critic/score/mean')
        stept=$(get 'timing_s/step')
    fi
    # [1031 换容器] 26.2.rc1 的 npu-smi 输出为 "3158 / 65536"（斜杠前有空格），兼容两种格式。
    hbm=$(npu-smi info 2>/dev/null | grep -o '[0-9]\+ */ *65536' | cut -d/ -f1 | tr -d ' ' | sort -n | tail -1)
    host=$(free -g | awk '/^Mem:/{print $3","$7}')
    alive=$(ps -eo cmd | grep -c "[m]ain_ppo")
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date '+%F %T')" "${LOG##*/}" "${step:-}" "${probs:-}" "${pearson:-}" "${ppl:-}" \
        "${reward:-}" "${stept:-}" "${hbm:-}" "${host:-}" "${alive}" >> "$OUT"
    sleep "$INTERVAL"
done
