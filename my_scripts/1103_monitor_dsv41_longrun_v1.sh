#!/usr/bin/env bash
# =============================================================================
# 1103 DSV4.1 全功能长跑监控 v1（CP=2 + Engram）
#
# 规则 3.2：长跑必须记录内存/HBM 并与预估对比；本脚本只采样，不干预训练。
# 与 1030_longrun_monitor.sh（v1，W3 500 步）差异：
#   * 识别 1103_run_dsv41_longrun_v1.sh 产出的 logs/1103_longrun_*.log；
#   * 追加 rollout_corr/kl、rollout_ppl、response_length/mean、actor HBM；
#   * NPU 采样兼容 npu-smi 26.0.rc1 的 "3158 / 65536" 与 "3158/65536" 两种格式。
#
# 用法：setsid -f bash my_scripts/1103_monitor_dsv41_longrun_v1.sh
#   MY_ENV_MONITOR_INTERVAL 默认 600 秒
# 输出：wiki/longrun_monitor_20261103_dsv41_cp2_engram.csv（追加写）
# 验证 commit：verl f72316da（长跑脚本基线）
# =============================================================================
set -o pipefail
REPO=/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
OUT=$REPO/wiki/longrun_monitor_20261103_dsv41_cp2_engram.csv
INTERVAL=${MY_ENV_MONITOR_INTERVAL:-600}
[ -f "$OUT" ] || echo "time,log,step,probs_diff_mean,pearson,kl,train_ppl,rollout_ppl,reward,resp_len_mean,step_time_s,hbm_max_mib,host_used_gib,host_avail_gib,runner_alive" > "$OUT"
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
while true; do
    cd "$REPO" || exit 1
    LOG=$(ls -t logs/1103_longrun_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        line=$(grep "training/rollout_probs_diff_mean" "$LOG" 2>/dev/null | tail -1)
        step=$(printf '%s' "$line" | grep -o 'training/global_step:[0-9]*' | tail -1 | cut -d: -f2)
        get() { printf '%s' "$line" | grep -o "$1:[0-9.eE+-]*" | tail -1 | cut -d: -f2; }
        probs=$(get 'training/rollout_probs_diff_mean')
        pearson=$(get 'training/rollout_actor_probs_pearson_corr')
        kl=$(get 'rollout_corr/kl')
        tppl=$(get 'rollout_corr/training_ppl')
        rppl=$(get 'rollout_corr/rollout_ppl')
        reward=$(get 'critic/score/mean')
        rlen=$(get 'response_length/mean')
        stept=$(get 'timing_s/step')
    fi
    hbm=$(npu-smi info 2>/dev/null | grep -o '[0-9]\+ */ *65536' | cut -d/ -f1 | tr -d ' ' | sort -n | tail -1)
    host=$(free -g | awk '/^Mem:/{print $3","$7}')
    alive=$(pgrep -fc "[T]askRunnerV1" 2>/dev/null || true); alive=${alive:-0}
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date '+%F %T')" "${LOG##*/}" "${step:-}" "${probs:-}" "${pearson:-}" "${kl:-}" \
        "${tppl:-}" "${rppl:-}" "${reward:-}" "${rlen:-}" "${stept:-}" "${hbm:-}" "${host:-}" "${alive:-}" >> "$OUT"
    sleep "$INTERVAL"
done
