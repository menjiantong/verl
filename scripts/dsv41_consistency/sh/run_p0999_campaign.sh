#!/bin/bash
# Run the whole p0999 A/B series with retries.
#
# 2026-09-27: another tenant holds ~20-37 GB on every card, so a run sometimes dies with
#   torch.OutOfMemoryError ... Tried to allocate 6.33 GiB   (FSDP2 foreach_reduce in the
#   actor backward) or with the engine's startup free-memory check. Both are contention, not
#   config errors, and the needed headroom (~40.4 GB) does come back. So: check the free HBM
#   first, and retry each variant until it produces the metric.
cd /workspace-verl/verl
RES=/tmp/p0999/campaign_results.txt
NEED_GB=${NEED_GB:-40}     # our measured peak is 40.4 GB allocated on the busiest rank
MAX_TRIES=${MAX_TRIES:-6}

free_hbm_gb() {  # min over the visible chips of (65536 - used) MB, in GB
    # npu-smi prints the usage either as "12345/ 65536" or as " 1234 / 65536" depending on width,
    # so match the optional whitespace on both sides.
    npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" \
        | awk '{gsub(/ */,"");split($0,a,"/");print (65536-a[1])/1024}' | sort -n | head -1
}

wait_for_memory() {
    for _ in $(seq 1 40); do
        free=$(free_hbm_gb)
        [ -n "$free" ] && awk -v f="$free" -v n="$NEED_GB" 'BEGIN{exit !(f>=n)}' && return 0
        echo "[campaign] waiting for HBM: free=${free:-?}GB need=${NEED_GB}GB $(date +%H:%M:%S)"
        sleep 60
    done
    return 1
}

run_one() {  # tag align nobi
    local tag="$1" align="$2" nobi="${3:-0}"
    for try in $(seq 1 "$MAX_TRIES"); do
        bash /tmp/p0999/cleanup.sh
        wait_for_memory || echo "[campaign] memory never freed up; trying anyway"
        echo "--- START ${tag} try=${try} align='${align}' nobi=${nobi} $(date +%H:%M:%S) ---" >> "$RES"
        TAG="$tag" ALIGN="$align" VERL_DSV41_ALIGN_NOBI="$nobi" \
            bash scripts/dsv41_consistency/sh/run_p0999_ab.sh \
            > "/tmp/p0999/run_${tag}.out" 2>&1
        local m
        m=$(grep -o "rollout_actor_probs_pearson_corr:[0-9.]*" "logs/p0999-${tag}.log" 2>/dev/null | tr '\n' ' ')
        if [ -n "$m" ]; then
            echo "--- OK ${tag} try=${try} pearson=[${m}] $(date +%H:%M:%S) ---" >> "$RES"
            return 0
        fi
        echo "--- RETRY ${tag} try=${try} (no metric) $(date +%H:%M:%S) ---" >> "$RES"
        tail -3 "/tmp/p0999/run_${tag}.out" | grep -i "memory\|killed" >> "$RES" 2>/dev/null
        sleep 30
    done
    echo "--- FAIL ${tag} after ${MAX_TRIES} tries $(date +%H:%M:%S) ---" >> "$RES"
}

echo "=== campaign start $(date) ===" >> "$RES"
run_one all2    "attn,router,moe,head"
run_one am      "attn,router,moe"
echo "=== campaign done $(date) ===" >> "$RES"
bash /tmp/p0999/cleanup.sh