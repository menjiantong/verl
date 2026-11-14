#!/usr/bin/env bash
# [验证结果] ✅ 成功：11/05 14:29 与 18:09 两轮跑通 2 步（STEPS=2、无 step0 验证）；11/04 14:49 直跑失败是主机内存wall，非本脚本。
# =============================================================================
# 1104 DSV4.1 云道 8 机 Engram+CP：整层重计算 + 关验证（快速验证 update 阶段）v2
#
# 基线（rule 7.3/7.4：本脚本只写差异）：
#   my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_v1.sh
#     → 1103 Engram+CP（8 机×16 卡=128 rank，FSDP128/EP128，CP2，engram 表节点内共享）
#       + recompute_plan=["model.layers.{*}"]（方案 B 整层重计算）
#
# 与基线的差异只有三条：
#   1) 关掉 step0 验证：VAL_BEFORE_TRAIN=False。11/04 10:22 那次 val 是 8 个 batch，
#      每批 8.5–11.4 分钟，合计约 75 分钟；跳过它，从启动到第一次 update 省约 75 分钟。
#   2) TEST_FREQ=-1：训练过程中不再插验证。
#   3) STEPS 默认 2：跑完 2 次 update 就正常退出，目的是确认"建优化器状态 → 更新 →
#      第二步"这条链能过。要改步数：MY_ENV_DSV41_STEPS=...
#
# 这次要验证的问题（11/04 10:22 运行失败）：
#   方案 B 已经解决了显存 OOM（反传 + clip_grads 全 128 worker 跑完），作业死在第一次
#   update 的 AdamW 状态创建：7.5 分钟里 128 个 rank 只有 6 个建完，单参数最长 46.47 s，
#   55 次 30 s 无进展，最后 Ray keepalive 超时判 actor 不可用。同代码在 11/01（无 engram）
#   是全部 rank ≤43 s 建完、单参数 >5 s 的 0 个。
#
# 关于 engram dp_shared_memory（本次已确认，不需要再开）：
#   它本来就是开着的。证据来自 11/04 10:22 的日志：
#     - vLLM 侧 EngramConfig(cpu_offload=True, embedding_across_dp=False,
#       dp_shared_memory=True, use_thp=False)
#     - "Engram table offloaded to registered host memory: 48002349 rows x 256,
#        22.89 GiB per rank"
#   语义见 vllm/vllm/config/engram.py：dp_shared_memory 就是"每个节点只存每份 TP 分片的
#   一个副本"；vllm-ascend/.../engram/embedding.py 里也写着 "One physical copy of the
#   mapped range"。所以当前 366 GiB/节点已经是共享后的数字，不共享会是 16×2×22.89≈733 GiB。
#   再往下只剩一个开关能省主机内存：embedding_across_dp=True（把表再按 16 个 DP 副本切，
#   全表从"每节点一份"变成"全集群一份"，每节点 366 → 约 46 GiB），代价是每次 engram
#   lookup 要走跨 DP（跨节点）collective，rollout 会变慢，且和 dp_shared_memory 互斥。
#   本次不动它：一次只改一个变量。
#
# 附带：主机内存探针（每 5 s 一次）
#   11/04 的失败怀疑是节点主机内存被 engram 顶满（预检 MemAvailable=1988 GiB，
#   engram 常驻约 366+46 GiB），但当时没有采样，只能算推断。这次把 /proc/meminfo 和
#   cgroup memory.current / memory.high 直接落到
#     logs/memprobe/<启动时间>/<主机名>_<pid>.log
#   下次卡住时可以直接判"是不是内存"。
#   关掉：MY_ENV_DSV41_MEM_PROBE=0
#
# 用法（云道"启动命令"只填这一行，8 机作业，每节点都会执行）：
#   bash /efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_noval_v2.sh \
#     |& tee /efs_rl/m00899630/workspace/dsv41-verl/verl/logs/dsv4-yundao_dapo_rebase_cp_engram_noval_$(date -u -d '+8 hours' +'%Y%m%d_%H%M%S').log
#
# 回滚：跑 1104_..._recompute_full_v1.sh（保留验证，步数默认 200）。
# =============================================================================

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE="${HERE}/1104_run_dsv41_yundao_8node_engram_cp_recompute_full_v1.sh"

if [ ! -f "$BASE" ]; then
  echo "[noval] ABORT: base wrapper not found: $BASE" >&2
  exit 3
fi

# 1) 关掉 step0 验证；2) 训练中不插验证；3) 只跑 2 步。
#    1103 里这三项分别由 MY_ENV_DSV41_VAL_BEFORE_TRAIN / TEST_FREQ / STEPS 控制，
#    这里在前，1103 用 ${...:-默认值} 读取，所以不会被覆盖回去。
export MY_ENV_DSV41_VAL_BEFORE_TRAIN=False
export MY_ENV_DSV41_TEST_FREQ=-1
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-2}

# 主机内存探针：每 5 s 记一行，单位 kB（meminfo 原始单位）；cgroup 单位是字节。
MY_ENV_DSV41_MEM_PROBE=${MY_ENV_DSV41_MEM_PROBE:-1}
if [ "$MY_ENV_DSV41_MEM_PROBE" = "1" ]; then
  PROBE_DIR="/efs_rl/m00899630/workspace/dsv41-verl/verl/logs/memprobe/$(date -u -d '+8 hours' +%Y%m%d_%H%M%S)"
  mkdir -p "$PROBE_DIR"
  PROBE_FILE="${PROBE_DIR}/$(hostname 2>/dev/null || echo unknown)_$$.log"
  (
    while true; do
      ts=$(date '+%F %T')
      avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      memfree=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
      cached=$(awk '/^Cached:/{print $2}' /proc/meminfo)
      swapfree=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
      ccur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null \
             || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo NA)
      chi=$(cat /sys/fs/cgroup/memory.high 2>/dev/null \
            || cat /sys/fs/cgroup/memory/memory.soft_limit_in_bytes 2>/dev/null || echo NA)
      # 每 NUMA 节点的 MemFree：分辨"整机不够"还是"单个 NUMA 被打满"
      numa=$(for d in /sys/devices/system/node/node[0-9]*; do
               [ -r "$d/meminfo" ] || continue
               awk -v n="${d##*/}" '/MemFree:/{printf "%s_free_kB=%s ", n, $4}' "$d/meminfo"
             done)
      echo "$ts avail_kB=$avail free_kB=$memfree cached_kB=$cached swapfree_kB=$swapfree cgroup_cur=$ccur cgroup_high=$chi $numa"
      sleep 5
    done
  ) > "$PROBE_FILE" 2>&1 &
  echo "[noval] MY_DEBUG_LOG mem_probe file=${PROBE_FILE} host=$(hostname 2>/dev/null || echo unknown) interval=5s"
fi

echo "[noval] MY_DEBUG_LOG val_before_train=${MY_ENV_DSV41_VAL_BEFORE_TRAIN} test_freq=${MY_ENV_DSV41_TEST_FREQ} steps=${MY_ENV_DSV41_STEPS} base=$(basename "$BASE") host=$(hostname 2>/dev/null || echo unknown) time=$(date '+%Y-%m-%d %H:%M:%S')"

exec bash "$BASE"
