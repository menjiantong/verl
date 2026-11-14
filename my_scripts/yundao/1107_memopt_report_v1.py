#!/usr/bin/env python3
# [启动结果] ⏳ 待验证：2026-11-07 由 1107_run_dsv41_yundao_1node_engram_memopt_v1.sh 自动调用
# =============================================================================
# 1107 E1 内存归因报告 v1（只读解析，不发任何命令）
#
# 输入：1106 单机主日志路径（dsv4-yundao_1106_p3s_<arm>_<ts>.log）
# 自动在同目录找最新的 memacct_v2_*.log（v2 探针输出）。
#
# 输出：
#   1) 每步 timing_s（gen/old_log_prob/ref/update_actor/update_weights/step）
#   2) [rt-probe] done 分布（wall/cpu/ratio/VmRSS）与 SLOW 汇总
#   3) [engram-probe] 分段 p50/p90/max
#   4) 节点账：MemAvail 最低点、Shmem/Anon/Cached、隐藏量、sum(VmRSS)/sum(VmPin)
#   5) engram 表加载次数（base/nopin 应为 2，P0 后应为 1）
# =============================================================================
import math
import re
import statistics
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_text(path: Path) -> str:
    try:
        return ANSI.sub("", path.read_text(errors="ignore"))
    except OSError as exc:
        print(f"WARN: cannot read {path}: {exc}")
        return ""


def pct(values, q):
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def fmt(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values):.2f} p50={pct(values,0.5):.2f} "
        f"p90={pct(values,0.9):.2f} max={max(values):.2f}"
    )


# ---------------------------- 主日志 -----------------------------------------
main_log = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if main_log is None or not main_log.exists():
    print("usage: 1107_memopt_report_v1.py <main_log>")
    sys.exit(2)
text = read_text(main_log)

print("=" * 100)
print(f"1107 E1 内存归因报告")
print(f"主日志: {main_log}")
print("=" * 100)

# 1) timing_s
print("\n[1] 每步 timing_s")
step_re = re.compile(r"step:(\d+) ")
key_re = re.compile(r"timing_s/([a-zA-Z_]+):([0-9.]+)")
rows = {}
for line in text.splitlines():
    if "timing_s/" not in line:
        continue
    m = step_re.search(line)
    if not m:
        continue
    step = int(m.group(1))
    for k, v in key_re.findall(line):
        rows.setdefault(step, {})[k] = float(v)
    # cpu_memory_used_gb / max_memory_allocated_gb 与 timing_s 同行，必须按行取，
    # 否则整篇 re.search 只会拿到第一步的值（v1 演练时踩过）。
    cm = re.search(r"cpu_memory_used_gb:([0-9.]+)", line)
    ma = re.search(r"max_memory_allocated_gb:([0-9.]+)", line)
    if cm:
        rows[step]["cpu_mem"] = float(cm.group(1))
    if ma:
        rows[step]["max_alloc"] = float(ma.group(1))
keys = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "step"]
print("step  " + "  ".join(f"{k:>14}" for k in keys) + "  cpu_mem_GiB  max_alloc_GB")
for step in sorted(rows):
    r = rows[step]
    print(
        f"{step:<5} "
        + "  ".join(f"{r.get(k, float('nan')):>14.1f}" for k in keys)
        + f"  {r.get('cpu_mem', float('nan')):>10.1f}"
        + f"  {r.get('max_alloc', float('nan')):>12.2f}"
    )

# 2) rt-probe done / SLOW
print("\n[2] [rt-probe] 优化器状态创建")
done_wall, done_cpu, done_ratio, done_rss = [], [], [], []
for line in text.splitlines():
    if "[rt-probe]" not in line or " done " not in line:
        continue
    m = re.search(
        r"created=(\d+)/(\d+).*?wall=([0-9.]+)s cpu=([0-9.]+)s ratio=([0-9.]+).*?VmPin=([0-9]+)MiB VmRSS=([0-9]+)MiB",
        line,
    )
    if m:
        done_wall.append(float(m.group(3)))
        done_cpu.append(float(m.group(4)))
        done_ratio.append(float(m.group(5)))
        done_rss.append(float(m.group(7)) / 1024.0)
print(f"  done wall(s):  {fmt(done_wall)}")
print(f"  done cpu(s):   {fmt(done_cpu)}")
print(f"  done ratio:    {fmt(done_ratio)}")
print(f"  done VmRSS(GiB): {fmt(done_rss)}")

slow_wall, slow_mbps = [], []
for line in text.splitlines():
    if "[rt-probe]" not in line or " SLOW " not in line:
        continue
    m = re.search(r"wall=([0-9.]+)s cpu=([0-9.]+)s mb=([0-9.]+) mbps=([0-9.]+)", line)
    if m:
        slow_wall.append(float(m.group(1)))
        slow_mbps.append(float(m.group(4)))
print(f"  SLOW wall(s):  {fmt(slow_wall)}  sum={sum(slow_wall):.0f}s")
print(f"  SLOW MB/s:     {fmt(slow_mbps)}")

# 3) engram-probe
print("\n[3] [engram-probe] 分段耗时 (ms)")
probe = {"total_ms": [], "t_d2h_list_ms": [], "t_buf_alloc_ms": [], "t_cpu_gather_ms": [], "t_gather_ms": []}
for line in text.splitlines():
    if "[engram-probe]" not in line:
        continue
    for k in probe:
        m = re.search(rf"{k}=([0-9.]+)", line)
        if m:
            probe[k].append(float(m.group(1)))
for k, vals in probe.items():
    print(f"  {k:<18} {fmt(vals)}")

# 4) engram table loads
loads = [l for l in text.splitlines() if "host-offload Engram tables loaded" in l]
print("\n[4] engram 表加载次数")
print(f"  host-offload Engram tables loaded: {len(loads)} 次")
for l in loads:
    m = re.search(r"(\d+) table\(s\), ([0-9.]+) GiB per rank in ([0-9.]+)s", l)
    if m:
        print(f"    tables={m.group(1)} GiB/rank={m.group(2)} load_s={m.group(3)}")

# 5) memacct v2
print("\n[5] 节点内存账（memacct v2）")
v2s = sorted(main_log.parent.glob("memacct_v2_*.log"), key=lambda p: p.stat().st_mtime)
if not v2s:
    print("  WARN: 同目录没有 memacct_v2_*.log")
else:
    v2 = v2s[-1]
    print(f"  探针日志: {v2.name}")
    txt = read_text(v2)
    fields = {
        "MemTotal": [], "MemFree": [], "MemAvailable": [], "Cached": [], "Shmem": [],
        "AnonPages": [], "SReclaimable": [], "SUnreclaim": [], "Mlocked": [],
        "Unevictable": [], "PageTables": [], "AnonHugePages": [], "HugePages_Total": [],
    }
    cg = {"total_cache": [], "total_rss": [], "total_shmem": []}
    sums = {"sum_VmRSS_MiB": [], "sum_VmPin_MiB": []}
    for line in txt.splitlines():
        for k in fields:
            m = re.search(rf"meminfo {k}=(\d+)", line)
            if m:
                fields[k].append(int(m.group(1)))
        for k in cg:
            m = re.search(rf"cgstat {k}=(\d+)", line)
            if m:
                cg[k].append(int(m.group(1)))
        for k in sums:
            m = re.search(rf"{k}=(\d+)", line)
            if m:
                sums[k].append(int(m.group(1)))

    def gib(k):
        return [v / 1048576.0 for v in fields[k]]

    def minmax(k, scale=1048576.0):
        vals = fields[k]
        return (min(vals) / scale, max(vals) / scale) if vals else (float("nan"), float("nan"))

    print(f"  MemTotal:      {minmax('MemTotal')[0]:.1f} GiB")
    for k in ["MemFree", "MemAvailable", "Cached", "Shmem", "AnonPages", "SReclaimable", "SUnreclaim", "Mlocked", "Unevictable", "PageTables", "AnonHugePages"]:
        lo, hi = minmax(k)
        print(f"  {k:<14} min={lo:.1f}  max={hi:.1f} GiB")
    print(f"  HugePages_Total max={max(fields['HugePages_Total']) if fields['HugePages_Total'] else 0}")
    for k, vals in cg.items():
        if vals:
            print(f"  cg {k:<13} min={min(vals)/2**30:.1f}  max={max(vals)/2**30:.1f} GiB")
    for k, vals in sums.items():
        if vals:
            print(f"  {k:<15} min={min(vals)/1024:.1f}  max={max(vals)/1024:.1f} GiB")

    # 隐藏量 = MemTotal - (MemFree + Cached + AnonPages + SReclaimable)
    if all(fields[k] for k in ["MemTotal", "MemFree", "Cached", "AnonPages", "SReclaimable"]):
        n = min(len(fields[k]) for k in ["MemTotal", "MemFree", "Cached", "AnonPages", "SReclaimable"])
        hidden = [
            (fields["MemTotal"][i] - fields["MemFree"][i] - fields["Cached"][i]
             - fields["AnonPages"][i] - fields["SReclaimable"][i]) / 1048576.0
            for i in range(n)
        ]
        print(f"  隐藏量(hidden)  min={min(hidden):.1f}  p50={pct(hidden,0.5):.1f}  max={max(hidden):.1f} GiB")
        print("  说明: hidden = MemTotal - MemFree - Cached - AnonPages - SReclaimable；")
        print("        不含 Slab/Kernel/PageTables，故比真实驱动侧内存略高几 GiB。")

# 6) 关键判据
print("\n[6] 判据")
print("  - base 预期: hidden max ~1200 GiB, MemAvail min ~0, done wall p90 >100 s")
print("  - nopin 预期: AnonPages 增加 ~732 GiB；hidden 若降到 ~166 ⇒ 表是 pinned；")
print("                若仍 ~470 ⇒ 另有独立 pinned 缓冲（FSDP full_state/HCCL）")
print("  - noengram 预期: hidden ~166 GiB, MemAvail min >1200 GiB, 3/3 过")
print("  - P0 后: 'loaded' 只 1 次；MemAvail min >300 GiB；done wall p90 <30 s")
print("=" * 100)
