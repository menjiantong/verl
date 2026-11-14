#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[1107] 把一次「1 机 × 16 卡对齐 124」的运行结果抽成一张表，与 124 的 P1 基线并排。

只读日志，不改任何东西。用法：
    python3 1107_align_report_v1.py <日志路径>

日志里会出现的东西（都已被 1106 基线那套探针打过）：
    timing_s/gen|old_log_prob|ref|adv|update_actor|update_weights|step
    [engram-probe] ... t_buf_alloc_ms= t_gather_ms= t_cpu_gather_ms= t_d2h_list_ms= t_uniq_ms= total_ms=
    [rt-probe][rankN] i=N/1192 wall= cpu= mb= mbps= ratio=
    [cg-probe] <time> node MemTotal=..MemFree=..MemAvail=..Cached=..Shmem=..
失败标记：Traceback / EL0004 / out of memory / Memory_Allocation_Failure / keepalive watchdog timeout
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------- 124 的 P1 基线（2026-11-06 02:52 那一轮，1 机 16 卡 EP16）--------
BASE_124 = {
    "engram":   {"update_actor": [106.7, 120.1, 101.6], "step": None},
    "noengram": {"update_actor": [67.5, 67.6, 68.0], "step": None},
}
BASE_124_MEMTOTAL = 2454  # GiB
BASE_124_MIN_AVAIL = 870  # GiB，MemAvailable 最低值

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FAIL_PAT = ("Traceback", "EL0004", "out of memory", "OutOfMemory",
            "Memory_Allocation_Failure", "keepalive watchdog timeout",
            "is killed by signal")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parse_steps(lines: list[str]) -> list[dict]:
    """每个 summary 行取一次；同一 step 号只保留最后一次（重跑会覆盖）。"""
    by_no: dict[int, dict] = {}
    for line in lines:
        if "timing_s/step:" not in line:
            continue
        m = re.search(r"training/global_step:(\d+)", line)
        if not m:
            continue
        row = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
        row["step_no"] = int(m.group(1))
        for key, pat in (("cpu_mem", r"cpu_memory_used_gb:([0-9.]+)"),
                         ("max_alloc", r"max_memory_allocated_gb:([0-9.]+)"),
                         ("pearson", r"rollout_actor_probs_pearson_corr:([0-9.]+)")):
            mm = re.search(pat, line)
            if mm:
                row[key] = float(mm.group(1))
        by_no[row["step_no"]] = row
    return [by_no[k] for k in sorted(by_no)]


def parse_kv(lines: list[str], tag: str) -> list[dict]:
    out = []
    for line in lines:
        i = line.find(tag)
        if i < 0:
            continue
        seg = line[i + len(tag):]
        d = {k: v for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", seg)}
        # shape=(32, 5120, 2304) 里有空格，上面的通配会截断成 "(32,"，单独捞一次。
        ms = re.search(r"shape=\(([^)]*)\)", line)
        if ms:
            d["shape"] = "(" + ms.group(1).strip() + ")"
        if d:
            d["_raw"] = line
            out.append(d)
    return out


def to_num(v: str) -> float | None:
    """'187.93s' / '25.0MiB' / '0.42' → float；取不到就 None。

    探针把单位直接贴在数值后面（wall=187.93s、mb=25.0），不剥掉单位
    会让整列变成 NaN —— 上一版就是这样把 wall 全丢了的。
    """
    if v is None:
        return None
    m = re.match(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", v)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def nums(rows: list[dict], key: str) -> list[float]:
    vals = []
    for r in rows:
        n = to_num(r.get(key))
        if n is not None:
            vals.append(n)
    return vals


def fmt(v: float, nd: int = 1) -> str:
    if v != v:  # NaN
        return "—"
    return f"{v:.{nd}f}"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: 1107_align_report_v1.py <log>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[report] log not found: {path}")
        return 2

    text = strip_ansi(path.read_text(errors="replace"))
    lines = text.splitlines()

    steps = parse_steps(lines)
    probes = parse_kv(lines, "[engram-probe]")
    rts = parse_kv(lines, "[rt-probe]")
    # cgroup 探针写在**旁边**的 .cg.log 里（基线脚本第 102 行），不在主日志。
    # 主日志里找不到就自动去读同名的 .cg.log，否则第 6 节永远是空的。
    cg = [l for l in lines if "[cg-probe]" in l and "node MemTotal=" in l]
    cg_src = path
    if not cg:
        sibling = path.parent / (path.stem + ".cg.log")
        if sibling.exists():
            cg_text = strip_ansi(sibling.read_text(errors="replace"))
            cg = [l for l in cg_text.splitlines() if "[cg-probe]" in l and "node MemTotal=" in l]
            cg_src = sibling
    fails = [l.strip()[:220] for l in lines if any(p in l for p in FAIL_PAT)]

    print(f"# 1107 单机对齐报告\n")
    print(f"- 日志：`{path}`")
    print(f"- 行数：{len(lines)}")
    print()

    # ---------------- 1. 每步耗时 ----------------
    print("## 1. 每步耗时（秒）\n")
    if steps:
        cols = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "step"]
        print("| step | " + " | ".join(cols) + " | cpu_mem GiB | max_alloc GB |")
        print("|---|" + "---|" * (len(cols) + 2))
        for r in steps:
            cells = [fmt(r.get(c, float("nan"))) for c in cols]
            print(f"| {r['step_no']} | " + " | ".join(cells)
                  + f" | {fmt(r.get('cpu_mem', float('nan')))} | {fmt(r.get('max_alloc', float('nan')), 2)} |")
    else:
        print("（没有完成任何一步 —— 见第 5 节失败标记）")
    print()

    # ---------------- 2. 与 124 并排 ----------------
    print("## 2. 与 124 P1 并排（update_actor，秒）\n")
    print("| arm | 124 P1 三次 | 本次 | 比值（本次均值 / 124 均值） |")
    print("|---|---|---|---|")
    for arm, base in BASE_124.items():
        b = base["update_actor"]
        bm = sum(b) / len(b)
        mine = [r["update_actor"] for r in steps if "update_actor" in r]
        if mine:
            mm = sum(mine) / len(mine)
            ratio = mm / bm
            print(f"| {arm} | {' / '.join(f'{x:.1f}' for x in b)} | "
                  f"{' / '.join(f'{x:.1f}' for x in mine)} | {ratio:.2f}× |")
        else:
            print(f"| {arm} | {' / '.join(f'{x:.1f}' for x in b)} | 未产出 | — |")
    print()

    # ---------------- 3. engram 分段 ----------------
    print("## 3. engram 分段（ms）\n")
    if probes:
        keys = ["t_uniq_ms", "t_a2a_cnt_ms", "t_d2h_list_ms", "t_a2a_ids_ms",
                "t_buf_wait_ms", "t_buf_alloc_ms", "t_gather_ms", "t_cpu_gather_ms", "total_ms"]
        print("| 指标 | n | p50 | p90 | max |")
        print("|---|---|---|---|---|")
        for k in keys:
            v = nums(probes, k)
            if v:
                print(f"| {k} | {len(v)} | {fmt(pct(v, .5), 2)} | {fmt(pct(v, .9), 2)} | {fmt(max(v), 2)} |")
        buf = [r.get("buf", "?") for r in probes]
        dist = {b: buf.count(b) for b in sorted(set(buf))}
        print(f"\n- `buf=` 分布：{dist}")
        print(f"- 采样条数：{len(probes)}")
    else:
        print("（没有 [engram-probe] 行 —— 本次没走到 engram 前向，或探针未开）")
    print()

    # ---------------- 4. 优化器状态创建 ----------------
    print("## 4. 优化器状态创建（[rt-probe]）\n")
    if rts:
        slow = [r for r in rts if "SLOW" in r.get("_raw", "")]
        done = [r for r in rts if "created=" in r.get("_raw", "")]
        walls = nums(done, "wall")
        if walls:
            print(f"- 整批完成：{len(walls)} 条  p50={fmt(pct(walls, .5), 1)}s  "
                  f"p90={fmt(pct(walls, .9), 1)}s  max={fmt(max(walls), 1)}s")
        ratio = nums(done, "ratio")
        if ratio:
            print(f"- cpu/wall 比值：min={fmt(min(ratio), 2)}  p50={fmt(pct(ratio, .5), 2)}  "
                  f"max={fmt(max(ratio), 2)}（>1 = 多线程烧 CPU，不是阻塞在驱动）")
        sw = nums(slow, "wall")
        if sw:
            print(f"- 单参数 SLOW：{len(sw)} 条  p50={fmt(pct(sw, .5), 1)}s  "
                  f"p90={fmt(pct(sw, .9), 1)}s  max={fmt(max(sw), 1)}s")
            srt = sorted(slow, key=lambda r: to_num(r.get("wall")) or 0, reverse=True)[:3]
            for r in srt:
                print(f"  - i={r.get('i', '?')} wall={r.get('wall', '?')} "
                      f"mb={r.get('mb', '?')} mbps={r.get('mbps', '?')} shape={r.get('shape', '?')}")
        print(f"- 原始 [rt-probe] 行：{len(rts)}（SLOW {len(slow)}，整批完成 {len(done)}）")
    else:
        print("（没有 [rt-probe] 行）")
    print()

    # ---------------- 5. 失败标记 ----------------
    print("## 5. 失败标记\n")
    if fails:
        for f in fails[:12]:
            print(f"- `{f}`")
        if len(fails) > 12:
            print(f"- …… 共 {len(fails)} 条")
    else:
        print("（无）")
    print()

    # ---------------- 6. 内存 ----------------
    print("## 6. 内存（节点视角）\n")
    if cg:
        print(f"- 来源：`{cg_src}`")
        print(f"- 采样条数：{len(cg)}")
        print(f"- 首条：`{cg[0].split(']')[-1].strip()}`")
        print(f"- 末条：`{cg[-1].split(']')[-1].strip()}`")
        low = min((to_num(m.group(1)) for l in cg
                   for m in [re.search(r"MemAvail=(\d+)GiB", l)] if m), default=None)
        if low is not None:
            print(f"- **MemAvail 最低：{low:.0f} GiB**（124 P1 是 {BASE_124_MIN_AVAIL} GiB）")
        print(f"\n124 P1 对照：MemTotal={BASE_124_MEMTOTAL} GiB，MemAvailable 最低 {BASE_124_MIN_AVAIL} GiB")
    else:
        print("（没有 [cg-probe] 节点行）")
    print()

    # ---------------- 7. 判定 ----------------
    print("## 7. 一句话判定\n")
    if not steps:
        print("- **没跑完任何一步。** 先看第 5 节失败标记：\n"
              "  - `keepalive watchdog timeout` / `is killed by signal` → 主机内存打穿，"
              "看 `memacct_*.log` 的进程 RSS 分解；\n"
              "  - `Traceback` 其他类型 → 代码/配置错误，与内存无关。")
    else:
        ua = [r["update_actor"] for r in steps]
        ua_m = sum(ua) / len(ua)
        base_m = 106.7
        print(f"- 本次 update_actor 均值 **{ua_m:.1f} s**，124 是 106.7 s ⇒ **{ua_m/base_m:.2f}×**。")
        if ua_m < 200:
            print("- 落在 124 同一量级（<2×）⇒ **宿主不是主因**，云道 8 机的 26× 来自规模（EP128 跨机）。")
        else:
            print("- 远高于 124（>2×）⇒ **单个云道节点的主机侧本身就有重尾**，"
                  "与 8 机的 26× 同源，不是规模问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
