#!/usr/bin/env python3
"""Live status line for a DeepSeek-V4.x GRPO run, every N seconds (default 5).

    python3 scripts/watch_grpo.py                     # newest logs/grpo_*.log
    python3 scripts/watch_grpo.py --log <file> --interval 5

Each cycle prints one line:

    [02:41:07] stage=12 STEP_METRICS | log +12 lines(2410) err=0 | host 2011/2454GB(82%) |
               npu hbm 41.2/64.0GB aicore 8% | alive drv=1 w=8 v=8 | [worker-log stage=06] <last line>

and the fields mean:

  stage      the furthest known phase marker found in the driver log (see STAGES below);
             the number is its index, so it can only grow -- a *decreasing* stage means I got
             the marker table wrong, not that the run went backwards.
  log +A (N lines) err=B  A = lines appended since the previous cycle, N = total lines,
             B = cumulative count of the error patterns below.
             +0 for several minutes while `aicore` is also 0 and host memory is flat == hung
             (the 02:22 hang was exactly this: engine workers dead, sender blocked on recv()).
  host       used/total GB from /proc/meminfo (used = MemTotal - MemAvailable). This is the
             binding resource for these runs: trainer Engram tables + CPU-offloaded experts +
             AdamW state + the engine's own host Engram tables. Watch for >2331 GB (Ray's
             default OOM kill line is 0.95 * 2454) and for it not coming back down after a
             phase (leftover registered host memory shows up here but not in `ps` RSS).
  npu hbm    average / peak HBM per card and average AICore utilisation. ~3 GB baseline = idle
             residue. Trainer materialisation peaks around 19 GB; rollout decode is the
             0.6*gmem engine allocation (tens of GB). High HBM + 0% AICore for minutes = a
             collective deadlock rather than progress.
  alive      driver process, `ray::WorkerDict` trainer workers, vLLM worker processes.
             drv=0 with the log still growing means the driver restarted; drv=1 w=0 = the
             trainer died and only the engine is left.
  err        cumulative count of error patterns (Traceback / OutOfMemory / killed due to /
             No backend type / IndexError / "no attribute").

Phase markers (STAGES) are the strings this integration actually prints; the trainer-side ones
come from the *worker* stdout, which Ray forwards into the driver log but not always promptly,
so --worker-logs additionally greps /tmp/ray/session_latest/logs/worker-*.out for them.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import time

"""(stage, markers, meaning, expected host GB, expected HBM GB) -- one tuple per phase.

The expectations are the *measured* anchors for the DeepSeek-V4.1 top32 run on this node
(8 NPUs, colocated engine), not guesses; see wiki/worklog_dsv41_top64_grpo.md §5.7/§5.8.
Model behind them: trainer Engram tables 8 x 45.8 GiB = 366 GB host (pinned, stays for the
whole run) + bf16 params on host ~105 GB + AdamW fp32 states ~421 GB (allocated at the first
optimizer step) + the engine's own Engram copy 366 GB.
"""
STAGES = [
    (
        "00 SCRIPT_START",
        ["run_deepseek_v41_", "DEVICE=npu"],
        "recipe shell is deriving config; nothing on devices yet",
        (0, 60),
        (2, 4),
    ),
    (
        "01 RAY_UP",
        ["Started a local Ray instance", " ray init"],
        "Ray up, actors being created; ranks import verl/vllm (each WorkerDict ~5 GB RSS)",
        (60, 400),
        (2, 4),
    ),
    (
        "02 TRAINER_BUILD",
        ["materializing ", "build_deepseek_v41_model"],
        "trainer building the 40-layer graph + FSDP/EP wrapping (meta -> materialized)",
        (150, 500),
        (15, 22),
    ),
    (
        "03 DCP_READ",
        ["checkpoint read: ", "broadcast_state_dict"],
        "rank0 reads ~150 GB of non-expert tensors from NFS and broadcasts; host RAM climbs fast",
        (400, 900),
        (15, 25),
    ),
    (
        "04 BUFFER_ALIGN",
        ["aligned ", "checkpoint buffers loaded"],
        "40 checkpoint buffers aligned + bit-exact verified against the ckpt",
        (500, 950),
        (15, 25),
    ),
    (
        "05 TRAIN_MATERIALIZED",
        ["parameters materialized in"],
        "params done; experts stay meta until refresh_dsv41_expert_metadata",
        (500, 950),
        (15, 25),
    ),
    (
        "06 ENGRAM_LOAD",
        ["engram layers.", "Engram tables loaded"],
        "each rank slice-reads its 1/8 row range of the two Engram tables into pinned host RAM (366 GB total) and freezes them",
        (700, 1400),
        (15, 25),
    ),
    (
        "07 ENGINE_BUILD",
        [
            "Initializing a V1 LLM engine",
            "vllm_async_server",
            "Warmup complete",
            "Engram table offloaded",
            "Loading weights took",
        ],
        "vLLM workers start, load their own copy of weights + Engram (another ~366 GB host)",
        (1000, 1800),
        (20, 45),
    ),
    (
        "08 ENGINE_KVCACHE",
        ["Available KV cache memory", "GPU KV cache size", "Peak/mean GPU block usage"],
        "engine profile run done, KV cache sized at 0.6*gmem; HBM then stops rising",
        (1200, 1900),
        (38, 48),
    ),
    (
        "09 READY_TO_FIT",
        ["all initialize finished, ready to fit"],
        "both sides resident; next thing that happens is the INITIAL WEIGHT SYNC (trainer -> engine)",
        (1400, 1900),
        (38, 48),
    ),
    (
        "10 WEIGHT_SYNC",
        ["bucketed-sync", "update_weights_from_ipc", "vLLM load weights, loaded_params"],
        "streaming ~105 GB of trainable weights over IPC; '[bucketed-sync] staged' lines = the E8 fix working",
        (1450, 2000),
        (38, 52),
    ),
    (
        "11 ROLLOUT",
        ["generate_sequences", "RolloutRequest"],
        "generation of 8 prompts x 5 samples; AICore goes high, trainer params get offloaded to host",
        (1400, 2000),
        (38, 52),
    ),
    (
        "12 STEP_METRICS",
        ["step:1 - ", "training/rollout_probs_diff", "actor/pg_loss"],
        "trainer forward/backward/optimizer_step done; AdamW fp32 states (+~421 GB host) now allocated",
        (1700, 2300),
        (30, 55),
    ),
]

ERROR_PATTERNS = [
    "Traceback (most recent call last)",
    "OutOfMemory",
    "killed due to",
    "No backend type associated",
    "IndexError",
    "RuntimeError",
    "has no attribute",
    "RayChannelTimeout",
    "heartbeat",
    "CUDA error",
    "HCCL error",
    "ErrCode",
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_meminfo() -> tuple[float, float]:
    total = avail = 0.0
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1])
    gb = 1024.0**2
    return (total - avail) / gb, total / gb


def npu_stats(visible: int, device_ids: list[int] | None = None) -> tuple[float, float, float]:
    """Return (mean HBM GB, peak HBM GB, mean AICore %) over the first `visible` cards.

    `device_ids` selects explicit chips by their position in `npu-smi info` output, which is
    Phy-ID order (chip 2 of this node can be held by another container's process, and the run
    then uses a non-contiguous visible set, e.g. 0,1,3,4,5,6,7,8)."""
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:  # noqa: BLE001 - monitoring must never break the run
        return float("nan"), float("nan"), float("nan")
    entries: list[tuple[float, float]] = []  # (hbm GB, aicore %); NaN aicore when unavailable
    for line in out.splitlines():
        # "| 0     0                   | 0000:9D:00.0  | 0                    0    / 0                3107 / 65536            |"
        m = re.search(r"\|\s*\d+\s+\d+\s+\|\s*[0-9A-Fa-f:.]+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/", line)
        if m:
            entries.append((float(m.group(2)) / 1024.0, float(m.group(1))))
        # the single-line variant some npu-smi builds emit
        elif "|" in line:
            m2 = re.search(r"\|\s*(\d+)\s+(\d+)\s*/\s*65536", line)
            if m2:
                entries.append((float(m2.group(2)) / 1024.0, float("nan")))
    if device_ids is not None:
        wanted = set(device_ids)
        entries = [e for i, e in enumerate(entries) if i in wanted]
    else:
        entries = entries[:visible] if visible else entries
    if not entries:
        return float("nan"), float("nan"), float("nan")
    hbm = [e[0] for e in entries]
    core = [e[1] for e in entries if e[1] == e[1]]
    return sum(hbm) / len(hbm), max(hbm), (sum(core) / len(core) if core else float("nan"))


def count(pattern: str, text: str) -> int:
    return text.count(pattern)


def tail_line(path: str, size: int) -> tuple[str, int]:
    """Return (last meaningful line, total line count) reading only the tail of the file."""
    if not os.path.exists(path):
        return "", 0
    with open(path, "r", errors="replace") as handle:
        data = handle.read()
    lines = [ANSI.sub("", l).strip() for l in data.splitlines() if l.strip()]
    return (lines[-1] if lines else ""), len(data.splitlines())


def proc_count(pattern: str) -> int:
    """Count live processes whose command line matches ``pattern``.

    Zombies are excluded on purpose: a killed vLLM worker keeps its command name
    (``[VLLM::Worker_TP] <defunct>``) until its parent is reaped, so counting them reads as
    "the engine is still up" exactly when it is not. This happened after the 02:22 failure --
    nine ``VLLM::`` entries stayed in ``ps`` for the whole next run.
    """
    try:
        out = subprocess.run(["ps", "-eo", "stat=,args="], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return 0
    return sum(
        1
        for line in out.splitlines()
        if not line.startswith("Z") and pattern in line and "watch_grpo.py" not in line
    )


def driver_pid() -> int:
    return proc_count("verl.trainer.main_ppo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="driver log (default: newest verl/logs/grpo_*.log)")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--devices", type=int, default=8, help="cards to summarise (0 = all)")
    parser.add_argument(
        "--device-ids",
        default=None,
        help="explicit Phy-IDs to summarise, e.g. 0,1,3,4,5,6,7,8 (default: first --devices cards)",
    )
    parser.add_argument("--worker-logs", action="store_true", help="also grep Ray worker stdout for stage markers")
    parser.add_argument("--tail", type=int, default=90, help="chars of the last log line to show")
    args = parser.parse_args()

    log = args.log
    if log is None:
        cands = sorted(
            glob.glob("/workspace-verl/verl/logs/grpo_*.log"), key=os.path.getmtime, reverse=True
        )
        if not cands:
            raise SystemExit("no verl/logs/grpo_*.log found; pass --log")
        log = cands[0]
    print(f"[watch] log={log} interval={args.interval}s (Ctrl-C to stop; output is safe to tee)")
    print(
        "[watch] stage | log delta + total lines + err count | host used GB (% of 2454) vs expected band | "
        "npu hbm mean GB (of 64) vs band, aicore % | alive driver/trainer-workers/vllm-workers | last log line",
        flush=True,
    )

    previous_lines = 0
    cycle = 0
    while True:
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        lines = len(text.splitlines())
        appended = lines - previous_lines
        previous_lines = lines

        # Stage = the furthest marker seen in EITHER log. The trainer's own prints
        # (`[fsdp_turbo_dsv41] materializing ...`, `checkpoint read: ...`, `Engram tables
        # loaded`) are written to the Ray *worker* files and only reach the driver log when Ray
        # flushes it -- minutes later. Reading the driver alone shows "stuck at RAY_UP" while
        # rank 0 is already broadcasting tensors, which is how I lost track of phase twice.
        hits: set[str] = set()
        if args.worker_logs:
            cmd = ["grep", "-ho"]
            for row in STAGES:
                for m in row[1]:
                    cmd += ["-e", m]
            cmd += ["--", *sorted(glob.glob("/tmp/ray/session_latest/logs/worker-*.out"))]
            try:
                hits = set(
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.splitlines()
                )
            except Exception:  # noqa: BLE001
                hits = set()

        stage_row = STAGES[0]
        for row in STAGES:
            if any(m in text for m in row[1]) or any(h in row[1] for h in hits):
                stage_row = row
        stage, meaning, want_host, want_hbm = stage_row[0], stage_row[2], stage_row[3], stage_row[4]
        errors = sum(count(p, text) for p in ERROR_PATTERNS)

        host_used, host_total = read_meminfo()
        device_ids = [int(x) for x in args.device_ids.split(",")] if args.device_ids else None
        hbm_mean, hbm_max, aicore = npu_stats(args.devices, device_ids)

        last, _ = tail_line(log, args.tail)
        host_flag = "!" if not (want_host[0] <= host_used <= want_host[1] * 1.15) else " "
        hbm_flag = "!" if hbm_mean == hbm_mean and not (want_hbm[0] <= hbm_mean <= want_hbm[1]) else " "
        print(
            f"[{time.strftime('%H:%M:%S')}] stage={stage} | log +{appended} lines({lines}) err={errors} | "
            f"host {host_used:.0f}/{host_total:.0f}GB({host_used / host_total * 100:.0f}%) "
            f"{host_flag}want {want_host[0]}-{want_host[1]} | "
            f"npu hbm {hbm_mean:.1f}/{hbm_max:.1f}GB{hbm_flag}want {want_hbm[0]}-{want_hbm[1]} aicore {aicore:.0f}% | "
            f"alive drv={driver_pid()} w={proc_count('ray::WorkerDict')} v={proc_count('VLLM::')} | "
            f"why={meaning} :: {last[: args.tail]}",
            flush=True,
        )
        cycle += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
