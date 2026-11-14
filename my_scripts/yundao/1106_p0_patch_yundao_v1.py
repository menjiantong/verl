#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[1106 P0-云道] 把同一套探针/开关打到云道的代码树（/efs_rl/m00899630/workspace/dsv41-verl）。

为什么单独一份：云道那棵树与本地 /mnt/share/... 不是同一份拷贝。
  本地 host_offload_engram.py = e7ea671f...（含 _ENGRAM_REUSE 与 t_alloc_ms 拆分）
  云道 host_offload_engram.py = 5a6650ce...（无 _ENGRAM_REUSE、无 t_alloc_ms）
  engram.py / fsdp_turbo_impl.py 两侧一致（3b530c3f / 97f514d8）。
所以 C2（复用缓冲 + 分段）必须按云道的原始文本另写一套匹配串。

与本地补丁语义完全一致，只换匹配串：
  C1 engram.py                pin_memory -> MY_ENV_DSV41_ENGRAM_PIN_MEMORY
  C2 host_offload_engram.py   pinned 缓冲可复用；t_cpu_gather 拆成 等待/分配/gather 三段
  C5 host_offload_engram.py   a2a 计数后补 _probe_sync（云道这版也没有）
  C3 fsdp_turbo_impl.py       新增 _dsv41_probe_optimizer_state()

用法：
  python3.11 1106_p0_patch_yundao_v1.py --apply --check
"""
from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import shutil
import sys
import time

REPO = "/efs_rl/m00899630/workspace/dsv41-verl"
ENGRAM = f"{REPO}/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/engram.py"
HOSTOFF = f"{REPO}/FSDPTurbo/fsdp_turbo/models/deepseek/deepseek_v41/engram/host_offload_engram.py"
FSDPIMPL = f"{REPO}/verl/verl/workers/engine/fsdp/fsdp_turbo_impl.py"
TS = time.strftime("%Y%m%d_%H%M%S")

C1_OLD = """            self.embed = embedding_type(
                table_rows,
                layout.head_dim,
                layer_id=layer_id,
                num_max_tokens_per_rank=(args.max_batch_size * args.max_seq_len * n_hash_cols),
                pin_memory=True,
            )"""
C1_NEW = """            # [1106 P0-C1] 原来是硬编码 pin_memory=True。云道脚本早就设了
            # VERL_DSV41_ENGRAM_PIN_MEMORY，但全仓库没有任何 .py 读它 —— 死开关，
            # 所以 E1 一直是全 pinned 表（46 GiB/节点 = 16 rank x 2.86 GiB）。
            # 反向思考：关 pinning 会让 H2D 从一跳变两跳会变慢；但若主因是 rtMallocHost
            # 撞 devmm 全局锁，收益远大于损失 —— 必须 A/B 量。默认 1 即行为不变。
            _pin_env = os.getenv(
                "MY_ENV_DSV41_ENGRAM_PIN_MEMORY",
                os.getenv("VERL_DSV41_ENGRAM_PIN_MEMORY", "1"),
            )
            _engram_pin_memory = _pin_env.strip().lower() not in ("0", "false", "off", "no")
            self.embed = embedding_type(
                table_rows,
                layout.head_dim,
                layer_id=layer_id,
                num_max_tokens_per_rank=(args.max_batch_size * args.max_seq_len * n_hash_cols),
                pin_memory=_engram_pin_memory,
            )"""
C1_IMP_OLD = "import hashlib\n"
C1_IMP_NEW = "import hashlib\nimport os\n"

C2_DEF_OLD = '''_ENGRAM_PROBE_COUNT = {"fwd": 0, "bwd": 0}'''
C2_DEF_NEW = '''_ENGRAM_PROBE_COUNT = {"fwd": 0, "bwd": 0}

# [1106 P0-C2] 复用 pinned 主机中转缓冲。
#   现象（云道 E1 日志）：t_cpu_gather_ms 单次 16 986 ms，而同一次调用 a2a_* 全部
#   <= 1.55 ms；该段包住 torch.empty(..., pin_memory=True) 即 rtMallocHost。
#   7 MB 的分配耗 17 s（0.4 MB/s）只能是阻塞，不是带宽。
#   反向思考：复用要防「上一跳 non_blocking H2D 还没拷完、CPU 就覆盖同一块缓冲」，
#   故给每个 (layer, 容量桶) 记 device event，下次复用前 synchronize；拿不到就退回同步拷贝。
_ENGRAM_HOST_BUF = _os.getenv("MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF", "0") == "1"
_ENGRAM_HOST_BUF_MAX_MB = int(_os.getenv("MY_ENV_DSV41_ENGRAM_HOST_BUF_MAX_MB", "256"))
_HOST_BUF_CACHE: dict = {}


def _host_buf_slot(layer_id: int, n_rows: int, dim: int, dtype) -> dict | None:
    """按「向上取整到 64 行的桶」找/建一块可复用 pinned 缓冲；超上限放弃复用。"""
    if n_rows <= 0:
        return None
    bucket = ((n_rows + 63) // 64) * 64
    need_mb = bucket * dim * dtype.itemsize / (1024 * 1024)
    if need_mb > _ENGRAM_HOST_BUF_MAX_MB:
        return None
    key = (int(layer_id), bucket)
    slot = _HOST_BUF_CACHE.get(key)
    if slot is None:
        try:
            buf = torch.empty(bucket, dim, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            return None
        slot = {"buf": buf, "event": None}
        _HOST_BUF_CACHE[key] = slot
    return slot


def _host_buf_release_event(slot: dict) -> None:
    """记录一次 device event，标记「这块主机缓冲的 H2D 何时拷完」。"""
    try:
        import torch_npu  # noqa: F401

        ev = torch.npu.Event()
        ev.record()
        slot["event"] = ev
    except Exception:
        slot["event"] = "unsafe"'''

C2_USE_OLD = '''        _t2 = _time.perf_counter()
        # Materialize both count vectors with one D2H synchronization. These
        # lists are reused by every collective in forward and backward.'''
C2_USE_NEW = '''        # [1106 P0-C5] all_to_all_single 是异步下发；原来的 _t2 紧跟其后，
        # t_a2a_cnt_ms 只测到「下发完成」（实测 0.17 ms），真实等待被推到后面的
        # split_values.to("cpu") 里，会被误读成「通信不慢」。
        # PROBE_SYNC=1 时这里同步一次，分段耗时才可信；=0 时本调用为空操作。
        _probe_sync(row_ids.device)
        _t2 = _time.perf_counter()
        # Materialize both count vectors with one D2H synchronization. These
        # lists are reused by every collective in forward and backward.'''

C2_GATHER_OLD = '''        local_ids = (recv_ids - table.row_start).to(device="cpu", dtype=torch.long)
        if table.pin_memory and weight.is_pinned():
            values_cpu = torch.empty(
                local_ids.numel(),
                table.embedding_dim,
                dtype=weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            torch.index_select(weight.detach(), 0, local_ids, out=values_cpu)
        else:
            values_cpu = weight.detach().index_select(0, local_ids)
        _t5 = _time.perf_counter()
        values = values_cpu.to(device=row_ids.device, non_blocking=table.pin_memory)'''
C2_GATHER_NEW = '''        local_ids = (recv_ids - table.row_start).to(device="cpu", dtype=torch.long)
        _t_alloc0 = _time.perf_counter()
        n_rows = int(local_ids.numel())
        values_cpu = None
        _slot = None
        if _ENGRAM_HOST_BUF and table.pin_memory and weight.is_pinned():
            _slot = _host_buf_slot(table.layer_id, n_rows, table.embedding_dim, weight.dtype)
            if _slot is not None:
                _ev = _slot.get("event")
                if _ev == "unsafe":
                    _slot = None
                elif _ev is not None:
                    try:
                        _ev.synchronize()
                    except Exception:
                        _slot = None
                if _slot is not None:
                    values_cpu = _slot["buf"][:n_rows]
        _t_alloc1 = _time.perf_counter()
        _t_g0 = _time.perf_counter()
        if values_cpu is None:
            if table.pin_memory and weight.is_pinned():
                # 原路径：每次 fetch 都 rtMallocHost（E1 实测单次 16.99 s）
                values_cpu = torch.empty(
                    n_rows,
                    table.embedding_dim,
                    dtype=weight.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                values_cpu = torch.empty(n_rows, table.embedding_dim, dtype=weight.dtype, device="cpu")
        _t_g1 = _time.perf_counter()
        torch.index_select(weight.detach(), 0, local_ids, out=values_cpu)
        _t5 = _time.perf_counter()
        # 复用模式下拿不到 event 就退成同步 H2D（安全，代价是不重叠）
        _nb = table.pin_memory
        if _slot is not None and _slot.get("event") is None:
            _nb = False
        values = values_cpu.to(device=row_ids.device, non_blocking=_nb)
        if _slot is not None:
            _host_buf_release_event(_slot)'''

C2_EMIT_OLD = '''                t_cpu_gather_ms=_ms(_t4, _t5),
                t_h2d_ms=_ms(_t5, _t6),'''
C2_EMIT_NEW = '''                # [1106 P0-C2] 把 t_cpu_gather 拆成三段，才能判定那 17 s 在「等上一跳
                # H2D」还是「rtMallocHost」还是「纯主机 gather」。
                t_buf_wait_ms=_ms(_t_alloc0, _t_alloc1),
                t_buf_alloc_ms=_ms(_t_g0, _t_g1),
                t_gather_ms=_ms(_t_g1, _t5),
                buf=("reuse" if _slot is not None else ("alloc" if (table.pin_memory and weight.is_pinned()) else "pageable")),
                t_cpu_gather_ms=_ms(_t4, _t5),
                t_h2d_ms=_ms(_t5, _t6),'''

C3_CALL_OLD = '''        if os.environ.get("VERL_DSV41_OPT_STATE_DEBUG", "0") != "1":
            return'''
C3_CALL_NEW = '''        # [1106 P0-C3] 新逐参数探针：wall/cpu/MB/MBps/VmPin，长卡顿自动落栈。
        # 默认关闭；打开时由它代替 torch 的惰性建状态（口径与 torch 完全一致）。
        if os.environ.get("MY_ENV_DSV41_OPT_STATE_PROBE", "0") == "1":
            self._dsv41_probe_optimizer_state()

        if os.environ.get("VERL_DSV41_OPT_STATE_DEBUG", "0") != "1":
            return'''

# 与本地补丁逐字一致的方法体（同一份代码，避免两边行为漂移）
C3_METHOD = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "1106_p0_method_body.py")).read()


def _patch(path, triples):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    for sentinel, old, new in triples:
        if sentinel in src:
            print(f"  [skip] already applied: {sentinel[:50]!r}")
            continue
        n = src.count(old)
        if n == 0:
            raise SystemExit(f"ABORT: pattern not found in {path}: {old.splitlines()[0][:90]!r}")
        if n != 1:
            raise SystemExit(f"ABORT: pattern occurs {n}x in {path}: {old.splitlines()[0][:90]!r}")
        src = src.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.apply:
        for path in (ENGRAM, HOSTOFF, FSDPIMPL):
            bak = f"{path}.bak.{TS}"
            shutil.copy2(path, bak)
            print(f"[backup] {bak}  md5={hashlib.md5(open(path,'rb').read()).hexdigest()}")

        print("[C1] engram.py")
        _patch(
            ENGRAM,
            [
                ("import hashlib\nimport os\n", C1_IMP_OLD, C1_IMP_NEW),
                ("_engram_pin_memory", C1_OLD, C1_NEW),
            ],
        )
        print("[C2/C5] host_offload_engram.py")
        _patch(
            HOSTOFF,
            [
                ("MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF", C2_DEF_OLD, C2_DEF_NEW),
                ("PROBE_SYNC=1 时这里同步一次", C2_USE_OLD, C2_USE_NEW),
                ("_host_buf_release_event(_slot)", C2_GATHER_OLD, C2_GATHER_NEW),
                ("t_buf_alloc_ms", C2_EMIT_OLD, C2_EMIT_NEW),
            ],
        )
        print("[C3] fsdp_turbo_impl.py")
        _patch(FSDPIMPL, [("self._dsv41_probe_optimizer_state()", C3_CALL_OLD, C3_CALL_NEW)])

        with open(FSDPIMPL, "r", encoding="utf-8") as fh:
            src = fh.read()
        marker = "    def _dsv41_prepare_optimizer_step(self):"
        if "_dsv41_probe_optimizer_state" not in src.split(marker)[0]:
            src = src.replace(marker, C3_METHOD + "\n" + marker, 1)
            with open(FSDPIMPL, "w", encoding="utf-8") as fh:
                fh.write(src)
            print("  [append] _dsv41_probe_optimizer_state inserted")
        else:
            print("  [skip] _dsv41_probe_optimizer_state already present")

    print("[compile]")
    for path in (ENGRAM, HOSTOFF, FSDPIMPL):
        py_compile.compile(path, doraise=True)
        print(f"  OK {os.path.basename(path)}  md5={hashlib.md5(open(path,'rb').read()).hexdigest()}")

    if args.check:
        for path in (ENGRAM, HOSTOFF, FSDPIMPL):
            with open(path, encoding="utf-8") as fh:
                s = fh.read()
            print(
                "[check] %s: C1=%s C2=%s C3=%s C5=%s"
                % (
                    os.path.basename(path),
                    "MY_ENV_DSV41_ENGRAM_PIN_MEMORY" in s,
                    "MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF" in s,
                    "MY_ENV_DSV41_OPT_STATE_PROBE" in s,
                    s.count("_probe_sync(row_ids.device)") >= 3,
                )
            )


if __name__ == "__main__":
    sys.exit(main())
