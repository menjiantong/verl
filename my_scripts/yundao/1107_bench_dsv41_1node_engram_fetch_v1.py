#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [验证结果] ⏳ 待验证：2026-11-07 E1-单机 engram fetch 微基准，尚未实跑。
"""E1-单机：在 1 机 16 卡上直接跑真实的 TorchHostOffloadEngram fetch 路径。

为什么不是"单机跑完整训练"（这是硬约束，不是选择）
--------------------------------------------------
本模型 384 专家，bf16 权重总量 1.388 TiB（268 个 safetensors 实测），
8 机 FSDP128 时每 rank 才 11.1 GiB；1 机 16 卡 FSDP16 时每 rank 88.9 GiB，
而单卡 HBM 只有 65.5 GB、整机 16 卡合计 1.02 TiB < 1.39 TiB。
另外 8 机实测 cpu_memory_used=1080 GiB/节点 × 8 = 8.6 TB 训练状态
（参数/梯度/优化器 offload 到主机），1 节点只有 2.01 TB。
→ **单机跑不动这个模型的训练，任何脚本都跑不动；能跑的最小拓扑就是 8 机。**

那"单机 E1"到底测什么
----------------------
E1 要回答的是：1429 s 的 update_actor 里，跨机 all_to_all、D2H 同步、
主机侧 CPU gather 各占多少。其中"跨机"这一项没法用单机消除，但可以用单机
把它**换掉**：同一份 `_TorchHostFetch` 代码、同样的表、同样的行数，
只把 EP 组从 128 卡跨 8 机换成 16 卡单机（片内 HCCS），段级耗时可逐项对比。

两种规模（--scale）：
  --scale 1  全尺寸表。EP16 → 每 rank 24.6 GiB（2 张表合计），节点 393 GiB。
             这是"真正单机跑 engram"时的每 rank 占用。
  --scale 8  表缩小 8 倍。EP16 → 每 rank 2 类表合计 3.07 GiB ≈ 8 机 EP128 的
             2.9 GiB/rank。**与 8 机探针数据唯一可比的那一档**：表大小、
             每 rank 行数完全一致，只有互连不同。

表内容用 `shard_()` 自带的正态初始化（同 shape / 同 dtype / 同 pin_memory），
不读 1.4 TB checkpoint —— 被测的是内存访问与集合通信，不是权重数值。

用法见同目录 1107_bench_dsv41_1node_engram_fetch_v1.sh。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# 必须在 import 模块之前设置：探针开关是模块级读取的
os.environ.setdefault("MY_ENV_DSV41_ENGRAM_FETCH_PROBE", "1")

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # 注册 npu backend

from torch.distributed.device_mesh import init_device_mesh

from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import (
    TorchHostOffloadEngram,
)

# 与 /efs_rl/weights/DeepSeek-V4.1-Flash-bf16/config.json 一致
EMBED_DIM = 256
LAYERS = ((1, 384006168), (14, 384016682))


def log(msg: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else -1
    if rank in (0, -1):
        print(f"[engram-bench] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=1, choices=[1, 8],
                    help="1=全尺寸表（单机 EP16 真实占用）；8=缩 8 倍（每 rank 行数等同 8 机 EP128）")
    ap.add_argument("--rows-per-rank", type=int, default=8192,
                    help="每次 fetch 的 row_ids 条数（= 每 rank 本 micro-batch 的 token 行数）")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % 16))
    torch.npu.set_device(local)

    mesh = init_device_mesh("npu", (world,))
    log(f"dist ready rank={rank} world={world} local_device={local} "
        f"scale={args.scale} rows_per_rank={args.rows_per_rank} "
        f"probe={os.environ.get('MY_ENV_DSV41_ENGRAM_FETCH_PROBE')} "
        f"probe_sync={os.environ.get('MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC')}")

    # ---- 建表（EP 按行切分，pinned host 表）----
    tables = []
    per_rank_bytes = 0
    t_alloc0 = time.perf_counter()
    for layer_id, rows in LAYERS:
        logical = rows // args.scale
        tbl = TorchHostOffloadEngram(
            logical,
            EMBED_DIM,
            layer_id=layer_id,
            num_max_tokens_per_rank=args.rows_per_rank,
            pin_memory=True,
        )
        tbl.shard_(mesh)
        per_rank_bytes += tbl.local_rows * EMBED_DIM * 2
        tables.append(tbl)
        log(f"table layer={layer_id} logical_rows={logical} physical_rows={tbl.num_embeddings} "
            f"local_rows={tbl.local_rows} row_start={tbl.row_start} "
            f"local_gib={tbl.local_rows * EMBED_DIM * 2 / 2**30:.3f} pinned={tbl.weight.is_pinned()}")
    torch.npu.synchronize()
    dist.barrier()
    t_alloc = time.perf_counter() - t_alloc0
    log(f"tables ready per_rank_total={per_rank_bytes / 2**30:.3f}GiB "
        f"node_total={per_rank_bytes * world / 2**30:.1f}GiB alloc_s={t_alloc:.1f}")

    # ---- 每 rank 每次 fetch 的 row_ids（真实路径就是 token 的 hash 行号，近似均匀）----
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)

    def make_ids(tbl):
        logical = tbl.logical_num_embeddings
        ids = torch.randint(0, logical, (args.rows_per_rank,), generator=gen, dtype=torch.int64)
        return ids.to(f"npu:{local}")

    fwd_ms, bwd_ms = [], []
    total_iters = args.warmup + args.iters
    for it in range(total_iters):
        ids_per_table = [make_ids(t) for t in tables]
        torch.npu.synchronize()
        t0 = time.perf_counter()
        outs = [tbl(ids) for tbl, ids in zip(tables, ids_per_table)]
        loss = sum(o.float().sum() for o in outs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        torch.npu.synchronize()
        t2 = time.perf_counter()
        for tbl in tables:
            tbl.clear_sparse_gradient()
        if it >= args.warmup:
            fwd_ms.append((t1 - t0) * 1000.0)
            bwd_ms.append((t2 - t1) * 1000.0)
        if rank == 0:
            log(f"iter {it + 1}/{total_iters} fwd_ms={(t1 - t0) * 1000:.2f} bwd_ms={(t2 - t1) * 1000:.2f}")

    def summary(name, xs):
        if not xs:
            return
        log(f"SUMMARY {name} n={len(xs)} mean_ms={statistics.mean(xs):.2f} "
            f"min_ms={min(xs):.2f} max_ms={max(xs):.2f} "
            f"p50_ms={statistics.median(xs):.2f}")

    summary("fwd2tables", fwd_ms)
    summary("bwd2tables", bwd_ms)
    if fwd_ms:
        log(f"SUMMARY per_fetch_mean_ms={(statistics.mean(fwd_ms) + statistics.mean(bwd_ms)) / (2 * len(tables)):.2f} "
            f"rows_per_rank={args.rows_per_rank} ep={world} scale={args.scale}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
