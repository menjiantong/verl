"""1103 Engram hash kernel 单卡复现 v2。

v1 结论：默认流上、与线上相同的入参（pad_id=2、cache 设备张量、lookback 全 -1）
跑 `_hash_ids_kernel` 通过，且命中了线上 13:52 编译的同一份缓存二进制
（cache key hash fb29a7d8…），说明不是 kernel 二进制/入参值的问题。

线上唯一未覆盖的差异：vllm-ascend 的 multistream engram overlap 会把
`prepare_engram`（含 hash/lookup）放到 `_engram_prepare_stream` 上执行
（plog 里 fault 的 stream_id=46，非默认流）。本版增加 aux-stream 变量，
并补上线上实际会先跑的 `_write_hash_cache_kernel`。

variant:
  full : 默认流，req_index + write_hash_cache + hash_ids（线上 kernel 序列）
  aux  : 同上，但在独立 torch.npu.Stream() 上执行（复刻 overlap 路径）
  uva  : 先注册两张 22.89 GiB HostUvaBuffer（层 1/14 的 per-rank 表），再跑 full
"""
import json
import os
import sys
import types

import torch
import torch_npu
import vllm_ascend.ops  # noqa: F401  # 先导入，避免 device_op <-> vllm_ascend.ops 循环导入
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    _write_hash_cache_kernel,
    compute_hash_multipliers,
)
from vllm.triton_utils import triton
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.ops.triton.engram_hash import _engram_req_index_kernel, _hash_ids_kernel


def log(tag, **kw):
    print("MY_DEBUG_LOG " + tag + " " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    stage = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # 8=原始完整实现
    model_dir = os.environ["MY_ENV_DSV41_ENGRAM_MODEL"]
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = raw if "engram_layer_ids" in raw else raw["text_config"]
    keys = [
        "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
        "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size",
        "engram_pad_token_id", "engram_vocab_size",
    ]
    config = types.SimpleNamespace(**{k: cfg[k] for k in keys})
    layout = EngramLayout(config)
    vocab_size = int(cfg.get("vocab_size", 129280))

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    gen = torch.Generator().manual_seed(20261103)

    # uva 变体：先按真实 per-rank 尺寸注册两张主机表（48,000,217 / 48,001,465 行 x 256），
    # 复刻 vLLM 进程里先有 45.8 GiB host UVA 映射、再申请小张量并跑 hash kernel 的顺序。
    uva_bufs = []
    if variant == "uva":
        for rows in (48_000_217, 48_001_465):
            uva_bufs.append(HostUvaBuffer((rows, 256), torch.bfloat16, device))
        log("engram_repro3_000", variant=variant, bufs=len(uva_bufs),
            gib=[round(b.tensor.numel() * 2 / 1024**3, 2) for b in uva_bufs])

    num_tokens = 15
    num_layers = len(layout.layer_ids)
    max_ngram = layout.max_ngram_size
    num_heads = layout.n_heads
    cache_block_size = 128
    num_blocks = 36932
    max_blocks = 13
    lookback_depth = max_ngram - 1

    input_ids = torch.randint(0, vocab_size, (num_tokens,), generator=gen, dtype=torch.int32).to(device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    dead_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    lookback_token_ids = torch.full((256, lookback_depth), -1, dtype=torch.int32, device=device)
    lookback_dead_mask = torch.zeros((256, lookback_depth), dtype=torch.bool, device=device)
    block_table = torch.zeros((1, max_blocks), dtype=torch.int32, device=device)
    block_table[0, 0] = 3
    cache = torch.zeros(num_blocks * cache_block_size, dtype=torch.int32, device=device)
    # 线上首个 prefill 的 slot_mapping：block 3 -> 物理 slot 384..398
    slot_mapping = (3 * cache_block_size + torch.arange(num_tokens, dtype=torch.int32, device=device))
    token_map = torch.randint(
        0, layout.compressed_vocab_size, (vocab_size,), generator=gen, dtype=torch.int32
    ).to(device)
    multipliers = compute_hash_multipliers(layout.layer_ids, max_ngram, layout.compressed_vocab_size).to(device)
    primes = torch.tensor(layout.primes).to(device)
    offsets = layout.offsets.to(device)
    pad_id = 2  # 与线上 engram_hash_001 打印一致
    output = torch.empty(
        (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32, device=device
    )
    req_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)

    def launch_all():
        _engram_req_index_kernel[(triton.cdiv(num_tokens, 32),)](
            query_start_loc, req_ids, num_tokens, query_start_loc.numel() - 1, query_start_loc.stride(0), BLOCK_T=32
        )
        _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
            input_ids,
            token_map,
            dead_mask,
            slot_mapping,
            cache,
            num_tokens,
            input_ids.stride(0),
            dead_mask.stride(0),
            slot_mapping.stride(0),
            256,
            DEAD_ID,
        )
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            req_ids,
            multipliers,
            primes,
            offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0],
            pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0),
            table_col_stride=block_table.stride(1),
            query_stride=query_start_loc.stride(0),
            num_table_rows=block_table.shape[0],
            max_blocks=block_table.shape[1],
            cache_block_size=cache_block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_depth,
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
            STAGE=stage,
        )

    main_stream = torch.npu.current_stream()
    if variant == "uva":
        variant = "full"
    aux = None
    if variant == "aux":
        aux = torch.npu.Stream(device=device)
        aux.wait_stream(main_stream)
        log("engram_repro2_001", variant=variant, main_sid=main_stream.stream_id, aux_sid=aux.stream_id)
    else:
        log("engram_repro2_001", variant=variant, main_sid=main_stream.stream_id)
    if aux is not None:
        with torch.npu.stream(aux):
            launch_all()
        main_stream.wait_stream(aux)
    else:
        launch_all()
    torch.npu.synchronize()
    log("engram_repro2_002", variant=variant, ok=1, req_ids=req_ids.tolist(), out_sum=int(output.sum().item()))


if __name__ == "__main__":
    main()
