"""1103 Engram hash kernel 单卡复现（v1）。

背景：8 卡 dummy 推理在 `_hash_ids_kernel` 触发 vector core exception
（scalar instruction accesses an invalid GM address）。真实入参已确认全部在
NPU 上、地址范围合法（qsl=[0,15]、positions=0..14、block_table=[[3,0...]]、
cache 全 0、lookback 全 -1），因此需要用单卡同形状复现，再逐段二分定位。

variant:
  full    : 与线上完全一致（cache 为设备张量）
  nocache : cache=None（编译器走 else 分支，跳过 slot cache 读取）
  known   : lookback 填真实 token（known=True，needs_cache 全 False）
"""
import json
import os
import sys
import types

import torch
import torch_npu
import vllm_ascend.ops  # noqa: F401  # 必须先导入，避免 device_op <-> vllm_ascend.ops 循环导入
from vllm.models.deepseek_v41.common.engram import DEAD_ID, EngramLayout, compute_hash_multipliers
from vllm.triton_utils import triton
from vllm_ascend.ops.triton.engram_hash import _engram_req_index_kernel, _hash_ids_kernel


def log(tag, **kw):
    print("MY_DEBUG_LOG " + tag + " " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
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

    num_tokens = 15          # 与线上首个 prefill 一致
    num_layers = len(layout.layer_ids)
    max_ngram = layout.max_ngram_size
    num_heads = layout.n_heads
    cache_block_size = 128
    num_blocks = 36932       # cache numel 4727296 = 36932 * 128
    max_blocks = 13
    lookback_depth = max_ngram - 1

    input_ids = torch.randint(0, vocab_size, (num_tokens,), generator=gen, dtype=torch.int32).to(device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    dead_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    lookback_token_ids = torch.full((256, lookback_depth), -1, dtype=torch.int32, device=device)
    if variant == "known":
        lookback_token_ids[0] = input_ids[:lookback_depth]
    lookback_dead_mask = torch.zeros((256, lookback_depth), dtype=torch.bool, device=device)
    block_table = torch.zeros((1, max_blocks), dtype=torch.int32, device=device)
    block_table[0, 0] = 3
    cache = None if variant == "nocache" else torch.zeros(
        num_blocks * cache_block_size, dtype=torch.int32, device=device
    )
    token_map = torch.randint(
        0, layout.compressed_vocab_size, (vocab_size,), generator=gen, dtype=torch.int32
    ).to(device)
    multipliers = compute_hash_multipliers(layout.layer_ids, max_ngram, layout.compressed_vocab_size).to(device)
    primes = torch.tensor(layout.primes).to(device)
    offsets = layout.offsets.to(device)
    # 线上 engram_hash_001 打印 pad_id=2（真实 tokenizer 的压缩 id）。
    # Triton 会按标量实参做特化，pad_id 不同会编译出不同二进制，
    # 因此复现必须用与线上一致的常量，否则复现的不是同一个 kernel。
    pad_id = 2
    output = torch.empty(
        (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32, device=device
    )
    req_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)

    _engram_req_index_kernel[(triton.cdiv(num_tokens, 32),)](
        query_start_loc, req_ids, num_tokens, query_start_loc.numel() - 1, query_start_loc.stride(0), BLOCK_T=32
    )
    log("engram_repro_001", variant=variant, req_ids=req_ids.tolist(), dev=input_ids.device,
        ids_dtype=input_ids.dtype, pos_dtype=positions.dtype, dead_dtype=dead_mask.dtype,
        qsl=query_start_loc.tolist(), bt=block_table.tolist(),
        cache=None if cache is None else tuple(cache.shape), pad_id=pad_id)

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
        cache.shape[0] if cache is not None else 0,
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
    )
    torch.npu.synchronize()
    log("engram_repro_002", variant=variant, ok=1, out_sum=int(output.sum().item()))


if __name__ == "__main__":
    main()
