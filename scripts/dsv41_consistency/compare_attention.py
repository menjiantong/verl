"""Compare the attention internals of the two stacks, layer by layer.

The trainer dump records what `indexed_sparse_attention` received and returned per layer
(`attn_op<N>.{query,key_value,attn_sink,topk_indices,output}`); the engine dump records the
eager attention implementation's q (`multistream_preprocess`), its selected sparse indices
(`_select_sparse_indices`) and its output (`_attention`). Both are put side by side here so
a divergence can be attributed to the projection/rope, the selection, or the kernel math.

Notes on layout: the trainer runs with world_size=1 (all 64 heads), the engine shards heads
over TP=8 (8 heads per rank), so only rank 0's heads are compared -- head order is assumed
to be the same contiguous split on both sides (checked by the q comparison itself).

Usage:
    python3 scripts/dsv41_consistency/compare_attention.py --len 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def stats(ref: torch.Tensor, other: torch.Tensor) -> str:
    ref_flat, other_flat = ref.float().reshape(-1), other.float().reshape(-1)
    if ref_flat.numel() != other_flat.numel():
        return f"size mismatch {tuple(ref.shape)} vs {tuple(other.shape)}"
    diff = other_flat - ref_flat
    cos = torch.nn.functional.cosine_similarity(ref_flat, other_flat, dim=0).item()
    return (f"rel_err={diff.norm().item() / (ref_flat.norm().item() + 1e-12):.4f} "
            f"max|d|={diff.abs().max().item():.4f} cos={cos:.4f}")


def first(store, key):
    value = store.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--trainer-dir", default="/mnt/share/m00899630/dsv41/dump/trainer")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--heads", type=int, default=8, help="local heads per engine rank to compare")
    args = ap.parse_args()

    trainer = torch.load(Path(args.trainer_dir) / f"trainer_len{args.len}.pt", map_location="cpu")["stages"]
    engine_path = Path(args.engine_dir) / f"engine_stages_len{args.len}_rank0.pt"
    engine = torch.load(engine_path, map_location="cpu")

    for layer in range(8):
        q_t = first(trainer, f"attn_op{layer}.query")
        if q_t is None:
            break
        q_e = first(engine, "DeepseekV41EagerAttentionImpl.multistream_preprocess[0]")
        out_t = first(trainer, f"attn_op{layer}.output")
        out_e = first(engine, "DeepseekV41EagerAttentionImpl._attention")
        sink_t = first(trainer, f"attn_op{layer}.attn_sink")
        idx_t = first(trainer, f"attn_op{layer}.topk_indices")
        idx_e = first(engine, "DeepseekV41EagerAttentionImpl._select_sparse_indices")
        print(f"--- layer {layer}")
        print(f"  trainer q {tuple(q_t.shape)}  kv {tuple(first(trainer, f'attn_op{layer}.key_value').shape)} "
              f"indices {tuple(idx_t.shape)}  output {tuple(out_t.shape)}")
        if q_e is not None:
            # trainer q: [1, S, 64, 512] -> local heads; engine q: [S, 8, 512]
            ref = q_t.reshape(q_t.shape[1], q_t.shape[2], q_t.shape[3])[:, : args.heads, :]
            print(f"  q        : {stats(ref, q_e)}")
        if out_e is not None:
            ref = out_t.reshape(out_t.shape[1], out_t.shape[2], out_t.shape[3])[:, : args.heads, :]
            print(f"  attn out : {stats(ref, out_e)}")
        if idx_e is not None:
            print(f"  engine selected indices: shape {tuple(idx_e.shape)} "
                  f"dtype {idx_e.dtype} min {idx_e.min().item()} max {idx_e.max().item()}")
            print(f"  trainer indices row0: {idx_t.reshape(idx_t.shape[1], -1)[0][:16].tolist()} ...")
            print(f"  engine  indices row0: {idx_e[0][:16].tolist()} ...")
        if sink_t is not None:
            print(f"  trainer sink[0:4]={sink_t[:4].tolist()} (engine sink is part of the fused kernel)")
        print()


if __name__ == "__main__":
    main()
