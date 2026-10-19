#!/usr/bin/env python3
"""Compressed-index (sparse attention) consistency: trainer vs engine.

Compares the per-token set of compressed positions each stack attends to:

* trainer: ``attn_opN.topk_indices`` -- columns [128:] hold compressed ids **numbered
  S + idx** (S = sequence length; the first S ids are raw positions, e.g. len700 ->
  700..799); -1 = padding.
* engine: ``AscendDSAV41Impl._select_sparse_indices@...layers.N.self_attn`` output --
  compressed ids are **0-based** (0..99); -1 = padding, padded to index_topk=512.

The two numbering bases differ -- forgetting that yields a bogus "0% match".

Usage:
    python3 scripts/dsv41_consistency/compare_sparse_indices.py --len 700 \
        --trainer-dir /mnt/share/m00899630/dsv41/dump/slice4/trainer \
        --engine-dir  /mnt/share/m00899630/dsv41/dump/slice4/engine
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def as2(value):
    t = value[0] if isinstance(value, list) else value
    return t[0] if t.dim() == 3 else t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--trainer-dir", default="/mnt/share/m00899630/dsv41/dump/slice4/trainer")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/slice4/engine")
    ap.add_argument("--engine-rank", type=int, default=0)
    ap.add_argument("--layers", default="2,3", help="compressed layers to compare")
    args = ap.parse_args()

    trainer = torch.load(Path(args.trainer_dir) / f"trainer_len{args.len}.pt",
                         map_location="cpu")["stages"]
    engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{args.len}_rank{args.engine_rank}.pt",
                        map_location="cpu", weights_only=False)
    S = args.len
    print(f"=== compressed indices, len={S} (trainer base={S}, engine base=0) ===")
    for layer in (int(x) for x in args.layers.split(",")):
        key_t = f"attn_op{layer}.topk_indices"
        key_e = (f"AscendDSAV41Impl._select_sparse_indices@"
                 f"language_model.model.layers.{layer}.self_attn")
        if key_t not in trainer or key_e not in engine:
            print(f"  layer {layer}: missing capture (trainer {key_t in trainer}, engine {key_e in engine})")
            continue
        t = as2(trainer[key_t]).long()
        e = as2(engine[key_e]).long()
        exact = 0
        jac = 0.0
        n = 0
        for i in range(t.shape[0]):
            st = {v.item() - S for v in t[i] if v.item() >= S}
            se = {v.item() for v in e[i] if v.item() >= 0}
            if not st and not se:
                continue
            n += 1
            exact += (st == se)
            jac += len(st & se) / max(1, len(st | se))
        print(f"  layer {layer}: compared={n}  exact set match={exact}/{n} "
              f"({100 * exact / max(1, n):.1f}%)  mean Jaccard={jac / max(1, n):.3f}")


if __name__ == "__main__":
    main()
