#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Split the pinned-input attention floor into q-projection vs attention-core.

Inputs: the probe dump with attention-op sub-stages (``probe_input_real4_fix_attn``,
variant ``in_all`` -> the attention module input is the engine's bit-exact value) and
the engine stage dump (``engine_real4``).

Engine attention runs **head-sharded across the 8 worker ranks** (q/core = (tokens, 8, 512)
per rank, all 200 tokens on every rank), while the trainer keeps all 64 heads locally.
The head mapping (contiguous rank-major vs stride) is picked empirically by minimum rel_err.

For every layer we report:
  eps_q      = rel_err(trainer query, concat engine q)      <- wq_a/q_norm/wq_b/rope kernel
  eps_core   = rel_err(trainer core output, concat engine)  <- everything inside SparseFlashMla
               given the (slightly different) q and each side's own KV stream
  amp        = eps_core / eps_q                             <- does the core add or just inherit
  eps_module = module output (attn vs self_attn) for reference
plus per-token/per-head structure of eps_core (uniform tail-free kernel noise, or
concentrated on a few heads/tokens = discrete selection disagreement).

Usage (CPU, ~1 min)::

    python3 scripts/dsv41_consistency/analyze_attn_floor_offline.py --length 200
"""

import argparse
import os

import torch


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def concat_heads(engine: dict, rank_key: str, n_ranks: int, mapping: str) -> torch.Tensor:
    parts = [engine[r][rank_key][0].float() for r in sorted(engine)]
    if mapping == "contig":  # rank r owns heads [r*h : (r+1)*h]
        return torch.cat(parts, dim=1)
    # stride: rank r owns heads r::8
    out = torch.empty(parts[0].shape[0], parts[0].shape[1] * len(parts), parts[0].shape[2])
    for r, p in enumerate(parts):
        out[:, r :: parts[0].shape[1], :] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/mnt/share/m00899630/dsv41/dump/probe_input_real4_fix_attn")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine_real4")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--variant", default="in_all")
    ap.add_argument("--ranks", type=int, default=8)
    args = ap.parse_args()

    t = torch.load(os.path.join(args.probe_dir, f"probe_len{args.length}_{args.variant}.pt"),
                   map_location="cpu", weights_only=False)
    stages = t["stages"]
    n_layers = len({k.split(".")[0] for k in stages if k.startswith("attn_op")})
    engine = {}
    for r in range(args.ranks):
        p = os.path.join(args.engine_dir, f"engine_stages_len{args.length}_rank{r}.pt")
        if os.path.exists(p):
            engine[r] = torch.load(p, map_location="cpu", weights_only=False)
    n_ranks = len(engine)

    # pick head mapping on layer-0 query
    q0 = stages["attn_op0.query"].squeeze(0).float()  # (T, H, D)
    cand = {}
    for mapping in ("contig", "stride"):
        cand[mapping] = rel_err(q0, concat_heads(engine, "DeepseekV41EagerAttentionImpl.multistream_preprocess@language_model.model.layers.0.self_attn[0]", n_ranks, mapping))
    mapping = min(cand, key=cand.get)
    print(f"[map] layer0 q rel_err: contiguous {cand['contig']:.4f} | strided {cand['stride']:.4f} -> using '{mapping}'")

    print(f"{'L':2} {'eps_q':>7} {'eps_core':>9} {'q≠0?':>5} {'eps_module':>11}   core per-token med/p99 | per-head rel_err spread")
    for L in range(n_layers):
        q = stages[f"attn_op{L}.query"].squeeze(0).float()
        o = stages[f"attn_op{L}.output"].squeeze(0).float()
        eq = concat_heads(engine, f"DeepseekV41EagerAttentionImpl.multistream_preprocess@language_model.model.layers.{L}.self_attn[0]", n_ranks, mapping)
        eo = concat_heads(engine, f"DeepseekV41EagerAttentionImpl._attention@language_model.model.layers.{L}.self_attn", n_ranks, mapping)
        eq = eq[: q.shape[0]]
        eo = eo[: o.shape[0]]
        eps_q = rel_err(q, eq)
        exact_q = (q == eq).all().item() or int((q != eq).sum())
        eps_core = rel_err(o, eo)
        tok = (o - eo).norm(dim=-1) / eo.norm(dim=-1).clamp_min(1e-9)
        heads = [(o[:, h] - eo[:, h]).norm() / eo[:, h].norm() for h in range(o.shape[1])]
        hv = torch.stack(heads)
        tr_a = stages[f"model.layers.{L}.attn"].squeeze(0).float()
        en_a = engine[0][f"language_model.model.layers.{L}.self_attn"][0].float()
        eps_mod = rel_err(tr_a, en_a)
        print(
            f"{L:<2} {eps_q:7.4f} {eps_core:9.4f} {'0' if exact_q is True else exact_q!s:>5} {eps_mod:11.4f}   "
            f"{tok.median():.4f}/{torch.quantile(tok, 0.99):.4f} | {hv.min():.4f}..{hv.max():.4f}  "
            f"q max|Δ| {(q - eq).abs().max():.2e}"
        )

    # sink: trainer passes the fp32 per-head sink vector into the op; check it equals the
    # checkpoint value the engine holds (attn_sink params are ckpt-fp32, bf16-rounded on the
    # trainer -- the negative-control group of the gate.bias fix).
    for L in range(n_layers):
        sink = stages.get(f"attn_op{L}.attn_sink")
        if sink is None:
            continue
        s_bf = sink.float().bfloat16().float()
        print(f"[sink L{L}] shape {tuple(sink.shape)}  |sink - bf16(sink)| max {(sink - s_bf).abs().max():.3e} "
              f"(rel {(sink - s_bf).norm()/sink.norm():.3e})")


if __name__ == "__main__":
    main()
