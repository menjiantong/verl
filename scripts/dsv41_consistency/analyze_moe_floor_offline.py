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
"""Offline decomposition of the post-fix "pinned-input floor" into sub-modules.

Answers, from the EXISTING dumps only (no NPU): with every module input pinned
to the engine's bit-exact values (probe variant ``in_all``), where does the
residual engine-vs-trainer difference actually live?

  * Layer-0 router decisions: engine ``AscendFusedTopKRouter._compute_routing``
    (sharded across the 8 DP ranks as (25, 6) per rank) vs the trainer's own
    ``ffn.gate[0]`` (weights) / ``ffn.gate[1]`` (ids). The engine hook fires once,
    so this covers layer 0 only -- which is exactly the layer whose MoE floor is
    free of any upstream input contamination.
  * MoE: trainer ``ffn.experts`` + ``ffn.shared_experts`` vs engine ``mlp.experts``
    (Ascend fused MoE output, shared expert included -- see the engine-dump trap
    note in wiki/worklog_dsv41_module_input_probe.md). Splits the per-layer floor
    by routed/shared magnitude and by token-error shape (uniform vs tail).
  * attn: trainer ``attn`` vs engine ``self_attn`` module outputs (engine-side
    sub-stages exist in the dump but the trainer probe does not record attn
    internals -- that half needs the next probe run with sub-stage hooks).
  * Output tail: ``model.norm`` vs engine ``model.norm``, ``model.head`` vs
    engine ``logits_processor``.

Usage (CPU, ~1 min, ~7 GB host for the 8 engine rank files)::

    python3 scripts/dsv41_consistency/analyze_moe_floor_offline.py \
        --probe-dir /mnt/share/m00899630/dsv41/dump/probe_input_real4_fix \
        --engine-dir /mnt/share/m00899630/dsv41/dump/engine_real4 \
        --length 200 --variant in_all
"""

import argparse
import os

import torch


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """‖a − b‖ / ‖b‖ with b as the reference (engine = the side we pin to)."""
    a = a.float().flatten()
    b = b.float().flatten()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def token_rel_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token rel_err over the last (hidden) dim; a, b: (tokens, hidden)."""
    a = a.float()
    b = b.float()
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)


def squeeze(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t


def describe_per_token(err: torch.Tensor) -> str:
    q = torch.quantile(err, torch.tensor([0.5, 0.9, 0.99]))
    top = err.pow(2).topk(max(1, err.numel() // 100)).values.sum().item()
    tot = err.pow(2).sum().item()
    return f"med {q[0]:.4f} p90 {q[1]:.4f} p99 {q[2]:.4f} top1%energy {top / tot:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/mnt/share/m00899630/dsv41/dump/probe_input_real4_fix")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine_real4")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--variant", default="in_all")
    ap.add_argument("--ranks", type=int, default=8, help="engine DP-rank files to load (router reconstruction)")
    args = ap.parse_args()

    probe_path = os.path.join(args.probe_dir, f"probe_len{args.length}_{args.variant}.pt")
    print(f"[analyze] trainer: {probe_path}")
    t = torch.load(probe_path, map_location="cpu", weights_only=False)
    stages = t["stages"]
    n_tokens = args.length

    engine = {}
    for r in range(args.ranks):
        ep = os.path.join(args.engine_dir, f"engine_stages_len{args.length}_rank{r}.pt")
        if not os.path.exists(ep):
            break
        engine[r] = torch.load(ep, map_location="cpu", weights_only=False)
    print(f"[analyze] engine ranks loaded: {sorted(engine)}")
    e0 = engine[0]

    # ---- 1. layer-0 router: engine (25,6) per rank -> (200,6), compared to trainer gate ----
    rkey_w = "AscendFusedTopKRouter._compute_routing[0]"
    rkey_i = "AscendFusedTopKRouter._compute_routing[1]"
    akey = "AscendFusedTopKRouter._compute_routing#arg0"  # the router's hidden input, for token mapping
    if rkey_w in e0:
        chunk = e0[rkey_w][0].shape[0]
        ids = torch.cat([engine[r][rkey_i][0] for r in sorted(engine)], 0)
        wts = torch.cat([engine[r][rkey_w][0] for r in sorted(engine)], 0)
        arg = torch.cat([engine[r][akey][0] for r in sorted(engine)], 0)
        tw = stages["model.layers.0.ffn.gate[0]"]
        ti = stages["model.layers.0.ffn.gate[1]"]
        # Verify the rank-contiguous token mapping by matching the router INPUT against
        # the trainer's layer-0 ffn input (in_all pins it bit-exactly).
        ref = squeeze(stages["model.layers.0.ffn_norm"]).float()
        gate_in = arg - ref
        map_ok = torch.equal(arg, ref)
        print(
            f"[router L0] shard={chunk}x{args.ranks}={ids.shape[0]} (engine router-input vs trainer ffn_norm: "
            f"{'bit-exact' if map_ok else f'max|Δ| {gate_in.abs().max():.2e} rel_err {rel_err(arg, ref):.2e}'})"
        )
        same_id = (ids.long() == ti.long()).all(dim=1)
        slot_overlap = (ids.long().unsqueeze(1) == ti.long().unsqueeze(2)).any(dim=2).float().sum(1).mean()
        werr = (wts - tw).abs()
        print(
            f"[router L0] id-set identical: {same_id.float().mean():.4f} ({int(same_id.sum())}/{len(same_id)})  "
            f"slot overlap {slot_overlap:.4f}  |Δw| max {werr.max():.2e} mean {werr.mean():.2e}"
        )
        # Is the engine's id ORDER identical too (slots)?
        exact_order = (ids.long() == ti.long()).all(dim=1)
        print(f"[router L0] exact slot-order identical: {exact_order.float().mean():.4f}")

    # ---- 2. MoE per layer: trainer routed+shared vs engine fused mlp.experts ----
    print(f"{'layer':6} {'ffn vs .mlp':>11} {'r+s vs .experts':>15} {'‖routed‖':>9} {'‖shared‖':>9} {'top1%':>6}  per-token (routed+shared vs engine fused)")
    for L in range(4):
        tr_ffn = squeeze(stages[f"model.layers.{L}.ffn"]).float()
        tr_r = squeeze(stages[f"model.layers.{L}.ffn.experts"]).float()
        tr_s = squeeze(stages[f"model.layers.{L}.ffn.shared_experts"]).float()
        en_mlp = e0[f"language_model.model.layers.{L}.mlp"][0].float()
        en_exp = e0[f"language_model.model.layers.{L}.mlp.experts"][0].float()
        comb = tr_r + tr_s
        err = token_rel_err(comb, en_exp)
        print(
            f"L{L:<5} {rel_err(tr_ffn, en_mlp):11.4f} {rel_err(comb, en_exp):15.4f} "
            f"{tr_r.norm():9.0f} {tr_s.norm():9.0f} {(err.topk(max(1, n_tokens // 100)).values.pow(2).sum() / err.pow(2).sum()):6.2f}  "
            + describe_per_token(err)
        )
        # Token-error shape: is the floor concentrated?
        if L == 0:
            worst = err.topk(6).indices.tolist()
            print(f"       worst-6 tokens (L0): {worst}")

    # ---- 3. attn per layer (module boundary only) ----
    print(f"{'layer':6} {'attn vs self_attn':>17}")
    for L in range(4):
        tr_a = squeeze(stages[f"model.layers.{L}.attn"]).float()
        en_a = e0[f"language_model.model.layers.{L}.self_attn"][0].float()
        err = token_rel_err(tr_a, en_a)
        print(f"L{L:<5} {rel_err(tr_a, en_a):17.4f}   " + describe_per_token(err))

    # ---- 4. output tail: norm + head ----
    if "model.norm" in stages and "language_model.model.norm" in e0:
        print(f"[tail] model.norm rel_err {rel_err(squeeze(stages['model.norm']).float(), e0['language_model.model.norm'][0].float()):.4f}")
    if "model.head" in stages and "language_model.logits_processor" in e0:
        # Engine-side logits are decode-phase (single row) -> locate its trainer row and compare.
        lh = e0["language_model.logits_processor"][0].float()
        th = squeeze(stages["model.head"]).float()
        lh = lh.reshape(-1, th.shape[-1])
        d = (th - lh).abs().amax(dim=1)
        row = int(d.argmin())
        print(
            f"[tail] logits: engine decode-row vs trainer row {row} (of {th.shape[0]}): "
            f"rel_err {rel_err(th[row], lh[0]):.4f}  next-token argmax agree: "
            f"{int(th[row].argmax() == lh[0].argmax())}"
        )

    # ---- 5. logprob/argmax on the pinned variant (already known, recomputed for the record) ----
    lp = t.get("next_logprobs")
    print(f"[done] trainer in_all next_logprobs mean {lp.float().mean():.4f} (dump present: {lp.shape})")


if __name__ == "__main__":
    main()
