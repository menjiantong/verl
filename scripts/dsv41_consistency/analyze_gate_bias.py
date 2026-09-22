"""Quantify the *parameter-precision* asymmetry of the MoE router (real-weights H2).

The real checkpoint stores `layers.<i>.ffn.gate.bias` (= engine's `e_score_correction_bias`)
in **fp32**, but `prepare_deepseek_v41_model_for_fsdp` normalizes every floating parameter to
bf16 (`adapter.py:395-398`) -- the architecture's own `Gate.__init__` declares it
`torch.float32` (`model.py:936`). The engine keeps fp32. Bias enters *selection only*
(`indices = (scores + bias).topk(...)`; the weights come from the unbiased scores), so the
entire effect of that rounding is discrete: tokens whose top-k expert set changes.

Random fixtures cannot show this (there the tensor is bf16 on both sides), which is why the
4-layer real-weights probe's forced-input MoE residual (0.9-4.5%, vs a uniform 0.54% floor
under random weights) has no counterpart in the earlier runs.

This script measures the effect offline on CPU, no NPU needed:

* `x`       = the engine's MoE input (`DeepseekV41DecoderLayer.rms_norm_cast@layer<i>[0]`,
              i.e. the exact tensor the probe's `in.all.ffn` variant forces into the trainer),
* `W, bias` = the slice's `layers.<i>.ffn.gate.{weight,bias}`,
* top-k sets are computed twice -- bias as fp32 (engine) vs bias rounded to bf16 (trainer) --
  and the differing sets are counted.

The engine's own router capture (8 decode-recompute tokens) is used to validate the recipe
(score function / gate temp) before any flip count is believed.

Usage:
    python3 scripts/dsv41_consistency/analyze_gate_bias.py --lengths 64,200 \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real \
        --engine-dir /mnt/share/m00899630/dsv41/dump/engine_real4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="64,200")
    ap.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine_real4")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--gate-temp", type=float, default=1.0)
    return ap.parse_args()


def unwrap(store, key):
    value = store.get(key)
    if isinstance(value, list):
        value = value[0]
    return value


def ckpt_tensor(model_path: str, name: str) -> torch.Tensor:
    from safetensors import safe_open

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
    with safe_open(Path(model_path) / index[name], framework="pt") as handle:
        return handle.get_tensor(name)


def score(logits: torch.Tensor, func: str) -> torch.Tensor:
    if func == "softmax":
        return logits.softmax(dim=-1)
    if func == "sigmoid":
        return logits.sigmoid()
    return F.softplus(logits).sqrt()  # "sqrtsoftplus" (V3-style)


def topk_sets(choice: torch.Tensor, k: int) -> list[set]:
    return [set(row.tolist()) for row in choice.topk(k, dim=-1)[1]]


def validate_recipe(engine, model_path, args) -> str:
    """Pick the score function that reproduces the engine's captured router decisions."""
    hidden = unwrap(engine, "AscendFusedTopKRouter._compute_routing#arg0").float()
    captured_ids = unwrap(engine, "AscendFusedTopKRouter._compute_routing[1]").long()
    print("recipe validation on the engine's own 8 router-capture tokens "
          f"(input {tuple(hidden.shape)}):")
    best = None
    for layer in range(4):
        weight = ckpt_tensor(model_path, f"layers.{layer}.ffn.gate.weight").float()
        bias = ckpt_tensor(model_path, f"layers.{layer}.ffn.gate.bias")
        logits = hidden @ weight.t() / args.gate_temp
        for func in ("sigmoid", "softmax", "sqrtsoftplus"):
            ids = (score(logits, func) + bias.float()).topk(args.topk, dim=-1)[1]
            match = (ids == captured_ids).all(dim=-1).float().mean().item()
            print(f"  layer {layer} {func:<12} exact top-{args.topk} match {match * 100:6.1f}%")
            if best is None or match > best[0]:
                best = (match, layer, func)
    print(f"  -> best: layer {best[1]}, score_func={best[2]} ({best[0] * 100:.1f}%)\n")
    return best[2]


def main():
    args = parse_args()
    for length in (int(x) for x in args.lengths.split(",") if x):
        engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{length}_rank0.pt",
                            map_location="cpu", weights_only=False)
        func = validate_recipe(engine, args.model_path, args) if length == int(args.lengths.split(",")[0]) else None
        print(f"===== len={length}: router bias precision, fp32 (engine) vs bf16 (trainer)")
        print(f"{'layer':<7} {'tokens':>7} {'flips':>7} {'flip rate':>10} {'slot overlap':>13} "
              f"{'bias |d|max':>12} {'bias |d|mean':>13}")
        for layer in range(4):
            x = unwrap(engine, f"DeepseekV41DecoderLayer.rms_norm_cast@layer{layer}[0]").float()
            x = x.reshape(-1, x.shape[-1])
            weight = ckpt_tensor(args.model_path, f"layers.{layer}.ffn.gate.weight").float()
            bias_fp32 = ckpt_tensor(args.model_path, f"layers.{layer}.ffn.gate.bias").float()
            bias_bf16 = bias_fp32.to(torch.bfloat16).float()
            logits = x @ weight.t() / args.gate_temp
            base = score(logits, func or "sigmoid")
            sets_fp32 = topk_sets(base + bias_fp32, args.topk)
            sets_bf16 = topk_sets(base + bias_bf16, args.topk)
            same = [a == b for a, b in zip(sets_fp32, sets_bf16)]
            overlap = sum(len(a & b) for a, b in zip(sets_fp32, sets_bf16)) / (len(sets_fp32) * args.topk)
            delta = (bias_bf16 - bias_fp32).abs()
            print(f"{layer:<7} {len(same):>7} {same.count(False):>7} "
                  f"{1 - sum(same) / len(same):>10.4f} {overlap:>13.4f} "
                  f"{delta.max().item():>12.4f} {delta.mean().item():>13.5f}")
        print()


if __name__ == "__main__":
    main()
