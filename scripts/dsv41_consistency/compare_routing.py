"""Compare MoE routing (gate scores and selected experts) between the two stacks.

The trainer dump carries the gate output (`model.layers.N.ffn.gate[0]` = top-k weights,
`[1]` = expert ids) and the gate's input (`model.layers.N.ffn_norm`), so the raw router
logits can be recomputed offline from the checkpoint. The engine dump carries the router's
own logits and its (weights, ids) through the `_compute_routing` wrapper. Matching top-k
sets mean the MoE sees the same experts; differing sets are a discrete source of train-vs-
inference divergence that no amount of precision tuning will hide.

Usage:
    python3 scripts/dsv41_consistency/compare_routing.py --len 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--trainer-dir", default="/mnt/share/m00899630/dsv41/dump/trainer")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32")
    ap.add_argument("--trainer-tag", default="")
    return ap.parse_args()


def load_gate_weight(model_path: str, layer: int) -> torch.Tensor:
    from safetensors import safe_open

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
    name = f"layers.{layer}.ffn.gate.weight"
    with safe_open(Path(model_path) / index[name], framework="pt") as handle:
        return handle.get_tensor(name).float()


def first(store, key):
    value = store.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def main():
    args = parse_args()
    trainer = torch.load(Path(args.trainer_dir) / f"trainer{args.trainer_tag}_len{args.len}.pt",
                         map_location="cpu")["stages"]
    engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{args.len}_rank0.pt", map_location="cpu")

    print(f"=== routing, len={args.len} ===")
    for layer in range(8):
        ids_t = first(trainer, f"model.layers.{layer}.ffn.gate[1]")
        if ids_t is None:
            break
        weights_t = first(trainer, f"model.layers.{layer}.ffn.gate[0]")
        norm_out = first(trainer, f"model.layers.{layer}.ffn_norm")
        hidden = norm_out.reshape(-1, norm_out.shape[-1]).float()
        gate_w = load_gate_weight(args.model_path, layer)
        raw_t = hidden @ gate_w.t()  # gate_temp is 1.0

        # engine side: the router wrapper records outputs [0]=weights [1]=ids and the logits
        prefix = f"language_model.model.layers.{layer}.mlp"
        engine_keys = [k for k in engine if "compute_routing" in k and prefix in k]
        if not engine_keys:
            print(f"  layer {layer}: no engine router capture (keys: "
                  f"{[k for k in engine if 'compute_routing' in k][:4]})")
            continue
        ids_e = first(engine, f"{engine_keys[0].rsplit('[', 1)[0]}[1]") if "[1]" in engine_keys[0] else None
        logits_e = None
        for key in engine_keys:
            if "#arg1" in key:
                logits_e = first(engine, key)
        if ids_e is None:
            base = next((k for k in engine_keys if k.endswith("[1]")), None)
            ids_e = first(engine, base) if base else None
        weights_e = None
        base0 = next((k for k in engine_keys if k.endswith("[0]")), None)
        if base0:
            weights_e = first(engine, base0)

        print(f"--- layer {layer}")
        if logits_e is not None:
            ref = raw_t
            other = logits_e.reshape(-1, logits_e.shape[-1]).float()
            if ref.shape == other.shape:
                diff = (other - ref).abs()
                print(f"  router logits: shape {tuple(ref.shape)} mean|d|={diff.mean():.5f} "
                      f"max|d|={diff.max():.5f} rel={diff.norm() / (ref.norm() + 1e-12):.5f}")
            else:
                print(f"  router logits shape mismatch: trainer {tuple(ref.shape)} engine {tuple(other.shape)}")
        if ids_t is not None and ids_e is not None:
            a = ids_t.long().reshape(ids_t.shape[-2], -1)
            b = ids_e.long().reshape(ids_e.shape[0], -1)
            same_rows = (a == b).all(dim=-1).float().mean().item() if a.shape == b.shape else float("nan")
            overlap = ((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)).float().mean().item()
            print(f"  selected expert ids: exact top-k set match {same_rows * 100:.2f}% of tokens, "
                  f"per-slot overlap {overlap * 100:.2f}%")
            if a.shape == b.shape and same_rows < 1.0:
                different = (a != b).any(dim=-1).nonzero().flatten()[:3]
                for row in different.tolist():
                    print(f"    token {row}: trainer {a[row].tolist()} vs engine {b[row].tolist()}")
        if weights_t is not None and weights_e is not None:
            wt = weights_t.float().reshape(-1, weights_t.shape[-1])
            we = weights_e.float().reshape(-1, weights_e.shape[-1])
            if wt.shape == we.shape:
                print(f"  top-k weights: mean|d|={(we - wt).abs().mean():.5f}")


if __name__ == "__main__":
    main()
