"""Verify the weights the engine holds against the checkpoint on disk.

`dump_engine_stages.py` fingerprints every engine parameter it was asked about
(`params.json`); this script maps each engine parameter to the checkpoint tensor it must
be (undoing TP/EP sharding and the Ascend layout transforms) and compares the
permutation-invariant fingerprint (sum / std / absmax). Equal fingerprints mean the engine
is holding exactly the checkpoint values -- which separates "the loader/sync is wrong"
from "the two stacks compute differently".

    python3 scripts/dsv41_consistency/check_engine_params.py \
        --params /mnt/share/m00899630/dsv41/dump/engine/params.json \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="/mnt/share/m00899630/dsv41/dump/engine/params.json")
    ap.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32")
    ap.add_argument("--rtol", type=float, default=0.02, help="relative tolerance on std/absmax/mean")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--n-experts", type=int, default=0,
                    help="Override the routed-expert count (default: read n_routed_experts from "
                         "--model-path/config.json). The old hard-coded 32 silently mis-checks "
                         "384-expert checkpoints.")
    return ap.parse_args()


def num_routed_experts(model_path: str, override: int) -> int:
    """The checkpoint's routed-expert count, so the EP shard mapping is right for any model."""
    if override:
        return override
    config = json.loads((Path(model_path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    return int(text_config["n_routed_experts"])


class Checkpoint:
    def __init__(self, model_path: str):
        from safetensors import safe_open

        self.dir = Path(model_path)
        index = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self.weight_map = index
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._handles:
            from safetensors import safe_open

            self._handles[shard] = safe_open(self.dir / shard, framework="pt")
        return self._handles[shard].get_tensor(name)


def fingerprint(tensor: torch.Tensor) -> dict:
    flat = tensor.float().reshape(-1)
    return {
        "std": float(flat.std().item()) if flat.numel() > 1 else 0.0,
        "absmax": float(flat.abs().max().item()),
        "mean": float(flat.mean().item()),
        "shape": list(tensor.shape),
    }


def close(a: float, b: float, rtol: float, atol: float = 1e-6) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare(name: str, engine: dict, candidates: list[tuple[str, torch.Tensor]], rtol: float):
    """Compare an engine fingerprint against candidate checkpoint tensors."""
    best = None
    for label, tensor in candidates:
        if tensor is None:
            continue
        ref = fingerprint(tensor)
        ok = all(close(engine[k], ref[k], rtol) for k in ("std", "absmax", "mean"))
        score = sum(abs(engine[k] - ref[k]) for k in ("std", "absmax"))
        if best is None or score < best[0]:
            best = (score, label, ref, ok)
        if ok:
            return True, label, ref
    if best is None:
        return False, "<no candidate>", None
    return False, best[1], best[2]


def main():
    args = parse_args()
    payload = json.load(open(args.params))
    ckpt = Checkpoint(args.model_path)

    results = []
    for rank_entry in payload:
        rank = rank_entry["rank"]
        for engine in rank_entry["params"]:
            name = engine["name"]
            candidates = []

            def add(label, tensor):
                if tensor is not None:
                    candidates.append((label, tensor))

            def ck(name_ck: str, dim: int | None = None):
                tensor = ckpt.get(name_ck) if name_ck in ckpt.weight_map else None
                if tensor is not None and dim is not None:
                    chunk = tensor.shape[dim] // args.tp
                    sl = slice(rank * chunk, (rank + 1) * chunk)
                    tensor = tensor[sl] if dim == 0 else tensor[:, sl]
                return tensor

            m = re.match(r"language_model\.model\.layers\.(\d+)\.(.+)", name)
            if name == "language_model.model.embed_tokens.weight":
                add("embed.weight[tp shard0]", ck("embed.weight", 0))
            elif name == "language_model.lm_head.weight":
                add("head.weight[tp shard0]", ck("head.weight", 0))
            elif name == "language_model.model.norm.weight":
                add("norm?", ck("norm.weight") if "norm.weight" in ckpt.weight_map else None)
            elif name.startswith("vision.") or name.startswith("aligner."):
                # The VL wrapper builds the tower when the config says vision_n_layers > 0 (F6),
                # so the engine holds vision tensors under the checkpoint's own names. It was
                # reported as a false FAIL until this branch existed.
                add(name, ck(name))
            elif m:
                layer, rest = m.group(1), m.group(2)
                base = f"layers.{layer}."
                simple = {
                    "input_layernorm.weight": "attn_norm.weight",
                    "post_attention_layernorm.weight": "ffn_norm.weight",
                    "self_attn.q_norm.weight": "attn.q_norm.weight",
                    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
                    "self_attn.wq_a.weight": "attn.wq_a.weight",
                    "self_attn.wkv.weight": "attn.wkv.weight",
                    "self_attn.compressor.wkv.weight": "attn.compressor.wkv.weight",
                    "self_attn.compressor.norm.weight": "attn.compressor.norm.weight",
                    "self_attn.compressor.wgate.weight": "attn.compressor.wgate.weight",
                    "self_attn.indexer.wq_b.weight": "attn.indexer.wq_b.weight",
                    "self_attn.indexer.weights_proj.weight": "attn.indexer.weights_proj.weight",
                    "self_attn.indexer.wk.weight": "attn.indexer.wk.weight",
                    "self_attn.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
                    "mlp.gate.weight": "ffn.gate.weight",
                    "mlp.gate.bias_vl": "ffn.gate.bias_vl",
                    "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
                    "hc_attn_fn": "hc_attn_fn",
                    "hc_ffn_fn": "hc_ffn_fn",
                    "hc_attn_base": "hc_attn_base",
                    "hc_ffn_base": "hc_ffn_base",
                    "hc_attn_scale": "hc_attn_scale",
                    "hc_ffn_scale": "hc_ffn_scale",
                }
                if rest in simple:
                    add(f"{base}{simple[rest]}", ck(base + simple[rest]))
                elif rest == "self_attn.attn_sink":
                    # TP-sharded along the head axis (engine holds n_heads/tp entries per rank).
                    # The random fixtures zero this tensor, which hid the missing shard until now.
                    add(f"{base}attn.attn_sink[tp]", ck(base + "attn.attn_sink", 0))
                elif rest == "self_attn.wq_b.weight":
                    add(f"{base}attn.wq_b.weight[tp0]", ck(base + "attn.wq_b.weight", 0))
                elif rest == "self_attn.wo_b.weight":
                    add(f"{base}attn.wo_b.weight[tp1]", ck(base + "attn.wo_b.weight", 1))
                elif rest == "self_attn.wo_a.weight":
                    tensor = ck(base + "attn.wo_a.weight", 0)
                    if tensor is not None:
                        # engine layout is [n_local_groups, o_lora_rank, in] (transposed)
                        add(f"{base}attn.wo_a.weight[tp0].T", tensor.t().contiguous())
                elif rest == "mlp.shared_experts.gate_up_proj.weight":
                    n = args.tp
                    w1 = ck(base + "ffn.shared_experts.w1.weight", 0)
                    w3 = ck(base + "ffn.shared_experts.w3.weight", 0)
                    if w1 is not None and w3 is not None:
                        add(f"{base}ffn.shared_experts.w1+w3[tp0]", torch.cat([w1, w3], dim=0))
                elif rest == "mlp.shared_experts.down_proj.weight":
                    tensor = ck(base + "ffn.shared_experts.w2.weight", 1)
                    if tensor is not None:
                        add(f"{base}ffn.shared_experts.w2[tp1]", tensor)
                elif rest in ("mlp.experts.routed_experts.w13_weight", "mlp.experts.routed_experts.w2_weight"):
                    n_experts = num_routed_experts(args.model_path, args.n_experts)
                    per_rank = n_experts // args.ep
                    mine = range(rank * per_rank, (rank + 1) * per_rank)
                    if rest.endswith("w13_weight"):
                        tensors = [
                            torch.cat([ckpt.get(f"{base}ffn.experts.{i}.w1.weight").t(),
                                       ckpt.get(f"{base}ffn.experts.{i}.w3.weight").t()], dim=0)
                            for i in mine
                        ]
                        label = f"{base}ffn.experts[{rank * per_rank}:{(rank + 1) * per_rank}].w1+w3 (transposed)"
                    else:
                        tensors = [ckpt.get(f"{base}ffn.experts.{i}.w2.weight").t() for i in mine]
                        label = f"{base}ffn.experts[{rank * per_rank}:{(rank + 1) * per_rank}].w2 (transposed)"
                    add(label, torch.stack(tensors, dim=0))

            ok, label, ref = compare(name, engine, candidates, args.rtol)
            results.append((rank, name, ok, label, engine, ref))

    n_ok = sum(1 for r in results if r[2])
    print(f"checked {len(results)} engine parameters, {n_ok} match the checkpoint\n")
    for rank, name, ok, label, engine, ref in results:
        status = "OK  " if ok else "FAIL"
        print(f"{status} [rank{rank}] {name}")
        print(f"        engine: std={engine.get('std'):.6f} absmax={engine.get('absmax'):.6f} "
              f"shape={engine['shape']}")
        if ref is None:
            print("        checkpoint: <no candidate>")
        else:
            print(f"        ckpt  : std={ref['std']:.6f} absmax={ref['absmax']:.6f} shape={ref['shape']}  <- {label}")
    if n_ok != len(results):
        print("\nmismatching parameters do NOT hold checkpoint values (loader problem)")


if __name__ == "__main__":
    main()
