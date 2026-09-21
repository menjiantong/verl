"""Stage-by-stage comparison of the trainer and engine dumps.

Pairs up the activations recorded by ``dump_trainer_stages.py`` (FSDPTurbo reference model)
with those of ``dump_engine_stages.py`` (live vLLM/vllm-ascend engine, Dsv41DumpExtension
hooks) for the same fixture length, and reports where the two forwards start to disagree.
That tells us whether the gap lives in the embedding/HC/attention/MoE/head stage instead of
only showing up as an end-to-end logprob difference.

Usage:
    python3 scripts/dsv41_consistency/compare_stages.py --len 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# trainer stage -> (engine module name, slice/squeeze applied to the engine tensor)
STAGE_MAP = {
    "model.embed": "language_model.model.embed_tokens",
    "model.norm": "language_model.model.norm",
    "model.head": "language_model.logits_processor",
}
for _layer in range(8):
    STAGE_MAP[f"model.layers.{_layer}"] = f"language_model.model.layers.{_layer}"
    STAGE_MAP[f"model.layers.{_layer}.attn"] = f"language_model.model.layers.{_layer}.self_attn"
    STAGE_MAP[f"model.layers.{_layer}.attn_norm"] = f"language_model.model.layers.{_layer}.input_layernorm"
    STAGE_MAP[f"model.layers.{_layer}.ffn"] = f"language_model.model.layers.{_layer}.mlp"
    STAGE_MAP[f"model.layers.{_layer}.ffn_norm"] = f"language_model.model.layers.{_layer}.post_attention_layernorm"

# engine-internal captures that have no trainer counterpart module (methods wrapped by the
# extension); they are reported for shape/sanity but not paired automatically.
ENGINE_ONLY = [
    "DeepseekV41DecoderLayer.hc_pre",
    "DeepseekV41DecoderLayer.hc_post",
    "DeepseekV41DecoderLayer.rms_norm_cast",
    "DeepseekV41EagerAttentionImpl",
]


def describe(ref: torch.Tensor, other: torch.Tensor) -> dict:
    diff = (other.float() - ref.float()).reshape(-1)
    ref_flat = ref.float().reshape(-1)
    other_flat = other.float().reshape(-1)
    ref_norm = ref_flat.norm().item()
    cos = torch.nn.functional.cosine_similarity(ref_flat, other_flat, dim=0).item()
    return {
        "ref_norm": ref_norm,
        "other_norm": other_flat.norm().item(),
        "rel_err": diff.norm().item() / (ref_norm + 1e-12),
        "max_abs": diff.abs().max().item(),
        "cos": cos,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--trainer-dir", default="/mnt/share/m00899630/dsv41/dump/trainer")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--engine-rank", type=int, default=0)
    ap.add_argument("--show-engine-only", action="store_true")
    args = ap.parse_args()

    trainer = torch.load(Path(args.trainer_dir) / f"trainer_len{args.len}.pt", map_location="cpu")
    engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{args.len}_rank{args.engine_rank}.pt",
                        map_location="cpu")
    rank1 = None
    other_rank = Path(args.engine_dir) / f"engine_stages_len{args.len}_rank1.pt"
    if args.engine_rank != 1 and other_rank.exists():
        rank1 = torch.load(other_rank, map_location="cpu")

    def stage_of(store, key):
        value = store.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    print(f"=== len={args.len} ===")
    print(f"{'trainer stage':<34} {'shape(t)':<22} {'shape(e)':<22} {'rel_err':>9} {'max|d|':>9} {'cos':>8}")
    for t_name, e_name in STAGE_MAP.items():
        t_tensor = stage_of(trainer["stages"], t_name)
        e_tensor = stage_of(engine, e_name)
        if t_tensor is None or e_tensor is None:
            continue
        if t_tensor.dim() == 3 and e_tensor.dim() == 3 and t_tensor.shape[1] != e_tensor.shape[1]:
            pass
        ref = t_tensor.reshape(-1).float()
        oth = e_tensor.reshape(-1).float()
        if ref.numel() != oth.numel():
            print(f"{t_name:<34} {str(tuple(t_tensor.shape)):<22} {str(tuple(e_tensor.shape)):<22} (size mismatch)")
            continue
        stats = describe(t_tensor, e_tensor)
        print(f"{t_name:<34} {str(tuple(t_tensor.shape)):<22} {str(tuple(e_tensor.shape)):<22} "
              f"{stats['rel_err']:>9.4f} {stats['max_abs']:>9.4f} {stats['cos']:>8.4f}")

    if rank1 is not None:
        print("\nengine rank0 vs rank1 (should be ~0 for replicated activations):")
        for key in sorted(k for k in engine if not k.startswith("__")):
            a, b = stage_of(engine, key), stage_of(rank1, key)
            if a is None or b is None or a.numel() != b.numel():
                continue
            stats = describe(a, b)
            if stats["rel_err"] > 1e-6:
                print(f"  {key:<70} rel_err={stats['rel_err']:.6f} max|d|={stats['max_abs']:.6f}")

    if args.show_engine_only:
        print("\nengine-internal captures:")
        for key in sorted(k for k in engine if not k.startswith("__")):
            if any(key.startswith(prefix) for prefix in ENGINE_ONLY):
                value = engine[key]
                tensors = value if isinstance(value, list) else [value]
                shapes = [tuple(t.shape) for t in tensors if torch.is_tensor(t)]
                print(f"  {key:<70} {shapes}")


if __name__ == "__main__":
    main()
