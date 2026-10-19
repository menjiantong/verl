"""Rebuild the full graft-variant table from the saved dumps.

Each ``graft_trainer_stages.py`` run writes one ``graft_len<L>_<variant>.pt`` per variant
plus a ``report_len<L>.json`` that is *overwritten* by the next run. This script reads all
``*.pt`` dumps instead, recomputes every variant against the engine dump, and prints one
table (so results from several runs can be combined without re-running anything).

Usage:
    python3 scripts/dsv41_consistency/summarize_graft.py --lengths 64,200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ENGINE_KEYS = {
    "attn": "language_model.model.layers.{i}.self_attn",
    "ffn": "language_model.model.layers.{i}.mlp",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="64,200")
    ap.add_argument("--graft-dir", default="/mnt/share/m00899630/dsv41/dump/graft")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--per-stage", action="store_true", help="Also print the per-stage rel_err grid.")
    return ap.parse_args()


def engine_tensor(engine: dict, key: str) -> torch.Tensor:
    value = engine[key]
    return (value[0] if isinstance(value, list) else value).float()


def rel_err(trainer: torch.Tensor, engine: torch.Tensor) -> float:
    return (engine.float() - trainer.float()).norm().item() / (trainer.float().norm().item() + 1e-12)


def main():
    args = parse_args()
    for length in (int(x) for x in args.lengths.split(",") if x):
        engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{length}_rank0.pt",
                            map_location="cpu", weights_only=False)
        scored = torch.tensor([lp for _, lp in json.load(open(Path(args.engine_dir) /
                                                               f"engine_len{length}.json"))["scored"]],
                              dtype=torch.float32)
        norm_e = engine_tensor(engine, "language_model.model.norm")
        rows = []
        for path in sorted(Path(args.graft_dir).glob(f"graft_len{length}_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            variant = payload["variant"]
            stages = payload["stages"]
            delta = (payload["next_logprobs"].float() - scored).float()
            stages_err = {}
            for index in range(8):
                for module in ("attn", "ffn"):
                    key = f"model.layers.{index}.{module}"
                    if key in stages:
                        stages_err[key] = rel_err(stages[key], engine_tensor(engine, ENGINE_KEYS[module].format(i=index)))
            rows.append({
                "variant": variant,
                "norm": rel_err(stages["model.norm"], norm_e),
                "mean": delta.mean().item(),
                "std": delta.std().item(),
                "abs_mean": delta.abs().mean().item(),
                "p99": delta.abs().quantile(0.99).item(),
                "max": delta.abs().max().item(),
                "stages": stages_err,
            })
        order = {"baseline": 0}
        rows.sort(key=lambda r: (order.get(r["variant"], 1), r["std"]))
        print(f"\n===== len={length} (σ = std of Δlogp trainer−engine; ‖·‖ rel_err of model.norm)")
        print(f"{'variant':<20} {'norm rel':>9} {'Δmean':>8} {'σ':>8} {'|Δ|mean':>8} {'p99':>7} {'max':>6}")
        for row in rows:
            print(f"{row['variant']:<20} {row['norm']:>9.4f} {row['mean']:>+8.4f} {row['std']:>8.4f} "
                  f"{row['abs_mean']:>8.4f} {row['p99']:>7.3f} {row['max']:>6.3f}")
        if args.per_stage:
            keys = [k for k in rows[0]["stages"]] if rows else []
            print(f"\n  per-stage rel_err ({len(keys)} stages)")
            print(f"  {'variant':<20} " + " ".join(f"{k.split('layers.')[-1]:>9}" for k in keys))
            for row in rows:
                print(f"  {row['variant']:<20} " + " ".join(f"{row['stages'][k]:>9.4f}" for k in keys))


if __name__ == "__main__":
    main()
