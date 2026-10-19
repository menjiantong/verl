"""Compare trainer-side and engine-side logprobs for one fixture length.

Reads ``trainer_len<L>.pt`` (dump_trainer_stages.py) and ``engine_len<L>.json``
(dump_engine_stages.py). The engine reports, for every prompt position, the logprob of the
actual token; the trainer dump carries both the canonical next-token convention
(``next_logprobs``, aligned with the engine) and the convention verl's padded path uses
(``logprobs``, i.e. labels from ``torch.roll``) so the two can be told apart.

Usage:
    python3 scripts/dsv41_consistency/compare_logprobs.py --len 64
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
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--convention", choices=["next", "roll"], default="next",
                    help="trainer logprob convention to compare against the engine")
    return ap.parse_args()


def main():
    args = parse_args()
    trainer = torch.load(Path(args.trainer_dir) / f"trainer_len{args.len}.pt", map_location="cpu")
    engine = json.load(open(Path(args.engine_dir) / f"engine_len{args.len}.json"))

    ids = trainer["input_ids"]
    t_all = trainer["next_logprobs"].float() if args.convention == "next" else trainer["logprobs"].float()
    # `next_logprobs[i]` scores ids[i+1]; `logprobs[i]` scores the rolled label at i.
    t_logp = t_all[:-1] if args.convention == "next" else t_all[1:]
    e_logp = torch.tensor([p for _, p in engine["scored"]], dtype=torch.float32)
    e_arg = torch.tensor([t for t, _ in engine["argmax"]], dtype=torch.long)
    t_arg = trainer["topk_ids"][:-1, 0]

    n = min(len(t_logp), len(e_logp))
    t_logp, e_logp, e_arg, t_arg = t_logp[:n], e_logp[:n], e_arg[:n], t_arg[:n]
    delta = e_logp - t_logp

    print(f"len={args.len} convention={args.convention} positions={n}")
    print(f"trainer: mean {t_logp.mean():.4f} std {t_logp.std():.4f}")
    print(f"engine : mean {e_logp.mean():.4f} std {e_logp.std():.4f}")
    print(f"delta (engine - trainer): mean {delta.mean():+.4f} std {delta.std():.4f} "
          f"min {delta.min():+.4f} max {delta.max():+.4f}")
    print(f"|delta| percentiles: p50 {delta.abs().median():.4f} p90 {delta.abs().quantile(0.9):.4f} "
          f"p99 {delta.abs().quantile(0.99):.4f} max {delta.abs().max():.4f}")
    print(f"argmax agreement: {(e_arg == t_arg).float().mean().item() * 100:.2f}%")
    for lo, hi in [(0, 1e-3), (1e-3, 1e-2), (1e-2, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 1e9)]:
        count = int(((delta.abs() >= lo) & (delta.abs() < hi)).sum())
        print(f"  |delta| in [{lo:g}, {hi:g}): {count}")

    order = torch.argsort(delta.abs(), descending=True)[: args.worst]
    print(f"\nworst {args.worst} positions:")
    print(f"{'pos':>5} {'token':>7} {'trainer':>10} {'engine':>10} {'delta':>9} {'argmax_t':>9} {'argmax_e':>9}")
    for pos in order.tolist():
        print(f"{pos:>5} {int(ids[pos + 1]):>7} {t_logp[pos]:>10.4f} {e_logp[pos]:>10.4f} "
              f"{delta[pos]:>+9.4f} {int(t_arg[pos]):>9} {int(e_arg[pos]):>9}")

    print(f"\nfirst 6 deltas: {[round(x, 4) for x in delta[:6].tolist()]}")
    print(f"last 6 deltas : {[round(x, 4) for x in delta[-6:].tolist()]}")

    # How much of the mean offset comes from the (trainer-side) rolled last position?
    if args.convention == "roll":
        alt = trainer["next_logprobs"].float()[:-1]
        print(f"\nroll vs next convention on the same dump: mean delta over positions = "
              f"{(torch.tensor([p for _, p in engine['scored']])[:len(alt)] - alt).mean():+.4f} "
              f"(next) vs {delta.mean():+.4f} (roll)")


if __name__ == "__main__":
    main()
