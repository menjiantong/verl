#!/usr/bin/env python3
"""Per-token decomposition of the rollout-vs-actor metrics, from `VERL_DSV41_DUMP_BATCH` dumps.

`verl/utils/debug/metrics.py::calculate_debug_metrics` reports four scalars per step:

    training/rollout_probs_diff_{mean,max,std}   = |exp(old_log_probs) - exp(rollout_log_probs)|
    training/rollout_actor_probs_pearson_corr    = pearson of those two *probability* vectors

They are heavy-tailed: the two stacks (vLLM-Ascend decode vs the trainer's padded forward) agree to
<0.1 nats on almost every token, but MoE routing flips make a few tokens disagree by O(1) nats, and
the probability-domain Pearson is dominated by the high-probability ones. This script reads the raw
tensors (the `VERL_DSV41_DUMP_BATCH=<dir>` hook in the same module writes `batch_step<N>.pt`) and
reports, per step:

  * the same four scalars, recomputed offline (they must match the trainer's log line);
  * the tail split -- share of sum(Δlogp²) / sum(Δp²) carried by the worst 1/5/10% of tokens;
  * the Pearson after dropping the worst-k tokens, i.e. how much of the gap is a handful of tokens;
  * the log-domain Pearson for contrast (it stays ~0.999 when the probability-domain one is 0.97).

Usage:
    python3 scripts/dsv41_consistency/analyze_rollout_actor_gap.py --dump-dir /tmp/dsv41_batch_real4
    python3 scripts/dsv41_consistency/analyze_rollout_actor_gap.py --dump-dir DIR --per-sequence
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """torch.corrcoef on masked-selected pairs -- same call the trainer's metric uses."""
    return float(torch.corrcoef(torch.stack([a, b], dim=0))[0][1])


def step_of(path: str) -> int:
    match = re.search(r"batch_step(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def analyze(path: str, per_sequence: bool) -> dict | None:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "old_log_probs" not in data or "rollout_log_probs" not in data:
        return None
    old = data["old_log_probs"].double()
    roll = data["rollout_log_probs"].double()
    mask = data.get("response_mask")
    mask = torch.ones_like(old) if mask is None else mask.double()
    keep = mask.bool().reshape(-1)
    if not bool(keep.any()):
        return None

    t_old, t_roll = old.reshape(-1)[keep], roll.reshape(-1)[keep]
    p_old, p_roll = t_old.exp(), t_roll.exp()
    d_logp = t_old - t_roll
    dp = (p_old - p_roll).abs()

    out = {
        "step": step_of(path),
        "tokens": int(t_old.numel()),
        "pearson_p": pearson(p_old, p_roll),
        "pearson_logp": pearson(t_old, t_roll),
        "diff_mean": float(dp.mean()),
        "diff_max": float(dp.max()),
        "diff_std": float(dp.std()),
        "dlogp_mean": float(d_logp.mean()),
        "dlogp_abs_mean": float(d_logp.abs().mean()),
        "dlogp_p99": float(d_logp.abs().quantile(0.99)),
        "dlogp_max": float(d_logp.abs().max()),
    }

    # tail concentration + leave-the-worst-out Pearson
    for tag, energy in (("logp", d_logp.pow(2)), ("p", dp.pow(2))):
        order = torch.argsort(energy, descending=True)
        total = float(energy.sum())
        for frac in (0.01, 0.05, 0.10):
            k = max(1, int(frac * t_old.numel()))
            out[f"share_{tag}_top{int(frac * 100)}"] = float(energy[order[:k]].sum()) / (total + 1e-300)
        if tag == "p":
            for k in (1, 2, 5, 10, 20):
                keep_idx = order[k:]
                out[f"pearson_p_drop{k}"] = pearson(p_old[keep_idx], p_roll[keep_idx]) if keep_idx.numel() > 2 else float("nan")

    if per_sequence:
        seq_pearson, seq_tokens = [], []
        for row in range(old.shape[0]):
            row_keep = mask[row].bool()
            if int(row_keep.sum()) < 4:
                continue
            seq_pearson.append(pearson(old[row][row_keep].exp(), roll[row][row_keep].exp()))
            seq_tokens.append(int(row_keep.sum()))
        if seq_pearson:
            s = torch.tensor(seq_pearson)
            out["seq_pearson_min"] = float(s.min())
            out["seq_pearson_med"] = float(s.median())
            out["seq_pearson_max"] = float(s.max())
            out["seq_tokens_min"] = min(seq_tokens)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/dsv41_batch_real4")
    ap.add_argument("--per-sequence", action="store_true", help="also report per-sequence Pearson spread")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "batch_step*.pt")), key=step_of)
    if not files:
        raise SystemExit(f"no batch_step*.pt under {args.dump_dir} (is VERL_DSV41_DUMP_BATCH set?)")

    rows = []
    for path in files:
        row = analyze(path, args.per_sequence)
        if row is not None:
            rows.append(row)

    head = (f"{'step':>4} {'tokens':>7} {'pearson_p':>10} {'pearson_logp':>12} "
            f"{'|dp|mean':>9} {'|dp|max':>8} {'dlogp_abs':>9} {'dlogp_p99':>9} {'dlogp_max':>9}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['step']:>4} {r['tokens']:>7} {r['pearson_p']:>10.4f} {r['pearson_logp']:>12.6f} "
              f"{r['diff_mean']:>9.5f} {r['diff_max']:>8.4f} {r['dlogp_abs_mean']:>9.4f} "
              f"{r['dlogp_p99']:>9.4f} {r['dlogp_max']:>9.4f}")

    print("\ntail concentration (share of the squared error carried by the worst tokens)")
    print(f"{'step':>4} {'logp top1%':>11} {'top5%':>8} {'top10%':>8} | {'p top1%':>9} {'top5%':>8} {'top10%':>8}")
    for r in rows:
        print(f"{r['step']:>4} {r['share_logp_top1']:>11.3f} {r['share_logp_top5']:>8.3f} "
              f"{r['share_logp_top10']:>8.3f} | {r['share_p_top1']:>9.3f} {r['share_p_top5']:>8.3f} "
              f"{r['share_p_top10']:>8.3f}")

    print("\npearson_p after dropping the worst-k probability-error tokens (how concentrated the gap is)")
    print(f"{'step':>4} {'drop1':>8} {'drop2':>8} {'drop5':>8} {'drop10':>8} {'drop20':>8}")
    for r in rows:
        print(f"{r['step']:>4} " + " ".join(f"{r[f'pearson_p_drop{k}']:>8.5f}" for k in (1, 2, 5, 10, 20)))

    if args.per_sequence and rows and "seq_pearson_min" in rows[0]:
        print("\nper-sequence pearson_p spread")
        print(f"{'step':>4} {'min':>8} {'median':>8} {'max':>8} {'min_tokens':>10}")
        for r in rows:
            print(f"{r['step']:>4} {r['seq_pearson_min']:>8.4f} {r['seq_pearson_med']:>8.4f} "
                  f"{r['seq_pearson_max']:>8.4f} {r['seq_tokens_min']:>10}")


if __name__ == "__main__":
    main()
