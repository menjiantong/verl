"""Rebuild the input-probe table from the saved dumps of ``probe_module_inputs.py``.

Each run writes one ``probe_len<L>_<variant>.pt`` per variant (``report_len<L>.json`` is
overwritten by the next run), so this reads the ``.pt`` files and re-derives everything:

* ``in_nat``  -- ||engine_input - trainer_natural_input|| / ||trainer_input||, from the
  ``baseline`` variant: how far apart the two stacks' module inputs were to begin with.
* output cells -- ||engine_out - trainer_out|| / ||trainer_out|| for every variant, with a
  ``*`` marking modules whose input was forced to the engine's value (so the remaining
  difference is the module's own kernel difference, not upstream propagation).
* Δlogp σ / argmax -- end-to-end effect of the forcing.

Usage:
    python3 scripts/dsv41_consistency/summarize_probe.py --lengths 64,200
    python3 scripts/dsv41_consistency/summarize_probe.py --lengths 200 \
        --probe-dir .../dump/probe_input_scaled --engine-dir .../dump/engine_scaled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ENGINE_INPUT_KEYS = {
    "attn": "language_model.model.layers.{i}.input_layernorm",
    "ffn": "DeepseekV41DecoderLayer.rms_norm_cast@layer{i}[0]",
}
ENGINE_OUTPUT_KEYS = {
    "attn": "language_model.model.layers.{i}.self_attn",
    "ffn": "language_model.model.layers.{i}.mlp",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="64,200")
    ap.add_argument("--probe-dir", default="/mnt/share/m00899630/dsv41/dump/probe_input")
    ap.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--sort", default="baseline,in.all.ffn,in.all.attn,in.all",
                    help="Preferred variant order (comma separated); unknown ones go last.")
    return ap.parse_args()


def engine_tensor(engine: dict, key: str) -> torch.Tensor:
    value = engine[key]
    return (value[0] if isinstance(value, list) else value).float()


def rel_err(trainer: torch.Tensor, engine: torch.Tensor) -> float:
    return (engine.float() - trainer.float()).norm().item() / (trainer.float().norm().item() + 1e-12)


def per_token_rel_err(trainer: torch.Tensor, engine: torch.Tensor) -> torch.Tensor:
    trainer = trainer.float().reshape(-1, trainer.shape[-1])
    engine = engine.float().reshape(-1, engine.shape[-1])
    diff = (engine - trainer).norm(dim=-1)
    return diff / (trainer.norm(dim=-1) + 1e-12)


def routing_section(payloads, engine, length, out=print):
    """Do tiny input differences flip the MoE's discrete top-6 choice?

    The trainer's own routing (``baseline``) is compared against the routing it computes when
    its MoE input is forced to the engine's value: same module, same weights, inputs 0.6-12%
    apart. The per-token output error is then split by whether that token's expert *set*
    changed, which shows whether the error is a continuous effect or a discrete one.
    """
    ref = next((p for p in payloads if p["variant"] == "baseline"), None)
    forced = next((p for p in payloads if any(key.endswith(".ffn") for key in p["graft"])), None)
    if ref is None or forced is None:
        return
    out(f"\n----- len={length}: routing stability, {ref['variant']} vs {forced['variant']} (MoE input forced)")
    out(f"{'layer':<7} {'expert-sets same':>16} {'slot overlap':>13} {'flips':>8} "
        f"{'rel_err|flip':>13} {'rel_err|same':>13} {'flip share of err²':>18}")
    for index in range(4):
        gate_key = f"model.layers.{index}.ffn.gate[1]"
        a, b = ref["stages"].get(gate_key), forced["stages"].get(gate_key)
        ffn_key = f"model.layers.{index}.ffn"
        if a is None or b is None or ffn_key not in ref["stages"]:
            continue
        sets_a = [set(row.tolist()) for row in a.long()]
        sets_b = [set(row.tolist()) for row in b.long()]
        same = torch.tensor([sa == sb for sa, sb in zip(sets_a, sets_b)])
        overlap = sum(len(sa & sb) / max(len(sa), 1) for sa, sb in zip(sets_a, sets_b)) / len(sets_a)
        errors = per_token_rel_err(ref["stages"][ffn_key], engine_tensor(engine, f"language_model.model.layers.{index}.mlp"))
        count = min(errors.numel(), same.numel())
        errors, same = errors[:count], same[:count]
        flip_mean = errors[~same].mean().item() if (~same).any() else float("nan")
        keep_mean = errors[same].mean().item() if same.any() else float("nan")
        # how much of the squared output error sits on the tokens whose expert set changed
        share = ((errors[~same] ** 2).sum() / (errors**2).sum()).item() if (~same).any() else 0.0
        out(f"{f'{index}.ffn':<7} {same.float().mean().item():>16.4f} {overlap:>13.4f} "
            f"{int((~same).sum()):>4}/{count:<3} {flip_mean:>13.4f} {keep_mean:>13.4f} {share:>18.3f}")


def main():
    args = parse_args()
    order = {name: index for index, name in enumerate(args.sort.split(",") if args.sort else [])}
    for length in (int(x) for x in args.lengths.split(",") if x):
        engine_path = Path(args.engine_dir) / f"engine_stages_len{length}_rank0.pt"
        engine = torch.load(engine_path, map_location="cpu", weights_only=False)
        engine_json = json.load(open(Path(args.engine_dir) / f"engine_len{length}.json"))
        scored = torch.tensor([lp for _, lp in engine_json["scored"]], dtype=torch.float32)
        argmax_ids = torch.tensor([token for token, _ in engine_json["argmax"]], dtype=torch.long)

        payloads = [torch.load(path, map_location="cpu", weights_only=False)
                    for path in sorted(Path(args.probe_dir).glob(f"probe_len{length}_*.pt"))]
        if not payloads:
            print(f"\n===== len={length}: no probe dumps in {args.probe_dir}")
            continue
        payloads.sort(key=lambda p: (order.get(p["variant"], len(order)), p["variant"]))

        modules = [f"model.layers.{i}.{m}" for i in range(4) for m in ("attn", "ffn")]
        baseline = next((p for p in payloads if p["variant"] == "baseline"), None)
        natural_in = {}
        if baseline is not None:
            for module in ("attn", "ffn"):
                for index in range(4):
                    key = f"model.layers.{index}.{module}"
                    value = baseline["inputs"].get(key)
                    if value is not None:
                        natural_in[key] = rel_err(value, engine_tensor(engine, ENGINE_INPUT_KEYS[module].format(i=index)))
        variants = [p["variant"] for p in payloads]
        print(f"\n===== len={length}  (rel_err = ‖engine − trainer‖/‖trainer‖; * = input forced to the engine's)")
        print(f"{'module':<20} {'in_nat':>8} " + " ".join(f"{name:>13}" for name in variants))
        for module_key in modules:
            index = int(module_key.split(".")[2])
            module = module_key.split(".")[3]
            cells = []
            for payload in payloads:
                stages = payload["stages"]
                if module_key not in stages:
                    cells.append(f"{'-':>13}")
                    continue
                value = rel_err(stages[module_key], engine_tensor(engine, ENGINE_OUTPUT_KEYS[module].format(i=index)))
                cells.append(f"{value:>12.4f}{'*' if module_key in payload['graft'] else ' '}")
            print(f"{module_key.split('.', 2)[2]:<20} {natural_in.get(module_key, float('nan')):>8.4f} " + " ".join(cells))
        norm_cells = []
        for payload in payloads:
            stages = payload["stages"]
            norm_cells.append(f"{rel_err(stages['model.norm'], engine_tensor(engine, 'language_model.model.norm')):>12.4f} ")
        print(f"{'model.norm':<20} {'-':>8} " + " ".join(norm_cells))

        print(f"\n{'variant':<20} {'Δmean':>8} {'σ':>8} {'σ vs base':>9} {'|Δ|mean':>8} {'p99':>7} {'max':>6} {'argmax':>7}")
        base_sigma = None
        for payload in payloads:
            delta = (payload["next_logprobs"].float() - scored).float()
            sigma = delta.std().item()
            if payload["variant"] == "baseline":
                base_sigma = sigma
            ratio = f"{base_sigma / sigma:.1f}×" if base_sigma else "-"
            pred_ids = payload.get("pred_ids")
            agree = float("nan")
            if pred_ids is not None:
                count = min(pred_ids.numel(), argmax_ids.numel())
                agree = (pred_ids[:count].long() == argmax_ids[:count]).float().mean().item()
            print(f"{payload['variant']:<20} {delta.mean().item():>+8.4f} {sigma:>8.4f} {ratio:>9} "
                  f"{delta.abs().mean().item():>8.4f} {delta.abs().quantile(0.99).item():>7.3f} "
                  f"{delta.abs().max().item():>6.3f} {agree:>7.3f}")
        routing_section(payloads, engine, length)


if __name__ == "__main__":
    main()
