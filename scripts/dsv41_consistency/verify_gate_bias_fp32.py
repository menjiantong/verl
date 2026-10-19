"""Regression check: the trainer holds the checkpoint's **fp32** router correction bias.

The fix (`Gate.bias`/`bias_vl` are fp32 *buffers*, see
`fsdp_turbo/models/deepseek_v41/model.py`) is invisible to every end-to-end metric in the
usual sense -- it does not change the shape of the gap, it removes one of its causes. So it
is checked here on the two things that actually decide the question, both CPU-only:

1. **Provenance, bit-exact.** The trainer dump records the loaded buffers verbatim
   (`dump_trainer_stages.py:checkpoint_buffers`). Its sha1 must equal the checkpoint
   tensor's sha1. A bf16 round-trip (what the parameter-based load produced before, and
   what `Gate.bias.to(torch.bfloat16)` still produces) hashes differently -- that column is
   printed for contrast.

2. **Identification, not correlation.** The probe dump records the routing the trainer
   *itself* computed (`model.layers.<i>.ffn.gate[1]`) together with the input its gate saw
   (`model.layers.<i>.ffn`). Recomputing top-k from that same input with the fp32 bias and
   with the bf16-rounded bias answers "which value does the trainer hold" directly: the
   value it holds reproduces its own decisions. Before the fix the bf16 column won
   (100% vs 21-62%, worklog §11.5); after it, the fp32 column must win.

3. **Cost of the rounding.** On those same inputs, how many tokens the bf16 bias would move
   to a different expert set -- the discrete perturbation the fix removes (the flip rate is
   input-independent enough that this matches the engine-input measurement in
   `analyze_gate_bias.py`).

Usage:
    python3 scripts/dsv41_consistency/verify_gate_bias_fp32.py --lengths 64,200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

N_LAYERS_DEFAULT = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real")
    parser.add_argument("--trainer-dir", default="/mnt/share/m00899630/dsv41/dump/trainer_real4_fix",
                        help="dump_trainer_stages.py output (records the loaded buffers).")
    parser.add_argument("--trainer-tag", default="_fix",
                        help="The dump's --out-tag: files are trainer<tag>_len<N>.pt.")
    parser.add_argument("--probe-dir", default="/mnt/share/m00899630/dsv41/dump/probe_input_real4",
                        help="probe_module_inputs.py output (records the trainer's own routing).")
    parser.add_argument("--lengths", default="64,200")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS_DEFAULT)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--gate-temp", type=float, default=1.0)
    return parser.parse_args()


def sha1(tensor: torch.Tensor) -> str:
    """sha1 of the raw bytes: this check is bit-exactness, not closeness."""
    return hashlib.sha1(tensor.detach().float().contiguous().numpy().tobytes()).hexdigest()[:16]


def ckpt_tensor(model_path: str, name: str) -> torch.Tensor:
    from safetensors import safe_open

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
    with safe_open(Path(model_path) / index[name], framework="pt") as handle:
        return handle.get_tensor(name)


def check_provenance(args) -> bool:
    """(1) the loaded bias is bit-identical to the checkpoint's fp32 tensor."""
    print("== 1. provenance: trainer-held bias vs checkpoint (bit-exact)")
    print(f"{'tensor':<38} {'ckpt':>6} {'trainer':>8} {'ckpt sha1(fp32)':>16} {'trainer sha1':>13} "
          f"{'sha1(bf16)':>12} {'same':>5}")
    ok = True
    for length in (int(x) for x in args.lengths.split(",") if x):
        path = Path(args.trainer_dir) / f"trainer{args.trainer_tag}_len{length}.pt"
        if not path.exists():
            print(f"  {path} missing -- run dump_trainer_stages.py first")
            ok = False
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("checkpoint_buffers") or {}
        if not recorded:
            print(f"  {path} has no checkpoint_buffers (pre-fix dump?)")
            ok = False
            continue
        for name, value in sorted(recorded.items()):
            ckpt_name = name.removeprefix("model.")
            ckpt = ckpt_tensor(args.model_path, ckpt_name)
            same = torch.equal(ckpt.float(), value.float())
            ok &= same
            print(f"  len{length} {ckpt_name:<32} {str(ckpt.dtype):>6} {str(value.dtype):>8} "
                  f"{sha1(ckpt):>16} {sha1(value):>13} {sha1(ckpt.to(torch.bfloat16).float()):>12} "
                  f"{'YES' if same else 'NO':>5}")
    print(f"  -> provenance {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_identification(args) -> bool:
    """(2)+(3) recompute the trainer's own routing with each bias value."""
    print("== 2. identification: whose bias reproduces the trainer's recorded routing")
    print("     (recomputed from the trainer's own recorded gate input, baseline variant)")
    print(f"{'len':>4} {'layer':>5} {'tokens':>7} {'match fp32':>11} {'match bf16':>11} "
          f"{'flips':>6} {'flip rate':>10} {'slot overlap':>13}")
    ok = True
    for length in (int(x) for x in args.lengths.split(",") if x):
        path = Path(args.probe_dir) / f"probe_len{length}_baseline.pt"
        if not path.exists():
            print(f"  {path} missing -- run the probe first")
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        inputs, stages = payload["inputs"], payload["stages"]
        for layer in range(args.n_layers):
            # Hook payloads: the FFN input is the gate's input (`MoE.forward` calls the gate
            # first), and the gate's top-k ids are stored under `...gate[1]` (index 1 of the
            # module's output tuple, flattened into the stage sink).
            x = inputs.get(f"model.layers.{layer}.ffn")
            recorded = stages.get(f"model.layers.{layer}.ffn.gate[1]")
            if x is None or recorded is None:
                print(f"  len{length} layer {layer}: dump lacks the ffn input / gate routing")
                ok = False
                continue
            x = x.reshape(-1, x.shape[-1]).float()
            weight = ckpt_tensor(args.model_path, f"layers.{layer}.ffn.gate.weight").float()
            bias_fp32 = ckpt_tensor(args.model_path, f"layers.{layer}.ffn.gate.bias").float()
            bias_bf16 = bias_fp32.to(torch.bfloat16).float()
            scores = F.softplus(x @ weight.t() / args.gate_temp).sqrt()  # sqrtsoftplus
            ids_fp32 = (scores + bias_fp32).topk(args.topk, dim=-1)[1]
            ids_bf16 = (scores + bias_bf16).topk(args.topk, dim=-1)[1]
            recorded = recorded.long()
            match_fp32 = (ids_fp32 == recorded).all(dim=-1).float().mean().item()
            match_bf16 = (ids_bf16 == recorded).all(dim=-1).float().mean().item()
            same_sets = [(set(a.tolist()) == set(b.tolist())) for a, b in zip(ids_fp32, ids_bf16)]
            overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids_fp32, ids_bf16))
            ok &= match_fp32 >= match_bf16
            print(f"{length:>4} {layer:>5} {len(x):>7} {match_fp32 * 100:>10.1f}% {match_bf16 * 100:>10.1f}% "
                  f"{same_sets.count(False):>6} {1 - sum(same_sets) / len(same_sets):>10.4f} "
                  f"{overlap / (len(same_sets) * args.topk):>13.4f}")
    print(f"  -> identification {'PASS' if ok else 'FAIL'} "
          f"(the trainer's own routing must be reproduced by the fp32 bias at least as well as by bf16)\n")
    return ok


def main():
    args = parse_args()
    if args.n_layers <= 0:
        config = json.load(open(Path(args.model_path) / "config.json"))
        args.n_layers = config["text_config"]["num_hidden_layers"]
    provenance = check_provenance(args)
    identification = check_identification(args)
    print(f"VERDICT: provenance={'PASS' if provenance else 'FAIL'} "
          f"identification={'PASS' if identification else 'FAIL'}")
    raise SystemExit(0 if (provenance and identification) else 1)


if __name__ == "__main__":
    main()
