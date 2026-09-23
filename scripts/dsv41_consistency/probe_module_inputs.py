# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Force the trainer's module *inputs* to the engine's values, then re-measure the outputs.

`graft_trainer_stages.py` replaces a module's *output* with the engine's, which answers "how
much would the rest of the stack drift if this module were perfect?". It cannot answer the
opposite question: "given the *same* input, does this module still compute a different
output?" -- because the engine's recorded output was produced from the engine's own input,
which by layer >= 1 is not the trainer's input (the hybrid-state caveat in the worklog 9.6).

This script grafts at the module *input* instead: a forward pre-hook overwrites the tensor
the trainer is about to feed `model.layers.<i>.<attn|ffn>` with the engine's recorded input
for the same module, and the engine's recorded output is still compared against whatever the
trainer produces from it. With the input bit-identical, any remaining difference is the
module's own kernel-level difference -- no upstream propagation involved. Variants:

    baseline           no injection (reference, and the natural input difference per module)
    in.all.ffn         force every layer's MoE input   <- "ffn 输入一致后还有没有误差"
    in.all.attn        force every layer's attention input
    in.all             both of the above (every module gets the engine's exact input)
    in.<layer>.<mod>   a single layer, e.g. in.3.ffn

Engine-side input sources (all already present in the staged engine dump):

    attn: language_model.model.layers.<i>.input_layernorm        (module output)
    ffn:  DeepseekV41DecoderLayer.rms_norm_cast@layer<i>[0]      (bf16 x; [1] is its
          fp32 twin, verified bit-identical after the fp32 upcast, so the fused MoE's
          `hidden_states_fp32` router input is covered by injecting [0] alone)

Readouts per module: `in` = ||engine_in - trainer_natural_in|| / ||trainer_in|| (what the
difference *was*), `out` = ||engine_out - trainer_out|| / ||trainer_out|| (what is left with
the input forced). Plus model.norm rel_err, logprob Δ stats and argmax agreement.

Usage (8 NPUs; same topology as the GRPO script):

    PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
    torchrun --nproc_per_node=8 scripts/dsv41_consistency/probe_module_inputs.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32 \
        --engine-dir /mnt/share/m00899630/dsv41/dump/engine \
        --out-dir /mnt/share/m00899630/dsv41/dump/probe_input \
        --lengths 64,200 --variants baseline,in.all.ffn,in.all.attn,in.all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_trainer_stages as dt  # noqa: E402  (sibling helper: build/load/hook plumbing)

# trainer module -> engine dump key. Inputs are what this experiment injects; outputs are what
# it measures against (the same keys graft_trainer_stages.py uses for its output graft).
ENGINE_INPUT_KEYS = {
    "attn": "language_model.model.layers.{i}.input_layernorm",
    "ffn": "DeepseekV41DecoderLayer.rms_norm_cast@layer{i}[0]",
}
ENGINE_OUTPUT_KEYS = {
    "attn": "language_model.model.layers.{i}.self_attn",
    "ffn": "language_model.model.layers.{i}.mlp",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--fixture", default="/mnt/share/m00899630/dsv41/dump/fixture.json")
    parser.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    parser.add_argument("--out-dir", default="/mnt/share/m00899630/dsv41/dump/probe_input")
    parser.add_argument("--lengths", default="64", help="Comma separated fixture lengths.")
    parser.add_argument(
        "--variants",
        default="baseline,in.all.ffn,in.all.attn,in.all",
        help="baseline | items joined by '+' | item = in.<all|layer>.<attn|ffn> | in.all",
    )
    parser.add_argument("--fsdp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--efsdp-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("VERL_DSV41_MAX_SEQ_LEN", "2048")))
    parser.add_argument("--max-elements", type=int, default=40_000_000)
    return parser.parse_args()


class _BuildArgs:
    """The subset of flags the shared build helpers read from the args namespace."""

    def __init__(self, args):
        self.fsdp_size = args.fsdp_size
        self.tp_size = args.tp_size
        self.ep_size = args.ep_size
        self.efsdp_size = args.efsdp_size


def parse_variant(spec: str, n_layers: int) -> dict[str, str]:
    """`in.all.ffn+in.0.attn` -> {"model.layers.<i>.ffn": <engine input key>, ...}."""
    if spec == "baseline":
        return {}
    graft: dict[str, str] = {}
    items: list[str] = []
    for raw in spec.split("+"):
        # `in.all` is shorthand for both modules at every layer
        items.extend(["in.all.attn", "in.all.ffn"] if raw == "in.all" else [raw])
    for item in items:
        parts = item.split(".")
        if len(parts) != 3 or parts[0] != "in" or parts[2] not in ENGINE_INPUT_KEYS:
            raise SystemExit(
                f"variant {spec!r}: expected baseline | in.<all|layer>.<attn|ffn> | in.all, got {item!r}"
            )
        layer, module = parts[1], parts[2]
        if layer != "all" and not layer.isdigit():
            raise SystemExit(
                f"variant {spec!r}: expected baseline | in.<all|layer>.<attn|ffn> | in.all, got {item!r}"
            )
        layers = range(n_layers) if layer == "all" else [int(layer)]
        for index in layers:
            graft[f"model.layers.{index}.{module}"] = ENGINE_INPUT_KEYS[module].format(i=index)
    return graft


def engine_tensor(engine: dict, key: str, label: str) -> torch.Tensor:
    value = engine.get(key)
    if value is None:
        raise SystemExit(f"engine dump has no key {key!r} ({label})")
    tensor = value[0] if isinstance(value, list) else value
    if not torch.is_tensor(tensor):
        raise SystemExit(f"engine key {key!r} is not a tensor: {type(value)}")
    return tensor


def describe(reference: torch.Tensor, other: torch.Tensor) -> dict:
    diff = (other.float() - reference.float()).reshape(-1)
    ref = reference.float().reshape(-1)
    return {
        "rel_err": diff.norm().item() / (ref.norm().item() + 1e-12),
        "max_abs": diff.abs().max().item(),
        "cos": torch.nn.functional.cosine_similarity(ref, other.float().reshape(-1), dim=0).item(),
    }


def summarize(variant, graft, input_sink, stages, engine, length, scored, engine_argmax,
              next_logprobs, pred_ids, baseline, log) -> dict:
    """Compare the forced forward against the engine: input diff (natural) + output diff (forced)."""
    report = {"__inputs__": {}, "__outputs__": {}}
    # (a) what each module input difference *was*, before the pre-hook overwrote it
    for trainer_key, natural in input_sink.items():
        parts = trainer_key.split(".")
        engine_key = ENGINE_INPUT_KEYS[parts[3]].format(i=parts[2])
        report["__inputs__"][trainer_key] = describe(
            engine_tensor(engine, engine_key, "module input"), natural
        )
    # (b) every module's output, forced input where grafted
    for index in range(8):
        for module in ("attn", "ffn"):
            trainer_key = f"model.layers.{index}.{module}"
            tensor = stages.get(trainer_key)
            if tensor is None:
                continue
            engine_tensor_out = engine_tensor(engine, ENGINE_OUTPUT_KEYS[module].format(i=index), "module output")
            if tensor.numel() != engine_tensor_out.numel():
                continue
            report["__outputs__"][trainer_key] = describe(tensor, engine_tensor_out)
    if "model.norm" in stages:
        report["__outputs__"]["model.norm"] = describe(
            stages["model.norm"], engine_tensor(engine, "language_model.model.norm", "model.norm")
        )
    delta = (next_logprobs - scored).float()
    agree = float("nan")
    if engine_argmax:
        ids = torch.tensor([token for token, _ in engine_argmax], dtype=torch.long)
        agree = (ids == pred_ids[: ids.numel()].long()).float().mean().item()
    report["__logprob__"] = {
        "mean": delta.mean().item(),
        "std": delta.std().item(),
        "abs_mean": delta.abs().mean().item(),
        "p99": delta.abs().quantile(0.99).item(),
        "max": delta.abs().max().item(),
        "argmax_agree": agree,
    }
    parts = []
    for key, value in report["__outputs__"].items():
        name = key.split("layers.")[-1] if key.startswith("model.layers") else key
        natural = report["__inputs__"].get(key)
        in_text = f" in={natural['rel_err']:.4f}" if natural else ""
        marker = "*" if key in graft else ""
        parts.append(f"{name}{in_text} out={value['rel_err']:.4f}{marker}")
    log(f"[{variant}] " + " | ".join(parts) + "   (* = input forced to the engine's)")
    sigma = report["__logprob__"]["std"]
    base_sigma = baseline.get("__logprob__", {}).get("std") if baseline else None
    log(f"[{variant}] logprob Δ: mean={report['__logprob__']['mean']:+.4f} std={sigma:.4f} "
        f"|Δ|p99={report['__logprob__']['p99']:.4f} argmax={report['__logprob__']['argmax_agree']:.3f}"
        + (f"  ({base_sigma / sigma:.1f}× better than baseline σ={base_sigma:.4f})" if base_sigma else ""))
    return report


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path (or MODEL_PATH) is required")
    # Fail on a bad --variants string before spending two minutes loading the checkpoint.
    variants = [v for v in args.variants.split(",") if v]
    for variant in variants:
        parse_variant(variant, 1)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    torch, dist, device = dt.init_distributed()
    rank = dist.get_rank()

    from fsdp_turbo.distributed.parallel_state import init_parallel_state
    from fsdp_turbo.fsdp_turbo import FSDPTurbo
    from fsdp_turbo.models.deepseek_v41 import (
        build_deepseek_v41_model,
        initialize_deepseek_v41_model,
        prepare_deepseek_v41_model_for_fsdp,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    from verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl import (
        checkpoint_buffer_meta,
        move_buffers_to_device,
        place_checkpoint_buffers_for_load,
        read_dsv41_checkpoint_state_dict,
        refresh_dsv41_expert_metadata,
    )

    def log(message):
        if rank == 0:
            print(f"[probe-input] {message}", flush=True)

    config = dt.build_fsdp_turbo_config(_BuildArgs(args))
    init_parallel_state(config)

    model = build_deepseek_v41_model(
        tokenizer=None,
        engram_meta_init=True,
        engram_storage_backend="row_sharded",
        use_sparse_flash_attn=False,  # the engine builds it with False (eager torch)
        experts_meta_init=True,
        config_path=args.model_path,
        max_seq_len=args.max_seq_len,
    )
    initialize_deepseek_v41_model(model)
    model = prepare_deepseek_v41_model_for_fsdp(model, device=device, parameter_dtype=torch.bfloat16)
    param_meta = {name: (tuple(param.shape), param.dtype) for name, param in model.named_parameters()}
    buffer_meta = checkpoint_buffer_meta(model)  # fp32 router correction bias (persistent buffers)
    model = FSDPTurbo(config, model).model

    started = time.time()
    if rank == 0:
        full_state, ckpt_report = read_dsv41_checkpoint_state_dict(Path(args.model_path), param_meta, buffer_meta)
        # DCP's strict=True would fail much later with a generic "Missing key(s)"; name them here.
        assert not ckpt_report["missing"], (
            f"checkpoint {args.model_path} lacks {len(ckpt_report['missing'])} tensors the model needs, "
            f"e.g. {ckpt_report['missing'][:5]}"
        )
    else:
        full_state = {}
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True)
    place_checkpoint_buffers_for_load(model, buffer_meta, log)
    set_model_state_dict(model, full_state, options=options)
    del full_state
    log(f"model built + checkpoint loaded in {time.time() - started:.1f}s")

    move_buffers_to_device(model, device, log)
    fp32_mode = os.environ.get("VERL_DSV41_KEEP_FP32_PARAMS", "")
    if fp32_mode:
        dt.restore_fp32_parameters(model, args.model_path, log, device, only="" if fp32_mode == "1" else fp32_mode)
    refresh_dsv41_expert_metadata(model, device)
    still_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    assert not still_meta, f"parameters left on meta: {still_meta[:5]}"

    module_names = {name for name, _ in model.named_modules()}
    n_layers = sum(1 for name in module_names if name.startswith("model.layers.") and name.count(".") == 2)
    log(f"n_layers={n_layers}")

    import fsdp_turbo.models.deepseek_v41.experts as _experts

    log(f"MoE arithmetic: {'BF16-act (engine recipe)' if _experts.MOE_BF16_ACT else 'FP32-act (reference)'} "
        f"[VERL_DSV41_MOE_BF16={os.environ.get('VERL_DSV41_MOE_BF16', '<unset>')}]")

    def find_module(key: str):
        if key not in module_names:
            raise SystemExit(f"module {key!r} not found in the wrapped model")
        return model.get_submodule(key)

    with open(args.fixture) as f:
        fixture = json.load(f)
    samples = [s for s in fixture["samples"] if s["target_len"] in lengths]

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for sample in samples:
        length = sample["target_len"]
        engine = torch.load(Path(args.engine_dir) / f"engine_stages_len{length}_rank0.pt",
                            map_location="cpu", weights_only=False)
        with open(Path(args.engine_dir) / f"engine_len{length}.json") as f:
            engine_logprobs = json.load(f)
        scored = torch.tensor([lp for _, lp in engine_logprobs["scored"]], dtype=torch.float32)
        engine_argmax = engine_logprobs.get("argmax")

        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        reports = {}
        baseline = None
        for variant in variants:
            graft = parse_variant(variant, n_layers)
            stage_sink: dict[str, torch.Tensor] = {}
            input_sink: dict[str, torch.Tensor] = {}
            handles = []
            fired: dict[str, int] = {}
            # `baseline` grafts nothing but still records every module input: that is the
            # "how far apart were the two stacks' inputs" column of the report.
            record_keys = list(graft) or [
                f"model.layers.{index}.{module}" for index in range(n_layers) for module in ENGINE_INPUT_KEYS
            ]
            # forward pre-hooks run in registration order: record the natural input first (that
            # is the "input difference" readout), then overwrite it with the engine's value.
            for trainer_key in record_keys:
                module = find_module(trainer_key)
                engine_key = graft.get(trainer_key)

                def record_in(_module, inputs, key=trainer_key):
                    if inputs and torch.is_tensor(inputs[0]):
                        payload = dt.to_cpu_fp32(inputs[0], args.max_elements)
                        if payload is not None:
                            input_sink[key] = payload

                def replace_in(_module, inputs, key=trainer_key, eng_key=engine_key):
                    hidden = inputs[0]
                    grafted = engine_tensor(engine, eng_key, "module input").to(
                        device=hidden.device, dtype=hidden.dtype
                    )
                    if grafted.numel() != hidden.numel():
                        raise SystemExit(
                            f"{key}: engine input {tuple(grafted.shape)} vs trainer input {tuple(hidden.shape)}"
                        )
                    fired[key] = fired.get(key, 0) + 1
                    return (grafted.reshape(hidden.shape), *inputs[1:])

                handles.append(module.register_forward_pre_hook(record_in))
                if engine_key is not None:
                    handles.append(module.register_forward_pre_hook(replace_in))
            handles += dt.install_hooks(model, torch, args.max_elements, stage_sink)
            # Record the inputs/output of `indexed_sparse_attention` per layer (the trainer-side
            # attention has no submodule to hook; same wrapper as the trainer smoke, which rebinds
            # to *this* variant's stage_sink on every call). This is what lets the ~0.005 attn
            # floor be decomposed offline into qkv / compressed-kv / core / sink contributions.
            dt.install_attention_op_hooks(torch, stage_sink, args.max_elements)

            started = time.time()
            try:
                with torch.no_grad():
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for handle in handles:
                    handle.remove()
            elapsed = time.time() - started

            logits = output.logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            next_logprobs = logprobs[0, :-1].gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            pred_ids = logits[0, :-1].argmax(-1).cpu()
            if fired != {key: 1 for key in graft}:
                raise SystemExit(f"input hooks did not fire as expected: fired={fired} graft={list(graft)}")
            if rank == 0:
                report = summarize(variant, graft, input_sink, stage_sink, engine, length, scored,
                                   engine_argmax, next_logprobs.cpu(), pred_ids, baseline, log)
                if variant == "baseline":
                    baseline = report
                reports[variant] = report
                torch.save(
                    {
                        "variant": variant,
                        "graft": graft,
                        "input_ids": input_ids[0].cpu(),
                        "next_logprobs": next_logprobs.float().cpu(),
                        "pred_ids": pred_ids,
                        "stages": stage_sink,
                        "inputs": input_sink,
                        "forward_s": elapsed,
                    },
                    out_dir / f"probe_len{length}_{variant.replace('+', '-').replace('.', '_')}.pt",
                )
            # fresh engine request <-> fresh ring buffers, same as the staged dumps
            for name, buffer in model.named_buffers():
                if name.endswith("kv_cache"):
                    buffer.zero_()
            dist.barrier()

        if rank == 0:
            with open(out_dir / f"report_len{length}.json", "w") as f:
                json.dump(reports, f, indent=1)
            log(f"len={length}: wrote report ({len(reports)} variants) to {out_dir}")

    dist.barrier()
    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
