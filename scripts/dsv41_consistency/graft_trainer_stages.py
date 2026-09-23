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

"""Graft the engine's per-layer attention / MoE outputs into the trainer forward.

The staged dumps (``dump_trainer_stages.py`` / ``dump_engine_stages.py``) show the two
stacks first disagree inside layer 0's attention and that the disagreement then grows
layer by layer. That is consistent with *either* (a) layer 0's attention being the only
seed -- the later layers just propagate it -- *or* (b) every layer adding its own seed.

This script separates the two by substituting the engine's own activation into the
trainer's forward for selected layers (``--variants 0.attn``), so everything downstream of
the graft runs on the engine's value. If the downstream stages (and the final logprobs)
then agree, that grafted module was the seed; if they still drift, the remaining layers
contribute on their own. ``baseline`` (no graft) runs in the same process for an
apples-to-apples reference.

Usage (8 NPUs; same topology as the GRPO script):

    PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
    torchrun --nproc_per_node=8 scripts/dsv41_consistency/graft_trainer_stages.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32 \
        --lengths 64,200 --variants baseline,0.attn,0.attn+0.ffn,all.attn
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

# trainer module -> engine dump key ([i] = layer index)
ENGINE_KEYS = {
    "attn": "language_model.model.layers.{i}.self_attn",
    "ffn": "language_model.model.layers.{i}.mlp",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--fixture", default="/mnt/share/m00899630/dsv41/dump/fixture.json")
    parser.add_argument("--engine-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    parser.add_argument("--out-dir", default="/mnt/share/m00899630/dsv41/dump/graft")
    parser.add_argument("--lengths", default="64", help="Comma separated fixture lengths.")
    parser.add_argument(
        "--variants",
        default="baseline,0.attn,0.attn+0.ffn,all.attn",
        help="baseline | <layer>.<attn|ffn> joined by '+' | all.attn | all.ffn",
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
    """`0.attn+0.ffn` / `all.attn` -> {"model.layers.0.attn": <engine key>, ...}."""
    if spec == "baseline":
        return {}
    graft: dict[str, str] = {}
    for item in spec.split("+"):
        layer, _, module = item.partition(".")
        if module not in ENGINE_KEYS:
            raise SystemExit(f"variant {spec!r}: module must be one of {sorted(ENGINE_KEYS)}")
        layers = range(n_layers) if layer == "all" else [int(layer)]
        for index in layers:
            graft[f"model.layers.{index}.{module}"] = ENGINE_KEYS[module].format(i=index)
    return graft


def engine_stage(engine: dict, key: str) -> torch.Tensor:
    value = engine.get(key)
    if value is None:
        raise SystemExit(f"engine dump has no key {key!r}")
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


def summarize(variant: str, stages: dict, engine: dict, lengths: int, scored: torch.Tensor,
              next_logprobs: torch.Tensor, log) -> dict:
    """Compare the grafted forward's downstream stages and logprobs against the engine."""
    report = {}
    for index in range(8):
        for module in ("attn", "ffn"):
            trainer_key = f"model.layers.{index}.{module}"
            tensor = stages.get(trainer_key)
            if tensor is None:
                continue
            engine_tensor = engine_stage(engine, ENGINE_KEYS[module].format(i=index))
            if tensor.numel() != engine_tensor.numel():
                continue
            # same convention as compare_stages.py: rel_err = ||engine - trainer|| / ||trainer||
            report[trainer_key] = describe(tensor, engine_tensor)
    if "model.norm" in stages:
        report["model.norm"] = describe(stages["model.norm"], engine_stage(engine, "language_model.model.norm"))
    delta = (next_logprobs - scored).float()
    report["__logprob__"] = {
        "mean": delta.mean().item(),
        "std": delta.std().item(),
        "abs_mean": delta.abs().mean().item(),
        "p99": delta.abs().quantile(0.99).item(),
        "max": delta.abs().max().item(),
    }
    log(f"[{variant}] " + " ".join(
        f"{k.split('layers.')[-1]}={v['rel_err']:.4f}" for k, v in report.items() if k != "__logprob__"
    ))
    log(f"[{variant}] logprob Δ: mean={report['__logprob__']['mean']:+.4f} "
        f"std={report['__logprob__']['std']:.4f} |Δ|p99={report['__logprob__']['p99']:.4f}")
    return report


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path (or MODEL_PATH) is required")
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
            print(f"[graft] {message}", flush=True)

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
    lengths = [int(x) for x in args.lengths.split(",") if x]
    samples = [s for s in fixture["samples"] if s["target_len"] in lengths]
    variants = [v for v in args.variants.split(",") if v]

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

        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        reports = {}
        for variant in variants:
            graft = parse_variant(variant, n_layers)
            stage_sink: dict[str, torch.Tensor] = {}
            original_sink: dict[str, torch.Tensor] = {}
            # No install_attention_op_hooks() here on purpose: its wrapper closes over the first
            # call's sink, so across variants the attn-op entries would land in the wrong file.
            # Those captures already exist in the staged trainer dump.
            handles = []
            # 1) record the module's own (pre-graft) output, 2) replace it, 3) dump the result:
            # forward hooks run in registration order, so the dump hooks added by install_hooks()
            # below see the grafted value, while `original_sink` keeps the reference.
            fired: dict[str, int] = {}
            for trainer_key in graft:
                module = find_module(trainer_key)

                def record(_module, _inputs, output, key=trainer_key):
                    payload = dt.to_cpu_fp32(output, args.max_elements)
                    if payload is not None:
                        original_sink[key] = payload

                def replace(_module, _inputs, output, key=trainer_key, engine_key=graft[trainer_key]):
                    grafted = engine_stage(engine, engine_key).to(device=output.device, dtype=output.dtype)
                    if grafted.numel() != output.numel():
                        raise SystemExit(f"{key}: engine {tuple(grafted.shape)} vs trainer {tuple(output.shape)}")
                    fired[key] = fired.get(key, 0) + 1
                    return grafted.reshape(output.shape)

                handles.append(module.register_forward_hook(record))
                handles.append(module.register_forward_hook(replace))
            handles += dt.install_hooks(model, torch, args.max_elements, stage_sink)

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
            expected_hits = {key: 1 for key in graft}
            if fired != expected_hits:
                raise SystemExit(f"graft hooks did not fire as expected: fired={fired} expected={expected_hits}")
            if rank == 0:
                report = summarize(variant, stage_sink, engine, length, scored, next_logprobs.cpu(), log)
                reports[variant] = report
                torch.save(
                    {
                        "variant": variant,
                        "graft": graft,
                        "input_ids": input_ids[0].cpu(),
                        "next_logprobs": next_logprobs.float().cpu(),
                        "stages": stage_sink,
                        "stages_pre_graft": original_sink,
                        "forward_s": elapsed,
                    },
                    out_dir / f"graft_len{length}_{variant.replace('+', '-')}.pt",
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
