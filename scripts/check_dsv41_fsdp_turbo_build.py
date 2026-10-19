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

"""Standalone (no Ray, no verl trainer) smoke test for the DeepSeek-V4.1 FSDPTurbo path.

Runs the three riskiest steps of ``fsdp_turbo_dsv41`` in isolation, so a failure here is
unambiguous:

1. build the training model from the rollout checkpoint's ``config.json`` with the routed
   experts deferred to the meta device,
2. wrap it with FSDPTurbo (FSDP x EP) and load the checkpoint through
   ``set_model_state_dict(broadcast_from_rank0=True)`` — rank 0 streams the safetensors,
   every rank keeps only its shard,
3. run one forward on a padded batch.

Usage (8 NPUs):
    torchrun --nproc_per_node=8 scripts/check_dsv41_fsdp_turbo_build.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random"),
        help="Rollout checkpoint directory (config.json + tokenizer + safetensors).",
    )
    parser.add_argument("--fsdp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--efsdp-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("VERL_DSV41_MAX_SEQ_LEN", "2048")))
    parser.add_argument("--forward", action="store_true", help="Also run one forward pass.")
    parser.add_argument("--random-init", action="store_true", help="Skip checkpoint IO, defer experts at N(0, 0.02).")
    return parser.parse_args()


def init_distributed():
    import torch
    import torch.distributed as dist

    try:
        import torch_npu  # noqa: F401

        device = torch.device("npu", int(os.environ.get("LOCAL_RANK", 0)))
        torch.npu.set_device(device)
        backend = "hccl"
    except ImportError:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        backend = "nccl"

    dist.init_process_group(backend=backend)
    torch.manual_seed(int(os.environ.get("RANK", 0)))
    return torch, dist, device


def build_fsdp_turbo_config(args):
    """The turbo_config the GRPO script passes through ``fsdp_config.turbo_config``."""
    from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass

    turbo_config = {
        "distributed": {
            "fully_shard_parallel_size": args.fsdp_size,
            "tensor_parallel_size": args.tp_size,
            "expert_parallel_size": args.ep_size,
            "expert_fully_shard_parallel_size": args.efsdp_size,
            "ulysses_parallel_size": 1,
            "fsdp_plan": {
                "apply_modules": {
                    "model.embed": {},
                    "model.layers.{*}": {},
                    "model.norm": {},
                    "model.head": {},
                },
                "hook_modules": ["model.layers.{*}"],
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "output_dtype": None,
                "cast_forward_inputs": False,
                "num_to_forward_prefetch": 1,
                "num_to_backward_prefetch": 1,
                "fsdp_implementation": "native",
            },
            "ep_plan": {
                "apply_modules": ["model.layers.{*}.ffn.experts"],
                "dispatcher": "fused",
            },
        },
        "memory": {"recompute": False, "recompute_plan": []},
    }
    config = _dict_to_dataclass(FSDPTurboConfig, turbo_config)
    config.distributed.fsdp_plan.cpu_offload = True
    return config


def main():
    args = parse_args()
    torch, dist, device = init_distributed()
    rank = dist.get_rank()

    from fsdp_turbo.distributed.parallel_state import init_parallel_state
    from fsdp_turbo.fsdp_turbo import FSDPTurbo
    from fsdp_turbo.models.deepseek_v41 import (
        build_deepseek_v41_model,
        initialize_deepseek_v41_model,
        prepare_deepseek_v41_model_for_fsdp,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl import (
        _random_state_dict,
        read_dsv41_checkpoint_state_dict,
        refresh_dsv41_expert_metadata,
    )

    def log(message):
        if rank == 0:
            print(f"[check] {message}", flush=True)

    def log_rank(message):
        print(f"[check][rank{rank}] {message}", flush=True)

    config = build_fsdp_turbo_config(args)
    init_parallel_state(config)

    log("building model (routed experts deferred to meta)")
    started = time.time()
    model = build_deepseek_v41_model(
        tokenizer=None,
        engram_meta_init=True,
        use_sparse_flash_attn=False,
        experts_meta_init=True,
        config_path=args.model_path,
        max_seq_len=args.max_seq_len,
    )
    initialize_deepseek_v41_model(model)
    model = prepare_deepseek_v41_model_for_fsdp(model, device=device, parameter_dtype=torch.bfloat16)
    param_meta = {name: (tuple(param.shape), param.dtype) for name, param in model.named_parameters()}
    log(f"model built in {time.time() - started:.1f}s ({len(param_meta)} parameters)")

    log("wrapping with FSDPTurbo")
    started = time.time()
    model = FSDPTurbo(config, model).model
    log(f"wrapped in {time.time() - started:.1f}s")

    log("materializing parameters")
    started = time.time()
    if args.random_init:
        full_state = _random_state_dict(param_meta) if rank == 0 else {}
    else:
        full_state = read_dsv41_checkpoint_state_dict(Path(args.model_path), param_meta)[0] if rank == 0 else {}
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True)
    set_model_state_dict(model, full_state, options=options)
    del full_state
    log(f"parameters materialized in {time.time() - started:.1f}s")

    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    refresh_dsv41_expert_metadata(model, device)

    still_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    assert not still_meta, f"parameters left on meta: {still_meta[:5]}"
    log("no meta parameters left")

    # Memory/placement report: experts are EP-sharded (resident), the rest is FSDP-sharded
    # and, with cpu_offload, stays on the host between steps.
    report = {}
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        kind = name.split(".")[1] if name.startswith("model.layers.") else name
        entry = report.setdefault(kind, {"count": 0, "bytes": 0, "devices": set(), "sample": None})
        entry["count"] += 1
        entry["bytes"] += local.numel() * local.element_size()
        entry["devices"].add(str(local.device))
        if entry["sample"] is None:
            entry["sample"] = f"{name} {tuple(local.shape)} {local.dtype}"
    if rank == 0:
        for kind, entry in sorted(report.items()):
            log(
                f"  {kind}: {entry['count']} params, {entry['bytes'] / 2**30:.2f} GiB/rank on {sorted(entry['devices'])}"
                f" | e.g. {entry['sample']}"
            )
    total_gib = sum(entry["bytes"] for entry in report.values()) / 2**30
    try:
        allocated = torch.npu.memory_allocated() / 2**30
    except Exception:  # noqa: BLE001 - cuda builds have no torch.npu
        allocated = torch.cuda.memory_allocated() / 2**30
    log_rank(f"parameters {total_gib:.2f} GiB this rank, accelerator allocated {allocated:.2f} GiB")

    if args.forward:
        log("running one forward pass (batch=1, seq=8)")
        from torch.distributed.tensor import DTensor

        parameters = [(name, parameter) for name, parameter in model.named_parameters() if "embed" in name]
        for name, parameter in parameters:
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            log(f"{name}: device={local.device} dtype={local.dtype} absmax={local.abs().max().item():.4f}")

        model.eval()
        input_ids = torch.randint(3, 1000, (1, 8), device=device)
        attention_mask = torch.ones_like(input_ids)
        # No position_ids: the V4.1 reference forward indexes RoPE by absolute position and
        # rejects that input (the engine strips it in `prepare_model_inputs`).
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        log(f"forward ok: logits {tuple(logits.shape)} dtype={logits.dtype} absmax={logits.abs().max().item():.4f}")

    dist.barrier()
    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
