#!/usr/bin/env python3
"""Trainer-side-only smoke test for the DeepSeek-V4.1 FSDP-Turbo training engine.

Why this exists: the GRPO recipe needs vLLM for rollouts, so when *that* stack is broken it
hides whether training itself is healthy. This driver talks to the **production** engine
class the actor worker uses (``FSDPTurboDSV41EngineWithLMHead``, registered as
``language_model`` / ``fsdp_turbo_dsv41``) and runs the trainer half end to end without any
rollout code in the process:

    build model -> DCP checkpoint load + host-offload Engram row load -> train_mode
      -> one forward + backward on a synthetic batch -> optimizer step -> sync payload check

Everything it prints is evidence about the *training* stack; nothing here touches vLLM.

Topology and the on/offload flags mirror
``verl/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh`` so a green run
here means the actor engine would come up inside GRPO too.

Usage (8 NPUs; see the sibling ``run_train_engine_smoke.sh`` for env and torchrun):

    torchrun --nproc_per_node=8 train_engine_smoke.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top64-bf16 \
        --tokenizer-path /mnt/share/m00899630/dsv41/rl_tokenizer \
        --seq-len 512 --batch-size 1

Checkpoint caveat: ``HFModelConfig`` parses the checkpoint's ``config.json`` through vLLM, so the
``model_type`` there has to be one vLLM knows. The top64 release declares ``deepseek_v41`` (fine);
the older 4-layer fixtures still say ``deepseek_v4.1`` and fail with "Transformers does not
recognize this architecture" before the engine is even built -- that is a fixture problem, not an
engine one.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "verl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top64-bf16",
        help="Rollout checkpoint directory (config.json + tokenizer + safetensors).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/mnt/share/m00899630/dsv41/rl_tokenizer",
        help="Tokenizer directory; the Engram token map is built from it.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fsdp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--efsdp-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep parameters/optimizer on device instead of the production CPU-offload setting.",
    )
    parser.add_argument(
        "--skip-export-check",
        action="store_true",
        help="Skip the weight-sync payload check (get_per_tensor_param).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the configs and construct the engine, but do not load the checkpoint or train. "
        "Fast plumbing check.",
    )
    return parser.parse_args()


def build_engine_config(args):
    """FSDPEngineConfig for this integration, i.e. the GRPO script's ``fsdp_config`` block."""
    from omegaconf import OmegaConf

    from verl.workers.config import FSDPEngineConfig

    turbo = OmegaConf.load(REPO / "verl/trainer/config/engine/fsdp.yaml").turbo_config
    # [2026-10-09] Memory experiments from the ~1.1 GB/layer saved-for-backward plateau:
    # SMOKE_RECOMPUTE="model.layers.{*}.ffn" (etc.) enables memory.recompute for exactly
    # that module set. Scoping to `ffn` keeps the attention's own custom-Function
    # checkpoint OUT of the recompute region (that nesting is what crashed the whole-
    # layer recompute attempt with "different number of tensors saved").
    if os.environ.get("SMOKE_RECOMPUTE"):
        # Assign the whole node: the YAML leaves `memory` absent, and setting a nested key
        # under a missing node raises "Missing key memory".
        from omegaconf import OmegaConf as _OC

        turbo.memory = _OC.create(
            {"recompute": True, "recompute_plan": [os.environ["SMOKE_RECOMPUTE"]]}
        )
    distributed = turbo.distributed
    distributed.fully_shard_parallel_size = args.fsdp_size
    distributed.tensor_parallel_size = args.tp_size
    distributed.expert_parallel_size = args.ep_size
    distributed.expert_fully_shard_parallel_size = args.efsdp_size
    distributed.ulysses_parallel_size = 1
    # Module names as FSDP-Turbo sees them (`model` is the adapter's backbone) -- same list as
    # FSDP_APPLY_MODULES / HOOK_MODULES in the GRPO script.
    distributed.fsdp_plan.apply_modules = {
        "model.embed": {},
        "model.layers.{*}": {},
        "model.norm": {},
        "model.head": {},
    }
    distributed.fsdp_plan.hook_modules = ["model.layers.{*}"]
    distributed.fsdp_plan.cast_forward_inputs = False
    distributed.fsdp_plan.output_dtype = None
    distributed.ep_plan = {
        "apply_modules": ["model.layers.{*}.ffn.experts"],
        "dispatcher": "fused",
    }

    engine_config = FSDPEngineConfig(
        strategy="fsdp_turbo",  # the dataclass only accepts fsdp/fsdp2/fsdp_turbo ...
        turbo_config=OmegaConf.to_container(turbo, resolve=True),
        offload_policy=not args.no_offload,
        param_offload=not args.no_offload,
        optimizer_offload=not args.no_offload,
        reshard_after_forward=True,
        entropy_checkpointing=True,
        entropy_from_logits_with_chunking=True,
        use_remove_padding=False,
        use_fused_kernels=False,
        use_torch_compile=False,
        fsdp_size=args.fsdp_size,
    )
    # ... the registry key for this engine is the fork's own backend name. BaseConfig freezes its
    # fields once set, so this goes through object.__setattr__ -- exactly how the production actor
    # config syncs it (verl/workers/config/actor.py, FSDPActorConfig.__post_init__).
    object.__setattr__(engine_config, "strategy", "fsdp_turbo_dsv41")
    return engine_config


def build_batch(args, vocab_size: int):
    """One padded synthetic batch in the layout the FSDP engine expects."""
    import torch
    from tensordict import TensorDict

    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils import tensordict_utils as tu

    torch.manual_seed(args.seed + int(os.environ.get("RANK", 0)))
    seq_len, bsz = args.seq_len, args.batch_size
    # ``input_ids``/``position_ids`` are jagged nested tensors even though the engine runs with
    # use_remove_padding=False: that branch pads them itself (``to_padded_tensor`` +
    # ``build_attention_mask_from_nested``) from the packed representation, which is what the
    # dataset pipeline hands over. Any id in range is a valid Engram lookup.
    values = torch.randint(1, min(vocab_size, 100000), (bsz * seq_len,), dtype=torch.long)
    offsets = torch.arange(0, (bsz + 1) * seq_len, seq_len, dtype=torch.long)
    input_ids = torch.nested.nested_tensor_from_jagged(values, offsets)
    position_ids = torch.nested.nested_tensor_from_jagged(
        torch.arange(seq_len, dtype=torch.long).repeat(bsz), offsets
    )
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.float32)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        },
        batch_size=[bsz],
    )
    # Scalars/lists are not tensors: they go in as non-tensor (meta) data, which is how the
    # engine reads them back (`micro_batch["temperature"]` unwraps to a float).
    tu.assign_non_tensor(
        data,
        temperature=1.0,
        use_remove_padding=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        # The micro-batch splitter defaults use_dynamic_bsz to True and then insists on
        # max_token_len_per_gpu; the GRPO recipe runs with dynamic batching off.
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
    )
    return data


def smoke_loss(model_output, data, dp_group=None):
    """Differentiable, scale-free loss: a smoke test must exercise the graph, not learn anything.

    The engine's ``prepare_model_outputs`` never exposes raw logits -- it hands over per-token
    ``log_probs`` (jagged over the real sequence lengths), optionally ``entropy`` /
    ``sum_pi_squared``. Any of them carries the gradient back through the network, which is all a
    smoke test needs; ``dp_group`` is accepted because production loss functions take it.
    """
    import torch

    for key in ("log_probs", "entropy", "sum_pi_squared"):
        if key in model_output:
            x = model_output[key].float()
            loss = (x * x).mean() * 1e-3
            return loss, {f"smoke/{key}_abs_mean": x.detach().abs().mean().item()}
    raise RuntimeError(
        "model_output carries none of log_probs/entropy/sum_pi_squared; "
        f"the engine's output contract changed. Keys: {list(model_output)}"
    )


def engram_report(module) -> list[str]:
    """Host-offload Engram tables as the engine sees them (after shard_() + row load)."""
    from verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl import host_offload_engram_tables

    lines = []
    for checkpoint_name, table in host_offload_engram_tables(module):
        weight = table.weight
        lines.append(
            f"{checkpoint_name}: local_rows={table.local_rows} row_start={table.row_start} "
            f"logical={table.logical_num_embeddings} rows_here=[{table.row_start}, "
            f"{min(table.row_start + table.local_rows, table.logical_num_embeddings)}) "
            f"dtype={weight.dtype} meta={weight.is_meta} requires_grad={weight.requires_grad} "
            f"device={weight.device}"
        )
    return lines


def main() -> None:
    args = parse_args()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401  (registers the npu backend)

    from verl.workers.engine import EngineRegistry
    from verl.workers.config import HFModelConfig, FSDPOptimizerConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    assert dist.get_world_size() == world, "torchrun world size mismatch"

    if rank == 0:
        print(f"[smoke] world={world} fsdp={args.fsdp_size} tp={args.tp_size} ep={args.ep_size} "
              f"efsdp={args.efsdp_size} seq_len={args.seq_len} bsz={args.batch_size}", flush=True)

    model_config = HFModelConfig(
        path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_type="language_model",
        trust_remote_code=True,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
    )
    engine_config = build_engine_config(args)
    optimizer_config = FSDPOptimizerConfig(
        lr=args.lr,
        optimizer="AdamW",
        clip_grad=1.0,
        lr_scheduler_type="constant",
    )

    started = time.time()
    engine = EngineRegistry.new(
        model_type="language_model",
        backend="fsdp_turbo_dsv41",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )
    if rank == 0:
        print(f"[smoke] engine created: {type(engine).__name__}", flush=True)
    if args.dry_run:
        dist.barrier()
        if rank == 0:
            print("[smoke] DRY RUN OK (configs + engine construction only)", flush=True)
        return
    engine.initialize()
    if rank == 0:
        print(f"[smoke] initialize() done in {time.time() - started:.1f}s", flush=True)

    module = engine.module
    for line in engram_report(module):
        print(f"[smoke][rank {rank}] {line}", flush=True)
    if rank == 0:
        n_params = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[smoke] parameters: total={n_params/1e9:.2f}B trainable={n_train/1e9:.2f}B", flush=True)

    vocab_size = getattr(getattr(module, "config", None), "vocab_size", 129280)
    data = build_batch(args, int(vocab_size))

    # train_mode() is a context manager: it performs the parameter/optimizer offload entry and
    # exit, which CPU-offloaded runs need around the step.
    step_started = time.time()
    with engine.train_mode():
        outputs = engine.train_batch(data, smoke_loss)
    metrics = dict(outputs["metrics"])
    if rank == 0:
        print(f"[smoke] train_batch done in {time.time() - step_started:.1f}s metrics={metrics}", flush=True)

    if not args.skip_export_check:
        sync_started = time.time()
        try:
            with engine.train_mode():
                gen, _ = engine.get_per_tensor_param()
                payload = [(name, tensor.nbytes, tensor.device.type) for name, tensor in gen]
            names = [name for name, _, _ in payload]
            # Match the same way the exporter excludes the tables: the exported key carries
            # FSDPTurbo's EP-shard suffix (`...embed.weight.ep_shard_00006_of_00008`), so an
            # `endswith` test silently reported "not in payload" while the tables were there
            # (worklog §5.7.4 -- that false green is why the sync bug survived this check).
            engram = [n for n in names if ".engram.embed.weight" in n]
            # Anything bigger than rollout.update_weights_bucket_megabytes (512) leaves the
            # bucket and is published as a single IPC handle; a host tensor there is exactly
            # the E8 failure, so report name + size + device.
            oversized = [
                (n, f"{nbytes / 2**30:.1f}GiB", device)
                for n, nbytes, device in payload
                if nbytes > 512 * 2**20
            ]
            if rank == 0:
                print(
                    f"[smoke] sync payload: {len(names)} tensors in {time.time() - sync_started:.1f}s; "
                    f"engram tables in payload: {engram}; "
                    f"oversized (>512MB bucket, direct-send path): {oversized}",
                    flush=True,
                )
            if engram:
                raise AssertionError(f"host Engram tables leaked into the sync payload: {engram}")
        except Exception as error:  # noqa: BLE001 - report, do not hide the green training step
            if rank == 0:
                print(f"[smoke] WARNING: sync payload check failed: {type(error).__name__}: {error}", flush=True)
            raise

    dist.barrier()
    if rank == 0:
        print("[smoke] TRAIN-ENGINE SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()