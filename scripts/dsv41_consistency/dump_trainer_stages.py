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

"""Dump every stage of the FSDPTurbo training forward for a fixed token stream.

This is the "training side" half of the train-vs-inference consistency harness
(``scripts/dsv41_consistency/``): it builds/lodes the model exactly like the
``fsdp_turbo_dsv41`` engine does (same builder kwargs, same FSDPTurbo topology, same
checkpoint loader), runs the fixture sequences from ``make_fixture.py`` and writes, for
every hooked module, the activation it produced -- so the engine-side dump can be
compared stage by stage instead of only end-to-end logprobs.

Output (rank 0 only), one file per fixture sequence:

    <out-dir>/trainer_len<LEN>.pt
      input_ids       [S]        int64
      logprobs        [S]        float32  (log_softmax of the final logits, gathered)
      top1_ids        [S]        int64    (argmax of the final logits)
      stages          {name: tensor}      float32 activations, see HOOK_PATTERNS

Usage (8 NPUs; must match the GRPO script's topology):

    PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
    torchrun --nproc_per_node=8 scripts/dsv41_consistency/dump_trainer_stages.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32 \
        --out-dir /mnt/share/m00899630/dsv41/dump/trainer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HOOK_SUFFIXES = [
    "model.embed",
    "model.norm",
    "model.head",
]
BLOCK_SUFFIXES = [
    "",
    ".attn",
    ".attn_norm",
    ".ffn_norm",
    ".ffn",
    ".ffn.gate",
    ".ffn.experts",
    ".ffn.shared_experts",
    # layer 2 owns the compressed KV and the index; layer 3 reads layer 2's selection.
    ".attn.compressor",
    ".attn.indexer",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--fixture", default="/mnt/share/m00899630/dsv41/dump/fixture.json")
    parser.add_argument("--out-dir", default="/mnt/share/m00899630/dsv41/dump/trainer")
    parser.add_argument("--lengths", default="", help="Comma separated subset of fixture lengths (default: all).")
    parser.add_argument("--fsdp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--efsdp-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("VERL_DSV41_MAX_SEQ_LEN", "2048")))
    parser.add_argument("--max-elements", type=int, default=40_000_000, help="Skip stages larger than this (safety).")
    parser.add_argument("--sparse-flash-attn", action="store_true",
                        help="Build the model with the fused NPU SparseFlashMla instead of the eager torch fallback.")
    parser.add_argument("--out-tag", default="", help="Suffix for the output files (A/B runs).")
    parser.add_argument("--force-cmp-mask-mode-3", action="store_true",
                        help="Work around the A2/A3 tiling check for window-only layers.")
    parser.add_argument("--fp32-attention", action="store_true",
                        help="Run the eager indexed attention with fp32 scores/probabilities (A/B).")
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


def force_cmp_mask_mode_3():
    """Pass cmp_mask_mode=3 to the SparseFlashMla metadata even without compressed KV.

    The A2/A3 tiling function rejects the dense-BSND layout with cmp_mask_mode=0
    ("cmpMaskMode should be 3 on A2/A3, but got 0"), and FSDPTurbo sets exactly that for
    layers whose compress_ratio is 0. The engine does not hit this because it runs the
    paged (PA_BBND) layout.
    """
    from cann_ops_transformer import ops

    originals = []
    for name in ("sparse_flash_mla_metadata", "sparse_flash_mla", "sparse_flash_mla_grad"):
        original = getattr(ops, name, None)
        if original is None:
            continue

        def patched(*args, __original=original, **kwargs):
            kwargs["cmp_mask_mode"] = 3
            return __original(*args, **kwargs)

        setattr(ops, name, patched)
        originals.append((name, original))
    return originals


def install_fp32_attention(torch):
    """Run the eager indexed attention in fp32 instead of bf16 scores / bf16 probabilities.

    `indexed_sparse_attention_torch` (fsdp_turbo/ops/cpu/sparse_attention.py:49-76) lets the
    QK einsum run in the autocast dtype (bf16) and casts the probabilities to bf16 before the
    PV product. This variant keeps both in fp32 to test whether the gap against the fused
    engine kernel is caused by that rounding.
    """
    import fsdp_turbo.ops.npu.sparse_attention as npu_attn

    def fp32_attention(
        query,
        key_value,
        attention_sink,
        topk_indices,
        softmax_scale,
        attention_mask=None,
        use_sparse_flash_attn=False,
        compressed_key_value=None,
        compressed_topk_indices=None,
        compress_ratio=1,
        window_size=128,
    ):
        del use_sparse_flash_attn, compress_ratio, window_size
        if compressed_key_value is not None:
            if compressed_topk_indices is None:
                raise ValueError("compressed_topk_indices is required with compressed_key_value")
            compressed_topk_indices = torch.where(
                compressed_topk_indices >= 0,
                compressed_topk_indices + key_value.size(1),
                compressed_topk_indices,
            )
            topk_indices = torch.cat((topk_indices, compressed_topk_indices), dim=-1)
            key_value = torch.cat((key_value, compressed_key_value), dim=1)
        batch_size, sequence_length, _, head_dim = query.shape
        topk_size = topk_indices.shape[-1]
        topk_indices = topk_indices.to(query.device)
        safe_indices = topk_indices.clamp_min(0).long()
        expanded_key_value = key_value.unsqueeze(1).expand(-1, sequence_length, -1, -1)
        selected_key_value = torch.gather(
            expanded_key_value, 2,
            safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk_size, head_dim),
        )
        attention_scores = (
            torch.einsum("bshd,bskd->bshk", query.float(), selected_key_value.float()) * softmax_scale
        )
        attention_scores = attention_scores.masked_fill(topk_indices.unsqueeze(2) < 0, -torch.inf)
        if attention_mask is not None and key_value.size(1) == attention_mask.size(1):
            selected_key_mask = torch.gather(attention_mask.bool(), 1, safe_indices.flatten(1)).view(
                batch_size, sequence_length, topk_size
            )
            attention_scores = attention_scores.masked_fill(~selected_key_mask.unsqueeze(2), -torch.inf)
        sink_scores = attention_sink.float().view(1, 1, -1, 1).expand(batch_size, sequence_length, -1, -1)
        attention_probabilities = torch.softmax(torch.cat((attention_scores, sink_scores), dim=-1), dim=-1)[..., :-1]
        attention_output = torch.einsum(
            "bshk,bskd->bshd", attention_probabilities, selected_key_value.float()
        )
        return attention_output.to(query.dtype)

    npu_attn.indexed_sparse_attention_torch = fp32_attention
    return npu_attn


def to_cpu_fp32(value, max_elements):
    """Best-effort conversion of a hook payload to a detachable fp32 CPU tensor."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() > max_elements:
            return None
        return value.detach().float().cpu()
    if isinstance(value, (tuple, list)):
        return {f"[{i}]": to_cpu_fp32(v, max_elements) for i, v in enumerate(value)}
    return None


def restore_fp32_parameters(model, model_path, log, device, only: str = ""):
    """Compute with the checkpoint's fp32 parameter values where the trainer holds bf16 (H2).

    `prepare_deepseek_v41_model_for_fsdp` normalizes *every* floating parameter to bf16, so the
    architecture's own fp32 declarations get rounded on load while the engine keeps the
    checkpoint's fp32 values. In the real 4-layer slice that is 32 parameter tensors the trainer
    actually has: the router's correction bias (`Gate.bias`, declared fp32 in `model.py:936`),
    `attn_sink` and the six `hc_*` per layer. The router bias is the one that matters most -- it
    steers expert *selection* only, so its rounding is a purely discrete perturbation that no
    input graft can remove (offline, on the engine's own MoE inputs: rounding it to bf16 changes
    the trainer's top-6 set on 20-62% of tokens -- and the trainer's *recorded* routing is
    reproduced 100% with the rounded value, 21-62% with the true fp32 one -- while the engine's
    own router capture reproduces with fp32).

    The parameter itself must stay bf16: FSDP2 asserts one original dtype per parameter group
    (`_fsdp_param_group.py::_init_mp_dtypes`, hit when this used to reassign `parameter.data`).
    So the fp32 value is swapped in around the owning module's forward instead -- same tensors
    the engine holds, and FSDP never sees a second dtype. Env-gated by the caller
    (`VERL_DSV41_KEEP_FP32_PARAMS=1`). Returns the names overridden.
    """
    import torch
    from safetensors import safe_open

    from verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl import _MODEL_PREFIX, _checkpoint_weight_map

    weight_map = _checkpoint_weight_map(Path(model_path))
    handles: dict[str, object] = {}
    overrides: dict[str, dict[str, torch.Tensor]] = {}  # module path -> {attr: fp32 tensor}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            ckpt_name = name[len(_MODEL_PREFIX):] if name.startswith(_MODEL_PREFIX) else name
            shard = weight_map.get(ckpt_name)
            if shard is None or parameter.is_meta or parameter.dtype == torch.float32:
                continue
            if shard not in handles:
                handles[shard] = safe_open(str(Path(model_path) / shard), framework="pt", device="cpu")
            handle = handles[shard]
            if handle.get_slice(ckpt_name).get_dtype() != "F32":
                continue
            value = handle.get_tensor(ckpt_name)
            if tuple(value.shape) != tuple(parameter.shape):
                log(f"skip fp32 override of {name}: ckpt shape {tuple(value.shape)} != {tuple(parameter.shape)}")
                continue
            module_path, _, attr = name.rpartition(".")
            if only == "gate" and attr != "bias":  # `=gate`: router correction bias only
                continue
            if only == "continuous" and attr == "bias":  # `=continuous`: hc_*/attn_sink only
                continue
            # The load uses cpu_offload=True, so `parameter.device` is cpu; the swap has to
            # land where the forward runs (the module's activations), not where it was loaded.
            overrides.setdefault(module_path, {})[attr] = value.to(device=device, dtype=torch.float32)

    module_names = {name for name, _ in model.named_modules()}
    for module_path, attrs in overrides.items():
        if module_path not in module_names:
            log(f"skip fp32 override for unknown module {module_path!r}")
            continue
        module = model.get_submodule(module_path)
        saved: dict[str, tuple[bool, object]] = {}

        def swap_in(mod, _inputs, _attrs=attrs, _saved=saved):  # pre-hook: (module, args)
            for attr, tensor in _attrs.items():
                _saved[attr] = (attr in mod.__dict__, mod.__dict__.get(attr))
                object.__setattr__(mod, attr, tensor)

        def swap_out(mod, _inputs, output, _saved=saved):  # post-hook: (module, args, output)
            for attr, (existed, old) in _saved.items():
                if existed:
                    object.__setattr__(mod, attr, old)
                else:
                    mod.__dict__.pop(attr, None)
            return output

        module.register_forward_pre_hook(swap_in)
        module.register_forward_hook(swap_out)

    names = [f"{path}.{attr}" for path, attrs in overrides.items() for attr in attrs]
    log(f"fp32 use-site override on {len(names)} parameters ({len(overrides)} modules): "
        f"{sorted({n.split('.')[-1] for n in names})}")
    return names


def install_attention_op_hooks(torch, stage_sink, max_elements):
    """Record the inputs/output of `indexed_sparse_attention` per layer.

    The attention itself is a plain function call inside `Attention.forward` (there is no
    submodule to hook), so wrap the module-level name the model imported. Calls happen in
    layer order during a forward, which is how the entries get their layer index.
    """
    import fsdp_turbo.models.deepseek_v41.model as model_module

    original = model_module.indexed_sparse_attention
    if getattr(original, "_dump_wrapped", False):
        return original

    def wrapper(query, key_value, attention_sink, topk_indices, softmax_scale, attention_mask=None, **kwargs):
        output = original(
            query, key_value, attention_sink, topk_indices, softmax_scale, attention_mask, **kwargs
        )
        index = len({key.split(".")[0] for key in stage_sink if key.startswith("attn_op")})
        payload = {
            "query": query,
            "key_value": key_value,
            "attn_sink": attention_sink,
            "topk_indices": topk_indices,
            "output": output,
        }
        for key, value in payload.items():
            tensor = to_cpu_fp32(value, max_elements)
            if isinstance(tensor, torch.Tensor):
                stage_sink[f"attn_op{index}.{key}"] = tensor
        return output

    wrapper._dump_wrapped = True
    model_module.indexed_sparse_attention = wrapper
    return original


def install_hooks(model, torch, max_elements, stage_sink):
    """Hook the milestone modules; store fp32 CPU copies under short stage names."""

    def make_hook(name):
        def hook(_module, _inputs, output):
            payload = to_cpu_fp32(output, max_elements)
            if payload is None:
                return
            if isinstance(payload, dict):
                for suffix, tensor in payload.items():
                    if tensor is not None:
                        stage_sink[f"{name}{suffix}"] = tensor
            else:
                stage_sink[name] = payload

        return hook

    handles = []
    for name, module in model.named_modules():
        short = None
        if name in HOOK_SUFFIXES:
            short = name
        elif name.startswith("model.layers."):
            parts = name.split(".", 3)
            if len(parts) == 4:
                rest = "." + parts[3]
                if rest in BLOCK_SUFFIXES:
                    short = name
        if short is not None:
            handles.append(module.register_forward_hook(make_hook(short)))
    return handles


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path (or MODEL_PATH) is required")
    if args.sparse_flash_attn:
        # `torch_npu` breaks the search path for `cann_ops_transformer` (verified: the same
        # import succeeds before torch_npu is loaded and fails after), and FSDPTurbo imports
        # the ops lazily at first attention call. Importing the package here, before
        # init_distributed() pulls in torch_npu, keeps it importable from then on.
        from cann_ops_transformer import ops as _cann_ops  # noqa: F401  (pre-import, see above)
        print("[dump-trainer] pre-imported cann_ops_transformer (fused sparse attention)", flush=True)
        if args.force_cmp_mask_mode_3:
            force_cmp_mask_mode_3()
            print("[dump-trainer] forcing cmp_mask_mode=3 on sparse_flash_mla_metadata", flush=True)
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

    from verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl import (
        read_dsv41_checkpoint_state_dict,
        refresh_dsv41_expert_metadata,
    )

    def log(message):
        if rank == 0:
            print(f"[dump-trainer] {message}", flush=True)

    config = build_fsdp_turbo_config(args)
    init_parallel_state(config)

    log("building model (routed experts deferred to meta)")
    started = time.time()
    model = build_deepseek_v41_model(
        tokenizer=None,
        engram_meta_init=True,
        engram_storage_backend="row_sharded",
        use_sparse_flash_attn=args.sparse_flash_attn,  # the engine builds it with False (eager torch)
        experts_meta_init=True,
        config_path=args.model_path,
        max_seq_len=args.max_seq_len,
    )
    initialize_deepseek_v41_model(model)
    model = prepare_deepseek_v41_model_for_fsdp(model, device=device, parameter_dtype=torch.bfloat16)
    param_meta = {name: (tuple(param.shape), param.dtype) for name, param in model.named_parameters()}
    log(f"model built in {time.time() - started:.1f}s ({len(param_meta)} parameters)")

    model = FSDPTurbo(config, model).model

    log("materializing parameters from checkpoint")
    started = time.time()
    if rank == 0:
        full_state, ckpt_report = read_dsv41_checkpoint_state_dict(Path(args.model_path), param_meta)
        # Otherwise DCP's strict=True raises a generic "Missing key(s)" much later (see H6 in
        # plans/dsv41-real-weights-4layer/plan.md); fail here with the actual names instead.
        assert not ckpt_report["missing"], (
            f"checkpoint {args.model_path} lacks {len(ckpt_report['missing'])} tensors the model needs, "
            f"e.g. {ckpt_report['missing'][:5]}"
        )
    else:
        full_state = {}
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True)
    set_model_state_dict(model, full_state, options=options)
    del full_state
    log(f"parameters materialized in {time.time() - started:.1f}s")

    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    fp32_mode = os.environ.get("VERL_DSV41_KEEP_FP32_PARAMS", "")
    if fp32_mode:
        restore_fp32_parameters(model, args.model_path, log, device, only="" if fp32_mode == "1" else fp32_mode)
    refresh_dsv41_expert_metadata(model, device)
    still_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    assert not still_meta, f"parameters left on meta: {still_meta[:5]}"

    with open(args.fixture) as f:
        fixture = json.load(f)
    want = {int(x) for x in args.lengths.split(",") if x} or None
    samples = [s for s in fixture["samples"] if want is None or s["target_len"] in want]

    model.eval()
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        length = sample["target_len"]
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        stage_sink: dict[str, torch.Tensor] = {}
        if args.fp32_attention and not getattr(install_fp32_attention, "_installed", False):
            install_fp32_attention(torch)
            install_fp32_attention._installed = True
            print("[dump-trainer] using fp32 eager attention", flush=True)
        install_attention_op_hooks(torch, stage_sink, args.max_elements)
        handles = install_hooks(model, torch, args.max_elements, stage_sink)
        try:
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        seq = input_ids.shape[1]
        # (a) canonical next-token convention: logprob of token t comes from logits[t-1].
        #     Position 0 has no context, so it is dropped -- this is what the engine reports.
        next_logprobs = logprobs[0, :-1].gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        # (b) the convention verl's engine uses for the padded path: labels are the *rolled*
        #     flat token stream (`torch.roll(input_ids.values(), -1)`), so the last position
        #     predicts the first token of the sequence (a wrap-around) instead of nothing.
        rolled_ids = torch.roll(input_ids, shifts=-1, dims=1)
        rolled_logprobs = logprobs[0].gather(-1, rolled_ids[0].unsqueeze(-1)).squeeze(-1)
        topk = torch.topk(logits[0], k=16, dim=-1)
        if rank == 0:
            payload = {
                "input_ids": input_ids[0].cpu(),
                "logprobs": rolled_logprobs.float().cpu(),
                "next_logprobs": next_logprobs.float().cpu(),
                "topk_ids": topk.indices.cpu(),
                "topk_logprobs": logprobs[0].gather(-1, topk.indices).cpu(),
                "logits_absmax": float(logits.abs().max().item()),
                "stages": stage_sink,
            }
            if length <= 64:
                payload["logits"] = logits[0].to(torch.float16).cpu()
            path = out_dir / f"trainer{args.out_tag}_len{length}.pt"
            torch.save(payload, path)
            sizes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(stage_sink.items())[:6])
            log(f"len={length}: next_logprobs[1:4]={[round(x, 4) for x in next_logprobs[:3].tolist()]} "
                f"roll_last={rolled_logprobs[-1].item():.4f} logits absmax={payload['logits_absmax']:.2f} "
                f"stages={len(stage_sink)} ({sizes} ...)")
            log(f"len={length}: wrote {path}")
        # The ring buffers (window/compress KV) must start clean for the next sample, the way a
        # fresh engine request would.
        for name, buffer in model.named_buffers():
            if name.endswith("kv_cache"):
                buffer.zero_()
        dist.barrier()

    dist.barrier()
    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
