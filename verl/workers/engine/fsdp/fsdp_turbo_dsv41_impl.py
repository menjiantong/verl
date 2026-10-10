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

"""FSDP-Turbo engine for DeepSeek-V4.1 training models.

DeepSeek-V4.1's trainable model lives in ``fsdp_turbo.models.deepseek_v41``
(``DeepseekV41ForCausalLMAdapter`` + the vendored reference architecture) and is
*not* a transformers ``AutoModel`` class, so verl's generic ``_build_module``
(HF ``from_pretrained``) cannot construct it. This engine builds the V4.1 adapter
through FSDP-Turbo's own build/initialize/prepare pipeline and lets the parent
``_build_fsdp_module`` wrap it with FSDP-Turbo parallelism.

Two things are specific to the full-size (rollout-sized) model:

* **Structure comes from the rollout checkpoint's ``config.json``**, not from
  FSDP-Turbo's vendored demo config, so the actor/ref model matches the model vLLM
  serves. ``actor_rollout_ref.model.path`` points at that directory (which also
  supplies the tokenizer).
* **The routed experts are ~98% of the parameters** (108B of 111B here). They are
  built on the meta device and materialized straight from the checkpoint, once per
  rank, by ``torch.distributed.checkpoint``'s ``set_model_state_dict`` with
  ``broadcast_from_rank0=True``: rank 0 streams the safetensors, everyone else
  receives the broadcast and keeps only its shard. Building a full replica per rank
  would need ~220GB of accelerator memory, which does not exist.

The checkpoint stores experts per expert (``w1``/``w3``/``w2``) while the training
model keeps them stacked (``gate_up_proj``/``down_proj``), so the loader fuses rows
while reading.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from verl.utils.device import get_device_id

from ..base import EngineRegistry
from .fsdp_turbo_impl import FSDPTurboEngineWithLMHead

logger = logging.getLogger(__name__)

# ``layers.<N>.ffn.experts.<E>.w1.weight`` … the routed experts are stored per expert in
# the checkpoint but stacked in the training model (``gate_up_proj`` / ``down_proj``).
_EXPERT_WEIGHT_PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
# The adapter wraps the backbone as ``self.model``, the checkpoint names the backbone.
_MODEL_PREFIX = "model."
# Host-offload Engram table suffix (both Engram layers, ``layers.{1,14}.engram.embed``).
_ENGRAM_EMBED_SUFFIX = ".engram.embed.weight"
# Fused (stacked) routed-expert parameters: the checkpoint stores one tensor per expert
# (``layers.N.ffn.experts.E.w1/w2/w3.weight``) and the model packs them per layer --
# ``gate_up_proj`` dim 1 is ``[w1 ; w3]``, ``down_proj`` is ``w2`` -- the same fused
# layout vLLM's loader uses (see read_dsv41_checkpoint_state_dict, which builds it by
# hand). The direct shard loader assembles only the window each rank owns.
_FUSED_EXPERT_LAYER_PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(gate_up_proj|down_proj)$")
_FUSED_EXPERT_PARTS = {
    "gate_up_proj": (("w1",), ("w3",)),
    "down_proj": (("w2",),),
}
# Fallback sequence length: covers prompt+response of the GRPO config; only sizes the
# RoPE tables and the per-layer attention scratch caches.
_DEFAULT_MAX_SEQ_LEN = 8192


def _log_rank0(message: str, *args) -> None:
    """Report progress from rank 0.

    Workers run under Ray with a WARNING-level root logger, and a 209GB checkpoint load
    gives no other feedback for minutes, so print explicitly instead of logging.
    """
    if dist.get_rank() == 0:
        print(f"[fsdp_turbo_dsv41] {message % args if args else message}", flush=True)


def _config_path_from_model_path(model_path: str) -> Path:
    """Return the ``config.json`` for a checkpoint directory (or the file itself)."""
    path = Path(model_path)
    return path / "config.json" if path.is_dir() else path


def _checkpoint_weight_map(ckpt_dir: Path) -> dict[str, str]:
    """Map checkpoint tensor name -> shard file name."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as index_file:
            return json.load(index_file)["weight_map"]
    shard_files = sorted(ckpt_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors checkpoint found in {ckpt_dir}.")
    from safetensors import safe_open

    weight_map = {}
    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                weight_map[name] = shard_file.name
    return weight_map


def _engram_checkpoint_rows(checkpoint_dir: Path, checkpoint_name: str) -> int:
    """Row count of an Engram table in the checkpoint, read from the shard header."""
    from safetensors import safe_open

    shard_file = _checkpoint_weight_map(checkpoint_dir)[checkpoint_name]
    with safe_open(str(checkpoint_dir / shard_file), framework="pt", device="cpu") as handle:
        return int(handle.get_slice(checkpoint_name).get_shape()[0])


def checkpoint_buffer_meta(module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes/dtypes of the buffers that the checkpoint owns.

    PyTorch's convention already separates checkpoint state from runtime scratch:
    every attention cache and RoPE table in this model is ``persistent=False``, so
    what is left is exactly the state the checkpoint must supply -- today the MoE
    router correction bias, which `Gate` keeps as an fp32 buffer so that the
    parameter-dtype normalization cannot round it (see the class docstring).

    The loader needs these because DCP will not report a model key that goes
    unassigned -- its ``strict`` only prunes the other direction (see
    :func:`read_dsv41_checkpoint_state_dict` and the callers, which check the values
    that landed). Routing with a zeroed correction bias would otherwise be silent.
    """
    parameters = {name for name, _ in module.named_parameters()}
    return {
        name: (tuple(buffer.shape), buffer.dtype)
        for name, buffer in module.named_buffers()
        if buffer is not None and name not in parameters and name in module.state_dict()
    }


def host_offload_engram_tables(module) -> list[tuple[str, object]]:
    """``(checkpoint tensor name, module)`` for every host-offload Engram table.

    The tables are ``TorchHostOffloadEmbedding`` modules: the weight is a plain CPU
    parameter (row-sharded over the EP mesh, one separate table per Engram layer), not
    a DTensor, and it is deliberately kept out of FSDP's managed parameter groups (see
    ``FSDPTurbo.apply_pre_fsdp_module_hooks``). Two consequences the engine has to
    handle: DCP must not be asked to materialize it (the loader below fills each
    rank's rows straight from the checkpoint), and the weight sync must not stream it.

    Names are taken before FSDP-Turbo wraps the model, while the module tree still is
    the adapter's own (``model.layers.<N>.engram.embed``); the checkpoint spells the
    same tensor without the adapter's ``model.`` prefix.
    """
    tables = []
    for module_name, submodule in module.named_modules():
        if not getattr(submodule, "uses_host_offload", False):
            continue
        checkpoint_name = (
            module_name[len(_MODEL_PREFIX) :] if module_name.startswith(_MODEL_PREFIX) else module_name
        )
        tables.append((f"{checkpoint_name}.weight", submodule))
    return tables


def _is_host_engram_param(name: str, param) -> bool:
    """Whether an exported parameter is one of the host-offload Engram tables.

    The name alone is not enough: ``TorchHostOffloadEmbedding._save_to_state_dict`` rewrites the
    key to ``<module>.weight`` + ``_checkpoint_suffix()``, i.e. ``...engram.embed.weight`` +
    ``.ep_shard_00006_of_00008``, so a test on the *end* of the name never fires and the table
    is exported anyway -- which is what killed the first weight sync of the top32/top64 runs
    (a 22.9 GiB host tensor handed to a device-IPC path, see
    wiki/worklog_dsv41_top64_grpo.md §5.7.4). Prefer the marker FSDPTurbo itself puts on the
    parameter (``_engram_host_offload``, the same attribute its optimizer uses to skip the
    tables) and keep a substring test as the backstop for a view that lost the attribute.
    """
    return bool(getattr(param, "_engram_host_offload", False)) or _ENGRAM_EMBED_SUFFIX in name


def _assemble_fused_expert_window(local_shape, offset, part_groups, part_rows, read_slice, dtype):
    """One rank's window of a fused expert tensor, built from per-expert checkpoint slices.

    Fused layout (identical to :func:`read_dsv41_checkpoint_state_dict` and vLLM's
    loader): dim 0 enumerates experts, so a dim-0 window is a plain range of expert ids;
    dim 1 concatenates the ``part_groups`` -- ``(("w1",), ("w3",))`` for gate_up_proj
    (w1 then w3, each ``part_rows[kind]`` rows), a single ``(("w2",),)`` for down_proj.
    ``read_slice(kind, expert_id, rows, cols)`` reads one source window. Windows may
    span the w1/w3 boundary or a subset of both; each expert's slab is assembled in the
    fused order and stacked along dim 0.
    """
    win1_lo, win1_hi = offset[1], offset[1] + local_shape[1]
    win2_lo, win2_hi = offset[2], offset[2] + local_shape[2]
    blocks = []
    for expert_id in range(offset[0], offset[0] + local_shape[0]):
        pieces = []
        cursor = 0
        for group in part_groups:
            kind = group[0]
            span = part_rows[kind]
            lo, hi = max(win1_lo, cursor), min(win1_hi, cursor + span)
            if hi > lo:
                pieces.append(
                    read_slice(kind, expert_id, slice(lo - cursor, hi - cursor), slice(win2_lo, win2_hi))
                )
            cursor += span
        if not pieces:
            raise RuntimeError("fused expert window intersects no source rows")
        slab = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        blocks.append(slab)
    if not blocks:
        return torch.empty(local_shape, dtype=dtype)
    stacked = blocks[0].unsqueeze(0) if len(blocks) == 1 else torch.stack(blocks, dim=0)
    return stacked.to(dtype)


def _plan_direct_shard_load(module, checkpoint_dir: Path, param_meta, buffer_meta):
    """Plan the per-rank slice reads for ``VERL_DSV41_DIRECT_SHARD_LOAD=1``.

    Every rank derives, from each tensor's own DTensor spec, the exact slice of the
    checkpoint tensor it needs. ``compute_local_shape_and_global_offset`` is the same
    helper DTensor itself uses, so the offsets match by construction, and each rank
    then pulls only that slice out of the safetensors shard -- the trick the Engram
    loader already uses. This replaces the rank-0-reads-the-full-state-dict-and-Gloo-
    broadcasts-it path, which is fast on one node (loopback) but collapses to
    ~30 MB/s across machines (measured 2026-10-09: 964 sequential per-tensor
    broadcasts, hours instead of minutes).

    Returns ``(plan, None)``, or ``(None, reason)`` when a layout cannot be reasoned
    about (nested placements, missing checkpoint key, local shape disagreement). The
    caller then falls back to the broadcast path rather than loading blind.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    weight_map = _checkpoint_weight_map(checkpoint_dir)

    def ckpt_name_of(module_name: str) -> str:
        return module_name[len(_MODEL_PREFIX):] if module_name.startswith(_MODEL_PREFIX) else module_name

    def build_entry(name, tensor, meta, plan) -> str | None:
        """Fill ``plan[name]``; returns a rejection reason or None."""
        from torch.distributed.tensor.placement_types import _StridedShard

        full_shape, dtype = meta[name]
        ckpt_name = ckpt_name_of(name)
        fused_match = _FUSED_EXPERT_LAYER_PATTERN.match(ckpt_name)
        entry = {"ckpt_name": ckpt_name, "full_shape": tuple(full_shape), "dtype": dtype}
        if fused_match:
            entry["fused_experts"] = {
                "layer": int(fused_match.group(1)),
                "parts": _FUSED_EXPERT_PARTS[fused_match.group(2)],
            }
        else:
            shard_file = weight_map.get(ckpt_name)
            if shard_file is None:
                return f"checkpoint has no tensor named {ckpt_name!r}"
            entry["shard_file"] = shard_file
        if isinstance(tensor, DTensor):
            if any(isinstance(p, (list, tuple)) for p in tensor.placements):
                return f"nested DTensor placements on {name}: {tensor.placements}"
            if any(isinstance(p, _StridedShard) and p.split_factor != 1 for p in tensor.placements):
                # A strided shard is not a contiguous window of the source tensor; the
                # committed loader cannot reconstruct it from one get_slice window.
                return f"strided shard placements on {name}: {tensor.placements}"
            local_shape, offset = compute_local_shape_and_global_offset(
                tuple(full_shape), tensor.device_mesh, tensor.placements
            )
            actual = tuple(tensor.to_local().shape)
            if actual != tuple(local_shape):
                return f"computed local shape {tuple(local_shape)} != actual {actual} on {name}"
            entry.update(
                kind="shard",
                local_shape=tuple(local_shape),
                offset=tuple(offset),
                mesh=tensor.device_mesh,
                placements=tuple(tensor.placements),
            )
        else:
            entry.update(kind="full")
        if fused_match:
            fused = entry["fused_experts"]
            if entry["kind"] == "shard":
                expert_lo, expert_hi = entry["offset"][0], entry["offset"][0] + entry["local_shape"][0]
            else:
                expert_lo, expert_hi = 0, full_shape[0]
            for expert_id in range(expert_lo, expert_hi):
                for group in fused["parts"]:
                    source = f"layers.{fused['layer']}.ffn.experts.{expert_id}.{group[0]}.weight"
                    if source not in weight_map:
                        return f"checkpoint has no tensor named {source!r}"
        entry["param"] = tensor
        plan[name] = entry
        return None

    plan: dict[str, dict] = {}
    for name, param in module.named_parameters():
        if name not in param_meta:
            continue  # routed experts stay meta; Engram tables were popped in _build_fsdp_module
        reason = build_entry(name, param, param_meta, plan)
        if reason:
            return None, reason
    for name, buffer in module.named_buffers():
        if name not in buffer_meta or name in plan:
            continue
        reason = build_entry(name, buffer, buffer_meta, plan)
        if reason:
            return None, reason
    expected = len(param_meta) + len(buffer_meta)
    if len(plan) != expected:
        return None, f"planned {len(plan)} tensors, expected {expected} (param_meta + buffer_meta)"
    return plan, None


def _read_direct_shards(plan, checkpoint_dir: Path) -> tuple[dict[str, torch.Tensor], int]:
    """Read this rank's share of the planned tensors (a slice for sharded ones).

    Fused expert parameters are assembled from their per-expert checkpoint tensors --
    only the experts and rows this rank's window covers are read, never the whole
    stacked tensor.
    """
    from safetensors import safe_open

    weight_map = _checkpoint_weight_map(checkpoint_dir)
    handles: dict[str, object] = {}

    def handle_for(shard_file: str):
        handle = handles.get(shard_file)
        if handle is None:
            handle = safe_open(str(checkpoint_dir / shard_file), framework="pt", device="cpu")
            handles[shard_file] = handle
        return handle

    values: dict[str, torch.Tensor] = {}
    total_bytes = 0
    try:
        for name, entry in plan.items():
            if entry.get("fused_experts"):
                fused = entry["fused_experts"]
                if entry["kind"] == "shard":
                    window_shape, window_offset = entry["local_shape"], entry["offset"]
                else:
                    window_shape = entry["full_shape"]
                    window_offset = (0,) * len(window_shape)
                part_rows = {}
                for group in fused["parts"]:
                    source = f"layers.{fused['layer']}.ffn.experts.{window_offset[0]}.{group[0]}.weight"
                    part_rows[group[0]] = int(handle_for(weight_map[source]).get_slice(source).get_shape()[0])

                def read_slice(kind, expert_id, rows, cols, _layer=fused["layer"]):
                    source = f"layers.{_layer}.ffn.experts.{expert_id}.{kind}.weight"
                    return handle_for(weight_map[source]).get_slice(source)[(rows, cols)]

                tensor = _assemble_fused_expert_window(
                    tuple(window_shape), tuple(window_offset), fused["parts"], part_rows, read_slice, entry["dtype"]
                )
                expected_shape = window_shape
            elif entry["kind"] == "shard":
                handle = handle_for(entry["shard_file"])
                local_shape, offset = entry["local_shape"], entry["offset"]
                if any(size == 0 for size in local_shape):
                    tensor = torch.empty(local_shape, dtype=entry["dtype"])
                else:
                    window = tuple(slice(o, o + size) for o, size in zip(offset, local_shape))
                    tensor = handle.get_slice(entry["ckpt_name"])[window].reshape(local_shape)
                expected_shape = entry["local_shape"]
            else:
                handle = handle_for(entry["shard_file"])
                tensor = handle.get_tensor(entry["ckpt_name"])
                expected_shape = entry["full_shape"]
            if tuple(tensor.shape) != tuple(expected_shape):
                raise RuntimeError(
                    f"direct shard read for {name}: got shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected_shape)}."
                )
            value = tensor.to(entry["dtype"])
            values[name] = value
            total_bytes += value.numel() * value.element_size()
    finally:
        del handles
    return values, total_bytes


def _load_direct_shards(module, plan, values, engram_tables, cpu_offload: bool = False) -> None:
    """Write this rank's shards into the model, mirroring torch's distribution protocol.

    Reference: ``_distribute_tensors`` in ``torch/distributed/_state_dict_utils.py``
    (the tail of the rank0-broadcast load). A real (non-meta) parameter is written
    **in place** through ``param.to_local().copy_`` and its object is left untouched;
    only a meta-initialized parameter is *replaced*, with a DTensor built from the
    existing parameter's own mesh/placements/shape/stride. Replacing the real
    parameters instead -- the first direct-load draft -- broke the FSDP-Turbo
    parameter bookkeeping: the weight-sync export then died with "Attempted to set
    the storage of a tensor on device cpu to a storage on different device npu:N"
    (2026-10-09 09:20). The partial ``assign=True`` load is limited to the replaced
    (meta) keys; completeness of the rest is enforced by the planning pass. The Engram
    tables stay hidden for the walk for the same reason as in the broadcast path:
    they sit in the module tree but are deliberately absent from the loaded state.

    ``cpu_offload`` mirrors `StateDictOptions(cpu_offload=...)` of the broadcast path:
    a replacement must end up with its local tensor on **CPU** for CPU-offloaded FSDP
    groups. ``DTensor.from_local`` silently moves a CPU value onto the mesh device
    ("convert the local tensor to desired device base on device mesh's device type"),
    which handed FSDP npu-local shards and blew up at the first forward with
    "FSDP parameters should be materialized on CPU when enabling CPU offloading"
    (2026-10-09 09:5x, single-node smoke). The broadcast path lands on CPU through
    DCP's own ``local_state_dict[key].cpu()`` after ``_distribute_tensors``; the
    explicit ``.cpu()`` below reproduces that end state (probe: from_local(cpu).cpu()
    and DCP cpu_offload=True give the same param device, both forward clean under
    fully_shard + CPUOffloadPolicy).
    """
    from torch.distributed.tensor import DTensor

    replacements: dict[str, torch.Tensor] = {}
    for name, entry in plan.items():
        param = entry["param"]
        value = values[name]
        if entry["kind"] == "shard" and isinstance(param, DTensor):
            local = param.to_local()
            if local.is_meta:
                replacement = DTensor.from_local(
                    value,
                    param.device_mesh,
                    list(param.placements),
                    shape=param.shape,
                    stride=param.stride(),
                )
                if cpu_offload:
                    replacement = replacement.cpu()
                replacements[name] = replacement
            else:
                with torch.no_grad():
                    local.copy_(value)
            continue
        if param.is_meta:
            replacements[name] = value
        else:
            with torch.no_grad():
                param.data.copy_(value)
    if replacements:
        detached_engram = hide_engram_tables(module, engram_tables)
        try:
            module.load_state_dict(replacements, strict=False, assign=True)
        finally:
            restore_modules(detached_engram)


def _verify_direct_shards(plan, checkpoint_dir: Path, module) -> int:
    """Rank-0 sample check: its loaded shard must equal the checkpoint slice exactly.

    The planning pass already proved every rank's *shape* against its DTensor spec;
    this catches the data-level mistakes shapes cannot see (offset off-by-one, slice
    interpretation). It compares rank 0's own loaded local tensor against the same
    window of a full read, for up to ``VERL_DSV41_DIRECT_SHARD_LOAD_VERIFY`` (default
    3) small parameters -- cheap, and it exercises the whole read->from_local->copy
    chain. Every other rank uses the identical code path and offset formula; the
    run's consistency metrics (training/rollout_probs_diff_*) are the end-to-end net.
    """
    if dist.get_rank() != 0:
        return 0
    sample = int(os.environ.get("VERL_DSV41_DIRECT_SHARD_LOAD_VERIFY", "3"))
    if sample <= 0:
        _log_rank0("direct shard load: verification disabled by VERL_DSV41_DIRECT_SHARD_LOAD_VERIFY=0")
        return 0
    from safetensors import safe_open

    parameters = dict(module.named_parameters())
    candidates = []
    for name, entry in plan.items():
        if entry["kind"] != "shard" or name not in parameters or len(entry["offset"]) == 0:
            continue
        numel = 1
        for size in entry["local_shape"]:
            numel *= size
        byte_size = numel * torch.empty((), dtype=entry["dtype"]).element_size()
        if 0 < byte_size <= 64 * 1024**2:
            candidates.append((byte_size, name))
    candidates.sort()
    handles: dict[str, object] = {}
    checked = 0
    try:
        for _, name in candidates[:sample]:
            entry = plan[name]
            handle = handles.get(entry["shard_file"])
            if handle is None:
                handle = safe_open(str(checkpoint_dir / entry["shard_file"]), framework="pt", device="cpu")
                handles[entry["shard_file"]] = handle
            full = handle.get_tensor(entry["ckpt_name"]).to(entry["dtype"])
            window = tuple(slice(o, o + size) for o, size in zip(entry["offset"], entry["local_shape"]))
            expected = full[window].reshape(entry["local_shape"])
            actual = parameters[name].to_local()
            if actual.is_meta:
                raise RuntimeError(
                    f"direct shard load left {name} on the meta device (the assign load did "
                    "not materialize it); refusing to continue with a meta parameter."
                )
            if actual.device.type != "cpu":
                actual = actual.cpu()
            if not torch.equal(expected, actual.to(entry["dtype"])):
                raise RuntimeError(
                    f"direct shard load verification failed for {name}: rank0's loaded shard does not "
                    "match the checkpoint slice. The slice mapping is wrong -- retry with "
                    "VERL_DSV41_DIRECT_SHARD_LOAD=0 and investigate before trusting this run."
                )
            checked += 1
    finally:
        del handles

    # Fused expert parameters get their own check: rank0 independently assembles the
    # whole stacked tensor (the same recurrence read_dsv41_checkpoint_state_dict uses)
    # and compares its own loaded window against it. This covers the window->per-expert
    # mapping, including windows that span the w1/w3 boundary.
    weight_map = _checkpoint_weight_map(checkpoint_dir)
    for suffix, label in (
        (".ffn.experts.gate_up_proj", "gate_up_proj"),
        (".ffn.experts.down_proj", "down_proj"),
    ):
        name = next((candidate for candidate in plan if candidate.endswith(suffix)), None)
        if name is None or name not in parameters:
            continue
        entry = plan[name]
        fused = entry["fused_experts"]
        kinds = [group[0] for group in fused["parts"]]
        handles = {}
        try:
            full = torch.empty(entry["full_shape"], dtype=entry["dtype"])
            for expert_id in range(entry["full_shape"][0]):
                cursor = 0
                for kind in kinds:
                    source = f"layers.{fused['layer']}.ffn.experts.{expert_id}.{kind}.weight"
                    shard_file = weight_map[source]
                    handle = handles.get(shard_file)
                    if handle is None:
                        handle = safe_open(str(checkpoint_dir / shard_file), framework="pt", device="cpu")
                        handles[shard_file] = handle
                    piece = handle.get_tensor(source).to(entry["dtype"])
                    if len(kinds) == 1:
                        full[expert_id] = piece
                    else:
                        full[expert_id, cursor : cursor + piece.shape[0]] = piece
                        cursor += piece.shape[0]
        finally:
            del handles
        if entry["kind"] == "shard":
            window = tuple(slice(o, o + s) for o, s in zip(entry["offset"], entry["local_shape"]))
            expected_shape = entry["local_shape"]
        else:
            window = tuple(slice(0, s) for s in entry["full_shape"])
            expected_shape = entry["full_shape"]
        expected = full[window].reshape(expected_shape)
        actual = parameters[name].to_local()
        if actual.is_meta:
            raise RuntimeError(
                f"direct shard load left {name} on the meta device (the assign load did not "
                "materialize it); refusing to continue with a meta parameter."
            )
        if actual.device.type != "cpu":
            actual = actual.cpu()
        if not torch.equal(expected, actual.to(entry["dtype"])):
            raise RuntimeError(
                f"direct shard load verification failed for {name} (fused experts): rank0's loaded "
                "shard does not match the checkpoint. Retry with VERL_DSV41_DIRECT_SHARD_LOAD=0 and "
                "investigate before trusting this run."
            )
        checked += 1
        _log_rank0("direct shard load verified: %s assembled independently and matched (fused expert)", label)

    if checked == 0:
        raise RuntimeError(
            "direct shard load: no verifiable sample parameter found; refusing to proceed "
            "unverified (set VERL_DSV41_DIRECT_SHARD_LOAD_VERIFY=0 to override explicitly)."
        )
    _log_rank0("direct shard load verified: %d sampled shard(s) bit-exact on rank0", checked)
    return checked


def hide_engram_tables(module, engram_tables) -> list[tuple[object, str, object]]:
    """Detach the host-offload Engram modules from the tree while DCP enumerates it.

    DCP derives the model-side keys by recursing the module tree
    (``_iterate_valid_model_state``) and loads with ``strict=True``. The Engram tables
    *are* parameters of the model but are deliberately absent from the state dict handed
    to DCP -- they are filled row-by-row from the checkpoint in
    :meth:`_load_host_engram_tables` -- so the load aborts with ``Missing key:
    model.layers.<N>.engram.embed.weight``. Taking the modules out of the tree for the
    duration of the load (rather than relaxing ``strict``) keeps the fail-closed
    behaviour for every other parameter. They are put back right after, so the forward
    graph, ``pre_fsdp_hook``'s ``shard_()`` result and FSDP-Turbo's ignored-module
    bookkeeping are untouched.

    Returns the ``(parent, attribute, submodule)`` triples for :func:`restore_modules`.
    """
    wanted = {id(table): checkpoint_name for checkpoint_name, table in engram_tables}
    detached = []
    found = set()
    # Located by identity, not by rebuilding the module path from the checkpoint name: the
    # path an FSDP wrapper exposes is not the caller's business, and a mismatch has to be
    # an error rather than a silently wrong key space.
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        if id(submodule) not in wanted:
            continue
        parent_path, _, attribute = module_name.rpartition(".")
        if not attribute:
            raise RuntimeError(f"Engram table {wanted[id(submodule)]} is the root module; cannot detach it.")
        parent = module.get_submodule(parent_path) if parent_path else module
        found.add(id(submodule))
        detached.append((parent, attribute, submodule))
    missing = sorted(name for table_id, name in wanted.items() if table_id not in found)
    if missing:
        raise RuntimeError(
            f"Engram tables {missing} are not in the module tree that DCP is about to walk; "
            "their keys would be reported missing."
        )
    for parent, attribute, _ in detached:
        setattr(parent, attribute, nn.Module())
    return detached


def restore_modules(detached) -> None:
    """Put back the modules :func:`hide_engram_tables` detached."""
    for parent, attribute, module in detached:
        setattr(parent, attribute, module)


def place_checkpoint_buffers_for_load(module, buffer_meta, log=None) -> list[str]:
    """Align the checkpoint-owned buffers with the parameters' device before the DCP load.

    ``set_model_state_dict(broadcast_from_rank0=True)`` collects the device of every dim>0
    tensor of the model state and refuses to guess when it sees more than one ("Multiple
    devices found"; meta is excluded -- that is how the deferred experts get assigned). The
    parameters are placed by the offload policy, i.e. CPU-side under ``cpu_offload=True``
    (the way this engine loads ~100 GB of experts), while
    ``prepare_deepseek_v41_model_for_fsdp`` moves every buffer to the accelerator. The other
    buffers are all ``persistent=False`` and therefore invisible to that scan; the
    checkpoint buffers are not, so align them here. Returns the names moved.

    The load itself replaces those tensors (``assign=True`` comes with the meta experts), so
    they end up where the load put them; the caller's post-load loop is what moves the
    buffers back to the compute device.
    """
    target = next((param.device for param in module.parameters() if not param.is_meta), None)
    if target is None:
        return []
    moved = []
    for name, buffer in module.named_buffers():
        if buffer is not None and name in buffer_meta and buffer.device != target:
            buffer.data = buffer.data.to(target)
            moved.append(name)
    if log is not None and moved:
        log(f"aligned {len(moved)} checkpoint buffers to {target} for the load (e.g. {moved[:2]})")
    return moved


def move_buffers_to_device(module, device, log=None) -> list[str]:
    """Put every buffer on the parameters' compute device (the load leaves them on the host).

    ``load_state_dict(assign=True)`` -- which comes with the deferred experts -- *replaces* the
    parameter and buffer tensors with the ones the broadcast produced, and with
    ``cpu_offload=True`` those are host tensors. Replacing the registered buffer is the move
    that always works: ``setattr`` on a buffer name updates ``_buffers[name]`` in place and
    keeps its persistence, whereas ``buffer.data = <tensor on another device>`` relies on
    ``set_data``'s type rules -- it rejects some targets outright ("Attempted to call
    `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type").
    Returns the names replaced.
    """
    moved = []
    for name, buffer in module.named_buffers():
        if buffer is None or buffer.device == device:
            continue
        owner, _, local = name.rpartition(".")
        target = module.get_submodule(owner) if owner else module
        setattr(target, local, buffer.to(device))
        moved.append(name)
    if log is not None and moved:
        log(f"moved {len(moved)} buffers to {device} (e.g. {moved[:2]})")
    return moved


def read_dsv41_checkpoint_state_dict(
    ckpt_dir: Path,
    param_meta: dict[str, tuple[tuple[int, ...], torch.dtype]],
    buffer_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Build the full, unsharded CPU state dict for the training model.

    Only rank 0 calls this; the result feeds ``set_model_state_dict`` with
    ``broadcast_from_rank0=True``, which broadcasts one tensor at a time and writes
    each rank's shard, so rank 0 is the only rank that ever holds the full model.

    ``buffer_meta`` carries the checkpoint-owned buffers (see
    :func:`checkpoint_buffer_meta`); they are loaded like parameters but keep their
    own dtype, so an fp32 checkpoint tensor is not rounded on the way in.

    Returns the state dict and a report of the checkpoint tensors that were skipped
    (vision tower / aligner / unused router biases) or expected but absent.
    """
    from safetensors import safe_open

    weight_map = _checkpoint_weight_map(ckpt_dir)
    handles: dict[str, object] = {}
    expected_meta = dict(param_meta)
    expected_meta.update(buffer_meta or {})

    def fetch(name: str) -> torch.Tensor:
        shard_file = weight_map[name]
        if shard_file not in handles:
            handles[shard_file] = safe_open(str(ckpt_dir / shard_file), framework="pt", device="cpu")
        return handles[shard_file].get_tensor(name)

    def cast(tensor: torch.Tensor, expected_name: str, shape, dtype) -> torch.Tensor:
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"Checkpoint tensor {expected_name} has shape {tuple(tensor.shape)}, "
                f"the training model expects {tuple(shape)}."
            )
        return tensor.to(dtype)

    state_dict: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    expert_names: dict[int, dict[str, dict[int, str]]] = {}
    for name in weight_map:
        expert_match = _EXPERT_WEIGHT_PATTERN.match(name)
        if expert_match:
            layer_id, expert_id, kind = int(expert_match.group(1)), int(expert_match.group(2)), expert_match.group(3)
            expert_names.setdefault(layer_id, {}).setdefault(kind, {})[expert_id] = name
            continue
        model_name = _MODEL_PREFIX + name
        if model_name not in expected_meta:
            # vision tower / aligner / image embeddings / vision router bias
            skipped.append(name)
            continue
        shape, dtype = expected_meta[model_name]
        state_dict[model_name] = cast(fetch(name), name, shape, dtype)

    for layer_id, kinds in sorted(expert_names.items()):
        for fused_name, fused_kinds in (
            (f"layers.{layer_id}.ffn.experts.gate_up_proj", (("w1",), ("w3",))),
            (f"layers.{layer_id}.ffn.experts.down_proj", (("w2",),)),
        ):
            model_name = _MODEL_PREFIX + fused_name
            if model_name not in param_meta:
                skipped.extend(
                    kinds[kind][expert_id] for parts in fused_kinds for kind in parts for expert_id in kinds[kind]
                )
                continue
            shape, dtype = param_meta[model_name]
            # ``gate_up_proj`` is [w1 ; w3] along the intermediate dim, ``down_proj`` is w2 --
            # exactly how vLLM's loader views the same tensors, so the fused layout is shared.
            per_kind = {kind: kinds[kind] for parts in fused_kinds for kind in parts}
            fused = torch.empty(shape, dtype=dtype)
            for expert_id in range(shape[0]):
                offset = 0
                for kind, names in per_kind.items():
                    tensor = fetch(names[expert_id]).to(dtype)
                    if len(per_kind) == 1:
                        fused[expert_id] = tensor
                    else:
                        fused[expert_id, offset : offset + tensor.shape[0]] = tensor
                        offset += tensor.shape[0]
            state_dict[model_name] = fused

    missing = [name for name in expected_meta if name not in state_dict]
    return state_dict, {"skipped": skipped, "missing": missing}


def _random_state_dict(
    param_meta: dict[str, tuple[tuple[int, ...], torch.dtype]]
) -> dict[str, torch.Tensor]:
    """Fallback initializer (``VERL_DSV41_RANDOM_INIT=1``): same recipe as
    ``fsdp_turbo.models.deepseek_v41.initialize_deepseek_v41_model``, applied to the
    deferred expert tensors only. Lets the RL loop be brought up without checkpoint IO.
    """
    state_dict: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in param_meta.items():
        if not name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            continue
        state_dict[name] = torch.empty(shape, dtype=dtype).normal_(mean=0.0, std=0.02)
    return state_dict


def refresh_dsv41_expert_metadata(module, device) -> None:
    """Rebuild the EP dispatcher bookkeeping that lives outside the state dict.

    ``expert_ids_per_ep_rank`` is an attribute, not a buffer: the EP dispatcher reads it
    at forward time, and meta-built expert tensors leave it on the meta device with
    uninitialized contents. Must run once the expert weights are real.
    """
    from fsdp_turbo.distributed.expert_parallel.expert_parallel import refresh_expert_parallel_metadata

    refresh_expert_parallel_metadata(module, device=device)


@EngineRegistry.register(
    model_type="language_model",
    backend="fsdp_turbo_dsv41",
    device=["npu", "cuda"],
)
class FSDPTurboDSV41EngineWithLMHead(FSDPTurboEngineWithLMHead):
    # Fused expert parameters, packed over the global expert dimension.
    _FUSED_EXPERT_SUFFIXES = (".ffn.experts.gate_up_proj", ".ffn.experts.down_proj")

    def _export_param(self, name, param):
        """Stream fused expert parameters one EP shard at a time.

        The default export all-gathers a packed expert tensor ``[384, 4608, 5120]``
        (18.1 GiB) onto every rank before unfusing it into per-expert keys, which does
        not fit next to the rollout engine's weights in the same HBM. Gathering one
        expert block at a time keeps the peak at a single block (2.26 GiB) while
        emitting the same per-expert keys, numbered with the global expert ids. Blocks
        that are not DTensor-sharded (or that are not expert parameters) take the
        default path.

        With ``offload_policy`` (parameters on the CPU) that gather is also *slow*:
        measured at 80 s per sync on a 14 GB model (~2% of that for the actual
        transfer and load), which grows to ~920 s on the production 114 GB checkpoint.
        ``VERL_DSV41_LOCAL_EXPERT_EXPORT=1`` switches to streaming *only this rank's*
        experts straight out of the local shard, which needs no collective at all. In
        the colocated layout the receiving engine rank holds the same expert range --
        the ZMQ endpoint pairs trainer rank i with engine rank i, and both sides shard
        the expert dimension contiguously by rank -- so the other seven eighths of the
        gather are waste. A wrong pairing cannot pass unnoticed: the engine would load
        the wrong experts and ``training/rollout_probs_diff_*`` /
        ``training/rollout_actor_probs_pearson_corr`` would jump from ~1e-2 / ~0.98 to
        O(1) / ~0.5.
        """
        from torch.distributed.tensor import DTensor

        from .utils import split_fused_expert_tensor

        if _is_host_engram_param(name, param):
            # Host-offload Engram tables: frozen (the rollout engine also keeps them
            # requires_grad=False) and host-resident. Streaming them would move
            # 2 x 24.6 GiB per rank *per step* to say what the engine already has from
            # the same checkpoint. A generated param the engine never receives simply
            # keeps its loaded value, which is the point.
            return

        if not name.endswith(self._FUSED_EXPERT_SUFFIXES) or not isinstance(param, DTensor) or param.dim() != 3:
            yield from super()._export_param(name, param)
            return

        local = param.to_local()
        if local.shape[0] <= 0 or param.shape[0] % local.shape[0] or local.shape[0] == param.shape[0]:
            yield from super()._export_param(name, param)
            return

        if os.environ.get("VERL_DSV41_LOCAL_EXPERT_EXPORT") == "1":
            yield from self._export_local_experts(name, param)
            return

        for first_expert_id in range(0, param.shape[0], local.shape[0]):
            block_tensor = param.narrow(0, first_expert_id, local.shape[0])
            materialized = block_tensor.to(get_device_id(), non_blocking=True).full_tensor()
            yield from split_fused_expert_tensor(name, materialized, first_expert_id=first_expert_id)

    def _export_local_experts(self, name, param):
        """Emit only the experts this rank's shard owns, with their global ids."""
        from torch.distributed.tensor import Shard

        from .utils import split_fused_expert_tensor

        local = param.to_local()
        shard_mesh_dims = [
            d for d, placement in enumerate(param.placements) if isinstance(placement, Shard) and placement.dim == 0
        ]
        if len(shard_mesh_dims) != 1 or local.shape[0] * param.device_mesh.size(shard_mesh_dims[0]) != param.shape[0]:
            # Not a single contiguous shard over the expert dim: keep the generic path.
            yield from super()._export_param(name, param)
            return

        first_expert_id = param.device_mesh.get_coordinate()[shard_mesh_dims[0]] * local.shape[0]
        materialized = local.to(get_device_id(), non_blocking=True)
        yield from split_fused_expert_tensor(name, materialized, first_expert_id=first_expert_id)

    def _build_module(self):
        # Keep verl's Qwen VLM monkey-patch guard the same way FSDPTurboEngine
        # does: do not slice the text model before FSDP-Turbo's own CP split.
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return self._build_dsv41_module()
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def prepare_model_inputs(self, micro_batch):
        """Drop the position ids verl hands to HF-style models.

        The V4.1 reference forward indexes its RoPE tables by absolute position and reads
        padding from ``attention_mask``; it has no ``position_ids`` input (passing one is
        rejected). verl's padded batch path always builds one, so remove it here. Packed
        (remove-padding) inputs would need per-sequence positions the reference forward
        cannot consume, so fail loudly instead of training on wrong positions.
        """
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        model_inputs.pop("position_ids", None)
        if model_inputs.get("cu_seqlens") is not None:
            raise NotImplementedError(
                "DeepSeek-V4.1 training does not support packed (remove-padding) inputs; "
                "set actor_rollout_ref.model.use_remove_padding=False."
            )
        return model_inputs, output_args

    def _build_dsv41_module(self):
        from fsdp_turbo.models.deepseek_v41 import (
            build_deepseek_v41_model,
            initialize_deepseek_v41_model,
            prepare_deepseek_v41_model_for_fsdp,
        )

        # Engram storage: ``host_offload`` is the only backend the current FSDP-Turbo
        # still implements (``row_sharded`` was dropped in 0769938; passing it now raises
        # for any model that declares Engram layers -- and the top64 checkpoint does,
        # layers 1 and 14). The host-offload tables stay in CPU memory and are sharded
        # over the EP mesh by FSDP-Turbo's ``pre_fsdp_hook``; this engine loads each
        # rank's row range from the checkpoint afterwards (see
        # :meth:`_load_host_engram_tables`).
        training_model = build_deepseek_v41_model(
            tokenizer=self.model_config.tokenizer,
            engram_meta_init=True,
            engram_storage_backend="host_offload",
            use_sparse_flash_attn=False,
            # The routed experts stay on meta until the checkpoint loader
            # materializes each rank's shard.
            experts_meta_init=True,
            config_path=_config_path_from_model_path(self._dsv41_model_path()),
            max_seq_len=self._dsv41_max_seq_len(),
        )
        initialize_deepseek_v41_model(training_model)
        training_model = prepare_deepseek_v41_model_for_fsdp(
            training_model,
            device=torch.accelerator.current_accelerator(),
            parameter_dtype=torch.bfloat16,
        )

        # [mem-debug 2026-10-09] The actor update plateaus at ~60 GB/card regardless of the
        # response length (512 x B8 in the 10th run and 1024 x B2 in the 12th both peaked
        # at ~60.8 GB and one of them OOM'd on a 300 MiB alloc), so the dominant consumer
        # is something per-layer that does not scale with tokens. These hooks print
        # alloc/reserved/peak at every decoder layer's forward entry/exit and after its
        # backward, which localizes the consumer to a layer index and a direction.
        # Enable with FSDP_TURBO_MEM_DEBUG=1 (rank filter: FSDP_TURBO_MEM_DEBUG_RANKS).
        if os.environ.get("FSDP_TURBO_MEM_DEBUG") == "1":
            import re as _re

            import torch.distributed as _dist

            _ranks = {
                int(r) for r in os.environ.get("FSDP_TURBO_MEM_DEBUG_RANKS", "0").split(",") if r.strip()
            }
            try:
                _rank = _dist.get_rank()
            except Exception:
                _rank = 0
            if _rank in _ranks:

                def _mem() -> str:
                    return (
                        f"alloc={torch.npu.memory_allocated() / 2**30:.2f} "
                        f"resv={torch.npu.memory_reserved() / 2**30:.2f} "
                        f"max={torch.npu.max_memory_allocated() / 2**30:.2f}"
                    )

                def _mk_hook(tag):
                    def _fn(module, *args):
                        print(f"[mem-debug] r{_rank} {tag} {_mem()}", flush=True)

                    return _fn

                _n = 0
                for _name, _mod in training_model.named_modules():
                    # Blocks ("...layers.7") and their direct children ("...layers.7.mlp"): the
                    # child granularity is what attributes the ~1.1 GB/layer saved-for-backward.
                    if _re.fullmatch(r".*layers\.\d+(\.[^.]+)?", _name):
                        _tag = _name.split("layers.", 1)[1]
                        _mod.register_forward_pre_hook(_mk_hook(f"L{_tag} FWD-in"))
                        _mod.register_forward_hook(_mk_hook(f"L{_tag} FWD-out"))
                        _mod.register_full_backward_hook(_mk_hook(f"L{_tag} BWD"))
                        _n += 1
                print(f"[mem-debug] installed {_n} layer hook sets on rank {_rank}", flush=True)
        return training_model

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        # Metadata of the unsharded model, captured before FSDP-Turbo rewrites
        # parameters into (nested) DTensors: the loader needs global shapes/dtypes.
        param_meta = {name: (tuple(param.shape), param.dtype) for name, param in module.named_parameters()}
        buffer_meta = checkpoint_buffer_meta(module)
        # Host-offload Engram tables are handled outside DCP (see
        # :func:`host_offload_engram_tables`); DCP must not see them, and they are
        # deliberately absent from ``param_meta`` so it never tries.
        engram_tables = host_offload_engram_tables(module)
        for checkpoint_name, _ in engram_tables:
            param_meta.pop(_MODEL_PREFIX + checkpoint_name, None)
        module = FSDPTurbo(self.fsdp_turbo_config, module).model

        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True

        self._materialize_dsv41_parameters(module, param_meta, buffer_meta, offload_policy, engram_tables)
        return module

    def _materialize_dsv41_parameters(self, module, param_meta, buffer_meta, offload_policy, engram_tables=()):
        """Load the rollout checkpoint into the sharded model and finish the device
        placement the meta construction skipped.

        Two load paths (2026-10-09): with ``VERL_DSV41_DIRECT_SHARD_LOAD=1`` every rank
        slice-reads its own shards straight from the checkpoint, so no weight data
        crosses the machines. Otherwise rank 0 reads the full state dict and DCP
        broadcasts it to everyone -- fast on one node (loopback), but the per-tensor
        Gloo broadcast collapses across machines (measured ~30 MB/s on the
        80.5.25.x link) and turns the load into hours.
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        checkpoint_dir = Path(self._dsv41_model_path())
        _log_rank0(
            "materializing %d parameters + %d checkpoint buffers from %s (experts deferred to meta)",
            len(param_meta),
            len(buffer_meta),
            checkpoint_dir,
        )
        started = time.time()
        if os.environ.get("VERL_DSV41_DIRECT_SHARD_LOAD", "0") == "1":
            if self._load_dsv41_parameters_directly(
                module, checkpoint_dir, param_meta, buffer_meta, offload_policy, engram_tables, started
            ):
                self._finish_dsv41_materialization(module, engram_tables, checkpoint_dir)
                return
        if os.environ.get("VERL_DSV41_RANDOM_INIT", "0") == "1":
            _log_rank0("VERL_DSV41_RANDOM_INIT=1: skipping checkpoint weights, deferring experts at N(0, 0.02)")
            full_state = _random_state_dict(param_meta) if dist.get_rank() == 0 else {}
        elif dist.get_rank() == 0:
            # Only rank 0 reads the checkpoint; the broadcast below feeds every other rank.
            full_state, report = read_dsv41_checkpoint_state_dict(checkpoint_dir, param_meta, buffer_meta)
            if report["missing"]:
                raise ValueError(
                    f"Checkpoint {checkpoint_dir} is missing {len(report['missing'])} tensors the training model "
                    f"needs, e.g. {report['missing'][:5]}."
                )
            _log_rank0(
                "checkpoint read: %d tensors (%d checkpoint tensors unused, e.g. vision tower)",
                len(full_state),
                len(report["skipped"]),
            )
        else:
            full_state = {}

        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=bool(offload_policy),
        )
        place_checkpoint_buffers_for_load(module, buffer_meta, log=_log_rank0)
        # DCP's strict load only prunes keys the model cannot take and never reports a model key
        # left unassigned (see _broadcast_state_dict); and it overwrites the dict handed to it,
        # so keep the expected values aside. Routing with a zeroed correction bias would be
        # silent and wrong -- check that the checkpoint buffers landed.
        expected_buffers = (
            {name: full_state[name].clone() for name in buffer_meta if name in full_state} if dist.get_rank() == 0 else {}
        )
        # DCP walks the module tree and loads ``strict=True``; the Engram tables are in the
        # tree but not in ``full_state`` (see :func:`host_offload_engram_tables`), so they are
        # detached for the duration of the load instead of relaxing the strictness.
        detached_engram = hide_engram_tables(module, engram_tables)
        try:
            set_model_state_dict(module, full_state, options=options)
        finally:
            restore_modules(detached_engram)
        if dist.get_rank() == 0:
            current = dict(module.named_buffers())
            unloaded = [
                name
                for name, expected in expected_buffers.items()
                if name not in current or not torch.equal(expected, current[name])
            ]
            if unloaded:
                raise RuntimeError(
                    f"checkpoint buffers were not loaded into the model: {unloaded[:3]}. "
                    "See place_checkpoint_buffers_for_load / checkpoint_buffer_meta."
                )
            _log_rank0("checkpoint buffers loaded: %d (bit-exact against the checkpoint)", len(expected_buffers))
        del full_state
        _log_rank0("parameters materialized in %.1fs", time.time() - started)

        self._finish_dsv41_materialization(module, engram_tables, checkpoint_dir)

    def _finish_dsv41_materialization(self, module, engram_tables, checkpoint_dir: Path) -> None:
        """Tail shared by both parameter-load paths (broadcast and direct shards)."""
        # Buffers are built with real values (only parameters were deferred) but on
        # the host: attention tables and scratch caches must follow the parameters'
        # compute device, the way fsdp2_load_full_state_dict does for HF models.
        device = torch.accelerator.current_accelerator()
        move_buffers_to_device(module, device, log=_log_rank0)

        refresh_dsv41_expert_metadata(module, device)

        self._load_host_engram_tables(engram_tables, checkpoint_dir)

        still_meta = [name for name, parameter in module.named_parameters() if parameter.is_meta]
        if still_meta:
            raise RuntimeError(
                "DeepSeek-V4.1 parameters were left on the meta device, e.g. {}. "
                "Check that the checkpoint covers every parameter.".format(still_meta[:5])
            )

    def _load_dsv41_parameters_directly(
        self, module, checkpoint_dir, param_meta, buffer_meta, offload_policy, engram_tables, started
    ) -> bool:
        """``VERL_DSV41_DIRECT_SHARD_LOAD=1``: every rank slice-reads its own shards.

        Returns False when the plan cannot be built, in which case the caller runs the
        rank0 read + broadcast path instead; True once the load completed, including
        the same checkpoint-buffer bit-exactness check the broadcast path runs.
        """
        plan, reason = _plan_direct_shard_load(module, checkpoint_dir, param_meta, buffer_meta)
        if plan is None:
            _log_rank0(
                "VERL_DSV41_DIRECT_SHARD_LOAD=1 rejected (%s); falling back to rank0 read + broadcast",
                reason,
            )
            return False
        place_checkpoint_buffers_for_load(module, buffer_meta, log=_log_rank0)
        values, total_bytes = _read_direct_shards(plan, checkpoint_dir)
        _log_rank0(
            "checkpoint read: %d tensors slice-read directly (%.2f GiB this rank; no cross-rank broadcast)",
            len(values),
            total_bytes / 1024**3,
        )
        _load_direct_shards(module, plan, values, engram_tables, cpu_offload=bool(offload_policy))
        if dist.get_rank() == 0:
            current = dict(module.named_buffers())
            unloaded = [
                name
                for name in buffer_meta
                if name in values and (name not in current or not torch.equal(values[name], current[name]))
            ]
            if unloaded:
                raise RuntimeError(
                    f"checkpoint buffers were not loaded into the model: {unloaded[:3]}. "
                    "See place_checkpoint_buffers_for_load / checkpoint_buffer_meta."
                )
            _log_rank0("checkpoint buffers loaded: %d (bit-exact against the checkpoint)", len(buffer_meta))
        del values
        _verify_direct_shards(plan, checkpoint_dir, module)
        _log_rank0("parameters materialized in %.1fs (direct shard load)", time.time() - started)
        return True

    def _load_host_engram_tables(self, engram_tables, checkpoint_dir: Path) -> None:
        """Fill each rank's host-offload Engram rows from the checkpoint, then freeze.

        The tables are EP row-sharded host tensors, so unlike the DCP-loaded parameters
        every rank reads *its own* row range straight out of the checkpoint shards: a
        ``safe_open`` slice read, never the whole 196 GB tensor. Rows past
        ``logical_num_embeddings`` are alignment padding the hash never addresses; they
        keep their initial values, the same way the engine's table leaves its padding.

        The tables are then frozen (``requires_grad=False`` + no autograd hookup). This
        matches the rollout engine, which builds its Engram parameters with
        ``requires_grad=False`` and loads them from this same checkpoint, so actor and
        engine stay identical without ever streaming ~393 GB per weight sync -- and no
        AdamW state for two 24.6 GB tables per rank is ever allocated. Detaching
        ``_grad_keepalive`` also keeps ``_TorchHostFetch`` from recording a backward, so
        no sparse Engram gradients pile up in host memory between steps.
        """
        if not engram_tables:
            return
        from safetensors import safe_open

        weight_map = _checkpoint_weight_map(checkpoint_dir)
        handles: dict[str, object] = {}
        started = time.time()
        layout: list[tuple[str, int, int]] = []
        total_bytes = 0
        for checkpoint_name, table in engram_tables:
            if table.weight.is_meta or table.mesh is None:
                raise RuntimeError(
                    f"Engram table {checkpoint_name} was not EP-sharded by FSDP-Turbo before the load "
                    "(shard_() runs in pre_fsdp_hook); cannot read it back."
                )
            start = table.row_start
            end = min(start + table.local_rows, table.logical_num_embeddings)
            if checkpoint_name not in weight_map:
                raise ValueError(f"Checkpoint {checkpoint_dir} has no tensor named {checkpoint_name!r}.")
            shard_file = weight_map[checkpoint_name]
            if shard_file not in handles:
                handles[shard_file] = safe_open(str(checkpoint_dir / shard_file), framework="pt", device="cpu")
            rows = handles[shard_file].get_slice(checkpoint_name)[start:end]
            with torch.no_grad():
                # ``rows`` are checkpoint rows [start, end); they land at the front of the
                # owner-local table, whose row 0 is exactly global row ``row_start``.
                table.weight[: end - start].copy_(rows)
                if not torch.equal(table.weight[0], rows[0]):
                    raise RuntimeError(f"Engram table {checkpoint_name}: checkpoint rows landed shifted.")
            layout.append((checkpoint_name, start, end))
            total_bytes += (end - start) * table.embedding_dim * table.weight.element_size()
            print(
                f"[fsdp_turbo_dsv41] engram {checkpoint_name}: checkpoint rows [{start}, {end}) "
                f"-> local rows [0, {end - start}) of {table.local_rows} "
                f"({table.local_rows - (end - start)} padding rows left as initialized); "
                f"logical {table.logical_num_embeddings}",
                flush=True,
            )
            table.weight.requires_grad_(False)
            table._grad_keepalive = torch.zeros((), dtype=torch.float32, device=table._grad_keepalive.device)
        del handles

        # Within one EP replica the ranks must tile the checkpoint's row space exactly: a
        # gap silently zeroes a row range for *every* lookup, and an overlap feeds
        # different values to trainer and engine for the same hash id. Replicas
        # (world_size / EP > 1, e.g. the FSDP=16 / EP=8 one-node topology) deliberately
        # hold the *same* rows, so each distinct range must be held by exactly one rank
        # per replica -- once in each of the EP groups -- and the distinct ranges must
        # cover [0, checkpoint rows) without gaps. (Comparing the raw sorted pieces
        # against a running cursor, as this check used to, treats a replica's second copy
        # of range 0 as "not tiled across ranks at 0".)
        copies_per_range = {
            name: dist.get_world_size() // table.mesh.size() for name, table in engram_tables
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, [(name, start, end) for name, start, end in layout])
        if dist.get_rank() == 0:
            by_name: dict[str, list[tuple[int, int]]] = {}
            for rank_layout in gathered:
                for name, start, end in rank_layout:
                    by_name.setdefault(name, []).append((start, end))
            for name, pieces in by_name.items():
                pieces.sort()
                counts: dict[tuple[int, int], int] = {}
                for piece in pieces:
                    counts[piece] = counts.get(piece, 0) + 1
                expected = 0
                for start, end in sorted(counts):
                    if start != expected:
                        raise RuntimeError(f"Engram table {name}: rows are not tiled across ranks at {start}.")
                    expected = end
                checkpoint_rows = _engram_checkpoint_rows(checkpoint_dir, name)
                if expected != checkpoint_rows:
                    raise RuntimeError(
                        f"Engram table {name}: ranks cover {expected} rows, the checkpoint has {checkpoint_rows}."
                    )
                expected_copies = copies_per_range[name]
                misheld = {piece: n for piece, n in counts.items() if n != expected_copies}
                if misheld:
                    raise RuntimeError(
                        f"Engram table {name}: row ranges {misheld} are held by a number of ranks other "
                        f"than the {expected_copies} EP replica(s); the rows are not tiled once per "
                        "replica."
                    )
            _log_rank0(
                "host-offload Engram tables loaded: %d table(s), %.1f GiB per rank in %.1fs (frozen)",
                len(engram_tables),
                total_bytes / 1024**3,
                time.time() - started,
            )

    def _dsv41_model_path(self) -> str:
        """Directory of the rollout checkpoint: config.json + tokenizer + safetensors."""
        model_config = self.model_config
        for attribute in ("local_hf_config_path", "hf_config_path", "local_path", "path"):
            candidate = getattr(model_config, attribute, None)
            if candidate:
                return candidate
        raise ValueError("actor_rollout_ref.model.path must point at the DeepSeek-V4.1 checkpoint directory.")

    def _dsv41_max_seq_len(self) -> int:
        override = os.environ.get("VERL_DSV41_MAX_SEQ_LEN")
        if override:
            return int(override)
        return _DEFAULT_MAX_SEQ_LEN
