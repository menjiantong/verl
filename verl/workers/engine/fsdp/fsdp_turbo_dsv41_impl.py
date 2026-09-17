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
from pathlib import Path

import torch
import torch.distributed as dist

from ..base import EngineRegistry
from .fsdp_turbo_impl import FSDPTurboEngineWithLMHead

logger = logging.getLogger(__name__)

# ``layers.<N>.ffn.experts.<E>.w1.weight`` … the routed experts are stored per expert in
# the checkpoint but stacked in the training model (``gate_up_proj`` / ``down_proj``).
_EXPERT_WEIGHT_PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
# The adapter wraps the backbone as ``self.model``, the checkpoint names the backbone.
_MODEL_PREFIX = "model."
# Fallback sequence length: covers prompt+response of the GRPO config; only sizes the
# RoPE tables and the per-layer attention scratch caches.
_DEFAULT_MAX_SEQ_LEN = 8192


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


def read_dsv41_checkpoint_state_dict(
    ckpt_dir: Path, param_meta: dict[str, tuple[tuple[int, ...], torch.dtype]]
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Build the full, unsharded CPU state dict for the training model.

    Only rank 0 calls this; the result feeds ``set_model_state_dict`` with
    ``broadcast_from_rank0=True``, which broadcasts one tensor at a time and writes
    each rank's shard, so rank 0 is the only rank that ever holds the full model.

    Returns the state dict and a report of the checkpoint tensors that were skipped
    (vision tower / aligner / unused router biases) or expected but absent.
    """
    from safetensors import safe_open

    weight_map = _checkpoint_weight_map(ckpt_dir)
    handles: dict[str, object] = {}

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
        if model_name not in param_meta:
            # vision tower / aligner / image embeddings / vision router bias
            skipped.append(name)
            continue
        shape, dtype = param_meta[model_name]
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

    missing = [name for name in param_meta if name not in state_dict]
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

        # Engram is a V4.1 structure that neither the text-only bring-up nor the
        # sparse-attention path uses yet; the checkpoint has no engram tables.
        training_model = build_deepseek_v41_model(
            tokenizer=self.model_config.tokenizer,
            engram_meta_init=True,
            engram_storage_backend="row_sharded",
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
        return training_model

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        # Metadata of the unsharded model, captured before FSDP-Turbo rewrites
        # parameters into (nested) DTensors: the loader needs global shapes/dtypes.
        param_meta = {name: (tuple(param.shape), param.dtype) for name, param in module.named_parameters()}
        module = FSDPTurbo(self.fsdp_turbo_config, module).model

        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True

        self._materialize_dsv41_parameters(module, param_meta, offload_policy)
        return module

    def _materialize_dsv41_parameters(self, module, param_meta, offload_policy):
        """Load the rollout checkpoint into the sharded model (rank 0 reads, everyone
        receives) and finish the device placement the meta construction skipped."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        checkpoint_dir = Path(self._dsv41_model_path())
        if os.environ.get("VERL_DSV41_RANDOM_INIT", "0") == "1":
            if dist.get_rank() == 0:
                logger.warning("VERL_DSV41_RANDOM_INIT=1: skipping checkpoint weights, deferring experts at N(0, 0.02).")
            full_state = _random_state_dict(param_meta) if dist.get_rank() == 0 else {}
        elif dist.get_rank() == 0:
            # Only rank 0 reads the checkpoint; the broadcast below feeds every other rank.
            full_state, report = read_dsv41_checkpoint_state_dict(checkpoint_dir, param_meta)
            if report["missing"]:
                raise ValueError(
                    f"Checkpoint {checkpoint_dir} is missing {len(report['missing'])} tensors the training model "
                    f"needs, e.g. {report['missing'][:5]}."
                )
            logger.info(
                "DeepSeek-V4.1 checkpoint loaded from %s (%d tensors; %d checkpoint tensors not used, e.g. vision tower).",
                checkpoint_dir,
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
        set_model_state_dict(module, full_state, options=options)
        del full_state

        # Buffers are built with real values (only parameters were deferred) but on
        # the host: attention tables and scratch caches must follow the parameters'
        # compute device, the way fsdp2_load_full_state_dict does for HF models.
        device = torch.accelerator.current_accelerator()
        for buffer in module.buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)

        refresh_dsv41_expert_metadata(module, device)

        still_meta = [name for name, parameter in module.named_parameters() if parameter.is_meta]
        if still_meta:
            raise RuntimeError(
                "DeepSeek-V4.1 parameters were left on the meta device, e.g. {}. "
                "Check that the checkpoint covers every parameter.".format(still_meta[:5])
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
