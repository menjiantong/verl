# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Host-swapped AdamW for the "parameters stay on the NPU" DSV4.1 layout.

The default FSDPTurbo plan offloads parameters and gradients to host memory
(``fsdp_plan.cpu_offload`` is wired from ``fsdp_config.offload_policy``), which
keeps one host copy of each for the whole run.  The alternative layout keeps
parameters and gradients on device and asks the optimizer to keep its moments
on the host, swapping them back in batches inside ``step()``:

    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.dsv41_swap_optim
    actor_rollout_ref.actor.optim.optimizer=AdamWSwapSafe

``fsdp_turbo.memory.swap_optimizer.AdamWSwap`` already implements the swap, but
it cannot be checkpointed: its device ``state`` entries are zero-storage
placeholders while the authoritative moments live in
``param_to_cpu_states_map``, so a plain ``Optimizer.state_dict()`` would save
empty tensors.  :class:`AdamWSwapSafe` materializes the host tensors for the
duration of ``state_dict()`` and refuses ``load_state_dict()`` instead of
silently restoring garbage.

``AdamWSwap`` also assumes every parameter it is given is an FSDP-managed
DTensor: ``opt_states_initialization`` allocates a device moment per parameter
and immediately calls ``.to_local()``.  The DSV4.1 VL engine passes
``module.parameters()`` straight from ``FSDPEngine._build_optimizer``, and that
list also contains plain tensors — the frozen, replicated vision tower/aligner
(``VERL_DSV41_VISION_FREEZE=1``) and the host-offload Engram tables
(``_engram_host_offload``, CPU parameters whose sparse gradients are applied
outside this optimizer).  :func:`split_swap_params` drops them before
construction; a trainable plain parameter that is neither frozen nor an Engram
table is rejected rather than silently left unoptimized.

Verified on the Yundao stack (torch 2.10.0+cpu + torch_npu 2.10.0.post4) with a
1-rank DTensor smoke test.  That torch pairing has no
``torch.accelerator.Stream``/``stream()``/``get_device_properties``; FSDPTurbo's
``swap_optimizer.py`` carries the 2026-11-17 fallbacks for those (see the
``swapopt_4node_20261117`` task record).

Environment knobs (read in every worker at optimizer construction):

``MY_ENV_DSV41_SWAP_MEM_FRACTION``
    Fraction of the free device memory used for one H2D batch of moments
    (default ``0.25``).  Keep it low in the colocated layout: the rest of the
    free memory has to hold activations and the rollout engine.
``MY_ENV_DSV41_SWAP_PARAM_PREFETCH``
    ``1`` prefetches a CPU-resident parameter while its moments are swapped
    in.  Only useful when parameters themselves are offloaded (default ``0``,
    which is the layout this module targets).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch

from fsdp_turbo.memory.swap_optimizer import AdamWSwap
try:  # allocation counters live in the implementation module
    from fsdp_turbo.memory.swap_optimizer.swap_optimizer import host_state_alloc_stats
except ImportError:  # pragma: no cover - older FSDPTurbo layouts
    host_state_alloc_stats = None

logger = logging.getLogger(__name__)
# Worker-side modules default to the Ray worker's root level (WARNING), which would
# swallow the informational swap markers below.  Follow verl's other worker modules
# and honour VERL_LOGGING_LEVEL.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

SWAP_MEM_FRACTION_ENV = "MY_ENV_DSV41_SWAP_MEM_FRACTION"
SWAP_PARAM_PREFETCH_ENV = "MY_ENV_DSV41_SWAP_PARAM_PREFETCH"

DEFAULT_MEM_FRACTION_STATIC = 0.25


def validate_mem_fraction(value: float, source: str) -> float:
    """Reject a fraction outside (0, 1]: 0 would make the swap batch unusable."""
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{source} must be in (0, 1], got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    return validate_mem_fraction(value, name)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def swap_optimizer_kwargs() -> dict:
    """Return the ``AdamWSwap`` knobs selected by the environment."""
    return {
        "mem_fraction_static": _env_float(SWAP_MEM_FRACTION_ENV, DEFAULT_MEM_FRACTION_STATIC),
        "param_prefetch": _env_bool(SWAP_PARAM_PREFETCH_ENV, False),
    }


def split_swap_params(params):
    """Split ``module.parameters()`` into FSDP-managed params and the rest.

    Only FSDP-managed, trainable parameters have device Dtensor moments to
    swap.  Frozen replicated parameters (vision tower/aligner) and the
    host-offload Engram tables are plain tensors that ``AdamWSwap`` cannot
    handle: it would call ``.to_local()`` on them, and for an Engram table it
    would first allocate a full table-sized moment on the device.

    Args:
        params: Iterable of ``torch.nn.Parameter`` (``module.parameters()``).

    Returns:
        ``(managed, skipped)`` — two lists in input order.

    Raises:
        ValueError: a trainable plain parameter that is neither frozen nor
            marked ``_engram_host_offload``; silently dropping it would stop
            training part of the model.
    """
    managed, skipped = [], []
    for param in params:
        is_sharded = hasattr(param, "to_local")
        trainable = bool(getattr(param, "requires_grad", False))
        if is_sharded and trainable:
            managed.append(param)
        elif is_sharded or not trainable or getattr(param, "_engram_host_offload", False):
            # Frozen FSDP param, frozen plain param, or a host Engram table
            # (its sparse rows are updated outside this optimizer).
            skipped.append(param)
        else:
            shape = tuple(getattr(param, "shape", ()))
            raise ValueError(
                "AdamWSwapSafe received a trainable parameter that is neither FSDP-managed nor a "
                f"host-offload Engram table: {type(param).__name__} shape={shape}. The host-swap "
                "optimizer can only update DTensor parameters; pass a filtered parameter list."
            )
    return managed, skipped


def host_state_bytes(optimizer: AdamWSwap) -> int:
    """Bytes held by the host-resident optimizer states."""
    total = 0
    for cpu_state in optimizer.param_to_cpu_states_map.values():
        for value in cpu_state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


@contextmanager
def host_states_materialized(optimizer: AdamWSwap):
    """Expose ``param_to_cpu_states_map`` through ``optimizer.state``.

    ``AdamWSwap`` keeps the authoritative moments on the host and leaves
    zero-storage DTensors in ``state``.  ``torch.optim.Optimizer.state_dict()``
    reads ``state`` as-is, so swap the host tensors in for the duration of the
    call and put the placeholders back afterwards.  The returned dict keeps
    references to the host tensors (``state_dict`` neither clones nor moves
    them), which is exactly what a checkpoint should store.

    ``swap_all_to_host()`` also waits for the outstanding D2H events, so the
    host copies are complete before they are read.  Each materialized entry is
    a *shallow copy* of the live state dict: ``Optimizer.state_dict()`` returns
    the inner dicts by reference, so mutating the live one back would also
    rewrite the caller's checkpoint.
    """
    optimizer.swap_all_to_host()
    saved = {}
    try:
        for param, cpu_state in optimizer.param_to_cpu_states_map.items():
            original = optimizer.state.get(param)
            if original is None:
                continue
            materialized = dict(original)
            for key in optimizer.state_keys:
                host_value = cpu_state.get(key)
                if host_value is not None:
                    materialized[key] = host_value
            saved[param] = original
            optimizer.state[param] = materialized
        yield
    finally:
        for param, original in saved.items():
            optimizer.state[param] = original


class AdamWSwapSafe(AdamWSwap):
    """``AdamWSwap`` with a checkpoint-safe ``state_dict``.

    ``load_state_dict`` is intentionally unsupported: the host moments are
    keyed by the live parameter objects, and mapping a checkpoint back onto
    them needs the FSDP2 sharded-state plumbing that this branch does not have
    yet.  Keep ``trainer.save_freq=-1`` until that lands; the DSV4.1 Yundao
    scripts already do.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize=False,
        mem_fraction_static=None,
        param_prefetch=None,
        **kwargs,
    ):
        managed, skipped = split_swap_params(list(params))
        if not managed:
            raise ValueError(
                "AdamWSwapSafe found no FSDP-managed trainable parameters; the actor must run with "
                "fsdp_turbo and at least one trainable sharded parameter."
            )

        knobs = swap_optimizer_kwargs()
        if mem_fraction_static is not None:
            knobs["mem_fraction_static"] = validate_mem_fraction(
                mem_fraction_static, "mem_fraction_static"
            )
        if param_prefetch is not None:
            knobs["param_prefetch"] = bool(param_prefetch)

        super().__init__(
            managed,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            mem_fraction_static=knobs["mem_fraction_static"],
            param_prefetch=knobs["param_prefetch"],
            **kwargs,
        )

        self._swap_skipped_plain = len(skipped)
        self._swap_first_step_logged = False
        stats = host_state_alloc_stats or {}
        logger.info(
            "[dsv41-swap] AdamWSwapSafe: params=%d skipped_plain=%d host_moments=%.2f GiB "
            "mem_fraction_static=%.2f param_prefetch=%s host_pinned_req=%s "
            "pinned_ok=%d pinned_failed=%d pageable=%d",
            len(self.param_to_cpu_states_map),
            len(skipped),
            host_state_bytes(self) / 1024**3,
            float(knobs["mem_fraction_static"]),
            bool(knobs["param_prefetch"]),
            os.environ.get("MY_ENV_DSV41_SWAP_HOST_PINNED", "1") != "0",
            stats.get("pinned", 0),
            stats.get("pinned_failed", 0),
            stats.get("pageable", 0),
        )

    def state_dict(self):
        with host_states_materialized(self):
            return super().state_dict()

    def load_state_dict(self, *args, **kwargs):
        raise RuntimeError(
            "AdamWSwapSafe cannot restore optimizer state yet: the authoritative Adam "
            "moments live in param_to_cpu_states_map keyed by live parameter objects, and "
            "mapping a checkpoint back onto them needs the FSDP2 sharded-state plumbing. "
            "Keep trainer.save_freq=-1 (the DSV4.1 Yundao scripts do) until that is "
            "implemented."
        )

    def step(self, closure=None):
        loss = super().step(closure)
        if not self._swap_first_step_logged:
            self._swap_first_step_logged = True
            swap_numel = int(getattr(self, "_max_swap_numel", self.swap_numel))
            batch_bytes = swap_numel * self.byte_param * (len(self.swap_state_keys) + 1)
            logger.info(
                "[dsv41-swap] first optimizer step: swap_numel=%d (%.3f GiB of moments+param "
                "buffer per device batch)",
                swap_numel,
                batch_bytes / 1024**3,
            )
        return loss
