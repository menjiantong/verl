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
"""CPU tests for the host-swapped AdamW used by the DSV4.1 device-param layout.

``AdamWSwap`` itself needs an accelerator (it creates streams and zero-storage
device DTensors at construction), so these tests cover the parts that decide
what a checkpoint and the environment knobs see: the host/state swap, the
``state_dict`` integration, the env parsing and the explicit refusal of
``load_state_dict``.
"""

import importlib
import inspect

import pytest
import torch

pytest.importorskip("fsdp_turbo")

from verl.utils import dsv41_swap_optim as swap_optim  # noqa: E402


class _HostStateOptimizer:
    """Stand-in for the attributes ``host_states_materialized`` touches."""

    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(2))
        self.max_placeholder = torch.full((1,), -1.0)  # sentinel: must survive a None host value
        self.state = {
            self.param: {
                "exp_avg": torch.empty(0),
                "exp_avg_sq": torch.empty(0),
                "max_exp_avg_sq": self.max_placeholder,
            }
        }
        self.param_to_cpu_states_map = {
            self.param: {
                "exp_avg": torch.ones(2),
                "exp_avg_sq": torch.full((2,), 2.0),
                "max_exp_avg_sq": None,
            }
        }
        self.state_keys = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
        self.swap_all_to_host_calls = 0

    def swap_all_to_host(self):
        self.swap_all_to_host_calls += 1


def test_materialize_swaps_host_tensors_into_state_and_restores():
    optimizer = _HostStateOptimizer()
    original_state = optimizer.state[optimizer.param]
    placeholder = original_state["exp_avg"]
    host_state = optimizer.param_to_cpu_states_map[optimizer.param]

    with swap_optim.host_states_materialized(optimizer):
        materialized = optimizer.state[optimizer.param]
        # A copy, not the live dict: Optimizer.state_dict() hands the inner dict
        # to the caller, and restoring the live one would rewrite the checkpoint.
        assert materialized is not original_state
        assert materialized["exp_avg"] is host_state["exp_avg"]
        assert materialized["exp_avg_sq"] is host_state["exp_avg_sq"]
        # A None host value (amsgrad off for max_exp_avg_sq) must not shadow the state entry.
        assert materialized["max_exp_avg_sq"] is optimizer.max_placeholder

    assert optimizer.state[optimizer.param] is original_state
    assert optimizer.state[optimizer.param]["exp_avg"] is placeholder
    # The dict handed to the caller keeps the host tensor after the context exits.
    assert materialized["exp_avg"] is host_state["exp_avg"]
    assert optimizer.swap_all_to_host_calls == 1


def test_materialize_restores_state_when_body_raises():
    optimizer = _HostStateOptimizer()
    original_state = optimizer.state[optimizer.param]

    with pytest.raises(RuntimeError, match="boom"):
        with swap_optim.host_states_materialized(optimizer):
            raise RuntimeError("boom")

    assert optimizer.state[optimizer.param] is original_state


def test_state_dict_sees_host_moments():
    """The real ``Optimizer.state_dict`` must serialize the host tensors, not the empty placeholders."""
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.state[param]["momentum_buffer"] = torch.empty(0)
    optimizer.state_keys = ["momentum_buffer"]
    host_momentum = torch.tensor([3.0, 4.0])
    optimizer.param_to_cpu_states_map = {param: {"momentum_buffer": host_momentum}}
    swap_calls = []
    optimizer.swap_all_to_host = lambda: swap_calls.append(1)

    with swap_optim.host_states_materialized(optimizer):
        state_dict = optimizer.state_dict()
        assert torch.equal(state_dict["state"][0]["momentum_buffer"], host_momentum)

    # The saved dict must survive the placeholder restore.
    assert state_dict["state"][0]["momentum_buffer"] is host_momentum
    assert optimizer.state[param]["momentum_buffer"].numel() == 0
    assert swap_calls == [1]


def test_swap_optimizer_kwargs_defaults(monkeypatch):
    monkeypatch.delenv(swap_optim.SWAP_MEM_FRACTION_ENV, raising=False)
    monkeypatch.delenv(swap_optim.SWAP_PARAM_PREFETCH_ENV, raising=False)

    assert swap_optim.swap_optimizer_kwargs() == {
        "mem_fraction_static": swap_optim.DEFAULT_MEM_FRACTION_STATIC,
        "param_prefetch": False,
    }


def test_swap_optimizer_kwargs_from_env(monkeypatch):
    monkeypatch.setenv(swap_optim.SWAP_MEM_FRACTION_ENV, "0.4")
    monkeypatch.setenv(swap_optim.SWAP_PARAM_PREFETCH_ENV, "1")

    assert swap_optim.swap_optimizer_kwargs() == {"mem_fraction_static": 0.4, "param_prefetch": True}


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.5", "not-a-number"])
def test_swap_optimizer_kwargs_rejects_bad_fraction(monkeypatch, raw):
    monkeypatch.setenv(swap_optim.SWAP_MEM_FRACTION_ENV, raw)

    with pytest.raises(ValueError):
        swap_optim.swap_optimizer_kwargs()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_validate_mem_fraction_rejects_out_of_range(value):
    """Explicit ``mem_fraction_static=`` bypasses the env parser; it must be checked too."""
    with pytest.raises(ValueError):
        swap_optim.validate_mem_fraction(value, "mem_fraction_static")


@pytest.mark.parametrize("value", [0.25, 1.0])
def test_validate_mem_fraction_accepts_in_range(value):
    assert swap_optim.validate_mem_fraction(value, "mem_fraction_static") == value


def test_load_state_dict_refuses_instead_of_restoring_placeholders():
    optimizer = swap_optim.AdamWSwapSafe.__new__(swap_optim.AdamWSwapSafe)

    with pytest.raises(RuntimeError, match="save_freq=-1"):
        optimizer.load_state_dict({})


def test_optimizer_is_resolvable_like_build_optimizer():
    """``build_optimizer`` resolves ``optimizer_impl`` + ``optimizer`` dynamically."""
    module = importlib.import_module("verl.utils.dsv41_swap_optim")
    optimizer_cls = getattr(module, "AdamWSwapSafe")

    from fsdp_turbo.memory.swap_optimizer import AdamWSwap

    assert issubclass(optimizer_cls, AdamWSwap)

    # build_optimizer passes lr/weight_decay (+ betas for names containing "adam").
    parameters = inspect.signature(optimizer_cls.__init__).parameters
    for name in ("lr", "weight_decay", "betas"):
        assert name in parameters


class _FakeShardedParam:
    """Stands in for an FSDP2 DTensor: the swapper only needs ``to_local``."""

    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def to_local(self):
        return torch.zeros(1)


def test_split_swap_params_keeps_only_trainable_sharded_params():
    sharded = _FakeShardedParam()
    frozen_sharded = _FakeShardedParam(requires_grad=False)
    frozen_plain = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    engram = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
    engram._engram_host_offload = True

    managed, skipped = swap_optim.split_swap_params(iter([sharded, frozen_sharded, frozen_plain, engram]))

    assert managed == [sharded]
    assert len(skipped) == 3
    assert skipped[0] is frozen_sharded
    assert skipped[1] is frozen_plain
    assert skipped[2] is engram


def test_split_swap_params_rejects_unknown_trainable_plain_param():
    plain = torch.nn.Parameter(torch.zeros(2), requires_grad=True)

    with pytest.raises(ValueError, match="neither FSDP-managed"):
        swap_optim.split_swap_params([plain])


def _swap_optimizer_impl():
    return importlib.import_module("fsdp_turbo.memory.swap_optimizer.swap_optimizer")


def test_alloc_host_state_like_pageable_when_disabled(monkeypatch):
    """``MY_ENV_DSV41_SWAP_HOST_PINNED=0`` must never touch the driver host-pin path."""
    impl = _swap_optimizer_impl()
    monkeypatch.setenv(impl._HOST_PIN_ENV, "0")
    monkeypatch.setattr(impl, "host_state_alloc_stats", {"pinned": 0, "pinned_failed": 0, "pageable": 0})

    src = torch.zeros(3, dtype=torch.bfloat16)
    host = impl._alloc_host_state_like(src)

    assert host.device.type == "cpu"
    assert host.dtype == src.dtype and host.shape == src.shape
    assert not host.is_pinned()
    assert impl.host_state_alloc_stats == {"pinned": 0, "pinned_failed": 0, "pageable": 1}


def test_alloc_host_state_like_latches_pageable_after_pinned_failure(monkeypatch):
    """207001 arrives after ~25 GiB pinned; later states must not keep retrying pinned."""
    impl = _swap_optimizer_impl()
    monkeypatch.delenv(impl._HOST_PIN_ENV, raising=False)
    monkeypatch.setattr(impl, "_pinned_disabled", False)
    monkeypatch.setattr(impl, "host_state_alloc_stats", {"pinned": 0, "pinned_failed": 0, "pageable": 0})

    calls = []
    real_empty_like = torch.empty_like

    def fake_empty_like(tensor, *args, **kwargs):
        calls.append(kwargs.get("pin_memory"))
        if kwargs.get("pin_memory"):
            raise RuntimeError("driver error: out of memory (207001)")
        return real_empty_like(tensor, *args, **kwargs)

    monkeypatch.setattr(impl.torch, "empty_like", fake_empty_like)

    src = torch.zeros(2)
    first = impl._alloc_host_state_like(src)
    second = impl._alloc_host_state_like(src)

    assert calls == [True, False, False], "pinned is tried once, then the latch skips it"
    for host in (first, second):
        assert host.device.type == "cpu" and host.shape == src.shape and not host.is_pinned()
    assert impl.host_state_alloc_stats == {"pinned": 0, "pinned_failed": 1, "pageable": 2}
