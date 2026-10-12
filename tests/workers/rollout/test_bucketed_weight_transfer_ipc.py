# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache Version 2.0 (the "License");
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
"""Oversized weights are published as device IPC handles, so a host tensor must be staged first.

Background (wiki/worklog_dsv41_top64_grpo.md §5.7.5): with ``offload_policy=True`` the generic
export path hands ``BucketedWeightSender._direct_send_large_weight`` parameters that are not
DTensors, and those still live on the host. ``reduce_tensor`` of a host tensor yields the CPU
reduction (``rebuild_tensor``, three arguments), and the receiver's ``rebuild_ipc`` patch of the
device index at position 6 then died with::

    IndexError: list assignment index out of range

in every rollout worker, which is the *real* reason the first DeepSeek-V4.1 top-N GRPO runs
failed during the initial weight sync (the 512 MB bucket leaves ``embed.weight`` and
``head.weight`` -- 1262.5 MiB each -- on the direct-send path).

Run with two visible accelerator devices:

    ASCEND_RT_VISIBLE_DEVICES=8,9 python -m pytest -q \
        tests/workers/rollout/test_bucketed_weight_transfer_ipc.py
"""

from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    _stage_on_device_for_ipc,
    rebuild_ipc,
)

device_count = torch.accelerator.device_count() if hasattr(torch, "accelerator") else 0
requires_two_devices = pytest.mark.skipif(
    device_count < 2, reason="needs two accelerator devices to exercise a cross-process IPC handle"
)

# A host tensor big enough that the sender would take the direct-send path.
_TENSOR_ELEMS = 50_000_000


def _child(conn, device_index: int) -> None:
    """Rebuild the handle the way the rollout worker does and report what came back."""
    handle = conn.recv()
    tensor = rebuild_ipc(handle, device_index)
    conn.send((str(tensor.device), float(tensor.to("cpu", non_blocking=False).float().sum())))
    conn.close()


def test_host_handle_is_rejected_not_misindex():
    """The failure mode must be a clear error, not the IndexError that hid the real cause."""
    from torch.multiprocessing.reductions import reduce_tensor

    func, args = reduce_tensor(torch.zeros(_TENSOR_ELEMS, dtype=torch.float32))
    assert len(args) < 7, "expected a host reduction to carry fewer arguments than a device handle"
    with pytest.raises(RuntimeError, match="not an accelerator IPC handle"):
        rebuild_ipc((func, args), 0)


@requires_two_devices
def test_staged_weight_survives_cross_process_ipc():
    torch.manual_seed(0)
    host = torch.randn(_TENSOR_ELEMS, dtype=torch.float32)

    staged = _stage_on_device_for_ipc(host, "test.embed.weight")
    assert staged.device.type != "cpu", "staging must move the tensor onto the accelerator"

    from torch.multiprocessing.reductions import reduce_tensor

    handle = reduce_tensor(staged)
    assert len(handle[1]) >= 7, "a device handle must carry a patchable device index"

    # The child has to pick the device up from its own environment; index 0 of the visible set
    # is the sender's card, which is the mapping rebuild_ipc is expected to honour.
    context = mp.get_context("spawn")
    parent, child_conn = context.Pipe()
    child = context.Process(target=_child, args=(child_conn, 0))
    child.start()
    try:
        parent.send(handle)
        device, total = parent.recv()
    finally:
        child.join(timeout=120)
        parent.close()
        child_conn.close()

    get_torch_device().synchronize()
    assert device.startswith(get_device_name())
    assert abs(total - float(staged.float().sum())) < 1.0, "tensor content changed across the IPC mapping"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
