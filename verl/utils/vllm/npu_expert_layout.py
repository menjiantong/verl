# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Ascend weight sync: put the fused MoE expert weights back into checkpoint layout.

``AscendUnquantizedFusedMoEMethod.process_weights_after_loading`` transposes the fused
expert weights into the inference layout -- ``w13_weight`` from ``[E, 2 * intermediate,
hidden]`` to ``[E, hidden, 2 * intermediate]``, ``w2_weight`` from ``[E, hidden,
intermediate]`` to ``[E, intermediate, hidden]``. ``RoutedExperts.weight_loader`` and
the trainer's per-expert ``w1``/``w3``/``w2`` tensors are both in checkpoint
orientation, so loading a sync bucket into the transposed parameter narrows the wrong
dimension (half of ``hidden`` instead of ``intermediate``) and the copy fails with a
shape error on the first weight sync.

Ascend's other linears handle a reload themselves: their loaders accept the checkpoint
layout and re-apply whatever layout the kernel wants (``AscendColumnParallelLinear``
reshapes ``wo_a`` to ``[n_local_groups, ., o_lora_rank]`` in place), and the runtime
performs the TP-weight switch from the layer's own state, which is why only the MoE
parameters need to be reverted here. vLLM re-runs ``process_weights_after_loading``
after the sync (see ``update_weights_from_ipc`` step 3), which transposes the experts
back into the inference layout.
"""

import logging

logger = logging.getLogger(__file__)


def restore_expert_checkpoint_layout(models) -> int:
    """Undo the Ascend MoE expert transpose on every layer of ``models``.

    Returns the number of modules whose parameters were reverted. Modules whose expert
    weights are already in checkpoint layout (or that are not fused-MoE modules at all)
    are left untouched, so calling this repeatedly is safe.
    """
    reverted = 0
    for model in models:
        for module in model.modules():
            w13 = getattr(module, "w13_weight", None)
            w2 = getattr(module, "w2_weight", None)
            if w13 is None or w2 is None or w13.dim() != 3 or w2.dim() != 3:
                continue
            # Inference layout: w13 [E, hidden, 2i], w2 [E, i, hidden]; the checkpoint
            # layout pairs w13.shape[2] with w2.shape[1] instead.
            if w13.shape[1] == w2.shape[2] and w13.shape[2] == 2 * w2.shape[1]:
                # Views on purpose: one layer's w13 is 2.11 GiB and the rollout engine
                # has already filled its pool, so materialising a contiguous copy runs
                # out of memory. The weight loaders narrow and copy into strided
                # tensors, and process_weights_after_loading transposes the same
                # storage back (also copy-free), so no copy is needed anywhere.
                module.w13_weight.data = w13.data.transpose(1, 2)
                module.w2_weight.data = w2.data.transpose(1, 2)
                reverted += 1
    if reverted:
        logger.info("Ascend MoE: reverted %d fused expert weights to checkpoint layout", reverted)
    return reverted
