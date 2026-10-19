# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU tests for ``verl/utils/vllm/npu_expert_layout.py``.

``AscendUnquantizedFusedMoEMethod.process_weights_after_loading`` transposes the fused
expert weights into the inference layout the Ascend kernels consume, while
``RoutedExperts.weight_loader`` and the trainer's per-expert tensors work in checkpoint
layout. The helper puts the expert weights back into checkpoint layout before a weight
sync; ``process_weights_after_loading`` re-applies the transpose afterwards.
"""

import pytest
import torch
import torch.nn as nn

from verl.utils.vllm.npu_expert_layout import restore_expert_checkpoint_layout

NUM_EXPERTS = 2
INTERMEDIATE = 4
HIDDEN = 8


class _ToyExpertLayer(nn.Module):
    """Stands in for ``RoutedExperts``: fused w13/w2 with a transposing post-load."""

    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN))
        self.w2_weight = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE))

    def process_weights_after_loading(self):
        # What the Ascend MoE method does after loading.
        self.w13_weight.data = self.w13_weight.data.transpose(1, 2).contiguous()
        self.w2_weight.data = self.w2_weight.data.transpose(1, 2).contiguous()


class _ToyModel(nn.Module):
    def __init__(self, *, with_experts: bool = True):
        super().__init__()
        self.layers = nn.ModuleDict({"experts": _ToyExpertLayer()} if with_experts else {"dense": nn.Identity()})


def _restore(module):
    """Revert and return the transposed tensors, to hand them back later."""
    transposed_w13 = module.w13_weight.data.clone()
    transposed_w2 = module.w2_weight.data.clone()
    restore_expert_checkpoint_layout([module])
    return transposed_w13, transposed_w2


def test_reverts_inference_layout_to_checkpoint_layout():
    model = _ToyModel()
    experts = model.layers["experts"]
    experts.process_weights_after_loading()
    assert tuple(experts.w13_weight.shape) == (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE)

    transposed_w13, transposed_w2 = _restore(experts)

    assert tuple(experts.w13_weight.shape) == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert tuple(experts.w2_weight.shape) == (NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    # The reverted weights are the transposed ones, not zeros.
    torch.testing.assert_close(experts.w13_weight.data, transposed_w13.transpose(1, 2).contiguous())
    torch.testing.assert_close(experts.w2_weight.data, transposed_w2.transpose(1, 2).contiguous())


def test_revert_is_idempotent_and_skips_non_expert_modules():
    model = _ToyModel()
    experts = model.layers["experts"]

    # Already in checkpoint layout: nothing to do.
    assert restore_expert_checkpoint_layout([model]) == 0
    assert tuple(experts.w13_weight.shape) == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)

    experts.process_weights_after_loading()
    assert restore_expert_checkpoint_layout([model]) == 1
    assert restore_expert_checkpoint_layout([model]) == 0, "second pass must be a no-op"

    dense_only = _ToyModel(with_experts=False)
    assert restore_expert_checkpoint_layout([dense_only]) == 0


def test_post_loading_processing_restores_the_inference_layout():
    """End to end for one sync round: revert, load checkpoint layout, post-process."""
    model = _ToyModel()
    experts = model.layers["experts"]
    experts.process_weights_after_loading()

    _restore(experts)
    new_w13 = torch.arange(NUM_EXPERTS * 2 * INTERMEDIATE * HIDDEN, dtype=torch.float32).reshape(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN
    )
    new_w2 = torch.arange(NUM_EXPERTS * HIDDEN * INTERMEDIATE, dtype=torch.float32).reshape(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE
    )
    # RoutedExperts.weight_loader writes slices of these into the parameter.
    experts.w13_weight.data.copy_(new_w13)
    experts.w2_weight.data.copy_(new_w2)
    experts.process_weights_after_loading()

    assert tuple(experts.w13_weight.shape) == (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE)
    torch.testing.assert_close(experts.w13_weight.data, new_w13.transpose(1, 2).contiguous())
    torch.testing.assert_close(experts.w2_weight.data, new_w2.transpose(1, 2).contiguous())


def test_loader_copy_fails_on_the_inference_layout():
    """The failure the revert avoids: the loader's slice does not fit the transpose."""
    model = _ToyModel()
    experts = model.layers["experts"]
    experts.process_weights_after_loading()

    # RoutedExperts._load_w13 narrows the fused (dim 0) half and then copies the
    # [intermediate, hidden] checkpoint slice into it. On the inference layout the
    # narrowed view is [E, intermediate, hidden] while the slice is [E, hidden,
    # intermediate], which cannot broadcast -- the sync dies here without the revert.
    narrowed = experts.w13_weight.data[:, :INTERMEDIATE]
    with pytest.raises(RuntimeError):
        narrowed.copy_(torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE))

    restore_expert_checkpoint_layout([model])
    experts.w13_weight.data[:NUM_EXPERTS, :INTERMEDIATE].copy_(
        torch.zeros(NUM_EXPERTS, INTERMEDIATE, HIDDEN)
    )
