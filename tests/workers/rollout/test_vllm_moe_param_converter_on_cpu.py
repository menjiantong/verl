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


import torch

from verl.workers.engine.fsdp.utils import split_fused_expert_tensor, unfuse_moe_params


def _collect(weights, model_type):
    return [item for item in unfuse_moe_params(weights, model_type)]


def test_qwen_moe_packed_weights_are_expanded_per_expert():
    gate_up = torch.randn(2, 6, 8)
    down = torch.randn(2, 8, 3)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_moe",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (3, 8),
        (3, 8),
        (3, 8),
        (3, 8),
        (8, 3),
        (8, 3),
    ]


def test_gpt_oss_packed_weights_are_not_expanded():
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 3, 8)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    converted = _collect(weights, "gpt_oss")

    assert [name for name, _ in converted] == [name for name, _ in weights]
    assert converted[0][1] is gate_up
    assert converted[1][1] is down
    assert converted[0][1].shape == (2, 8, 6)
    assert converted[1][1].shape == (2, 3, 8)


def test_expert_block_keeps_global_expert_numbering():
    """The DSV41 engine streams one EP block at a time; ids must stay global.

    vLLM maps the name's expert id onto the local expert slot, so a block that is not
    the first one has to emit the ids it actually carries.
    """
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 6, 3)

    converted = (
        split_fused_expert_tensor("model.layers.0.ffn.experts.gate_up_proj", gate_up, first_expert_id=48)
        + split_fused_expert_tensor("model.layers.0.ffn.experts.down_proj", down, first_expert_id=48)
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.ffn.experts.48.w1.weight",
        "model.layers.0.ffn.experts.48.w3.weight",
        "model.layers.0.ffn.experts.49.w1.weight",
        "model.layers.0.ffn.experts.49.w3.weight",
        "model.layers.0.ffn.experts.48.w2.weight",
        "model.layers.0.ffn.experts.49.w2.weight",
    ]
    torch.testing.assert_close(converted[0][1], gate_up[0, :4])
    torch.testing.assert_close(converted[1][1], gate_up[0, 4:])
    torch.testing.assert_close(converted[4][1], down[0])

    # Non-expert tensors are left to the caller (the engine streams them as-is).
    assert split_fused_expert_tensor("model.layers.0.self_attn.q_proj.weight", gate_up) is None
    assert split_fused_expert_tensor("model.layers.0.ffn.experts.gate_up_proj", gate_up[0]) is None
