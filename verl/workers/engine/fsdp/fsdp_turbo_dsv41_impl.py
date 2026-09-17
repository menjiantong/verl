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
(HF ``from_pretrained``) cannot construct it. This engine overrides
``_build_module`` to build the V4.1 adapter through FSDP-Turbo's own
build/initialize/prepare pipeline and lets the parent ``_build_fsdp_module``
wrap it with FSDP-Turbo parallelism as usual.

Model structure is currently read from FSDP-Turbo's vendored ``config.json``
(``fsdp_turbo/models/deepseek_v41/config.json``, a 4-layer demo), not from
``actor_rollout_ref.model.path``. ``model.path`` still supplies the tokenizer.
"""

from __future__ import annotations

import torch

from ..base import EngineRegistry
from .fsdp_turbo_impl import FSDPTurboEngineWithLMHead


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

    def _build_dsv41_module(self):
        from fsdp_turbo.models.deepseek_v41 import (
            build_deepseek_v41_model,
            initialize_deepseek_v41_model,
            prepare_deepseek_v41_model_for_fsdp,
        )

        # Hard code the memory-optimized path for now: keep deferred Engram
        # tables on meta (sharded/initialized later by FSDP-Turbo after it has
        # the data-parallel mesh). Disable sparse flash attention (engram
        # remains disabled while we first bring up the base RL loop).
        training_model = build_deepseek_v41_model(
            tokenizer=self.model_config.tokenizer,
            engram_meta_init=True,
            engram_storage_backend="row_sharded",
            use_sparse_flash_attn=False,
        )
        initialize_deepseek_v41_model(training_model)
        training_model = prepare_deepseek_v41_model_for_fsdp(
            training_model,
            device=torch.accelerator.current_accelerator(),
            parameter_dtype=torch.bfloat16,
        )
        return training_model
