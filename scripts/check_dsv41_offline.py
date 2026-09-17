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

"""Offline self-checks for the DeepSeek-V4.1 FSDPTurbo integration (no NPU needed).

Covers everything that can be verified without accelerators:

1. engine registration (``fsdp_turbo_dsv41``),
2. HF config loading for the 8-layer checkpoint (vLLM registry fallback),
3. external-config -> training ``ModelArgs`` translation,
4. checkpoint name coverage and expert fusion (``w1|w3 -> gate_up_proj``, ``w2 -> down_proj``),
5. the build -> initialize -> prepare pipeline with the routed experts deferred to meta.

Usage:
    python3 scripts/check_dsv41_offline.py [--model-path <ckpt dir>]

The full 8-layer checkpoint is 209GB; the reader check only touches a couple of layers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser.parse_args()


class Checker:
    def __init__(self):
        self.failures = []

    def check(self, name, fn):
        try:
            detail = fn()
        except Exception as error:  # noqa: BLE001 - report and keep going
            self.failures.append((name, error))
            print(f"[FAIL] {name}: {type(error).__name__}: {error}", flush=True)
        else:
            print(f"[ OK ] {name}{f' — {detail}' if detail else ''}", flush=True)


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    checker = Checker()

    def engine_registration():
        from verl.workers.engine import EngineRegistry

        cls = EngineRegistry.get_engine_cls("language_model", "fsdp_turbo_dsv41")
        assert cls.__name__ == "FSDPTurboDSV41EngineWithLMHead", cls
        return cls.__name__

    def hf_config():
        from verl.workers.config.model import HFModelConfig

        config = HFModelConfig(path=str(model_path), trust_remote_code=True)
        assert config.architectures == ["DeepseekV41ForCausalLM"], config.architectures
        return f"{type(config.hf_config).__name__}, tokenizer vocab={config.tokenizer.vocab_size}"

    def model_args():
        from fsdp_turbo.models.deepseek_v41 import build_deepseek_v41_model_args

        args_ = build_deepseek_v41_model_args(str(model_path), max_seq_len=args.max_seq_len)
        assert args_.n_layers == 8 and args_.n_routed_experts == 384, (args_.n_layers, args_.n_routed_experts)
        assert not args_.engram_layer_ids and args_.dspark_block_size == 0
        return (
            f"n_layers={args_.n_layers}, experts={args_.n_routed_experts}, "
            f"engram={args_.engram_layer_ids}, dspark={args_.dspark_block_size}"
        )

    def name_coverage_and_fusion():
        from safetensors import safe_open

        import verl.workers.engine.fsdp.fsdp_turbo_dsv41_impl as impl
        from fsdp_turbo.models.deepseek_v41 import build_deepseek_v41_model

        with torch.device("meta"):
            model = build_deepseek_v41_model(config_path=str(model_path), max_seq_len=512)
        param_meta = {name: (tuple(p.shape), p.dtype) for name, p in model.named_parameters()}

        weight_map = impl._checkpoint_weight_map(model_path)
        mapped, skipped = set(), set()
        for name in weight_map:
            if impl._EXPERT_WEIGHT_PATTERN.match(name):
                continue
            candidate = impl._MODEL_PREFIX + name
            (mapped if candidate in param_meta else skipped).add(candidate)
        missing = [
            n for n in param_meta if n not in mapped and not n.endswith(("gate_up_proj", "down_proj"))
        ]
        assert not missing, f"unmapped model params: {missing[:5]}"

        # Fusion is verified against the file contents on a single layer.
        subset = {k: v for k, v in weight_map.items() if k.startswith(("layers.0.", "embed.", "head.", "norm."))}
        original = impl._checkpoint_weight_map
        impl._checkpoint_weight_map = lambda _path: subset
        try:
            subset_meta = {
                n: param_meta[n]
                for n in param_meta
                if n.startswith(("model.embed", "model.head", "model.norm", "model.layers.0."))
            }
            state, report = impl.read_dsv41_checkpoint_state_dict(model_path, subset_meta)
        finally:
            impl._checkpoint_weight_map = original

        with safe_open(str(model_path / subset["layers.0.ffn.experts.383.w1.weight"]), framework="pt", device="cpu") as f:
            w1 = f.get_tensor("layers.0.ffn.experts.383.w1.weight")
        with safe_open(str(model_path / subset["layers.0.ffn.experts.383.w3.weight"]), framework="pt", device="cpu") as f:
            w3 = f.get_tensor("layers.0.ffn.experts.383.w3.weight")
        gate_up = state["model.layers.0.ffn.experts.gate_up_proj"]
        assert torch.equal(gate_up[383, : w1.shape[0]], w1), "w1 half mismatch"
        assert torch.equal(gate_up[383, w1.shape[0] :], w3), "w3 half mismatch"
        with safe_open(str(model_path / subset["layers.0.ffn.experts.383.w2.weight"]), framework="pt", device="cpu") as f:
            w2 = f.get_tensor("layers.0.ffn.experts.383.w2.weight")
        assert torch.equal(state["model.layers.0.ffn.experts.down_proj"][383], w2), "w2 mismatch"
        with safe_open(str(model_path / subset["embed.weight"]), framework="pt", device="cpu") as f:
            assert torch.equal(state["model.embed.weight"], f.get_tensor("embed.weight")), "embed mismatch"
        return (
            f"{len(param_meta)} model params: {len(mapped)} direct + expert-fused, "
            f"{len(skipped)} checkpoint tensors unused (vision); fusion byte-exact"
        )

    def build_prepare():
        from fsdp_turbo.models.deepseek_v41 import (
            build_deepseek_v41_model,
            initialize_deepseek_v41_model,
            prepare_deepseek_v41_model_for_fsdp,
        )

        model = build_deepseek_v41_model(
            config_path=str(model_path), experts_meta_init=True, max_seq_len=1024
        )
        initialize_deepseek_v41_model(model)
        model = prepare_deepseek_v41_model_for_fsdp(
            model, device=torch.device("cpu"), parameter_dtype=torch.bfloat16
        )
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        expected_meta = [n for n, _ in model.named_parameters() if n.endswith(("gate_up_proj", "down_proj"))]
        assert sorted(meta_params) == sorted(expected_meta), "unexpected meta parameters"
        assert not any(b.is_meta for b in model.buffers()), "buffers must stay materialized"
        assert not any(p.dtype != torch.bfloat16 for p in model.parameters() if not p.is_meta)
        return f"{len(meta_params)} experts deferred to meta, {len(list(model.buffers()))} buffers real"

    checker.check("engine registration", engine_registration)
    checker.check("hf config (vLLM registry fallback)", hf_config)
    checker.check("external config -> ModelArgs", model_args)
    checker.check("checkpoint names + expert fusion", name_coverage_and_fusion)
    checker.check("build -> prepare (meta experts)", build_prepare)

    if checker.failures:
        print(f"\n{len(checker.failures)} check(s) failed", flush=True)
        return 1
    print("\nall offline checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
