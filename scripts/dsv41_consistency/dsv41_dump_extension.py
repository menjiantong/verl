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

"""vLLM worker extension: read-only introspection of the live V4.1 engine.

Mixed into the Ascend worker via ``--worker-extension-cls`` (vLLM calls the methods
through ``collective_rpc``), so it runs *inside* each tensor-parallel worker with direct
access to the loaded model. Two jobs:

* ``dsv41_param_stats``  - fingerprint the weights the engine actually holds, so the
  checkpoint/loader/sync path can be checked numerically instead of inferred from outputs.
* ``dsv41_hook_begin`` / ``dsv41_hook_end`` - record the activation of every milestone
  module (plus the eager attention / hyper-connection internals) for the *prefill* of
  whatever request runs in between, and write them to disk for the trainer-side dump to
  be compared against.

Only method names prefixed ``dsv41_`` are used: vLLM asserts that an extension class does
not collide with any existing worker attribute.
"""

from __future__ import annotations

import os
import time

import torch

# Modules whose output is worth keeping. Matched against the *qualified* module name inside
# `model_runner.model` (vLLM names: model.embed_tokens, model.layers.N.self_attn, ...).
_MILESTONE_RE = None


def _milestone_match(name: str) -> bool:
    global _MILESTONE_RE
    if _MILESTONE_RE is None:
        import re

        _MILESTONE_RE = re.compile(
            r"(^model$|embed_tokens$|^lm_head$|\.norm$|logits_processor"
            r"|\.layers\.\d+$"
            r"|\.layers\.\d+\.(self_attn|mlp|input_layernorm|post_attention_layernorm"
            r"|compressor|indexer|gate|experts|shared_experts|attn|ffn))$"
        )
    return bool(_MILESTONE_RE.search(name))


# Modules worth hooking by *class* rather than by name: the MoE router is where expert
# selection happens, and the sparse indexer/compressor decide which compressed positions a
# query can see -- the two places where the two stacks are most likely to make different
# discrete choices.
_CLASS_PATTERNS = ("Router", "MoERunner", "Experts", "Indexer", "Compressor", "MoE")


class Dsv41DumpExtension:
    # ------------------------------------------------------------------ params
    def dsv41_param_stats(self, patterns=None, max_items=4000, first_values=4):
        """Return name/shape/dtype/summary for the parameters matching `patterns`."""
        model = self.model_runner.model
        out = []
        for name, param in model.named_parameters():
            if patterns and not any(p in name for p in patterns):
                continue
            tensor = param.detach()
            flat = tensor.reshape(-1)
            entry = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
            }
            if tensor.dtype.is_floating_point:
                sample = flat[:first_values].float().cpu().tolist()
                entry.update(
                    {
                        "mean": float(flat.float().mean().item()),
                        "std": float(flat.float().std().item()) if flat.numel() > 1 else 0.0,
                        "absmax": float(flat.float().abs().max().item()),
                        "first_values": [round(v, 6) for v in sample],
                    }
                )
            out.append(entry)
            if len(out) >= max_items:
                break
        return out

    def dsv41_local_expert_placement(self):
        """Which expert ids this rank owns (EP sharding) - for sync-layout questions."""
        info = {}
        model = self.model_runner.model
        for name, module in model.named_modules():
            if "experts" in name and hasattr(module, "local_expert_ids"):
                info[name] = list(module.local_expert_ids)
        return info

    # ------------------------------------------------------------------ hooks
    def _dsv41_store(self):
        store = getattr(self, "_dsv41_captures", None)
        if store is None:
            store = self._dsv41_captures = {}
        return store

    def dsv41_hook_begin(self, milestone_only=True, max_calls=2):
        """Start recording activations; the next forward(s) fill the buffer."""
        model = self.model_runner.model
        store = self._dsv41_store()
        store.clear()
        store["__meta__"] = {"rank": int(getattr(self, "rank", -1)), "milestone_only": milestone_only}
        handles = []
        max_elements = int(os.environ.get("DSV41_DUMP_MAX_ELEMENTS", 40_000_000))

        def record(key, value, call_index):
            if not isinstance(value, torch.Tensor) or value.numel() > max_elements:
                return
            bucket = store.setdefault(key, [])
            if len(bucket) > call_index:
                return
            try:
                bucket.append(value.detach().float().cpu())
            except Exception as exc:  # noqa: BLE001 - never break the engine
                store.setdefault("__errors__", []).append(f"{key}: {exc}")

        for name, module in model.named_modules():
            if not name:
                continue
            by_class = any(pattern in type(module).__name__ for pattern in _CLASS_PATTERNS)
            if milestone_only and not by_class and not _milestone_match(name):
                continue

            def make_hook(mod_name):
                def hook(_module, _inputs, output):
                    values = output if isinstance(output, (tuple, list)) else (output,)
                    calls = len(store.get(mod_name, []))
                    if calls >= max_calls:
                        return
                    for index, value in enumerate(values):
                        record(f"{mod_name}[{index}]" if len(values) > 1 else mod_name, value, calls)

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

        # The eager attention impl and the hyper-connection helpers are plain Python, so
        # their exact inputs/outputs can be captured by wrapping the methods.
        handles.extend(self._dsv41_wrap_methods(store, record, max_calls))
        handles.extend(self._dsv41_wrap_functions(store, record, max_calls))
        self._dsv41_handles = handles
        return {"hooked_modules": len(handles)}

    def _dsv41_wrap_functions(self, store, record, max_calls):
        """Patch module-level functions that the MoE router goes through.

        The routed-expert selection happens inside the fused MoE, not in a hookable
        submodule, so the router kernel itself is where the top-k expert ids can be read.
        """
        import functools

        targets = []
        try:
            from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router as bias_router

            targets.append(
                (
                    bias_router,
                    ["vllm_topk_softplus_sqrt", "_topk_softplus_sqrt_torch"],
                )
            )
        except Exception:  # noqa: BLE001
            pass

        handles = []
        for module, function_names in targets:
            for function_name in function_names:
                original = getattr(module, function_name, None)
                if original is None or getattr(original, "_dsv41_wrapped", False):
                    continue
                key = f"{module.__name__}.{function_name}"

                @functools.wraps(original)
                def wrapper(*args, __original=original, __key=key, **kwargs):
                    result = __original(*args, **kwargs)
                    calls = len(store.get(__key, []))
                    if calls < max_calls:
                        for index, item in enumerate(result if isinstance(result, (tuple, list)) else (result,)):
                            record(f"{__key}[{index}]", item, calls)
                    return result

                wrapper._dsv41_wrapped = True
                setattr(module, function_name, wrapper)
                handles.append((module, function_name, original))
        return handles

    def _dsv41_wrap_methods(self, store, record, max_calls):
        """Wrap the interesting engine-internal methods with recording proxies."""
        import functools

        targets = []
        try:
            from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl

            targets.append(
                (
                    DeepseekV41EagerAttentionImpl,
                    ["preprocess", "multistream_preprocess", "_select_sparse_indices", "_attention", "forward"],
                )
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer

            targets.append((DeepseekV41DecoderLayer, ["hc_pre", "hc_post", "rms_norm_cast"]))
        except Exception:  # noqa: BLE001
            pass
        # The Ascend MoE has its own routers (`vllm_topk_softplus_sqrt` is only the vLLM
        # default path); `_compute_routing` returns (topk_weights, topk_ids) and takes the
        # raw router logits, which is exactly the discrete decision to compare.
        try:
            from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter

            targets.append((AscendFusedTopKRouter, ["_compute_routing"]))
        except Exception:  # noqa: BLE001
            pass
        try:
            from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter

            targets.append((AscendGroupedTopKRouter, ["_compute_routing"]))
        except Exception:  # noqa: BLE001
            pass
        # `select_experts` is the template-method entry every router goes through (the
        # Ascend classes only override `_compute_routing`), so it captures the routing for
        # every layer even when the fused `npu_moe_gating_top_k` path is taken.
        try:
            from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
            from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

            targets.append((FusedMoERouter, ["select_experts"]))
            targets.append((BaseRouter, ["select_experts"]))
        except Exception:  # noqa: BLE001
            pass

        handles = []
        for cls, method_names in targets:
            for method_name in method_names:
                original = getattr(cls, method_name, None)
                if original is None or getattr(original, "_dsv41_wrapped", False):
                    continue
                key_prefix = f"{cls.__name__}.{method_name}"

                @functools.wraps(original)
                def wrapper(self_or_cls, *args, __original=original, __key=key_prefix, **kwargs):
                    result = __original(self_or_cls, *args, **kwargs)
                    # Include the layer identity so each layer's captures stay separate:
                    # the attention impl carries `prefix`, the decoder layer only `layer_idx`.
                    prefix = getattr(self_or_cls, "prefix", None)
                    if not isinstance(prefix, str) or not prefix:
                        layer_idx = getattr(self_or_cls, "layer_idx", None)
                        prefix = f"layer{layer_idx}" if layer_idx is not None else ""
                    key = f"{__key}@{prefix}" if prefix else __key
                    calls = len(store.get(key, []))
                    if calls < max_calls:
                        if isinstance(result, (tuple, list)):
                            for index, item in enumerate(result):
                                record(f"{key}[{index}]", item, calls)
                        else:
                            record(key, result, calls)
                        # Also keep the floating-point tensor arguments: for the routers this
                        # is the raw gate score, which is what actually decides the top-k.
                        for index, argument in enumerate(args):
                            if isinstance(argument, torch.Tensor) and argument.is_floating_point():
                                record(f"{key}#arg{index}", argument, calls)
                    return result

                wrapper._dsv41_wrapped = True
                setattr(cls, method_name, wrapper)
                handles.append((cls, method_name, original))
        return handles

    def dsv41_hook_end(self, out_dir, tag, include_prefixes=None):
        """Stop recording and write this rank's captures to `out_dir`."""
        for handle in getattr(self, "_dsv41_handles", []):
            if isinstance(handle, tuple):  # method wrapper
                cls, method_name, original = handle
                setattr(cls, method_name, original)
            else:
                handle.remove()
        self._dsv41_handles = []

        store = self._dsv41_store()
        keep = {}
        for key, value in store.items():
            if key.startswith("__"):
                keep[key] = value
                continue
            if include_prefixes and not any(key.startswith(p) for p in include_prefixes):
                continue
            if isinstance(value, list) and value:
                keep[key] = value
        os.makedirs(out_dir, exist_ok=True)
        rank = int(getattr(self, "rank", -1))
        path = os.path.join(out_dir, f"engine_stages_{tag}_rank{rank}.pt")
        torch.save(keep, path)
        total = sum(t.numel() for tensors in keep.values() if isinstance(tensors, list) for t in tensors)
        return {"path": path, "tensors": sum(len(v) for v in keep.values() if isinstance(v, list)),
                "elements": int(total), "time": time.time()}

    def dsv41_hook_names(self):
        store = self._dsv41_store()
        return sorted(k for k in store)
