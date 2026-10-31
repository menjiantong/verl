"""MY_ENV_OP_DUMP: dump selected module inputs/outputs for DSV41 train-vs-infer.

Loaded from sitecustomize.py in every Ray/vLLM worker.  The wrapper is
class-based so it works for both FSDPTurbo (training) and vLLM-Ascend
(inference) without touching either model implementation.

Enable with:
  MY_ENV_OP_DUMP=1
  MY_ENV_OP_DUMP_DIR=/mnt/share/.../logs/op_dump_xxx
  PYTHONPATH=<this directory>:$PYTHONPATH

Only rank 0 of each process group writes dumps.  One .pt per module call plus
a JSONL index; large tensors are sliced to keep the dump bounded.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

import torch
import torch.nn as nn

DUMP_DIR = os.environ.get("MY_ENV_OP_DUMP_DIR", "")
MAX_FORWARD = int(os.environ.get("MY_ENV_OP_DUMP_MAX_FORWARD", "4"))
MAX_NUMEL = int(os.environ.get("MY_ENV_OP_DUMP_MAX_NUMEL", str(64 * 1024 * 1024)))
MAX_CALLS = int(os.environ.get("MY_ENV_OP_DUMP_MAX_CALLS", "600"))
MAX_FILES = int(os.environ.get("MY_ENV_OP_DUMP_MAX_FILES", "4000"))
MAX_SECONDS = float(os.environ.get("MY_ENV_OP_DUMP_MAX_SECONDS", "120"))
# 默认只落 rank0：训练 16 rank / 推理 8 worker 若全落盘，数据量会放大 8~16 倍。
# 需要逐 rank 对照时设 MY_ENV_OP_DUMP_ALL_RANKS=1。
DUMP_ALL_RANKS = os.environ.get("MY_ENV_OP_DUMP_ALL_RANKS", "0") == "1"
# 逐专家输出量大且对定位融合算子误差帮助有限，默认跳过；需要时设 =1。
DUMP_EXPERTS = os.environ.get("MY_ENV_OP_DUMP_EXPERTS", "0") == "1"
# 可选：只落这些类（embedding 仍会落，用于提取 input_ids），逗号分隔。
_ONLY_CLASSES = {
    x.strip() for x in os.environ.get("MY_ENV_OP_DUMP_CLASSES", "").split(",") if x.strip()
}

TRAIN_CLASSES = {
    "Transformer", "Block", "Attention", "Indexer", "Compressor", "MoE",
    "Expert", "Gate", "ParallelHead", "ParallelEmbedding", "RMSNorm",
}
INFER_CLASSES = {
    "DeepseekV41Model", "AscendDeepseekV41LLMForCausalLM",
    "DeepseekV41DecoderLayer", "DeepseekV41Attention", "DeepseekV41SWAAttention",
    "DeepseekV41Compressor", "Indexer", "DeepseekV41MoE", "DeepseekV41MLP",
    "FusedTopKRouter", "GroupedTopKRouter", "ParallelLMHead",
    "VocabParallelEmbedding", "RMSNorm", "ReplicatedLinear",
}
ALL_CLASSES = TRAIN_CLASSES | INFER_CLASSES
EMBED_CLASSES = {"ParallelEmbedding", "VocabParallelEmbedding"}
HEAD_CLASSES = {"ParallelHead", "ParallelLMHead"}

_LOCK = threading.RLock()
_ORIG_CALL = nn.Module.__call__
_TLS = threading.local()
_FORWARD = 0
_FILE_N = 0
_INDEX = None
_RANK = None
_PID = os.getpid()


def _rank() -> int:
    global _RANK
    if _RANK is not None:
        return _RANK
    # 不能在 install 时缓存 0：vLLM 的 dist 在模型第一次 forward 前才初始化。
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            _RANK = int(dist.get_rank())
            return _RANK
    except Exception:
        pass
    # 这里绝不能 import vllm：install() 发生在“首次 import torch”期间，此时
    # import vllm 可能只完成一半并留下 partially initialized 的 torch_npu，
    # 后续 verl import torch_npu 会报 _C 缺失。dump 时 dist 一定已初始化。
    # 不读 RANK/LOCAL_RANK：Ray 下这些变量不保证是全局 rank。
    # dist 不可用时返回 0，下一次 dump 会重新探测。
    return 0


def _matched_class(mod):
    for cls in type(mod).__mro__:
        if cls.__name__ in ALL_CLASSES:
            return cls.__name__, cls.__module__
    return None, None


def _layer_id(mod):
    for attr in ("layer_id", "layer_idx", "layer_number"):
        val = getattr(mod, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            try:
                return int(val.item())
            except Exception:
                pass
    return None


def _ctx_layer():
    stack = getattr(_TLS, "stack", None)
    return stack[-1] if stack else None


def _should_dump(cls, mod):
    if cls in EMBED_CLASSES:
        return True
    if cls == "Expert" and not DUMP_EXPERTS:
        return False
    if _ONLY_CLASSES and cls not in _ONLY_CLASSES:
        return False
    if cls in HEAD_CLASSES:
        return True
    if cls == "ReplicatedLinear":
        weight = getattr(mod, "weight", None)
        return weight is not None and weight.ndim == 2 and weight.shape[0] in (32, 64, 256, 384)
    return True


def _find_input_ids(args, kwargs):
    for val in list(kwargs.values()) + list(args):
        if isinstance(val, torch.Tensor) and val.dtype in (torch.int64, torch.int32) and val.ndim >= 1:
            return val
    return None


def _tensor_stats(t):
    tf = t.float()
    return {
        "mean": float(tf.mean().item()),
        "absmax": float(tf.abs().max().item()),
        "l2": float(tf.norm().item()),
    }


def _serialize(obj, depth=0):
    if depth > 3:
        return repr(obj)[:200]
    if isinstance(obj, torch.Tensor):
        t = obj
        if hasattr(t, "to_local"):
            try:
                t = t.to_local()
            except Exception:
                pass
        t = t.detach()
        meta = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if t.numel() <= MAX_NUMEL:
            return {"tensor": t.to("cpu"), **meta}
        try:
            meta["stats"] = _tensor_stats(t)
        except Exception:
            pass
        flat = t.reshape(-1)
        meta["head"] = flat[:4096].to("cpu")
        meta["tail"] = flat[-4096:].to("cpu")
        if t.ndim >= 2 and t.shape[-1] >= 1024:
            meta["last_row"] = t.reshape(-1, t.shape[-1])[-1].to("cpu")
        return {"tensor_slice": True, **meta}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v, depth + 1) for v in obj[:8]]
    if isinstance(obj, dict):
        return {str(k): _serialize(v, depth + 1) for k, v in list(obj.items())[:16]}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)[:200]


def _shape_tree(obj, depth=0):
    if depth > 2:
        return None
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape_tree(v, depth + 1) for v in obj[:8]]
    if isinstance(obj, dict):
        return {str(k): _shape_tree(v, depth + 1) for k, v in list(obj.items())[:8]}
    return None


def _write_index(rec):
    global _INDEX
    if _INDEX is None:
        os.makedirs(DUMP_DIR, exist_ok=True)
        _INDEX = open(os.path.join(DUMP_DIR, f"index_{_PID}.jsonl"), "a", encoding="utf-8")
    _INDEX.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _INDEX.flush()


def _dump(mod, cls, role, layer, args, kwargs, out):
    global _FILE_N
    if role not in ("train", "infer"):
        return
    rank = _rank()
    if not DUMP_ALL_RANKS and rank != 0:
        return
    tls = _TLS
    fwd = getattr(tls, "forward_id", 0)
    if not getattr(tls, "prefill", False) or fwd <= 0 or fwd > MAX_FORWARD:
        return
    if not _should_dump(cls, mod):
        return
    if time.time() > getattr(tls, "deadline", 0.0):
        tls.prefill = False
        return
    remaining = getattr(tls, "remaining", 0)
    if remaining <= 0:
        tls.prefill = False
        return
    tls.remaining = remaining - 1
    key = (fwd, cls, layer)
    counts = getattr(tls, "counts", None)
    if counts is None:
        counts = tls.counts = {}
    count = counts.get(key, 0)
    if count >= 32:
        return
    counts[key] = count + 1
    with _LOCK:
        if _FILE_N >= MAX_FILES:
            return
        _FILE_N += 1
        layer_tag = f"L{layer:02d}" if layer is not None else "Lna"
        rel = os.path.join(
            role, f"rank{rank}", f"pid{_PID}", f"fwd{fwd:02d}", layer_tag,
            f"{cls}_{count}.pt",
        )
        path = os.path.join(DUMP_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {
            "role": role, "rank": rank, "pid": _PID, "forward": fwd,
            "cls": cls, "layer": layer, "call": count, "path": rel,
            "module_module": type(mod).__module__,
            "module_class": type(mod).__name__,
            "args_shapes": [list(a.shape) if isinstance(a, torch.Tensor) else None for a in args[:8]],
            "out_shapes": _shape_tree(out),
        }
        try:
            torch.save(
                {
                    "meta": rec,
                    "args": _serialize(args),
                    "kwargs": _serialize(kwargs),
                    "output": _serialize(out),
                },
                path,
            )
            _write_index(rec)
        except Exception:
            print(f"[MY_DEBUG_LOG opdump] save failed {path}", file=sys.stderr, flush=True)
            traceback.print_exc()


def _wrapped_call(self, *args, **kwargs):
    global _FORWARD
    cls, modname = _matched_class(self)
    if cls is None:
        return _ORIG_CALL(self, *args, **kwargs)
    modname = modname or ""
    # 视觉塔是冻结的，且其 Block/Attention/RMSNorm 会挤占语言模型的 dump 配额。
    if ".vision" in modname:
        return _ORIG_CALL(self, *args, **kwargs)
    if modname.startswith("fsdp_turbo"):
        role = "train"
    elif modname.startswith(("vllm", "vllm_ascend")):
        role = "infer"
    else:
        return _ORIG_CALL(self, *args, **kwargs)
    layer = _layer_id(self)
    tls = _TLS
    pushed = False
    if layer is not None:
        stack = getattr(tls, "stack", None)
        if stack is None:
            stack = tls.stack = []
        stack.append(layer)
        pushed = True
    try:
        if cls in EMBED_CLASSES:
            ids = _find_input_ids(args, kwargs)
            if ids is not None:
                tls.prefill = int(ids.numel()) > 1
                if tls.prefill:
                    with _LOCK:
                        _FORWARD += 1
                    tls.forward_id = _FORWARD
                    tls.counts = {}
                    tls.remaining = MAX_CALLS
                    tls.deadline = time.time() + MAX_SECONDS
                else:
                    tls.remaining = 0
        out = _ORIG_CALL(self, *args, **kwargs)
        try:
            _dump(self, cls, role, layer if layer is not None else _ctx_layer(), args, kwargs, out)
        except Exception:
            print("[MY_DEBUG_LOG opdump] dump failed", file=sys.stderr, flush=True)
            traceback.print_exc()
        return out
    finally:
        if pushed:
            tls.stack.pop()


def install():
    if os.environ.get("MY_ENV_OP_DUMP") != "1":
        return
    if not DUMP_DIR:
        raise RuntimeError("MY_ENV_OP_DUMP_DIR is required when MY_ENV_OP_DUMP=1")
    os.makedirs(DUMP_DIR, exist_ok=True)
    nn.Module.__call__ = _wrapped_call
    # 不要在 install 时解析 rank：此刻 dist 往往还没初始化，且需避免任何
    # 额外 import 干扰 torch/torch_npu 的初始化顺序。
    print(
        f"[MY_DEBUG_LOG opdump] installed pid={_PID} dir={DUMP_DIR}",
        file=sys.stderr,
        flush=True,
    )
