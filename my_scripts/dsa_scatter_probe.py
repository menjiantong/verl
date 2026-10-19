#!/usr/bin/env python3
"""Diagnostic probe for E9: the bounds of every ``npu_scatter_nd_update_sk`` write.

Why: the 10-08 GRPO run (and the standalone repro of it) dies inside
``torch.ops._C_ascend.npu_scatter_nd_update_sk`` -- the device reports "MTE accesses an
invalid GM address" in the *index-K cache scatter* of the DSA V4.1 path
(``vllm_ascend/models/deepseek_v41/indexer.py:109`` ->
``vllm_ascend/attention/dsa_v41.py:222``). The fault is asynchronous, so the log names the
next op (RmsNorm) rather than the offender; with ASCEND_LAUNCH_BLOCKING=1 the traceback does
name ``scatter_cache_sk`` but not *which* coordinate was out of range or against *which*
buffer. This probe answers that without touching the vllm-ascend working copy: it wraps the
function, checks ``indices`` against ``cache``'s shape/strides on the host before the device
sees them, prints a one-line summary per call and raises a plain RuntimeError -- with numbers
-- the first time a write would land outside the tensor.

Enabled by ``DSA_SCATTER_PROBE=1`` and installed from the rollout worker process via
``ProbeWorkerExtension`` (used as vLLM's ``--worker-extension-cls``). The check is
deliberately not a torch op: ``.min()/.max()`` sync the stream, which is fine for a debug run
(and this run already sets ASCEND_LAUNCH_BLOCKING=1).
"""

import os
import sys
from types import ModuleType

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

_MAX_REPORTS = int(os.environ.get("DSA_SCATTER_PROBE_MAX", "400"))

_state = {"installed": False, "reports": 0, "calls": 0}


def _describe(tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} strides={tuple(tensor.stride())} dtype={tensor.dtype} "
        f"dev={tensor.device} ptr={tensor.data_ptr():#x} nbytes={tensor.numel() * tensor.element_size()}"
    )


def _extent_check(cache) -> tuple[str | None, str]:
    """Does the strided view stay inside its own storage?

    The DSA V4.1 planes are views of one layer-outermost buffer with a page stride larger than
    the plane's own rows, so ``numel()*element_size`` is *not* the span the op walks. The highest
    page must still land inside the backing storage; otherwise the op writes past mapped memory
    ("MTE accesses an invalid GM address") no matter how sane its coordinates look.
    """
    storage = cache.untyped_storage()
    nbytes = storage.nbytes()
    offset = cache.storage_offset()
    stride0 = cache.stride(0)
    pages = cache.shape[0]
    per_page = cache.numel() // pages if pages else 0
    last_row_end = offset + (pages - 1) * stride0 + per_page
    span_bytes = last_row_end * cache.element_size()
    summary = (
        f"storage={nbytes}B off={offset}elem stride0={stride0}elem pages={pages} "
        f"per_page={per_page}elem span_end={span_bytes}B"
    )
    if span_bytes > nbytes:
        return (
            f"plane view overshoots its storage by {span_bytes - nbytes}B "
            f"(span_end={span_bytes}B > storage={nbytes}B)",
            summary,
        )
    return None, summary


def _touch_rows(cache, indices) -> str | None:
    """Read the rows the scatter would write (plus the plane's last page).

    Returns None when every read succeeded, else the description of the first failing read.
    """
    rows = cache.squeeze(-2)
    targets = []
    if indices.dim() == 2 and indices.shape[-1] == 2:
        for i in range(min(int(indices.shape[0]), 6)):
            page, col = int(indices[i, 0].item()), int(indices[i, 1].item())
            if page >= 0 and col >= 0:
                targets.append((page, col))
        targets.append((rows.shape[0] - 1, rows.shape[1] - 1))  # the plane's last row
    else:
        return None
    for page, col in targets:
        try:
            _ = rows[page, col, 0].item()
        except Exception as exc:
            return f"read of ({page},{col}) failed: {type(exc).__name__}: {str(exc)[:200]}"
    return None


def _mapped_ranges(device) -> list[tuple[int, int]]:
    """(address, nbytes) of the blocks the caching allocator currently owns, best effort."""
    try:
        snapshot = torch.npu.memory.memory_snapshot()
    except Exception:
        return []
    out = []
    for blk in snapshot.get("segments", []) or []:
        for sub in blk.get("blocks", []) or []:
            if sub.get("size"):
                out.append((int(sub["address"]), int(sub["size"])))
    return out


def _marker_forecast(cache, indices) -> str:
    """Predict, per ``[-1, -1]`` row, the address the op will actually write.

    The op has no skip semantics -- measured 10-08 with ``scripts/test_scatter_nd_update_sk_skip.py``.
    It sums the signed coordinates against the tiling masks (``ComputeLinearIndexFromIndices``) and
    reinterprets the sum as ``uint64`` elements, so a marker row's destination is a fixed offset
    below the plane -- ``-(mask0+1)*scatterLength*itemsize`` when dim0 is contiguous, and a
    different (larger) number when dim0 is a view stride. Whether that address is mapped is what
    decides "device fault" (E9) vs "silent write into a neighbour", and that depends on where the
    allocator put the plane, i.e. on the run's config. This prints the arithmetic instead of the story.
    """
    rows = cache.squeeze(-2)
    rpp, dim = int(rows.shape[1]), int(rows.shape[-1])
    marker = -rpp - 1                       # linear index of a [-1,-1] row
    u = marker % (1 << 64)
    stride0 = int(cache.stride(0))
    if stride0 > rpp * dim:                 # tiling sets isViewStride0 = 1 here
        i0 = u // rpp
        off = (i0 * stride0 + (u - i0 * rpp) * dim) % (1 << 64)
        branch = "view"
    else:
        off = (u * dim) % (1 << 64)
        branch = "contig"
    delta_bytes = ((off - (1 << 64)) if off >= (1 << 63) else off) * cache.element_size()
    target = cache.data_ptr() + delta_bytes
    start = cache.data_ptr() - cache.storage_offset() * cache.element_size()
    end = start + cache.untyped_storage().nbytes()
    hits = [f"{base:#x}+{size}B" for base, size in _mapped_ranges(cache.device)
            if base <= target < base + size]
    return (
        f"marker_dest[{branch}]={delta_bytes:+d}B->{target:#x} "
        f"plane_storage=[{start:#x},{end:#x}) in_own_storage={start <= target < end} "
        f"in_mapped_block={hits[:2] if hits else 'NO'}"
    )


def _check(cache, slot_mapping, values):
    """Return ``(violation, summary)``; ``violation`` is None when the write is in range."""
    indices = slot_mapping[: values.shape[0]]
    if indices.numel() == 0:
        return None, "no rows"
    extent_violation, extent_summary = _extent_check(cache)
    if extent_violation is not None:
        return extent_violation, extent_summary
    if indices.dim() == 2 and indices.shape[-1] == 2:
        rows = cache.squeeze(-2)
        # Coordinate pairs: (page, offset within page). The builder uses [-1, -1] to mean "skip
        # this row", but the op has no such semantics (see _marker_forecast) -- these rows are
        # written somewhere, so they are counted, not excused.
        page = indices[:, 0]
        offset = indices[:, 1]
        skip = (page < 0) & (offset < 0)
        bad_page = (page >= rows.shape[0]) & ~skip
        bad_offset = (offset >= rows.shape[1]) & ~skip
        n_bad = int((bad_page | bad_offset).sum().item())
        n_skip = int(skip.sum().item())
        summary = (
            f"rows[{int(page.min().item())},{int(page.max().item())}] "
            f"cols[{int(offset.min().item())},{int(offset.max().item())}] "
            f"skipped={n_skip}/{indices.shape[0]}"
        )
        if n_skip:
            summary += " " + _marker_forecast(cache, indices)
        if n_bad == 0:
            return None, summary
        first = int((bad_page | bad_offset).nonzero()[0, 0].item())
        violation = (
            f"{n_bad} of {indices.shape[0]} rows out of range; first bad row {first}: "
            f"coord=({int(page[first].item())}, {int(offset[first].item())}) vs cache rows "
            f"{tuple(rows.shape)}"
        )
        return violation, summary
    flat = indices.reshape(-1)
    n = cache.numel() if cache.dim() == 1 else int(cache.shape[0])
    worst = int(flat.max().item())
    summary = f"linear max={worst} min={int(flat.min().item())} < rows={n}"
    if worst >= n:
        return f"{flat.numel()} linear indices, max={worst} >= {n} rows", summary
    return None, summary


def wrap(fn):
    def probed(cache, slot_mapping, values):
        _state["calls"] += 1
        if os.environ.get("DSA_SCATTER_PROBE") != "1":
            return fn(cache, slot_mapping, values)
        where = f"call#{_state['calls']} from {_caller()}"
        try:
            violation, summary = _check(cache, slot_mapping, values)
        except Exception as exc:  # never let the probe hide the real error
            print(f"[dsa-scatter-probe] {where}: probe itself failed: {exc!r}", flush=True)
            return fn(cache, slot_mapping, values)
        if violation is not None:
            print(
                f"[dsa-scatter-probe] VIOLATION {where}\n"
                f"  cache : {_describe(cache)}\n"
                f"  values: {_describe(values)}\n"
                f"  idx   : {_describe(slot_mapping)} rows_used={values.shape[0]}\n"
                f"  {violation}",
                flush=True,
            )
            # STRICT=1 (default) aborts. Set DSA_SCATTER_PROBE_STRICT=0 to keep going when the
            # point of the run is to observe the op's own behaviour (e.g. does the marker row
            # fault here?) instead of being stopped by the probe.
            if os.environ.get("DSA_SCATTER_PROBE_STRICT", "1") == "1":
                raise RuntimeError(f"dsa_scatter_probe: {violation} ({where})")
        if _state["reports"] < _MAX_REPORTS:
            _state["reports"] += 1
            print(
                f"[dsa-scatter-probe] {where} cache={_describe(cache)} "
                f"values={tuple(values.shape)} storage={_extent_check(cache)[1]} {summary}",
                flush=True,
            )
        # Touch the exact rows the op is about to write: a read fault here means the plane's
        # backing pages are not mapped for this process/context at all, which would explain an
        # "invalid GM address" that sane coordinates cannot.
        if os.environ.get("DSA_SCATTER_PROBE_TOUCH") == "1":
            touched = _touch_rows(cache, slot_mapping[: values.shape[0]])
            if touched is not None:
                print(f"[dsa-scatter-probe] TOUCH-FAIL {where} {touched}", flush=True)
                raise RuntimeError(f"dsa_scatter_probe touch: {touched} ({where})")
        return fn(cache, slot_mapping, values)

    probed.__wrapped__ = fn
    probed.__name__ = getattr(fn, "__name__", "scatter_cache_sk")
    return probed


def _caller() -> str:
    """Nearest vllm_ascend frame above the scatter, e.g. ``indexer.py:109 update_keys``."""
    import traceback

    for entry in reversed(traceback.extract_stack()):
        if "vllm_ascend" in entry.filename and entry.name not in {"scatter_cache_sk", "probed"}:
            return f"{os.path.basename(entry.filename)}:{entry.lineno} {entry.name}"
    return "?"


def install():
    """Replace ``scatter_cache_sk`` in dsa_v41 and in every module that already holds it."""
    if _state["installed"] or os.environ.get("DSA_SCATTER_PROBE") != "1":
        return
    import vllm_ascend.attention.dsa_v41 as dsa

    original = dsa.scatter_cache_sk
    wrapped = wrap(original)
    dsa.scatter_cache_sk = wrapped
    patched = ["vllm_ascend.attention.dsa_v41"]
    for name, module in list(sys.modules.items()):
        # Modules that did ``from ... import scatter_cache_sk`` hold the original function
        # object; swap those too. ``hasattr(torch.ops, ...)`` is True for anything (its
        # __getattr__ fabricates namespaces), so only real modules may be rewritten.
        if not isinstance(module, ModuleType) or module is dsa:
            continue
        if getattr(module, "scatter_cache_sk", None) is original:
            module.scatter_cache_sk = wrapped
            patched.append(name)
    _state["installed"] = True
    print(f"[dsa-scatter-probe] installed, patched: {patched}", flush=True)


class ProbeWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM ``worker_extension_cls``: install the probe, then behave as verl's extension.

    Subclassing (rather than delegating) is required: verl's ``__new__`` calls the zero-arg
    ``super().__new__(cls)``, which checks the instance against the class in its ``__class__``
    cell -- a standalone delegating class trips that ("obj must be an instance or subtype of
    type").
    """

    def __new__(cls, **kwargs):
        install()
        return super().__new__(cls, **kwargs)
