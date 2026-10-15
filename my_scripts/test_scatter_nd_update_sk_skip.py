#!/usr/bin/env python3
"""Micro-repro of E9: what does ``npu_scatter_nd_update_sk`` do with [-1, -1] rows, on a plane
whose dim0 is a view stride?

E9 is the engine-side device fault that killed every GRPO rollout attempt on 10-08, and that the
10-08 repro harness (``scripts/run_vllm_dsv41_rl_dummy.sh``) showed is reachable with the *engine
alone* -- dummy weights, no trainer, no Ray: "MTE accesses an invalid GM address" / "vector core
exception" during the first real prefill, at the int8 index-K scatter
(``vllm_ascend/models/deepseek_v41/indexer.py:109`` -> ``dsa_v41.scatter_cache_sk`` ->
``torch.ops._C_ascend.npu_scatter_nd_update_sk``).

The bounds probe (``scripts/dsa_scatter_probe.py``) on those runs says every coordinate is inside
the plane, so this is not a plain out-of-range index. Two properties of the failing call were
never exercised together by the calls that succeeded:

  * some rows are ``[-1, -1]`` -- the metadata builder's "skip this row" marker, which
    ``dsa_v41.scatter_cache_sk``'s docstring claims the op honors;
  * the plane is a **view**: dim0 stride (131072 elements) is far larger than the contiguous
    expectation (rows_per_page*dim = 8192), which the tiling handles via ``isViewStride0_``.

Source reading of this fork's own op agrees with the "skip is not honored" side:
``csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_common.h`` computes
``linear = sum_i idx[i] * mask[i]`` over signed ints with no negative check, and the scatter loop
then does ``static_cast<uint64_t>(linearIndex)``; nothing in the aclnn wrapper, the tiling or any
of the three kernels mentions the marker. And ``vllm_ascend/device/device_op.py:46`` states the
contract outright: "this helper does not filter negative slots, and neither fast operator is
assumed to skip them".

Run the whole thing (one process per case) with:

    ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/test_scatter_nd_update_sk_skip.py --matrix

Measured on 10-08 (Ascend910, single card):

    layout  skip    result
    contig  no      OK
    contig  yes     exit 0 *only because the check looked at the plane*: the marker row is
                    written 65 rows * 128 B = 8320 bytes BELOW the plane base. ``--guard``
                    catches it; the write is misdirected, not skipped.
    view    no      OK
    view    yes     FAULT: vector core exception / MTE accesses an invalid GM address  (= E9)

So ``[-1, -1]`` is not a skip marker to this op at all; the view stride only decides whether the
same garbage offset stays inside the process's mapped range (quiet corruption) or leaves it (loud
device fault, every worker dies). That is exactly the bf16-vs-int8 asymmetry the probe saw: the
bf16 plane scatter (``skipped=0/5``, contiguous stride) survived, the int8 index-K scatter
(``skipped=3/5``, view stride) killed all eight workers.

``--fix`` exercises the two candidate repairs on the faulting shape:
    drop  filter marker rows out before the call. Correct, but the mask lives on the device, so
          ``indices[mask]`` costs a stream sync per call -- and there are 4 scatters per layer.
    sink  keep the row count and retarget marker rows at a reserved extra page. No sync; needs
          one more page per plane and a block pool that never hands that page out.

Exit codes: 0 = op completed, plane and guard both match expectation; 1 = the op raised (device
fault); 2 = the op "succeeded" but the plane or the guard does not match (silent corruption).
``--engine`` presets the shapes the probe captured on the real run.
"""

import argparse
import os
import sys
import time

# torch's device-backend autoload crashes on this container's torch_npu pairing; importing the
# extension explicitly works (and registers torch.ops._C_ascend).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

try:  # registers the torch.ops._C_ascend namespace
    # The vendor op libs (this op lives in vllm_ascend/_cann_ops_custom) are only found if the
    # custom-op env is bootstrapped before the op is first resolved -- without it the call raises
    # "aclnnScatterNdUpdateSk ... not in libopapi.so" instead of running. Same incantation as the
    # fork's own single-card op tests (tests/e2e/.../singlecard_ops/test_compressor_metadata.py).
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
except ImportError:  # pragma: no cover - depends on how the fork was built
    import vllm_ascend  # noqa: F401

SENTINEL = 3  # guard fill; the payload value is 7, the untouched plane is 0

DTYPE = {"int8": torch.int8, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", action="store_true", help="presets the shapes the probe captured")
    parser.add_argument("--dtype", choices=["int8", "bf16", "fp16"], default="int8")
    parser.add_argument("--layout", choices=["contig", "view"], default="view")
    parser.add_argument(
        "--skip",
        choices=["no", "yes"],
        default="yes",
        help="include [-1,-1] rows, as the DSA V4.1 metadata builder does",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=None,
        help="how many [-1,-1] rows (default: 3 with --engine, 1 otherwise)",
    )
    parser.add_argument("--pages", type=int, default=4096)
    parser.add_argument("--rows-per-page", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--page-stride", type=int, default=131072, help="elements; only for --layout view")
    parser.add_argument(
        "--op",
        choices=["sk", "builtin"],
        default="sk",
        help="sk = torch.ops._C_ascend.npu_scatter_nd_update_sk (the engine's fast path); "
        "builtin = torch_npu.npu_scatter_nd_update_ (device_op.py's generic fallback)",
    )
    parser.add_argument(
        "--guard",
        type=int,
        default=16384,
        help="elements of SENTINEL-filled padding before/after the plane's storage, to catch a "
        "misdirected write that lands outside the plane (0 disables)",
    )
    parser.add_argument("--fix", choices=["none", "drop", "sink"], default="none")
    args = parser.parse_args()
    if args.engine:
        # scripts/dsa_scatter_probe.py, 10-08 04:55 repro run, worker TP0:
        #   indexer.py:109 update_keys cache=shape=(33357, 64, 1, 128) strides=(131072, 128, 128, 1)
        #   dtype=torch.int8 values=(5, 128) rows[-1,1] cols[-1,1] skipped=3/5
        args.pages, args.rows_per_page, args.dim = 33357, 64, 128
        args.page_stride, args.dtype, args.layout = 131072, "int8", "view"
        args.skip, args.skip_rows = "yes", 3
    if args.skip_rows is None:
        args.skip_rows = 3 if args.engine else 1
    return args


def build_plane(args, dtype):
    """The engine's plane is a view of one big layer-outermost buffer: a page's payload is
    rows_per_page*dim elements but pages sit page_stride apart. Reproduce that, with SENTINEL
    padding on both sides so a write that misses the plane is still visible."""
    pages, rpp, dim = args.pages, args.rows_per_page, args.dim
    page_elems = args.page_stride if args.layout == "view" else rpp * dim
    total_pages = pages + (1 if args.fix == "sink" else 0)
    backing = torch.zeros(args.guard + total_pages * page_elems + args.guard, dtype=dtype, device="npu")
    backing[: args.guard] = SENTINEL
    backing[-args.guard :] = SENTINEL
    plane = backing.as_strided((total_pages, rpp, dim), (page_elems, dim, 1), storage_offset=args.guard)
    return backing, plane


def build_rows(args, sink_page):
    """Return (indices, values, rows_expected_to_be_written).

    Markers are placed *between* valid rows, not trailing -- that is what the probe saw on the
    real run (rows[-1,1]: the max valid page is 1 while 3 of 5 rows are markers)."""
    pages, rpp, dim = args.pages, args.rows_per_page, args.dim
    coords = [[0, 0], [-1, -1] if args.skip == "yes" else [1, 3], [pages - 1, rpp - 1], [2, 1]]
    for extra in range(1, args.skip_rows if args.skip == "yes" else 0):
        coords.insert(1, [-1, -1])
    expected = [c for c in coords if c[0] >= 0]
    indices = torch.tensor(coords, dtype=torch.int32, device="npu")
    values = torch.full((indices.shape[0], dim), 7, dtype=DTYPE[args.dtype], device="npu")
    if args.fix == "drop":
        keep = (indices >= 0).all(dim=-1)
        indices, values = indices[keep], values[keep]  # bool mask -> device sync
    elif args.fix == "sink":
        # The whole point of the sink variant: no host sync and no shape change, so the op stays
        # in the graph and the block pool only has to reserve one page per plane.
        marker = (indices < 0).any(dim=-1)
        sink = torch.full_like(indices[:, 0], sink_page)
        zero = torch.zeros_like(indices[:, 0])
        indices = indices.clone()
        indices[:, 0] = torch.where(marker, sink, indices[:, 0])
        indices[:, 1] = torch.where(marker, zero, indices[:, 1])
    return indices, values, expected


MATRIX = [
    ("contig", "no", "none"),
    ("contig", "yes", "none"),
    ("view", "no", "none"),
    ("view", "yes", "none"),
    ("view", "yes", "drop"),
    ("view", "yes", "sink"),
]
VERDICT = {0: "OK", 1: "FAULT", 2: "SILENT-CORRUPTION", 3: "timeout"}


def run_matrix():
    """One process per case: a device fault poisons the whole context, so cases cannot share one."""
    import subprocess

    extra = [a for a in sys.argv[1:] if a != "--matrix"]
    print(f"matrix on device {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<all>')}")
    for layout, skip, fix in MATRIX:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--layout",
            layout,
            "--skip",
            skip,
            "--fix",
            fix,
            *extra,
        ]
        cmd = [c for c in cmd if c != "--matrix"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            code, stream = out.returncode, out.stdout + out.stderr
        except subprocess.TimeoutExpired:
            code, stream = 3, ""
        lines = [ln for ln in stream.splitlines() if ln.startswith(("FAULT", "WRONG", "OK:"))]
        detail = (
            lines[0]
            if lines
            else (stream.strip().splitlines() or ["<no output>"])[-1]  # python crash, not the op
        )
        print(f"  layout={layout:6s} skip={skip:3s} fix={fix:4s} -> {VERDICT.get(code, code)}: {detail[:150]}", flush=True)
        time.sleep(5)  # let the device settle after a fault before the next context is created
    return 0


def main():
    if "--matrix" in sys.argv:
        return run_matrix()
    args = parse_args()
    if args.op == "builtin" and args.layout == "view":
        # device_op.py's fallback needs cache.view(-1, width) to be a *view*; on a strided plane it
        # copies, and the writes would land in the copy -- a misleading "OK". Skip the case.
        print("SKIP: builtin op is only meaningful with --layout contig", flush=True)
        return 0
    dtype = DTYPE[args.dtype]
    backing, plane = build_plane(args, dtype)
    indices, values, expected = build_rows(args, args.pages if args.fix == "sink" else None)

    print(
        f"case: op={args.op} dtype={args.dtype} layout={args.layout} skip={args.skip}"
        f"(n={args.skip_rows if args.skip == 'yes' else 0}) fix={args.fix} "
        f"plane={tuple(plane.shape)}/{tuple(plane.stride())} guard={args.guard} rows={indices.shape[0]}",
        flush=True,
    )
    target = plane[: args.pages]  # the sink page is excluded from the comparison
    try:
        if args.op == "sk":
            torch.ops._C_ascend.npu_scatter_nd_update_sk(plane, indices, values)
        else:  # generic fallback: linear row indices against a 2-D reshaped plane
            flat = plane.reshape(-1, args.dim)
            lin = (indices[:, 0].long() * args.rows_per_page + indices[:, 1].long()).view(-1, 1)
            torch_npu.npu_scatter_nd_update_(flat, lin, values)
        torch.npu.synchronize()
    except Exception as exc:
        print(f"FAULT: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return 1

    problems = []
    expected_plane = torch.zeros_like(target)
    for page, row in expected:
        expected_plane[page, row] = 7
    if not torch.equal(target, expected_plane):
        diff = (target != expected_plane).flatten().nonzero()
        first = int(diff[0, 0].item())
        problems.append(
            f"plane: {int(diff.shape[0])} element(s) differ, first at flat elem {first} "
            f"= {first // args.dim}th row of the plane"
        )
    # A write below/above the plane never shows up in the plane comparison, so check the padding.
    lo = backing[: args.guard]
    hi = backing[-args.guard :]
    for name, region, base in (("below", lo, -args.guard), ("above", hi, plane.numel())):
        dirty = (region != SENTINEL).nonzero()
        if dirty.numel():
            at = int(dirty[0, 0].item())
            problems.append(
                f"guard {name}: {int(dirty.shape[0])} byte(s) overwritten; first at "
                f"plane_base{at + base:+d} elements (value={region[at].item()})"
            )
    if problems:
        print("WRONG RESULTS: " + "; ".join(problems), flush=True)
        return 2
    print(f"OK: {len(expected)} rows written, guard intact", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
