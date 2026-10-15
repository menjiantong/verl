#!/usr/bin/env python3
"""Export a vLLM-loadable HF checkpoint from FSDPTurbo SFT shards (offline, CPU).

Why offline assembly is exact
-----------------------------
The SFT save (verl/utils/checkpoint/fsdp_checkpoint_manager.py) writes a SHARDED
state dict: one `model_world_size_32_rank_*.pt` per rank, each entry either a
DTensor or a plain tensor.  A DTensor carries its DeviceMesh + placements, and
torch's own `compute_local_shape_and_global_offset` turns those into the exact
(local_shape, global_offset) the training loader used -- the very same helper
`_plan_direct_shard_load` (fsdp_turbo_dsv41_impl.py:243) loads with.  So the 32
files can be stitched back with zero distributed machinery: for every rank r and
name, place `to_local()` at its global offset.

Name mapping (verified against the original checkpoint index)
-------------------------------------------------------------
* training params carry a ``model.`` prefix; checkpoint names do not.
* fused experts ``...ffn.experts.gate_up_proj [ne, 2*i, h]`` split into the
  checkpoint's per-expert ``...ffn.experts.{E}.w1.weight`` (gate half) and
  ``...ffn.experts.{E}.w3.weight`` (up half); ``...down_proj`` -> ``w2.weight``.
* Engram tables (``layers.{1,14}.engram.{embed,wkv,q_weight,k_weight}``, 8 tensors)
  are frozen: copied from the original checkpoint files, never from the shards.
  Any other checkpoint tensor the export does not produce is kept from the
  original files as well ("leftover" set, reported).

Output layout (OUT_DIR)
-----------------------
* new safetensors shards with every exported tensor, named exactly like the
  original checkpoint;
* the original files that carry Engram/leftover tensors: symlinked when pure
  (no exported tensor inside), otherwise their needed tensors are copied out;
* config.json / tokenizer / other non-safetensors files: symlinked from the
  original directory;
* model.safetensors.index.json rebuilt over the two sources.

Usage
-----
    python3 scripts/export_sft_hf.py inspect --ckpt-dir <global_step_N dir>
    python3 scripts/export_sft_hf.py dry-map --ckpt-dir <...> --orig-dir <...>
    python3 scripts/export_sft_hf.py export  --ckpt-dir <...> --orig-dir <...> --out-dir <...>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_ORIG_DIR_DEFAULT = "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16"

# Fused expert params (training side).  See fsdp_turbo_dsv41_impl.py:74.
_FUSED_UP = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.gate_up_proj$")
_FUSED_DOWN = re.compile(r"^(?:model\.)?layers\.(\d+)\.ffn\.experts\.down_proj$")


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def _strip_model_prefix(name: str) -> str:
    return name[len("model."):] if name.startswith("model.") else name


def load_orig_index(orig_dir: str) -> dict:
    with open(os.path.join(orig_dir, "model.safetensors.index.json")) as fh:
        return json.load(fh)


def orig_header(orig_dir: str, filename: str) -> dict:
    with safe_open(os.path.join(orig_dir, filename), framework="pt") as fh:
        return {"__metadata__": fh.metadata(), **{k: None for k in fh.keys()}}


def orig_tensor_meta(orig_dir: str, index: dict) -> dict:
    """name -> (torch dtype, shape) for every tensor in the original checkpoint."""
    sf_to_torch = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    by_file: dict[str, list[str]] = {}
    for name, fn in index["weight_map"].items():
        by_file.setdefault(fn, []).append(name)
    meta: dict[str, tuple] = {}
    for fn, names in by_file.items():
        with safe_open(os.path.join(orig_dir, fn), framework="pt") as fh:
            for name in names:
                sl = fh.get_slice(name)
                meta[name] = (sf_to_torch[sl.get_dtype()], tuple(sl.get_shape()))
    return meta


def is_engram_name(name: str) -> bool:
    return ".engram." in name


def _load_pt(path: str):
    """Load a shard, mmap-first: the engram storages we skip are ~46 GiB/rank, and
    mmap keeps them (and every untouched tensor) off the page cache read path."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:  # pragma: no cover - format dependent
        log(f"mmap load failed for {os.path.basename(path)} ({exc!r}); falling back to full load")
        return torch.load(path, map_location="cpu", weights_only=False)


def map_param_to_orig(name: str, orig_names: set[str], expert_ids: int | None):
    """Map one training param name to [(orig_name, part)] entries.

    part is None for plain tensors, "gate"/"up" for the two halves of a fused
    gate_up_proj.  Returns None when the name has no checkpoint counterpart.
    """
    m_up = _FUSED_UP.match(name)
    if m_up:
        base = f"layers.{m_up.group(1)}.ffn.experts"
        out = []
        n_experts = expert_ids
        for e in range(n_experts):
            out.append((f"{base}.{e}.w1.weight", ("gate", e)))
            out.append((f"{base}.{e}.w3.weight", ("up", e)))
        return out
    m_down = _FUSED_DOWN.match(name)
    if m_down:
        base = f"layers.{m_down.group(1)}.ffn.experts"
        return [(f"{base}.{e}.w2.weight", ("down", e)) for e in range(expert_ids)]
    plain = _strip_model_prefix(name)
    if plain in orig_names:
        return [(plain, None)]
    return None


# --------------------------------------------------------------------------
# shard loading
# --------------------------------------------------------------------------

def _iter_dtensor_placements(v):
    from torch.distributed.tensor import DTensor

    if isinstance(v, DTensor):
        return v
    return None


def inspect_shard(ckpt_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")))
    log(f"{len(files)} shard file(s) in {ckpt_dir}")
    f0 = files[0]
    log(f"loading {os.path.basename(f0)} (map_location=cpu, weights_only=False) ...")
    sd = _load_pt(f0)
    log(f"entries: {len(sd)}")
    from torch.distributed.tensor import DTensor

    n_dt = sum(1 for v in sd.values() if isinstance(v, DTensor))
    log(f"DTensor entries: {n_dt} / plain: {len(sd) - n_dt}")
    shown = 0
    for name, v in sd.items():
        if shown >= 12:
            break
        if isinstance(v, DTensor):
            loc = v.to_local()
            log(
                f"  DT {name} global={tuple(v.shape)} loc={tuple(loc.shape)} dt={v.dtype} "
                f"placements={v.placements} local_device={loc.device}"
            )
        else:
            log(f"  plain {name} shape={tuple(v.shape)} dt={v.dtype}")
        shown += 1
    engram = [n for n in sd if ".engram." in n]
    log(f"engram-named entries in state_dict: {len(engram)} e.g. {engram[:4]}")
    fused = [n for n in sd if _FUSED_UP.match(n) or _FUSED_DOWN.match(n)]
    log(f"fused-expert entries: {len(fused)} e.g. {fused[:3]}")


def build_name_map(sd_names: list[str], orig_names: set[str], expert_ids: int | None):
    """training name -> list[(orig_name, part)]; returns (mapping, unmapped)."""
    mapping: dict[str, list] = {}
    unmapped: list[str] = []
    for name in sd_names:
        if is_engram_name(name):
            continue
        m = map_param_to_orig(name, orig_names, expert_ids)
        if m is None:
            unmapped.append(name)
        else:
            mapping[name] = m
    return mapping, unmapped


def assemble(ckpt_dir: str, index: dict, expert_ids: int, orig_meta: dict):
    """Stitch all shards into full CPU tensors. Returns dict[name] = tensor."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")))
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    buffers: dict[str, torch.Tensor] = {}
    orig_names = set(index["weight_map"])
    for fi, f in enumerate(files):
        t0 = time.time()
        sd = _load_pt(f)
        rank = re.search(r"_rank_(\d+)\.pt$", f).group(1)
        new_names = 0
        for name, v in sd.items():
            if is_engram_name(name):
                continue
            if name not in buffers:
                # allocate using the ORIGINAL dtype/shape of the first mapped target
                entries = keys_cache.get(name)
                if entries is None:
                    entries = map_param_to_orig(name, orig_names, expert_ids)
                    keys_cache[name] = entries
                if entries is None:
                    continue  # unmapped -> reported separately
                oname0 = entries[0][0]
                dt, shp = orig_meta[oname0]
                if _FUSED_UP.match(name):
                    # fused [ne, 2i, h]; checkpoint w1/w3 are [i, h] each
                    gshape = (expert_ids, 2 * shp[0], shp[1])
                elif _FUSED_DOWN.match(name):
                    gshape = (expert_ids, shp[0], shp[1])
                else:
                    gshape = shp
                buffers[name] = torch.empty(gshape, dtype=dt)
                new_names += 1
            buf = buffers[name]
            if isinstance(v, DTensor):
                lshape, off = compute_local_shape_and_global_offset(
                    tuple(v.shape), v.device_mesh, v.placements
                )
                local = v.to_local()
                if tuple(local.shape) != tuple(lshape):
                    lshape = tuple(local.shape)
            else:
                lshape, off = tuple(v.shape), (0,) * v.dim()
                local = v
            if tuple(buf.shape) != tuple(v.shape):
                raise RuntimeError(
                    f"{name}: buffer {tuple(buf.shape)} != DTensor global {tuple(v.shape)}"
                )
            sl = tuple(slice(o, o + s) for o, s in zip(off, lshape))
            buf[sl] = local.to(buf.dtype)
        log(
            f"shard {fi + 1}/{len(files)} (rank {rank}): {len(sd)} entries, "
            f"+{new_names} new buffers, {time.time() - t0:.1f}s, "
            f"buffers={len(buffers)}, ram≈{sum(b.numel() * b.element_size() for b in buffers.values()) / 2**30:.1f} GiB"
        )
    return buffers


keys_cache: dict[str, list | None] = {}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def do_export(args) -> None:
    ckpt_dir = args.ckpt_dir
    orig_dir = args.orig_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    index = load_orig_index(orig_dir)
    orig_names = set(index["weight_map"])
    log(f"original checkpoint: {len(orig_names)} tensors, {len(set(index['weight_map'].values()))} files")
    log("reading original tensor metadata (dtype/shape) ...")
    orig_meta = orig_tensor_meta(orig_dir, index)

    n_experts = args.n_experts
    files = sorted(glob.glob(os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")))
    if not files:
        sys.exit(f"no shards under {ckpt_dir}")

    # ---- name plan: from rank0's keys, before paying for the full assembly
    sd0 = _load_pt(files[0])
    mapping, unmapped = build_name_map(list(sd0.keys()), orig_names, n_experts)
    del sd0
    exported_orig_names: set[str] = set()
    for entries in mapping.values():
        for oname, _ in entries:
            exported_orig_names.add(oname)
    log(f"exported tensors: {len(exported_orig_names)} (from {len(mapping)} train params)")
    if unmapped:
        log(f"UNMAPPED train params ({len(unmapped)}), dropped from export: {unmapped[:10]}")

    leftover = sorted(orig_names - exported_orig_names)
    log(f"leftover checkpoint tensors kept from original files: {len(leftover)}")
    leftover_engram = [n for n in leftover if is_engram_name(n)]
    leftover_other = [n for n in leftover if not is_engram_name(n)]
    log(f"  of which engram: {len(leftover_engram)}, other: {len(leftover_other)}")
    if leftover_other:
        log(f"  other (kept from original): {leftover_other[:12]}")

    # ---- assemble shards
    t0 = time.time()
    buffers = assemble(ckpt_dir, index, n_experts, orig_meta)
    log(f"assembly done in {(time.time() - t0) / 60:.1f} min, {len(buffers)} buffers")

    # ---- expand fused experts into per-expert checkpoint tensors
    out_tensors: dict[str, torch.Tensor] = {}
    for name, buf in buffers.items():
        entries = map_param_to_orig(name, orig_names, n_experts)
        m_up = _FUSED_UP.match(name)
        m_down = _FUSED_DOWN.match(name)
        if m_up:
            gate, up = buf.chunk(2, dim=1)
            for oname, part in entries:
                kind, e = part
                out_tensors[oname] = (gate[e] if kind == "gate" else up[e]).contiguous()
        elif m_down:
            for oname, part in entries:
                _, e = part
                out_tensors[oname] = buf[e].contiguous()
        else:
            (oname, _part), = entries
            out_tensors[oname] = buf.contiguous()
    del buffers
    total_bytes = sum(t.numel() * t.element_size() for t in out_tensors.values())
    log(f"expanded to {len(out_tensors)} checkpoint tensors, {total_bytes / 2**30:.1f} GiB")

    # ---- write new safetensors shards
    max_bytes = int(args.max_shard_gb * 2**30)
    names_sorted = sorted(out_tensors)
    shard_files: list[str] = []
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    new_weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal cur, cur_bytes
        if not cur:
            return
        fn = f"model-sft-{len(shard_files):05d}.safetensors"
        save_file(cur, os.path.join(out_dir, fn), metadata={"format": "pt"})
        shard_files.append(fn)
        log(f"wrote {fn}: {len(cur)} tensors, {cur_bytes / 2**30:.2f} GiB")
        for k in cur:
            new_weight_map[k] = fn
        cur, cur_bytes = {}, 0

    for n in names_sorted:
        t = out_tensors[n]
        b = t.numel() * t.element_size()
        if cur and cur_bytes + b > max_bytes:
            flush()
        cur[n] = t
        cur_bytes += b
    flush()
    del out_tensors

    # ---- original files: symlink when they hold no exported tensor, else copy needed tensors
    def orig_files_for(names: list[str]) -> dict[str, list[str]]:
        byfile: dict[str, list[str]] = {}
        for n in names:
            byfile.setdefault(index["weight_map"][n], []).append(n)
        return byfile

    orig_weight_map: dict[str, str] = {}
    exported_by_file: dict[str, int] = {}
    for n in exported_orig_names:
        exported_by_file.setdefault(index["weight_map"][n], 0)
        exported_by_file[index["weight_map"][n]] += 1
    for fn, names in orig_files_for(leftover).items():
        exported_in_file = exported_by_file.get(fn, 0) > 0
        if not exported_in_file:
            dst = os.path.join(out_dir, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(orig_dir, fn), dst)
            for n in names:
                orig_weight_map[n] = fn
        else:
            extra = f"model-sft-extra-{fn}"
            with safe_open(os.path.join(orig_dir, fn), framework="pt") as fh:
                tens = {n: fh.get_tensor(n) for n in names}
            save_file(tens, os.path.join(out_dir, extra), metadata={"format": "pt"})
            log(f"copied {len(names)} leftover tensors out of mixed file {fn} -> {extra}")
            for n in names:
                orig_weight_map[n] = extra

    weight_map = dict(new_weight_map)
    weight_map.update(orig_weight_map)
    import math

    total_size = 0
    for n, (dt, shp) in orig_meta.items():
        total_size += math.prod(shp) * torch.tensor([], dtype=dt).element_size()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fh)
    log(f"index rebuilt: {len(weight_map)} tensors ({len(new_weight_map)} new + {len(orig_weight_map)} original-derived)")

    # ---- side files: config / tokenizer / etc (symlink everything non-safetensors)
    for entry in os.listdir(orig_dir):
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            continue
        src = os.path.join(orig_dir, entry)
        dst = os.path.join(out_dir, entry)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            os.symlink(src, dst)
    log("side files symlinked/copied")

    # ---- verification
    with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
        vmap = json.load(fh)["weight_map"]
    missing_files = {f for f in vmap.values() if not os.path.exists(os.path.join(out_dir, f))}
    if missing_files:
        sys.exit(f"VERIFY FAIL: missing files {missing_files}")
    engram_out = [n for n in vmap if is_engram_name(n)]
    if len(engram_out) != len(leftover_engram):
        log(f"WARN: engram count in built index {len(engram_out)} != expected {len(leftover_engram)}")
    # spot-check: one trained tensor vs original (expect DIFFERENT), one frozen vs original (expect EQUAL)
    probe = args.verify_probe
    if probe:
        name = probe
        fn = vmap[name]
        with safe_open(os.path.join(out_dir, fn), framework="pt") as fh:
            new_t = fh.get_tensor(name)
        with safe_open(os.path.join(orig_dir, index["weight_map"][name]), framework="pt") as fh:
            old_t = fh.get_tensor(name)
        same = torch.equal(new_t.float(), old_t.float())
        diff = (new_t.float() - old_t.float()).abs().max().item() if new_t.shape == old_t.shape else float("nan")
        log(f"probe {name}: shape {tuple(new_t.shape)} equal_to_original={same} max_abs_diff={diff}")
    log(f"EXPORT DONE: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--ckpt-dir", required=True, help="directory holding model_world_size_*_rank_*.pt")
        p.add_argument("--orig-dir", default=_ORIG_DIR_DEFAULT)
        p.add_argument("--n-experts", type=int, default=32)

    p = sub.add_parser("inspect")
    add_common(p)

    p = sub.add_parser("dry-map")
    add_common(p)

    p = sub.add_parser("export")
    add_common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-shard-gb", type=float, default=3.0)
    p.add_argument("--verify-probe", default=None)

    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect_shard(args.ckpt_dir)
        return
    if args.cmd == "dry-map":
        index = load_orig_index(args.orig_dir)
        orig_names = set(index["weight_map"])
        files = sorted(glob.glob(os.path.join(args.ckpt_dir, "model_world_size_*_rank_*.pt")))
        sd0 = _load_pt(files[0])
        mapping, unmapped = build_name_map(list(sd0.keys()), orig_names, args.n_experts)
        cov = set()
        for es in mapping.values():
            for on, _ in es:
                cov.add(on)
        log(f"train params (rank0): {len(sd0)} -> mapped {len(mapping)}, unmapped {len(unmapped)}")
        log(f"checkpoint tensors covered: {len(cov)} / {len(orig_names)}; leftover {len(orig_names - cov)}")
        log(f"sample mapping: {list(mapping.items())[:2]}")
        if unmapped:
            log(f"unmapped sample: {unmapped[:20]}")
        return
    do_export(args)


if __name__ == "__main__":
    main()
