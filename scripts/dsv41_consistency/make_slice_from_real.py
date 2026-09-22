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

"""Slice N layers out of a real DeepSeek-V4.1 checkpoint into a standalone model dir.

The consistency harness runs on a 4-layer model (see ``worklog_dsv41_module_input_probe.md``),
so testing it on the real weights needs a 4-layer checkpoint that *contains the real tensors*.
This script builds one from ``/mnt/share/DeepSeek-V4.1-Flash-bf16`` without copying 113 GB.

**Why "just rewrite config.json" is not enough.** The two consumers read weights differently:

* trainer (``read_dsv41_checkpoint_state_dict``) is *index-driven*: it iterates
  ``model.safetensors.index.json`` and routes every name the 4-layer model does not declare to
  ``skipped`` without reading those bytes -- a rewritten config alone would work there;
* engine (vLLM) is *file-driven*: ``safetensors_weights_iterator`` yields **every key of every
  shard** (``weight_utils.py:968-972``; the only filter is ``should_skip_weight``, which drops
  non-local *experts*, ``ep_weight_filter.py:64-86``) and the v4.1 loader then does a bare
  ``params_dict[name]`` (``vllm_ascend/models/deepseek_v4/model.py:1331``). Any extra tensor in a
  linked shard is a hard ``KeyError``.

So the slice must satisfy the invariant both existing 4-layer checkpoints satisfy:
**for every ``*.safetensors`` in the directory, ``set(file keys) == {names the index maps to it}``**.
Shards that already satisfy it are hard-linked (0 bytes, immune to the source dir being renamed);
the few that also carry tensors of the other layers are rewritten with only the needed names,
dtype preserved bit-for-bit.

Usage:

    python3 scripts/dsv41_consistency/make_slice_from_real.py \
      --src /mnt/share/DeepSeek-V4.1-Flash-bf16 \
      --out /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real \
      --layers 0,1,2,3 \
      --config-template /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled/config.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

SKIP_PREFIXES = ("mtp.",)  # MTP/DSpark heads are never part of a backbone slice
SKIP_SUBSTRINGS = (".engram.",)  # the 4-layer config disables engram (its tables are 197 GB)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/mnt/share/DeepSeek-V4.1-Flash-bf16")
    ap.add_argument("--out", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real")
    ap.add_argument("--layers", default="0,1,2,3", help="Comma separated backbone layer ids (0-based prefix).")
    ap.add_argument("--config-template", required=True,
                    help="config.json of a working slice with the same shape (defines both the config and, "
                         "via its sibling index, the exact tensor-name set).")
    ap.add_argument("--mode", choices=("hybrid", "copy"), default="hybrid",
                    help="hybrid: hard-link clean shards and rewrite the rest (default). copy: rewrite all.")
    return ap.parse_args()


def read_header(path: Path) -> dict[str, tuple[int, str]]:
    """{tensor name: (nbytes, dtype)} from a safetensors header, without touching the data."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return {
        name: (entry["data_offsets"][1] - entry["data_offsets"][0], entry["dtype"])
        for name, entry in header.items()
        if name != "__metadata__"
    }


def is_excluded(name: str) -> bool:
    return name.startswith(SKIP_PREFIXES) or any(s in name for s in SKIP_SUBSTRINGS)


def link_or_copy(src: Path, dst: Path, mode: str, log) -> str:
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError as exc:  # cross-device or unsupported -> symlink keeps the slice usable
        log(f"  hardlink failed ({exc}); falling back to symlink for {dst.name}")
        os.symlink(src.resolve(), dst)
        return "symlinked"


def rewrite_shard(src: Path, dst: Path, names: list[str]) -> int:
    """Write a new shard holding only `names`, preserving dtype and bytes exactly."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = [n for n in names if n not in available]
        if missing:
            raise SystemExit(f"{src.name}: missing {len(missing)} wanted tensors, e.g. {missing[:3]}")
        for name in names:
            tensors[name] = handle.get_tensor(name).contiguous().clone()
    save_file(tensors, str(dst), metadata={"format": "pt"})
    written = sum(t.numel() * t.element_size() for t in tensors.values())
    del tensors
    return written


def pick_names(src_index: dict[str, str], template_names: list[str], layers: list[int], log) -> list[str]:
    wanted = [n for n in template_names
              if not is_excluded(n) and (not n.startswith("layers.") or int(n.split(".")[1]) in layers)]
    missing = [n for n in wanted if n not in src_index]
    if missing:
        raise SystemExit(f"{len(missing)} wanted tensors absent from the source, e.g. {missing[:5]}")
    # The source must not hold layer tensors the template lacks (that would mean an incomplete slice).
    wanted_set = set(wanted)
    src_layer = {n for n in src_index
                 if n.startswith("layers.") and int(n.split(".")[1]) in layers and not is_excluded(n)}
    extra = sorted(src_layer - wanted_set)
    if extra:
        raise SystemExit(f"{len(extra)} layer tensors exist in the source but not in the template "
                         f"(slice would be incomplete), e.g. {extra[:5]}")
    return wanted


def build_config(template_path: Path, real_path: Path, layers: list[int], log) -> dict:
    """Template byte-for-byte for a 0-based prefix; a validated rewrite otherwise.

    Copying the *real* config would be fatal for the trainer: `engram_meta_init=True` builds meta
    Engram modules for layers 1/14 that no slice tensor can materialize, so DCP fails.
    """
    template = json.loads(template_path.read_text())
    text = template.get("text_config", template)
    real_text = json.loads(real_path.read_text()).get("text_config", json.loads(real_path.read_text()))
    if layers == list(range(len(layers))):
        ratios = list(real_text["compress_ratios"])[: len(layers)]
        if list(text["compress_ratios"]) != ratios:
            raise SystemExit(f"template compress_ratios {text['compress_ratios']} != real prefix {ratios}")
        for key in ("hidden_size", "n_routed_experts", "num_experts_per_tok", "vocab_size"):
            if text.get(key) != real_text.get(key):
                raise SystemExit(f"template {key}={text.get(key)} != real {real_text.get(key)}")
        log("config: template used as-is (0-based prefix; ratios/expert count verified against real)")
        return template

    local = {layer: index for index, layer in enumerate(layers)}
    kv = [local[l] for l in real_text["kv_source_layer_ids"] if l in local]
    idx = [local[l] for l in real_text["index_source_layer_ids"] if l in local]
    if not kv:
        raise SystemExit("no kv_source layer inside the requested slice: candidate/compressor semantics "
                         "are undefined -- decide the remap explicitly before slicing")
    if not set(kv).issubset(set(idx)):
        raise SystemExit(f"kv sources {kv} are not a subset of index sources {idx}")
    candidate = max(kv)
    log(f"config: general rewrite -> kv={kv} index={idx} candidate={candidate}")
    text["num_hidden_layers"] = len(layers)
    text["compress_ratios"] = [real_text["compress_ratios"][l] for l in layers]
    for new_key, alias_key, values in (("kv_source_layer_ids", "kv_source_layers", kv),
                                       ("index_source_layer_ids", "index_source_layers", idx)):
        for key in (new_key, alias_key):
            if key in text:
                text[key] = values
    for key in ("candidate_source_layer_id", "candidate_source_layer"):
        if key in text:
            text[key] = candidate
    return template


def audit(out: Path, needed: dict[str, list[str]], wanted: list[str], index: dict, src: Path,
          src_index: dict[str, str], template_dir: Path, log) -> list[str]:
    """L1/L2/L4/L5 checks. Returns a list of problems (empty = pass)."""
    from safetensors import safe_open

    problems = []
    # L1: index <-> files <-> directory agree, and nothing else is in the directory
    if sorted(index["weight_map"]) != sorted(wanted):
        problems.append("index weight_map != selected names")
    present = {p.name for p in out.glob("*.safetensors")}
    if present != set(needed):
        problems.append(f"directory shards != needed shards (extra={present - set(needed)}, "
                        f"missing={set(needed) - present})")
    total = 0
    for shard_name, names in needed.items():
        sizes = read_header(out / shard_name)
        if set(sizes) != set(names):
            problems.append(f"{shard_name}: file keys != index names "
                            f"(extra={len(set(sizes) - set(names))}, missing={len(set(names) - set(sizes))})")
        total += sum(sizes[n][0] for n in names)
    if total != index["metadata"]["total_size"]:
        problems.append(f"total_size {index['metadata']['total_size']} != summed tensors {total}")

    # L2: the slice config must equal the template's (0-based prefix case)
    if out.name and (out / "config.json").is_file():
        got = json.loads((out / "config.json").read_text())
        want = json.loads((template_dir / "config.json").read_text())
        if got != want:
            diff = [k for k in set(got) | set(want) if got.get(k) != want.get(k)]
            problems.append(f"config differs from template in {diff}")

    # L4: sampled tensors must be bit-identical to the source
    sample = [n for n in wanted if n.endswith("hc_attn_fn")][:1] + \
             [n for n in wanted if n.endswith("embed.weight")] + \
             [n for n in wanted if "compressor.wkv" in n][:1]
    for name in sample:
        with safe_open(str(out / src_index[name]), framework="pt") as left, \
                safe_open(str(src / src_index[name]), framework="pt") as right:
            got, want = left.get_tensor(name), right.get_tensor(name)
        if got.dtype != want.dtype or got.shape != want.shape or not bool((got == want).all()):
            problems.append(f"{name}: slice value/dtype != source")

    # L5: dtype audit against the template, from headers only (no data reads).
    # The real checkpoint stores hc_*/attn_sink/gate.bias{,_vl} in fp32 while the template has bf16;
    # the trainer rounds those to bf16 (param_dtype) but the engine keeps fp32 -- a train-vs-engine
    # difference that only shows up with real weights (plan.md H2).
    template_index = json.loads((template_dir / "model.safetensors.index.json").read_text())["weight_map"]
    real_dtypes: dict[str, str] = {}
    template_dtypes: dict[str, str] = {}
    for shard_name in {src_index[n] for n in wanted}:
        real_dtypes.update({k: v[1] for k, v in read_header(src / shard_name).items()})
    for shard_name in {template_index[n] for n in wanted}:
        template_dtypes.update({k: v[1] for k, v in read_header(template_dir / shard_name).items()})
    gaps = collections.Counter()
    for name in wanted:
        if real_dtypes[name] != template_dtypes[name]:
            gaps[f"{template_dtypes[name]} -> {real_dtypes[name]}"] += 1
    log(f"dtype audit vs template (template dtype -> real dtype): {dict(gaps) if gaps else 'no differences'} "
        "(expected {BF16 -> F32: 36, F32 -> BF16: 3} with the real checkpoint)")
    return problems


def write_slice_doc(out: Path, src: Path, template: Path, layers: list[int], mode: str,
                    wanted: list[str], needed: dict[str, list[str]], index: dict,
                    linked: int, rewritten: int, new_bytes: int) -> None:
    total = index["metadata"]["total_size"]
    lines = [
        f"# Slice: {out.name}",
        "",
        f"- 生成时间：{index['metadata']['dsv41_slice']['created']}（`make_slice_from_real.py --mode {mode}`）",
        f"- 源：`{src}`（真实 DeepSeek-V4.1-Flash bf16，40 层 + 3 MTP，384 专家）",
        f"- config 模板：`{template}`（4 层、384 专家；**不要**改用真实 config —— `engram_meta_init=True` "
        f"会给层 1/14 建 meta Engram 模块，切片里没有对应张量，DCP 会失败）",
        f"- 层：{layers}；张量 {len(wanted)} 个；逻辑大小 {total / 2**30:.2f} GiB（{total} B）",
        f"- shard：{linked} 个 hardlink（0 字节）+ {rewritten} 个重写（新增磁盘 {new_bytes / 2**30:.2f} GiB）",
        "",
        "## 为什么要重写 shard（而不是只改 config.json）",
        "",
        "trainer 侧按 `model.safetensors.index.json` 取数（多余名字进 `skipped`、不读字节），",
        "而引擎侧 vLLM 是**按文件**枚举 `f.keys()` 后裸查 `params_dict[name]` → shard 里任何一个多余张量都是 KeyError。",
        "所以切片必须满足：每个 `*.safetensors` 的 key 集合 == index 指给它的名字（与两个已跑通的 4 层 ckpt 一致）。",
        "",
        "## 语义偏差（与真实模型前 4 层不完全等价）",
        "",
        "- `candidate_source_layer_id = 2`（真实是 20）→ 层 3 开了候选预筛，真实模型层 0–3 是关的；",
        "- engram 关闭（真实层 1 有 197 GB engram 表）；MTP 关闭；",
        "- 引擎走的是 config 里 `deepseek_v4.1`（带点）拼写的代码路径；真实 ckpt 的 config 是下划线拼写；",
        "- 真实权重与模板有 39 个 dtype 差异（36 个 `hc_*`/`attn_sink`/`gate.bias{,_vl}` 真实=fp32、模板=bf16；",
        "  `image_start/end/newline` 反向）→ 训练侧会把 fp32 舍入成 bf16，引擎保持 fp32，",
        "  这是真实权重下**新增的一个训推差异来源**（详见 plan.md 的 H2）。",
        "- 结论：两栈之间的对照是公平的（同一个 4 层模型），但绝对数值不代表 40 层真实模型的训推差。",
        "",
        "## 复现",
        "",
        "```bash",
        "python3 scripts/dsv41_consistency/make_slice_from_real.py \\",
        f"  --src {src} --out {out} --layers {','.join(str(l) for l in layers)} \\",
        f"  --config-template {template} --mode {mode}",
        "```",
        "",
        f"排除项：其它 {40 - len(layers)} 层的全部张量、`mtp.*`、`*.engram.*`、`optional/quarot.safetensors`、`config.json_bak`。",
    ]
    (out / "SLICE.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    started = time.time()
    src, out, template_config = Path(args.src), Path(args.out), Path(args.config_template)
    layers = [int(x) for x in args.layers.split(",") if x]

    def log(message):
        print(f"[slice] {message}", flush=True)

    if not (src / "model.safetensors.index.json").is_file():
        raise SystemExit(f"{src} has no model.safetensors.index.json")
    if not template_config.is_file():
        raise SystemExit(f"missing --config-template {template_config}")
    template_dir = template_config.parent
    src_index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    template_index = json.loads((template_dir / "model.safetensors.index.json").read_text())["weight_map"]
    log(f"source: {src} ({len(src_index)} tensors) | template: {template_dir} ({len(template_index)} tensors)")

    wanted = pick_names(src_index, sorted(template_index), layers, log)
    log(f"selected {len(wanted)} tensors for layers {layers}")
    needed: dict[str, list[str]] = collections.defaultdict(list)
    for name in wanted:
        needed[src_index[name]].append(name)

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.safetensors"):  # never mix shards from an earlier slice
        stale.unlink()

    total_bytes = new_bytes = 0
    linked = rewritten = 0
    for shard_name, names in sorted(needed.items()):
        src_shard, dst_shard = src / shard_name, out / shard_name
        sizes = read_header(src_shard)
        mine = sum(sizes[n][0] for n in names)
        total_bytes += mine
        if args.mode == "hybrid" and set(sizes) == set(names):
            linked += 1
            how = link_or_copy(src_shard, dst_shard, args.mode, log)
            log(f"  {shard_name}: {len(names)} tensors, {mine / 2**30:.2f} GiB ({how})")
        else:
            written = rewrite_shard(src_shard, dst_shard, names)
            new_bytes += written
            rewritten += 1
            log(f"  {shard_name}: rewrote {len(names)}/{len(sizes)} tensors "
                f"({len(set(sizes) - set(names))} extras dropped), {written / 2**30:.2f} GiB")

    index = {
        "metadata": {
            "total_size": total_bytes,
            "dsv41_slice": {
                "source": str(src), "config_template": str(template_config), "layers": layers,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": args.mode, "tensors": len(wanted),
                "linked_shards": linked, "rewritten_shards": rewritten,
            },
        },
        "weight_map": {name: src_index[name] for name in wanted},
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    (out / "config.json").write_text(json.dumps(build_config(template_config, src / "config.json", layers, log),
                                                indent=1))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = src / name
        if source.is_file():
            target = out / name
            if target.exists():
                target.unlink()
            link_or_copy(source, target, "copy", log)

    problems = audit(out, needed, wanted, index, src, src_index, template_dir, log)
    write_slice_doc(out, src, template_config, layers, args.mode, wanted, needed, index,
                    linked, rewritten, new_bytes)
    log(f"index: {len(wanted)} tensors, total_size={total_bytes} ({total_bytes / 2**30:.2f} GiB); "
        f"new disk {new_bytes / 2**30:.2f} GiB")
    if problems:
        for problem in problems:
            log(f"VALIDATION FAILED: {problem}")
        raise SystemExit(1)
    log(f"validation OK (shard key sets == index names; config == template; sampled tensors bit-identical)")
    log(f"slice dir: {out} ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
