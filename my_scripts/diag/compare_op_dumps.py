"""Compare training vs inference per-op dumps at the last prompt token.

本脚本只读 op-dump 目录，不加载模型权重。用法：
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 compare_op_dumps.py <dump_dir> \
      --infer-pid 1931290 --train-pid 1918739 --infer-fwd 3 --train-fwd 1 --pos 254
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F


def load_index(root: str, pid: int):
    path = os.path.join(root, f"index_{pid}.jsonl")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def pick(rows, role, fwd, cls, layer=None, call=0):
    for r in rows:
        if (
            r["role"] == role
            and r["forward"] == fwd
            and r["cls"] == cls
            and r["layer"] == layer
            and r["call"] == call
        ):
            return r
    return None


def tensors(x):
    out = []
    if isinstance(x, dict):
        if "tensor" in x:
            out.append(x["tensor"])
        else:
            for v in x.values():
                out += tensors(v)
    elif isinstance(x, list):
        for v in x:
            out += tensors(v)
    return out


def first_tensor(x, integer=False):
    for t in tensors(x):
        if integer and not torch.is_floating_point(t):
            return t
        if not integer and torch.is_floating_point(t):
            return t
    return None


def seq_feat(t):
    if t is None:
        return None
    t = t.float()
    while t.ndim > 2 and t.shape[0] == 1:
        t = t[0]
    if t.ndim == 2:
        return t
    return None


def row(t, pos):
    t = seq_feat(t)
    if t is None or t.shape[0] <= pos:
        return None
    return t[pos].flatten()


def metrics(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    d = a - b
    return {
        "rel_l2": float(d.norm() / (a.norm() + 1e-8)),
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "cos": float(F.cosine_similarity(a, b, dim=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--infer-pid", type=int, required=True)
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--infer-fwd", type=int, default=3)
    ap.add_argument("--train-fwd", type=int, default=1)
    ap.add_argument("--pos", type=int, default=254)
    args = ap.parse_args()

    inf = load_index(args.dump_dir, args.infer_pid)
    tr = load_index(args.dump_dir, args.train_pid)

    # 对齐检查：推理 prefill 的 input_ids 必须是训练序列的前缀。
    emb_i = pick(inf, "infer", args.infer_fwd, "VocabParallelEmbedding")
    emb_t = pick(tr, "train", args.train_fwd, "ParallelEmbedding")
    di = torch.load(os.path.join(args.dump_dir, emb_i["path"]), map_location="cpu", weights_only=False)
    dt = torch.load(os.path.join(args.dump_dir, emb_t["path"]), map_location="cpu", weights_only=False)
    ids_i = first_tensor(di["args"], integer=True).flatten()
    ids_t = first_tensor(dt["args"], integer=True).flatten()
    prefix_ok = bool(torch.equal(ids_i, ids_t[: ids_i.numel()]))
    print(f"infer_ids={ids_i.numel()} train_ids={ids_t.numel()} prefix_equal={prefix_ok} pos={args.pos}")

    # 直接可对齐的模块：推理类 -> 训练类。
    pairs = [
        ("DeepseekV41Attention", "Attention"),
        ("DeepseekV41MoE", "MoE"),
        ("DeepseekV41Compressor", "Compressor"),
        ("DeepseekV41Indexer", "Indexer"),
    ]

    rows = []
    for icls, tcls in pairs:
        for layer in range(40):
            ri = pick(inf, "infer", args.infer_fwd, icls, layer)
            rt = pick(tr, "train", args.train_fwd, tcls, layer)
            if ri is None or rt is None:
                continue
            di = torch.load(os.path.join(args.dump_dir, ri["path"]), map_location="cpu", weights_only=False)
            dt = torch.load(os.path.join(args.dump_dir, rt["path"]), map_location="cpu", weights_only=False)
            oi = first_tensor(di["output"])
            ot = first_tensor(dt["output"])
            a = row(ot, args.pos)
            b = row(oi, args.pos)
            m = metrics(a, b)
            rows.append(
                {
                    "op": icls,
                    "layer": layer,
                    "shape_infer": None if oi is None else list(oi.shape),
                    "shape_train": None if ot is None else list(ot.shape),
                    **(m or {"rel_l2": None, "max_abs": None, "mean_abs": None, "cos": None}),
                }
            )

    # 最终 hidden：训练 ParallelHead 的输入 vs 推理模型输出。
    rh = pick(tr, "train", args.train_fwd, "ParallelHead")
    rm = pick(inf, "infer", args.infer_fwd, "AscendDeepseekV41LLMForCausalLM")
    if rh and rm:
        dh = torch.load(os.path.join(args.dump_dir, rh["path"]), map_location="cpu", weights_only=False)
        dm = torch.load(os.path.join(args.dump_dir, rm["path"]), map_location="cpu", weights_only=False)
        a = row(first_tensor(dh["args"]), args.pos)
        b = row(first_tensor(dm["output"]), args.pos)
        m = metrics(a, b) or {}
        rows.append({"op": "FinalHidden", "layer": None, "shape_infer": [255, 5120], "shape_train": [1, 767, 5120], **m})

    # embedding
    a = row(first_tensor(dt["output"]), args.pos)
    b = row(first_tensor(di["output"]), args.pos)
    m = metrics(a, b) or {}
    rows.append({"op": "Embedding", "layer": None, "shape_infer": [255, 5120], "shape_train": [1, 767, 5120], **m})

    rows = [r for r in rows if r["rel_l2"] is not None]
    rows.sort(key=lambda r: r["rel_l2"], reverse=True)
    csv_path = os.path.join(args.dump_dir, "op_compare.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("op,layer,rel_l2,max_abs,mean_abs,cos,shape_infer,shape_train\n")
        for r in rows:
            f.write(
                f"{r['op']},{r['layer']},{r['rel_l2']:.8f},{r['max_abs']:.8f},"
                f"{r['mean_abs']:.8f},{r['cos']:.8f},{r['shape_infer']},{r['shape_train']}\n"
            )
    print(f"wrote {csv_path}")
    print("top errors:")
    for r in rows[:20]:
        print(
            f"  {r['op']:<22} L{str(r['layer']):>2} rel_l2={r['rel_l2']:.5f} "
            f"max={r['max_abs']:.5f} cos={r['cos']:.6f}"
        )


if __name__ == "__main__":
    main()
