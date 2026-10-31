"""对比 base / fused 两次 1-step dump：量化视觉融合 attention 对误差传播的影响。

用法（在 164 上，TORCH_DEVICE_BACKEND_AUTOLOAD=0）：
  python3 1031_compare_ab_vision.py <base_dump_dir> <fused_dump_dir> [--pos 254]

比较项：
  1. L0 图像/文本 token 的视觉特征误差（train Block0 输入 - embedding vs infer DecoderLayer0 输入 - embedding）。
  2. 每层 Block 输出 base vs fused 的 rel_l2（pos 与全部图像 token）。
  3. 最终 hidden（ParallelHead 输入）base vs fused，以及各自 vs 推理模型输出。
只读 dump，不加载模型。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F


def rows_of(root: str):
    rows = []
    for fn in sorted(glob.glob(os.path.join(root, "index_*.jsonl"))):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def find_one(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def load(root: str, r):
    return torch.load(os.path.join(root, r["path"]), map_location="cpu", weights_only=False)


def first_tensor(obj):
    if isinstance(obj, dict):
        if "tensor" in obj:
            return obj["tensor"]
        for v in obj.values():
            t = first_tensor(v)
            if t is not None:
                return t
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def tensor_at(obj, idx):
    """args/output 是 list 时按位置取 tensor（跳过非 tensor 元数据）。"""
    if isinstance(obj, (list, tuple)) and idx < len(obj):
        return first_tensor(obj[idx])
    return None


def train_pid(rows):
    for r in rows:
        if r["role"] == "train" and r["cls"] == "ParallelEmbedding" and r["forward"] == 1:
            return r["pid"]
    raise SystemExit("no train ParallelEmbedding record")


def seq_metrics(a, b):
    """整段/逐位置的 rel_l2 与 cos；a,b 形状相同，最后一维是 hidden。"""
    af = a.float()
    bf = b.float()
    d = af - bf
    rel = float(d.norm() / (af.norm() + 1e-8))
    cos = float(F.cosine_similarity(af.reshape(-1), bf.reshape(-1), dim=0))
    return rel, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("fused_dir")
    ap.add_argument("--pos", type=int, default=254)
    args = ap.parse_args()

    base_rows = rows_of(args.base_dir)
    fused_rows = rows_of(args.fused_dir)
    for r in base_rows:
        r["_root"] = args.base_dir
    for r in fused_rows:
        r["_root"] = args.fused_dir

    tb = train_pid(base_rows)
    tf = train_pid(fused_rows)
    # 推理 prefill 用 base/fused 各自的真实 prefill；fwd 号在两次运行中应一致。
    ib = find_one(base_rows, role="infer", cls="VocabParallelEmbedding", forward=3)
    iff = find_one(fused_rows, role="infer", cls="VocabParallelEmbedding", forward=3)
    if ib is None or iff is None:
        raise SystemExit("infer prefill fwd3 not found")

    # 训练 embedding / Block0 输入
    eb = load(args.base_dir, find_one(base_rows, role="train", cls="ParallelEmbedding", forward=1, pid=tb))
    ef = load(args.fused_dir, find_one(fused_rows, role="train", cls="ParallelEmbedding", forward=1, pid=tf))
    b0 = load(args.base_dir, find_one(base_rows, role="train", cls="Block", layer=0, forward=1, pid=tb))
    f0 = load(args.fused_dir, find_one(fused_rows, role="train", cls="Block", layer=0, forward=1, pid=tf))
    ii = load(args.base_dir, ib)
    i0 = load(args.base_dir, find_one(base_rows, role="infer", cls="DeepseekV41DecoderLayer", layer=0, forward=3, pid=ib["pid"]))

    ids = first_tensor(eb["args"]).flatten()
    img = ids == 129264
    emb_b = first_tensor(eb["output"]).float().reshape(-1, 5120)
    emb_f = first_tensor(ef["output"]).float().reshape(-1, 5120)
    xb = tensor_at(b0["args"], 0).float().reshape(-1, 4, 5120)[:, 0]
    xf = tensor_at(f0["args"], 0).float().reshape(-1, 4, 5120)[:, 0]
    emb_i = first_tensor(ii["output"]).float().reshape(-1, 5120)
    xi = tensor_at(i0["args"], 1).float().reshape(-1, 4, 5120)[:, 0]

    n = min(emb_b.shape[0], emb_f.shape[0], emb_i.shape[0], xi.shape[0], ids.numel())
    img = img[:n]
    fb = xb[:n] - emb_b[:n]
    ff = xf[:n] - emb_f[:n]
    fi = xi[:n] - emb_i[:n]

    def feat_err(feat, ref, mask):
        a = feat[mask]
        b = ref[mask]
        d = a - b
        rel = d.norm(dim=1) / (a.norm(dim=1) + 1e-8)
        return float(rel.mean()), float(rel.max()), float(F.cosine_similarity(a, b, dim=1).mean())

    print(f"ids={n} image={int(img.sum())} train_pids=({tb},{tf}) infer_pid={ib['pid']}")
    for name, mask in [("image", img), ("text", ~img)]:
        if mask.sum() == 0:
            continue
        rb = feat_err(fb, fi, mask)
        rf = feat_err(ff, fi, mask)
        print(f"{name:5s} base_err(mean/max/cos)={rb[0]:.6f}/{rb[1]:.6f}/{rb[2]:.6f}  "
              f"fused_err={rf[0]:.6f}/{rf[1]:.6f}/{rf[2]:.6f}")
    # base vs fused 视觉特征本身的差异
    d = (xb[:n][img] - xf[:n][img]).norm(dim=1) / (xb[:n][img].norm(dim=1) + 1e-8)
    print(f"image base-vs-fused rel mean={float(d.mean()):.6f} max={float(d.max()):.6f}")

    # 逐层 Block 输出
    out_rows = []
    p = args.pos
    for layer in range(40):
        rb = find_one(base_rows, role="train", cls="Block", layer=layer, forward=1, pid=tb)
        rf = find_one(fused_rows, role="train", cls="Block", layer=layer, forward=1, pid=tf)
        if rb is None or rf is None:
            continue
        ob = first_tensor(load(args.base_dir, rb)["output"]).float().reshape(-1, 4, 5120)
        of = first_tensor(load(args.fused_dir, rf)["output"]).float().reshape(-1, 4, 5120)
        # Block 输出含 response（767），图像 mask 只覆盖 prompt（255），取前 255 位。
        m = min(ob.shape[0], of.shape[0], img.numel())
        rel_img, cos_img = seq_metrics(ob[:m][img[:m]].reshape(-1, 5120), of[:m][img[:m]].reshape(-1, 5120))
        rel_p, cos_p = seq_metrics(ob[p], of[p])
        out_rows.append((layer, rel_img, cos_img, rel_p, cos_p))
        print(f"L{layer:02d} image rel={rel_img:.6f} cos={cos_img:.6f}  pos{p} rel={rel_p:.6f} cos={cos_p:.6f}")

    # 最终 hidden
    hb = first_tensor(load(args.base_dir, find_one(base_rows, role="train", cls="ParallelHead", forward=1, pid=tb))["args"]).float().reshape(-1, 5120)
    hf = first_tensor(load(args.fused_dir, find_one(fused_rows, role="train", cls="ParallelHead", forward=1, pid=tf))["args"]).float().reshape(-1, 5120)
    hi = first_tensor(load(args.base_dir, find_one(base_rows, role="infer", cls="AscendDeepseekV41LLMForCausalLM", forward=3, pid=ib["pid"]))["output"]).float().reshape(-1, 5120)
    print("final hidden pos%d: base-vs-fused rel=%.6f cos=%.6f" % (p, *seq_metrics(hb[p], hf[p])))
    print("final hidden pos%d: base-vs-infer rel=%.6f cos=%.6f" % (p, *seq_metrics(hb[p], hi[p])))
    print("final hidden pos%d: fused-vs-infer rel=%.6f cos=%.6f" % (p, *seq_metrics(hf[p], hi[p])))

    csv = os.path.join(args.fused_dir, "ab_vision_compare.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("layer,image_rel,image_cos,pos_rel,pos_cos\n")
        for row in out_rows:
            f.write("%d,%.8f,%.8f,%.8f,%.8f\n" % row)
    print("wrote", csv)


if __name__ == "__main__":
    main()
