#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Kernel-form parity microbench for the attention tail (the ~0.0045 attn floor).

The probe (wiki/worklog_dsv41_real4_prod_consistency.md §4.4) localized the biggest
attention difference to the segment AFTER the attention core: inverse-RoPE + the
block-diagonal o_proj. The two stacks compute the SAME math with different kernels:

    trainer (FSDPTurbo model.py:911-917):
        apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)   # complex64 mult, in-place copy_
        einsum("bsgd,grd->bsgr", o.view(B,T,G,Dg), wo_a.view(G,R,Dg))
    engine (vllm-ascend dsa_v41.py:655 + dsa_v1.py:1561):
        torch.ops._C_ascend.inplace_partial_rotary_mul(o.unsqueeze(1), cos, -sin,
            rotary_mode="interleave", partial_slice=[nope, head_dim])
        torch_npu.npu_transpose_batchmatmul(o.view(T,G,Dg), wo_a[G,Dg,R],
            perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2))

If either pair is bit-exact on this hardware, the corresponding piece of the floor is
NOT these two call forms (look elsewhere); if it is 1-ULP-different, unifying the form
is a concrete lever -- and because production probs-pearson is driven by ~1% high-prob
tokens whose MoE routing flips (§3.1), a single-form win can move the metric far more
than proportionally (threshold effect).

Runs on ONE NPU, no distributed. Inputs: real ckpt weights + the recorded in_all
attention-core output (layer 0) from the probe dump, plus a synthetic control.

Usage::

    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:/workspace-verl/verl \
    python3 scripts/dsv41_consistency/oproj_rope_parity_microbench.py \
        --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real
"""

import argparse
import math

import torch


def report(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float()
    b = b.float()
    same = (a == b)
    diff = (a - b).abs()
    rel = diff.norm() / b.norm().clamp_min(1e-12)
    print(
        f"  [{tag}] bit-exact {int(same.sum())}/{same.numel()} ({same.float().mean()*100:.2f}% same) "
        f"rel_err {rel:.3e} max|Δ| {diff.max():.3e}"
    )


def load_weights(model_path: str, layer: int, device):
    import json
    import os

    from safetensors import safe_open

    index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        with safe_open(os.path.join(model_path, index[name]), framework="pt") as f:
            return f.get_tensor(name)

    wo_a = get(f"layers.{layer}.attn.wo_a.weight").to(device)  # (8192, 4096) bf16
    return wo_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real")
    ap.add_argument("--probe-dump", default="/mnt/share/m00899630/dsv41/dump/probe_input_real4_fix_attn/probe_len200_in_all.pt")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    import torch_npu  # noqa: F401

    dev = torch.device("npu:0")
    torch.manual_seed(0)

    # --- config pieces (values verified against the checkpoint config 2026-09-23) ---
    from fsdp_turbo.models.deepseek_v41.model import apply_rotary_emb, precompute_freqs_cis

    H, D, RD, G, R = 64, 512, 64, 8, 1024  # heads, head_dim, rope_dim, o_groups, o_lora_rank
    T = 200
    use_yarn = args.layer >= 2  # model.py:774-779: compress layers use YaRN @ compress_rope_theta
    freqs_cis = precompute_freqs_cis(
        RD, T,
        original_seq_len=65536 if use_yarn else 0,
        base=40000.0 if use_yarn else 10000.0,
        factor=16.0, beta_fast=32, beta_slow=1,
    ).to(dev)  # complex64 (T, RD/2)
    print(f"[setup] layer {args.layer} YaRN={use_yarn} freqs {tuple(freqs_cis.shape)} {freqs_cis.dtype}")

    # --- input: recorded attention-core output (trainer side, saved fp32-of-bf16) + synthetic ---
    core_tr = None
    try:
        d = torch.load(args.probe_dump, map_location="cpu", weights_only=False)
        core_tr = d["stages"][f"attn_op{args.layer}.output"]
        print(f"[setup] core input from {args.probe_dump.split('/')[-1]}: {tuple(core_tr.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] no probe dump ({exc}); synthetic only")
    inputs = {}
    if core_tr is not None and core_tr.shape[1] >= T:
        inputs["recorded"] = core_tr[:1, :T].bfloat16().to(dev)
    inputs["synthetic"] = (torch.randn(1, T, H, D) * 0.5).bfloat16().to(dev)

    # ================= T1: inverse-RoPE, complex path vs CANN interleave op =================
    import vllm_ascend  # noqa: F401  (registers torch.ops._C_ascend)

    c, s = freqs_cis.real, freqs_cis.imag  # (T, 32)
    layouts = {
        "dup-adjacent (T,1,1,64)": (c.repeat_interleave(2, dim=-1), s.repeat_interleave(2, dim=-1)),
        "half (T,1,1,32)": (c, s),
    }
    print("[T1] inverse rope: trainer apply_rotary_emb(inverse) vs inplace_partial_rotary_mul(-sin, interleave)")
    for name, x in inputs.items():
        xa = x.clone()
        apply_rotary_emb(xa[..., -RD:], freqs_cis, inverse=True)
        for lname, (cos, sin) in layouts.items():
            try:
                xb = x.clone()
                torch.ops._C_ascend.inplace_partial_rotary_mul(
                    xb.squeeze(0).unsqueeze(1),  # (T,H,1,D) like the engine's attention_output.unsqueeze(1)
                    cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1),
                    rotary_mode="interleave",
                    partial_slice=[D - RD, D],
                )
                report(f"{name} vs {lname}", xa, xb)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name} vs {lname}] op failed: {exc}")

    # ================= T2: grouped o_proj GEMM, einsum vs npu_transpose_batchmatmul =========
    wo_a2d = load_weights(args.model_path, args.layer, dev)
    wa_grd = wo_a2d.view(G, R, H * D // G)          # trainer view: (G, R, Dg=4096)
    wa_gdr = wa_grd.transpose(1, 2).contiguous()    # engine A3 layout: (G, Dg, R)
    print("[T2] o_proj grouped GEMM: einsum(bsgd,grd->bsgr) vs npu_transpose_batchmatmul")
    for name, x in inputs.items():
        oin = x.reshape(T, G, H * D // G)  # both stacks view the core output (T,64,512)->(T,8,4096)
        ea = torch.einsum("sgd,grd->sgr", oin, wa_grd)
        eb = torch_npu.npu_transpose_batchmatmul(
            oin, wa_gdr, bias=None, scale=None,
            perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2), batch_split_factor=1,
        ).reshape(T, G, R)
        report(f"{name}", ea, eb)
        # fp32 reference to see which side is closer (informational only)
        ref = torch.einsum("sgd,grd->sgr", oin.float(), wa_grd.float())
        print(
            f"        vs fp32 ref: einsum rel {(ea.float()-ref).norm()/ref.norm():.3e} | "
            f"transpose_bmm rel {(eb.float()-ref).norm()/ref.norm():.3e}"
        )


if __name__ == "__main__":
    main()
