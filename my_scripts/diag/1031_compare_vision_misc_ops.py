# [1031 调试] 视觉塔其余算子的训练/推理实现数值对比：RMSNorm、RoPE、SwiGLU。
# 用法：source my_scripts/1030_ray_dsv41_env.sh && python3 my_scripts/diag/1031_compare_vision_misc_ops.py
import torch
import torch.nn.functional as F
import torch_npu

from fsdp_turbo.ops.rope import rope
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.models.deepseek.deepseek_v41.activation import packed_swiglu

torch.manual_seed(0)
torch.npu.set_device(0)
dev = "npu:0"


def rel(name, a, b):
    a = a.float().flatten(); b = b.float().flatten()
    print(f"  {name}: rel={float((a-b).norm()/(b.norm()+1e-8)):.7f} cos={float(F.cosine_similarity(a,b,dim=0)):.8f}", flush=True)


# ---- RMSNorm ----
S, D = 1764, 1024
x = torch.randn(S, D, dtype=torch.bfloat16, device=dev) * 1.0
w = torch.randn(D, dtype=torch.float32, device=dev) * 0.1 + 1.0
train_norm = rms_norm(x, w, 1e-6)
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)
print("RMSNorm")
rel("train_npu_vs_ref", train_norm, ref)
try:
    from vllm.model_executor.layers.layernorm import RMSNorm
    m = RMSNorm(D, eps=1e-6, dtype=torch.float32).to(dev)
    with torch.no_grad():
        m.weight.copy_(w)
    infer_norm = m.forward_native(x)
    rel("infer_native_vs_ref", infer_norm, ref)
    rel("infer_native_vs_train", infer_norm, train_norm)
except Exception as e:
    print("  vllm RMSNorm failed:", repr(e), flush=True)

# ---- RoPE ----
S, N, Dh = 1764, 16, 64
q = torch.randn(S, N, Dh, dtype=torch.bfloat16, device=dev)
half = Dh // 2
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, Dh, 2, dtype=torch.float32, device=dev) / Dh))
pos = torch.arange(S, dtype=torch.float32, device=dev).unsqueeze(1)
cos = torch.cos(pos * inv_freq).to(torch.float32); sin = torch.sin(pos * inv_freq).to(torch.float32)
# 训练视觉 apply_rotary：x[1,S,N,D] + cos[1,S,1,half]
train_rope = rope(q.unsqueeze(0), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2), interleaved=False).squeeze(0)
print("RoPE")
try:
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    infer_rope = ApplyRotaryEmb.forward_static(q.unsqueeze(0), cos, sin, is_neox_style=True, enable_fp32_compute=True).squeeze(0)
    qf = q.float(); x1, x2 = qf.chunk(2, dim=-1)
    c = cos.unsqueeze(1); s = sin.unsqueeze(1)
    ref_rope = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(torch.bfloat16)
    rel("train_vs_ref", train_rope, ref_rope)
    rel("infer_static_vs_ref", infer_rope, ref_rope)
    rel("infer_static_vs_train", infer_rope, train_rope)
except Exception as e:
    print("  vllm ApplyRotaryEmb failed:", repr(e), flush=True)

# ---- SwiGLU ----
S, I = 1764, 2816
gu = torch.randn(S, 2 * I, dtype=torch.bfloat16, device=dev) * 1.0
train_sw = packed_swiglu(gu)
print("SwiGLU")
try:
    from vllm.model_executor.layers.activation import SiluAndMul
    infer_sw = SiluAndMul.forward_native(gu)
    ref_sw = (F.silu(gu.float()[..., :I]) * gu.float()[..., I:]).to(torch.bfloat16)
    rel("train_vs_ref", train_sw, ref_sw)
    rel("infer_native_vs_ref", infer_sw, ref_sw)
    rel("infer_native_vs_train", infer_sw, train_sw)
except Exception as e:
    print("  vllm SiluAndMul failed:", repr(e), flush=True)
