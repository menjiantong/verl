# [1031 调试] 对比视觉塔 attention 三种实现：训练默认 SDPA、训练融合
# FlashAttentionScore、推理 FIA(npu_fused_infer_attention_score)。
# 目的：量化训练/推理视觉 attention 的数值差，判断是否启用融合算子能减小误差。
# 用法：source my_scripts/1030_ray_dsv41_env.sh && python3 my_scripts/diag/1031_compare_vision_ops.py
import math
import torch
import torch.nn.functional as F
import torch_npu

torch.manual_seed(0)
torch.npu.set_device(0)
dev = "npu:0"
S, N, D = 1764, 16, 64
scale = 1.0 / math.sqrt(D)
FULL = 2**31 - 1


def metrics(name, a, b):
    a = a.float().flatten(); b = b.float().flatten()
    rel = float((a - b).norm() / (b.norm() + 1e-8))
    cos = float(F.cosine_similarity(a, b, dim=0))
    print(f"  {name}: rel={rel:.6f} cos={cos:.7f}", flush=True)
    return rel, cos


def run_case(tag, outl=False):
    q = torch.randn(S, N, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(S, N, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(S, N, D, dtype=torch.bfloat16, device=dev)
    if outl:
        # 模拟视觉特征中的少量大值：只改少数通道，观察 softmax 饱和时的差异。
        q[0, 0, 0] = 30.0; k[0, 0, 0] = 30.0; v[1, 1, 1] = 30.0
        q[100, 2, 3] = -25.0; k[100, 2, 3] = 25.0
    sdpa = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1)
    fa = torch_npu.npu_fusion_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), N,
        input_layout="BSND", pse=None, padding_mask=None, atten_mask=None,
        scale=scale, keep_prob=1.0, pre_tockens=FULL, next_tockens=FULL,
        inner_precise=0, prefix=None, sparse_mode=0,
    )[0].squeeze(0)
    fia, _ = torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=N, num_key_value_heads=N, input_layout="TND",
        block_size=128, actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        scale=scale, sparse_mode=0, pre_tokens=FULL, next_tokens=FULL,
    )
    qf, kf, vf = q.float(), k.float(), v.float()
    ref = (torch.softmax((qf @ kf.transpose(1, 2)) * scale, dim=-1) @ vf)
    print(f"CASE {tag} outl={outl}", flush=True)
    metrics("sdpa-vs-ref", sdpa, ref)
    metrics("fa-vs-ref", fa, ref)
    metrics("fia-vs-ref", fia, ref)
    metrics("fa-vs-fia(train-fused vs infer)", fa, fia)
    metrics("sdpa-vs-fia(train-default vs infer)", sdpa, fia)
    metrics("sdpa-vs-fa(train-default vs train-fused)", sdpa, fa)


run_case("normal", False)
run_case("outlier", True)
