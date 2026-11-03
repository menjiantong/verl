# 1103 DSV4.1 Engram HostUVA 大表探针 v1（单卡）
# 目的：真实 RL 的 per-rank 表是 22.89 GiB（48,001,041 x 256 bf16），
#       小表探针通过不代表大表可用。本探针在单卡上分配/注册同尺寸主机表，
#       并从不连续位置（首、尾行）gather，验证：
#         1) aclrtHostRegisterV2 对 22.89 GiB 是否完整映射；
#         2) 跨 CHUNK_ROWS(1<<22) 的 ptrs 表项是否有效；
#         3) 末尾行（第 48,001,040 行）是否可读。
# 判据：三个 id 的 gather 结果与 CPU 写入值逐元素相等。
# 用法：bash my_scripts/1103_run_dsv41_engram_uva_large_probe_v1.sh [NPU_ID]
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401  # 先导入，避免 device_op <-> ops 循环导入
from vllm_ascend.models.deepseek_v41.engram.npu import (
    HostUvaBuffer,
    gather_dequantize_host_uva,
)


def _log(tag, **kw):
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"MY_DEBUG_LOG engram_uva_large_{tag} {extra}", flush=True)


def main():
    torch.npu.set_device(0)
    rows, width = 48_001_041, 256
    buf = HostUvaBuffer((rows, width), torch.bfloat16, "npu:0")
    try:
        _log("alloc", rows=rows, width=width, gib=round(rows * width * 2 / 1024**3, 2), ptrs=tuple(buf.ptrs.shape))
        ids_list = [0, 1, 4_194_303, 4_194_304, rows - 1]
        for i in ids_list:
            buf.tensor[i].fill_(float(i % 1000))
        ids = torch.tensor(ids_list, dtype=torch.int64, device="npu:0")
        out = gather_dequantize_host_uva(
            buf, None, ids, head_start=0, local_heads=1, pad_heads=1, vocab_start=0, vocab_end=rows
        )
        torch.npu.synchronize()
        got = out.cpu()
        want = buf.tensor[ids.cpu()].float()
        exact = bool(torch.equal(got, want))
        maxdiff = (got.float() - want).abs().max().item()
        _log("gather", ids=ids_list, exact=exact, maxdiff=maxdiff)
        assert exact, f"mismatch maxdiff={maxdiff}"
    finally:
        buf.close()
        _log("closed", ok=1)
    _log("done", ok=1)


if __name__ == "__main__":
    main()
