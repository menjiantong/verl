# 1103 DSV4.1 Engram HostUVA 探针 v1
# 目的：单独验证 vllm-ascend 的 HostUvaBuffer（aclrtHostRegisterV2 +
#       aclrtHostGetDevicePointer）与 Triton engram gather 内核在 .254 可用。
#       这是 Engram RL 的前置条件：vLLM 侧 engram 表放在注册过的主机内存里，
#       NPU 通过设备地址直接 gather，不走 H2D 拷贝。
# 判据：gather 输出与 CPU 参考逐元素一致（bf16 精确相等），buffer.close() 成功。
# 用法：bash my_scripts/1103_run_dsv41_engram_uva_probe_v1.sh
import torch
import torch_npu  # noqa: F401  # 注册 npu 设备后端

# 先导入 vllm_ascend.ops，避免 device_op <-> ops 的循环导入：
# 直接导入 engram.npu 会先加载 vllm_ascend.device.device_op，它在初始化中
# 反向触发 vllm_ascend.ops.__init__，后者又 import device_op 造成部分初始化。
import vllm_ascend.ops  # noqa: F401

from vllm_ascend.models.deepseek_v41.engram.npu import (
    HostUvaBuffer,
    gather_dequantize_host_uva,
)


def _log(tag, **kw):
    # 每行唯一标识符，便于 grep（rule 8）
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"MY_DEBUG_LOG engram_uva_{tag} {extra}", flush=True)


def main():
    torch.npu.set_device(0)
    rows, width = 256, 256
    buf = HostUvaBuffer((rows, width), torch.bfloat16, "npu:0")
    try:
        ref = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width).to(torch.bfloat16)
        buf.tensor.copy_(ref)  # buf.tensor 是主机侧 pin 内存的 CPU 视图
        ids = torch.tensor([0, 1, 17, 255], dtype=torch.int64, device="npu:0")
        out = gather_dequantize_host_uva(
            buf,
            None,
            ids,
            head_start=0,
            local_heads=1,
            pad_heads=1,
            vocab_start=0,
            vocab_end=rows,
        )
        torch.npu.synchronize()
        got = out.cpu()
        want = ref[ids.cpu()]
        exact = bool(torch.equal(got, want))
        maxdiff = (got.float() - want.float()).abs().max().item()
        _log("gather", rows=rows, width=width, out_shape=tuple(out.shape), exact=exact, maxdiff=maxdiff)
        assert exact, f"gather mismatch maxdiff={maxdiff}"
    finally:
        buf.close()
        _log("closed", ok=1)
    _log("done", ok=1)


if __name__ == "__main__":
    main()
