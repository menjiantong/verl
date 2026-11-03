# 1103 CP 补齐切片单测 v1（纯 CPU，不需要 NPU）
# 目的：锁定 no_padding_2_padding 在“每条样本末尾补 pad_token”时的正确切片语义。
# 背景：CP 的 _cp_pad_jagged 把 pad 追加在 prompt+response 之后，所以响应
#       不在补齐序列的尾部；values 是补齐后布局，切片必须用真实内容末端
#       P_i+prompt_i+response_i，而不是 P_i+padded_i。
# 判据：每条样本取出的 resp_len 个 logprob 与原始 values 对应位置逐元素相等。
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding


def _round_up(x, m):
    return (x + m - 1) // m * m


def main():
    prompt_lens = [5, 3, 7, 4]
    resp_lens = [2, 4, 1, 3]
    multiple = 4

    prompts = torch.nested.as_nested_tensor(
        [torch.arange(100 * i, 100 * i + n) for i, n in enumerate(prompt_lens)], layout=torch.jagged
    )
    responses = torch.nested.as_nested_tensor(
        [torch.arange(100 * i + 50, 100 * i + 50 + n) for i, n in enumerate(resp_lens)], layout=torch.jagged
    )
    padded = []
    for p, r in zip(prompts.unbind(), responses.unbind()):
        seq = torch.cat([p, r])
        pad = _round_up(seq.numel(), multiple) - seq.numel()
        padded.append(torch.cat([seq, torch.full((pad,), -1, dtype=seq.dtype)]))
    input_ids = torch.nested.as_nested_tensor(padded, layout=torch.jagged)

    data = TensorDict({"prompts": prompts, "responses": responses, "input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor_data(data, "max_response_len", max(resp_lens))

    total = sum(t.numel() for t in padded)
    raw = torch.arange(total, dtype=torch.float32)
    values = torch.nested.nested_tensor_from_jagged(raw, input_ids.offsets())

    got = no_padding_2_padding(values, data)
    want = torch.zeros(len(prompt_lens), max(resp_lens))
    start = 0
    for i, (pl, rl) in enumerate(zip(prompt_lens, resp_lens)):
        # 样本 i 的响应 logprob 在补齐布局中的位置：[P_i+pl-1, P_i+pl+rl-1)
        want[i, :rl] = raw[start + pl - 1 : start + pl + rl - 1]
        start += _round_up(pl + rl, multiple)

    print("MY_DEBUG_LOG cp_pad_test_001 got=", got.tolist(), flush=True)
    print("MY_DEBUG_LOG cp_pad_test_002 want=", want.tolist(), flush=True)
    assert torch.equal(got, want), f"mismatch\n got={got}\nwant={want}"
    print("MY_DEBUG_LOG cp_pad_test_003 ok=1", flush=True)


if __name__ == "__main__":
    main()
