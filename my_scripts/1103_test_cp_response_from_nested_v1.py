#!/usr/bin/env python3
# =============================================================================
# 1103 CP 补齐下 response_from_nested 切片语义单测 v1
#
# 背景：v1 trainer（trainer_base.py:1640/1641/1689/1711）用
#   response_from_nested(data.pop("log_probs"), data["response_mask"])
# 把嵌套输出切成响应；该函数按 offsets[1:]（含每条样本末尾 CP pad）切片，
# 未扣除 pad。B6 修的是 no_padding_2_padding（v0 路径），v1 未覆盖。
#
# 本单测用 3 条样本（prompt/resp/pad 各不同）复现：
#   want = values[prompt_len-1 : prompt_len-1+resp_len]
#   got  = values[seq_offset-resp_len-1 : seq_offset-1]（seq_offset 含 pad）
# 期望修复前 ok=0、修复后 ok=1。
#
# 用法：bash my_scripts/1103_run_cp_response_from_nested_test_v1.sh
# 验证 commit：verl c064ca65（脚本基线）；修复 commit verl 8c1518c8
# =============================================================================
import os

import torch

from verl.workers.utils.padding import response_from_nested


def jagged(rows):
    lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
    vals = torch.cat([torch.tensor(r, dtype=torch.long) for r in rows])
    off = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    return torch.nested.nested_tensor_from_jagged(vals, off)


prompt_lens = [3, 4, 5]
resp_lens = [2, 3, 1]
pad_multiple = int(os.environ.get("MY_ENV_DSV41_CP_PAD_MULTIPLE", "4") or "4")

rows, want, prompts = [], [], []
for i, (pl, rl) in enumerate(zip(prompt_lens, resp_lens, strict=True)):
    total = pl + rl
    padded = (total + pad_multiple - 1) // pad_multiple * pad_multiple
    pad = padded - total
    vals = torch.arange(i * 100, i * 100 + padded, dtype=torch.long)
    rows.append(vals.tolist())
    prompts.append(list(range(i * 10, i * 10 + pl)))
    # log_probs 左移一位：首个响应 token 的 logprob 在 prompt_len-1
    want.append(vals[pl - 1 : pl - 1 + rl].tolist())

tensor = jagged(rows)
resp_mask = jagged([[1] * rl for rl in resp_lens])
prompt_t = jagged(prompts)

got = [t.tolist() for t in response_from_nested(tensor, resp_mask).unbind()]
print(f"MY_DEBUG_LOG cp_resp_test_001 pad_multiple={pad_multiple} got={got}")
print(f"MY_DEBUG_LOG cp_resp_test_002 want={want}")
print(f"MY_DEBUG_LOG cp_resp_test_003 ok={int(got == want)}")

# 修复后：传入 prompts 时应按真实内容末端切片
if "prompts" in response_from_nested.__code__.co_varnames:
    got2 = [t.tolist() for t in response_from_nested(tensor, resp_mask, prompts=prompt_t).unbind()]
    print(f"MY_DEBUG_LOG cp_resp_test_004 with_prompts_ok={int(got2 == want)} got={got2}")
else:
    print("MY_DEBUG_LOG cp_resp_test_004 with_prompts_ok=not_supported")
