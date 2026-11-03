#!/usr/bin/env python3
# 1103 一次性补丁脚本：让 response_from_nested 支持 CP 末尾 pad 修正
import ast

p = "verl/workers/utils/padding.py"
s = open(p).read()
old = '''def response_from_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Extract response from nested model output.

    Args:
        tensor: a nested tensor with shape (bsz, prompt_len + response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)'''
new = '''def response_from_nested(
    tensor: torch.Tensor, response_mask: torch.Tensor, prompts: torch.Tensor | None = None
) -> torch.Tensor:
    """Extract response from nested model output.

    Args:
        tensor: a nested tensor with shape (bsz, prompt_len + response_len [ + cp_pad])
        response_mask: a nested tensor with shape (bsz, response_len)
        prompts: optional nested prompts; when given, strips the CP tail padding
            (see transformer_impl._cp_pad_jagged) before slicing.

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    # [1103 CP] CP 对齐会在每条样本末尾补 pad_token（transformer_impl:_cp_pad_jagged），
    # 因此 tensor 的 offsets 含 pad；真实内容末端 = 累计(prompt+resp) + 累计(pad) - pad。
    # 未开启补齐（pad 全 0）时与旧行为完全一致。
    if prompts is not None and prompts.is_nested:
        prompt_lens = prompts.offsets().diff().to(response_lens.device)
        seq_lens = prompt_lens + response_lens
        pad_lens = offsets.diff() - seq_lens
        assert pad_lens.min().item() >= 0, f"[1103 CP] negative pad: {pad_lens.tolist()}"
        content_ends = seq_lens.cumsum(0) + pad_lens.cumsum(0) - pad_lens
    else:
        content_ends = offsets[1:]
    response_list = []
    for resp_len, seq_offset in zip(response_lens, content_ends, strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)'''
assert s.count(old) == 1, ("padding old count", s.count(old))
open(p, "w").write(s.replace(old, new))

p2 = "verl/trainer/ppo/v1/trainer_base.py"
t = open(p2).read()
reps = [
    ('fields = ["entropy", "log_probs", "response_mask"]',
     'fields = ["entropy", "log_probs", "response_mask", "prompts"]'),
    ('data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])',
     'data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"], data.get("prompts"))'),
    ('data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])',
     'data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"], data.get("prompts"))'),
    ('select_fields=["log_probs", "response_mask"]',
     'select_fields=["log_probs", "response_mask", "prompts"]'),
    ('data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])',
     'data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"], data.get("prompts"))'),
    ('select_fields=["values", "response_mask"]',
     'select_fields=["values", "response_mask", "prompts"]'),
    ('data["values"] = response_from_nested(data.pop("values"), data["response_mask"])',
     'data["values"] = response_from_nested(data.pop("values"), data["response_mask"], data.get("prompts"))'),
]
for o, n in reps:
    assert t.count(o) == 1, ("trainer old count", o, t.count(o))
    t = t.replace(o, n)
open(p2, "w").write(t)
ast.parse(open(p).read())
ast.parse(open(p2).read())
print("PATCH_OK")
