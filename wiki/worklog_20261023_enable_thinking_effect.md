# `data.apply_chat_template_kwargs.enable_thinking` 是否影响（2026-10-23）

## 1. 结论

**会影响，且是 prompt 级别的实质性影响；但不是 119 这次 Ray 崩溃的原因。**

- `enable_thinking=False` 时，实际使用的 `DeepSeekV41VLProcessor.apply_chat_template` 会调用
  `vllm.tokenizers.deepseek_v41_encoding.encode_messages(..., thinking_mode="chat", drop_thinking=True)`，
  渲染出的 generation prompt 以 `<｜Assistant｜></think>` 结尾 —— think 块在 prompt 中已被关闭，
  模型只能直接作答。
- `enable_thinking=True` 时会渲染 `<｜System｜>Reasoning Effort: 50 (range 1-100, ...)` 前置说明，
  并以 `<｜Assistant｜><think>` 结尾 —— 模型可以进入推理。
- 因此该参数不是 no-op；它会改变模型看到的 prompt 和可用输出空间。

## 2. 代码路径证据

- 配置：`+data.apply_chat_template_kwargs.enable_thinking=false`（119 run 命令行与 resolved config）。
- `verl/experimental/agent_loop/agent_loop.py:246-255`：
  `self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})`，
  并作为 `chat_template_kwargs` 传给 `create_continuous_token_builder`。
- `verl/utils/tokenizer/continuous_token.py:1153,1277`：`template_kwargs = dict(self.chat_template_kwargs)`，
  再 `apply_chat_template(self.processor, messages, ..., **template_kwargs)`。
- `verl/utils/tokenizer/deepseek_v41_vl.py:139-183`：
  `DeepSeekV41VLProcessor.apply_chat_template` 显式接收 `enable_thinking`，并调用
  `encode_messages(rendered_messages, thinking_mode="thinking" if enable_thinking else "chat", drop_thinking=True, reasoning_effort=...)`。

## 3. 实测渲染（同一条 countbenchqa_lite prompt）

```text
enable_thinking=False (chat, 310 chars):
<｜begin▁of▁sentence｜><｜User｜><image>
How many dogs ... oxed{}.<｜Assistant｜></think>

enable_thinking=True (thinking, 410 chars):
<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)

<｜User｜><image>
How many dogs ... oxed{}.<｜Assistant｜><think>
```

- 数据集 prompt 本身要求 “You FIRST think ... enclosed within `<think> </think>` tags”；在 chat 模式下
  模板已追加 `</think>`，与该指令冲突，模型只能直接给答案。
- 与 119 后期的短回答塌缩一致：response_length 从 ~300 降到 25–60，entropy 0.09–0.12，
  reward 仍 0.84–0.90（countbenchqa 只校验最终整数，不检查 think/格式）。
- 但 119 与 123 都使用 False，dummy A/B 不受该变量干扰；Ray 崩溃也与该参数无关。

## 4. 建议

1. 若要训练推理行为：显式设 `enable_thinking=true`，并给 reward 增加 `<think>` 非空/格式检查；
   仅打开 thinking 而 reward 仍只看最终整数，模型仍可能学会空 `<think></think>` 后直接作答。
2. 若要保持非 thinking：同步修改数据 prompt（去掉强制 `<think>` 指令）和 reward，否则 prompt 与模板冲突。
3. 不要用 tokenizer 的 Jinja 模板判断该参数是否生效——本模型实际走 `DeepSeekV41VLProcessor`，
   Jinja 模板（`/mnt/share/m00899630/dsv41/rl_tokenizer/tokenizer_config.json`）根本不引用 `enable_thinking`，
   渲染 True/False 完全相同（309 chars, identical）；真正生效的是自定义 processor。
