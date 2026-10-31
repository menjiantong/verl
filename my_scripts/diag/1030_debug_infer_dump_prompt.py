# [1031 调试] 用 op-dump 中提取的 255 token prompt 重放推理，补齐
# indexer / 融合 router / lm_head GEMM 的输入输出。定位完成后删除本脚本。
# 用法: MY_ENV_PROMPT_IDS=/path/prompt_ids_255.json python3 -c "exec(open(...).read())"
import json
import os

from vllm import LLM, SamplingParams

MODEL = "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram"
TOKENIZER = "/mnt/share/m00899630/dsv41/rl_tokenizer"

ids = json.load(open(os.environ["MY_ENV_PROMPT_IDS"], encoding="utf-8"))
print("[infer-dump] prompt_len", len(ids), flush=True)

cfg = {"rl_config": {"enabled": True, "enable_batch_invariant": True}}
mc2 = os.environ.get("MY_ENV_SINGLE_INFER_MC2", "fullmesh_v2")
if mc2:
    cfg["mc2_comm_alg"] = mc2

llm = LLM(
    model=MODEL,
    tokenizer=TOKENIZER,
    tokenizer_mode="deepseek_v41",
    trust_remote_code=False,
    dtype="bfloat16",
    max_model_len=1664,
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    gpu_memory_utilization=0.6,
    enforce_eager=True,
    load_format="auto",
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    max_num_batched_tokens=1664,
    additional_config=cfg,
)
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1)
out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
print("[infer-dump] text", repr(out.outputs[0].text), flush=True)
print("[infer-dump] token_ids", out.outputs[0].token_ids[:4], flush=True)
print("[infer-dump] finish_reason", out.outputs[0].finish_reason, flush=True)
