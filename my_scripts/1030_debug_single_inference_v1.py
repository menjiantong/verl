# [1030 调试 v1] 单推理：验证新 vllm/vllm-ascend 对 top32-bf16-noengram 的加载与前向。
# 用法: source my_scripts/1030_ray_dsv41_env.sh; python3 my_scripts/1030_debug_single_inference_v1.py
# 定位完成后删除本脚本。
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram"
TOKENIZER = "/mnt/share/m00899630/dsv41/rl_tokenizer"


def _additional_config():
    # MY_ENV_SINGLE_INFER_MC2 为空表示不启用 MC2 融合通信算法（对照实验用）
    cfg = {"rl_config": {"enabled": True, "enable_batch_invariant": True}}
    mc2 = os.environ.get("MY_ENV_SINGLE_INFER_MC2", "fullmesh_v2")
    if mc2:
        cfg["mc2_comm_alg"] = mc2
    return cfg


tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
messages = [{"role": "user", "content": "2+3等于几？请直接给出答案。"}]
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]
print("[single-infer] prompt_len", len(ids), flush=True)

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
    additional_config=_additional_config(),
)
sp = SamplingParams(temperature=0.0, max_tokens=64, logprobs=1)
out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
print("[single-infer] output:", repr(out.outputs[0].text), flush=True)
lps = out.outputs[0].logprobs
if lps:
    vals = [next(iter(d.values())).logprob for d in lps]
    print("[single-infer] logprob mean", sum(vals)/len(vals), "min", min(vals), "max", max(vals), flush=True)

print("[single-infer] token_ids:", out.outputs[0].token_ids[:64], flush=True)
print("[single-infer] finish_reason:", out.outputs[0].finish_reason, flush=True)
if out.outputs[0].logprobs:
    print("[single-infer] first5_logprob_dicts:", out.outputs[0].logprobs[:5], flush=True)
try:
    cfg = llm.llm_engine.model_config.hf_config
    for k in ["num_hidden_layers", "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "first_k_dense_replace", "moe_layer_freq", "engram_layer_ids", "compress_ratios", "kv_source_layer_ids", "index_source_layer_ids", "num_nextn_predict_layers", "vocab_size"]:
        print("[single-infer] cfg." + k + ":", getattr(cfg, k, "<missing>"), flush=True)
except Exception as e:
    print("[single-infer] cfg print failed:", repr(e), flush=True)
