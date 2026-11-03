# 1103 快速生成自检 v1（隔离 verl，只验 vLLM 栈）
# 目的：用 countbenchqa_lite 验证集第 0 条的 prompt（纯文本、同 RL 模板），
#   让 noengram 基座生成 512 token，检查三件事：
#     1) finish_reason 是否出现 stop（EOS 是否生效）；
#     2) 文本是否连贯、是否给出 \boxed{}；
#     3) 返回的 logprobs 是否含 nan / 是否正常。
# 用法：bash my_scripts/1103_run_quick_vllm_gen_check_v1.sh [model_dir]
import io
import os

import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = os.environ["MY_ENV_DSV41_MODEL"]
TOKENIZER = os.environ["MY_ENV_DSV41_TOKENIZER"]
VAL = os.environ["MY_ENV_DSV41_VAL_PARQUET"]
MAXTOK = int(os.environ.get("MY_ENV_DSV41_MAXTOK", "512"))
USE_IMAGE = os.environ.get("MY_ENV_DSV41_USE_IMAGE", "0") == "1"
# [1103 采样 A/B] 1=打开 batch invariant（会走 vLLM 原生 top-k/top-p），0=走 Ascend random_sample
BATCH_INV = os.environ.get("MY_ENV_DSV41_BATCH_INVARIANT", "1") == "1"


def _log(tag, **kw):
    # 每行唯一标识符，便于 grep（rule 8）
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"MY_DEBUG_LOG quickgen_{tag} {extra}", flush=True)


def summarize(out):
    o = out.outputs[0]
    lps = []
    for lp in (o.logprobs or []):
        if lp is None:
            continue
        lps.extend(lp.values())
    vals = [x.logprob for x in lps]
    nan = sum(1 for v in vals if v != v)
    return dict(
        finish=o.finish_reason,
        n_tokens=len(o.token_ids),
        nan_logprob=nan,
        n_logprob=len(vals),
        lp_min=round(min(vals), 4) if vals else None,
        lp_max=round(max(vals), 4) if vals else None,
        lp_mean=round(sum(vals) / len(vals), 4) if vals else None,
        text=repr(o.text[:300]),
    )


def main():
    df = pd.read_parquet(VAL)
    r = df.iloc[0]
    q = r["extra_info"]["question"]
    _log("dataset", n=len(df), question=repr(q), gt=r["reward_model"]["ground_truth"], image=USE_IMAGE)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    # 直接用数据集原始 prompt（去掉 <image> 前缀），保证与 RL 模板一致
    content = r["prompt"][0]["content"].replace("<image>" + chr(92) + "n", "", 1)
    messages = [{"role": "user", "content": content}]
    text_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]
    _log("prompt", len_ids=len(ids), head=repr(text_prompt[-120:]))

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
        additional_config={"rl_config": {"enabled": True, "enable_batch_invariant": BATCH_INV}, "mc2_comm_alg": "fullmesh_v2"},
    )
    _log("loaded", batch_invariant=BATCH_INV)

    inputs = {"prompt_token_ids": ids}
    if USE_IMAGE:
        # 用数据集里的原图；prompt 里的 <image> 由 tokenizer 编码，vLLM 侧按 image token 展开
        img = Image.open(io.BytesIO(r["images"][0]["bytes"])).convert("RGB")
        inputs = {"prompt": text_prompt, "multi_modal_data": {"image": img}}

    # 分档：区分“采样坏了”与“只有 logprob 上报坏了”；MY_ENV_DSV41_CASE 可只跑其中一档
    cases = {
        "greedy_nologp": ("greedy_nologp", SamplingParams(temperature=0.0, max_tokens=64)),
        "sample_nologp": ("sample_nologp", SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=128)),
        "sample_logp": ("sample_logp", SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=128, logprobs=1)),
        "sample_t07_logp": ("sample_t07_logp", SamplingParams(temperature=0.7, top_p=0.95, top_k=-1, max_tokens=128, logprobs=1)),
    }
    want = os.environ.get("MY_ENV_DSV41_CASE", "all")
    if want == "seq":
        # 顺序验证：先采样 -> 再贪心 -> 再采样（同一引擎）
        cases = (
            ("seq_s1", SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=32, logprobs=1)),
            ("seq_g", SamplingParams(temperature=0.0, max_tokens=32, logprobs=1)),
            ("seq_s2", SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=32, logprobs=1)),
        )
    else:
        cases = tuple(cases.values()) if want == "all" else (cases[want],)
    for tag, sp in cases:
        out = llm.generate([inputs], sp, use_tqdm=False)[0]
        o = out.outputs[0]
        raw = llm.get_tokenizer().decode(o.token_ids, skip_special_tokens=False)
        _log(tag, **summarize(out), ids_head=repr(o.token_ids[:16]), raw=repr(raw[:160]))
    _log("done", ok=1)


if __name__ == "__main__":
    main()
