# 1103 DSV4.1 Engram 单推理验证 v1
# 目的：不经过 RL，直接用 vLLM 8 卡 TP=8 加载带 Engram 的完整检查点，验证：
#   1) vllm-ascend 自动解析 EngramConfig（cpu_offload=True，主机 UVA 表）；
#   2) HostUvaBuffer 的 49.2 GB/rank 注册成功；
#   3) 前向/采样可用（engram 层的 hash 查表参与计算）。
# 模式（MY_ENV_DSV41_ENGRAM_LOAD_FORMAT）：
#   dummy = 不读权重，快速验证表分配 + UVA + 前向链路（输出无意义）
#   real  = load_format=auto，真实读盘（494 GB，其中 engram 表 393 GB）
# 用法：bash my_scripts/1103_run_dsv41_engram_infer_validate_v1.sh [dummy|real]
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = os.environ["MY_ENV_DSV41_ENGRAM_MODEL"]
TOKENIZER = "/mnt/share/m00899630/dsv41/rl_tokenizer"
MODE = os.environ.get("MY_ENV_DSV41_ENGRAM_LOAD_FORMAT", "dummy")
assert MODE in ("dummy", "real"), MODE


def _log(tag, **kw):
    # 每行唯一标识符，便于 grep（rule 8）
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"MY_DEBUG_LOG engram_infer_{tag} {extra}", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    messages = [{"role": "user", "content": "2+3等于几？请直接给出答案。"}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]
    _log("prompt", mode=MODE, prompt_len=len(ids))

    additional = {"rl_config": {"enabled": True, "enable_batch_invariant": True}, "mc2_comm_alg": "fullmesh_v2"}
    # [1103 Engram 调试] 允许脚本覆盖多流 engram overlap 开关：
    # 默认（不设置）= 走 vllm-ascend 的硬件默认；0/1 显式关闭/打开。
    _overlap = os.environ.get("MY_ENV_DSV41_ENGRAM_OVERLAP", "")
    if _overlap != "":
        additional["multistream_engram_overlap"] = _overlap == "1"
        _log("overlap_override", value=additional["multistream_engram_overlap"])
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
        load_format="dummy" if MODE == "dummy" else "auto",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1664,
        additional_config=additional,
    )
    cfg = llm.llm_engine.model_config
    eng = getattr(cfg, "engram_config", None)
    _log(
        "loaded",
        mode=MODE,
        engram_layer_ids=getattr(cfg.hf_text_config, "engram_layer_ids", None),
        eng_cpu_offload=getattr(eng, "cpu_offload", None),
        eng_dp_shared=getattr(eng, "dp_shared_memory", None),
        eng_dtype=getattr(eng, "storage_dtype", None),
    )

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
    _log(
        "generate",
        text=repr(out.outputs[0].text),
        n_tokens=len(out.outputs[0].token_ids),
        finish_reason=out.outputs[0].finish_reason,
    )
    _log("done", ok=1)

if __name__ == "__main__":
    # vLLM 的 spawn 子进程会重新导入 __main__，必须有这个 guard，
    # 否则 RuntimeError: An attempt has been made to start a new process...
    main()
