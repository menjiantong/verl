#!/usr/bin/env python3
"""Standalone V4.1 VL smoke test (no verl trainer, no weight sync).

Loads the checkpoint with vLLM directly (load_format=auto) on 8 chips and asks:
  1. raw completion  "35+28="                      -> base-model coherence
  2. official-encoder text chat                    -> chat/thinking format
  3. official-encoder image chat (countbench row)  -> vision path

If (1) is coherent and (3) is not, the prompt/vision path is at fault; if (1) is
already incoherent, the checkpoint is a base model that the RL prompt format
cannot drive.

Usage (on a free node):
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 dsv41_vl_smoke.py
"""

from __future__ import annotations

import io
import os
import sys

import pyarrow.parquet as pq
import torch
from PIL import Image

MODEL = "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram"
DATA = "/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet"


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages

    print(f"[smoke] loading {MODEL} ...", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v41",
        load_format="auto",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        additional_config={"enable_cpu_binding": True},
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, ignore_eos=False)

    print("=" * 70)
    print("[1] raw completion: '35+28='", flush=True)
    out = llm.generate(["35+28="], SamplingParams(temperature=0.0, max_tokens=32))
    print(repr(out[0].outputs[0].text))

    print("=" * 70)
    print("[2] official encoder, text thinking chat", flush=True)
    text = encode_messages(
        [{"role": "user", "content": "What is 35+28? Answer with only the number."}],
        thinking_mode="thinking",
        reasoning_effort="high",
    )
    out = llm.generate([text], sp)
    print("prompt tail:", repr(text[-80:]))
    print(repr(out[0].outputs[0].text[:600]))

    print("=" * 70)
    print("[3] official encoder, image thinking chat", flush=True)
    row = pq.read_table(DATA).slice(0, 1).to_pylist()[0]
    img = Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB")
    question = row["prompt"][0]["content"].replace("<image>", "").strip()
    text = encode_messages(
        [{"role": "user", "content": f"{IMAGE_PLACEHOLDER}\n{question}"}],
        thinking_mode="thinking",
        reasoning_effort="high",
    )
    print("image size:", img.size, "| question:", question[:90])
    out = llm.generate(
        [{"prompt": text, "multi_modal_data": {"image": [img]}}],
        sp,
    )
    print(repr(out[0].outputs[0].text[:800]))


if __name__ == "__main__":
    main()
