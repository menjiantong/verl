#!/usr/bin/env python3
"""GSM8K official parquet (question/answer) -> verl MultiTurnSFTDataset 'messages' parquet.

Conversion rules (keep SFT targets consistent with the RL "Answer:" protocol):
  * answer: drop the ``<<expr=res>>`` calculator annotations -- the official
    text repeats the result right after each annotation, so the sentence stays
    intact; rewrite the trailing ``#### N`` into a final ``Answer: N`` line,
    which is exactly the format the RL prompt asks the model to produce.
  * messages: [{'role': 'user',      'content': question},
               {'role': 'assistant', 'content': cleaned_answer}]
    The assistant turn gets its <|end of sentence|> from the tokenizer's chat
    template when rendered -- that EOS-in-target is the point of this SFT
    (the pruned checkpoint currently never emits EOS).
  * rows whose rendered conversation exceeds --max-length tokens are dropped
    (the trainer config uses truncation: error, so overlong rows must go now).

Usage:
  python scripts/prepare_gsm8k_sft.py \
      --src /mnt/share/m00899630/dataset/gsm8k_official \
      --dst /workspace-verl/data/gsm8k_sft \
      --tokenizer /mnt/share/m00899630/dsv41/rl_tokenizer
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_answer(answer: str) -> str:
    answer = re.sub(r"<<[^>]*?>>", "", answer).strip()
    if "####" in answer:
        head, num = answer.rsplit("####", 1)
        answer = head.rstrip() + "\nAnswer: " + num.strip()
    return answer


def convert_one(src: Path, dst: Path, tok, max_length: int) -> dict:
    df = pd.read_parquet(src)
    rows = []
    dropped = 0
    for _, r in df.iterrows():
        msgs = [
            {"role": "user", "content": str(r["question"]).strip()},
            {"role": "assistant", "content": clean_answer(str(r["answer"]))},
        ]
        text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        n = len(tok.encode(text, add_special_tokens=False))
        if n > max_length:
            dropped += 1
            continue
        rows.append({"messages": msgs, "data_source": "gsm8k", "token_len": n})
    out = pd.DataFrame(rows)
    out.to_parquet(dst, index=False)
    lens = out["token_len"]
    return {
        "file": dst.name,
        "rows": len(out),
        "dropped": dropped,
        "tok_min": int(lens.min()) if len(out) else 0,
        "tok_mean": round(float(lens.mean()), 1) if len(out) else 0,
        "tok_p95": int(lens.quantile(0.95)) if len(out) else 0,
        "tok_max": int(lens.max()) if len(out) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/share/m00899630/dataset/gsm8k_official")
    ap.add_argument("--dst", default="/workspace-verl/data/gsm8k_sft")
    ap.add_argument("--tokenizer", default="/mnt/share/m00899630/dsv41/rl_tokenizer")
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("main/train-00000-of-00001.parquet", "gsm8k_main_train_sft.parquet"),
        ("main/test-00000-of-00001.parquet", "gsm8k_main_test_sft.parquet"),
        ("socratic/train-00000-of-00001.parquet", "gsm8k_socratic_train_sft.parquet"),
        ("socratic/test-00000-of-00001.parquet", "gsm8k_socratic_test_sft.parquet"),
    ]
    stats = []
    for rel, name in jobs:
        stats.append(convert_one(Path(args.src) / rel, dst / name, tok, args.max_length))
    print(pd.DataFrame(stats).to_string(index=False))

    # human spot-check: first sample of main train, rendered
    df = pd.read_parquet(dst / "gsm8k_main_train_sft.parquet")
    msgs = list(df.iloc[0]["messages"])
    text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    print("\n=== sample[0] rendered (tail 400 chars) ===")
    print(text[-400:])
    print("\n=== sample[0] assistant content ===")
    print(msgs[1]["content"])


if __name__ == "__main__":
    main()
