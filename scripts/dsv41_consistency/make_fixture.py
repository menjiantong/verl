"""Build the fixed input fixture shared by every train-vs-inference comparison.

Both stacks (FSDPTurbo training model and vLLM/vllm-ascend) must consume *identical*
token ids, so the fixture is plain token ids, no prompt text, no chat template at
comparison time. We take real DAPO-math prompts, render the single-turn template (the
same one the RL dataset uses, see scripts/make_dsv41_rl_tokenizer.py) and truncate the
token stream to a ladder of lengths that exercises every branch of the V4.1 attention:

    64    - shorter than sliding_window (128)
    200   - window < len < index_topk (512)
    700   - beyond index_topk(512) and beyond the compression boundary (compress_ratios=2)
    1500  - beyond candidate_topk_blocks * candidate_block_size boundaries

Writes /mnt/share/m00899630/dsv41/dump/fixture.json (shared NFS path, both stacks read it).
"""

import argparse
import json

import pandas as pd
from transformers import AutoTokenizer

TARGET_LENS = [64, 200, 700, 1500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/mnt/share/m00899630/dsv41/rl_tokenizer")
    ap.add_argument("--data", default="/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet")
    ap.add_argument("--out", default="/mnt/share/m00899630/dsv41/dump/fixture.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    df = pd.read_parquet(args.data)
    print(f"dataset: {len(df)} rows, columns={list(df.columns)}")

    # DAPO-math prompts are ~90-200 tokens; stack consecutive prompts into one stream so the
    # long fixtures still carry real text (a repeated prompt would make top-k selection
    # degenerate, which is exactly what we want to avoid when comparing sparse attention).
    pool, rows = [], []
    for i, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):  # some parquet revisions store raw strings
            prompt = [{"role": "user", "content": prompt}]
        else:
            prompt = list(prompt)  # parquet hands back a numpy array of dicts
        ids = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):  # BatchEncoding when the input is not a plain list
            ids = list(ids["input_ids"])
        pool.extend(ids)
        rows.append(i)
        if len(pool) >= max(TARGET_LENS):
            break
    print(f"token pool: {len(pool)} tokens from rows {rows[0]}..{rows[-1]}")
    if len(pool) < max(TARGET_LENS):
        raise SystemExit(f"not enough prompt tokens: {len(pool)} < {max(TARGET_LENS)}")

    samples = []
    for target in TARGET_LENS:
        samples.append({"target_len": target, "input_ids": pool[:target]})

    with open(args.out, "w") as f:
        json.dump({"tokenizer": args.tokenizer, "samples": samples}, f)
    for s in samples:
        print(f"target_len={s['target_len']:5d} -> actual {len(s['input_ids']):5d} tokens")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
