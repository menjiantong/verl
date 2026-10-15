#!/usr/bin/env python3
"""Convert the official GSM8K parquets into verl-GRPO RL data (GSM8K scorer schema).

Why this schema: rows are `{data_source, prompt, ability, reward_model, extra_info}`.
[2026-10-14 fix] `data_source` is now 'openai/gsm8k' so `default_compute_score`
dispatches to the GSM8K scorer (verl/utils/reward_score/gsm8k.py, run with
method="flexible": last number in the answer tail, trailing "."/","/"$" stripped).
The previous version copied the dapo wiring ('math_dapo' + 'rule-lighteval/MATH_v2'),
which routed every GSM8K rollout through the DAPO/MATH scorer -- wrong extraction
semantics for this protocol (e.g. "Answer: 18." scored as WRONG there).

Differences from prepare_gsm8k_sft.py (that one is for SFT `messages` supervision):
  * reads the RAW official `question`/`answer` columns (not the SFT conversion),
  * rewrites the trailing ``#### N`` into ``ground_truth=N`` (commas stripped),
  * emits one row per question with the RL instruction wrapper (same wording as the
    dapo file and as the served-model eval prompts; the SFT'd model was verified to
    follow it: 15/16 clean stops on GSM8K in the rl-prompt style, 2026-10-11).

Usage:
    python3 scripts/prepare_gsm8k_rl.py \
        --src /mnt/share/m00899630/dataset/gsm8k_official \
        --dst /mnt/share/m00899630/dsv41/rl_data
    # -> {dst}/gsm8k_main_train_rl.parquet  (7473 rows)
    #    {dst}/gsm8k_main_test_rl.parquet   (1319 rows, for data.val_files)
"""

from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response should "
    "be of the form Answer: $Answer (without quotes) where $Answer is the answer to the "
    "problem.\n\n"
)
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def official_to_gt(raw_answer: str) -> str | None:
    """Official answers end with ``#### 72`` (may carry thousands separators)."""
    tail = raw_answer.strip().rsplit("####", 1)
    if len(tail) != 2:
        return None
    gt = tail[1].strip().replace(",", "")
    return gt if re.fullmatch(r"-?\d+(\.\d+)?", gt) else None


def convert_one(src: Path, dst: Path, limit: int | None) -> dict:
    df = pd.read_parquet(src)
    rows, dropped = [], 0
    for _, r in df.iterrows():
        gt = official_to_gt(str(r["answer"]))
        if gt is None:
            dropped += 1
            continue
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": np.array(
                    [{"content": INSTRUCTION + str(r["question"]).strip() + SUFFIX, "role": "user"}],
                    dtype=object,
                ),
                "ability": "MATH",
                "reward_model": {"ground_truth": gt, "style": "rule"},
                "extra_info": {"index": str(uuid.uuid4())},
            }
        )
        if limit and len(rows) >= limit:
            break
    out = pd.DataFrame(rows)
    out.to_parquet(dst, index=False)
    return {"file": dst.name, "rows": len(out), "dropped": dropped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/mnt/share/m00899630/dataset/gsm8k_official")
    ap.add_argument("--dst", default="/mnt/share/m00899630/dsv41/rl_data")
    ap.add_argument("--limit", type=int, default=0, help="rows per split (0 = all)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    limit = args.limit or None
    report = []
    for split in ("train", "test"):
        f = src / "main" / f"{split}-00000-of-00001.parquet"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        r = convert_one(f, dst / f"gsm8k_main_{split}_rl.parquet", limit)
        report.append(r)
        print(f"[ok] {r['file']}: {r['rows']} rows ({r['dropped']} dropped w/o numeric ####)")

    # Self-check: schema identical to the production dapo file, and a sample row
    ref = Path("/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet")
    if ref.exists() and report:
        ref_cols = list(pd.read_parquet(ref).columns)
        got = pd.read_parquet(dst / report[0]["file"])
        assert list(got.columns) == ref_cols, f"schema mismatch: {list(got.columns)} vs {ref_cols}"
        s = got.iloc[0]
        print(f"[check] columns match dapo: {ref_cols}")
        print(f"[check] sample gt={s['reward_model']['ground_truth']!r} prompt_tail={s['prompt'][0]['content'][-60:]!r}")


if __name__ == "__main__":
    main()
