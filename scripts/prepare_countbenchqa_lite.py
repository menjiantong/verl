#!/usr/bin/env python3
"""Prepare countbenchqa_lite for the DeepSeek-V4.1 VLM GRPO run.

The upstream parquet is a 500-row evaluation file with an embedded JPEG per row
and a prompt that starts with the literal ``<image>`` marker.  RLHFDataset
splits that marker into an image content part + the remaining text, so the file
needs no schema change.  This script only makes a deterministic train/val split
and keeps every column (prompt/images/reward_model/extra_info) untouched.

Output:
    /mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet (450)
    /mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet   (50)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_SRC = "/mnt/share/m00899630/workspace/mjt_workplace/datasets/countbenchqa_lite/train.parquet"
DEFAULT_OUT_DIR = "/mnt/share/m00899630/dsv41/rl_data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--val-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    table = pq.read_table(args.src)
    rows = table.num_rows
    if not 0 < args.val_size < rows:
        raise SystemExit(f"val_size must be in (0, {rows}), got {args.val_size}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(rows)
    val_idx = np.sort(order[: args.val_size])
    train_idx = np.sort(order[args.val_size :])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "countbenchqa_lite_train.parquet"
    val_path = out / "countbenchqa_lite_val.parquet"
    pq.write_table(table.take(train_idx.tolist()), train_path)
    pq.write_table(table.take(val_idx.tolist()), val_path)

    print(f"src      : {args.src} ({rows} rows)")
    print(f"train    : {train_path} ({len(train_idx)} rows)")
    print(f"val      : {val_path} ({len(val_idx)} rows)")
    print(f"seed     : {args.seed}")
    print(f"schema   : {table.schema.names}")


if __name__ == "__main__":
    main()
