#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[1106-单机] 给云道最终基线脚本加两个 MY_ENV_ 覆盖口，默认值逐字不变。

为什么必须加（而且只加这两处）：
  1) TRAIN_FILE / TEST_FILE 是硬编码。124 的 P1 用的是 dapo_math_train_512.parquet，
     云道上也有同名副本；若 1 机跑全量 dapo-math-17k，engram 表的 uniq 行数与
     all_to_all 载荷就和 124 不同，update_actor 没法比。
  2) FREE_CACHE_ENGINE 硬编码 True。1 机要复现 124 的 P1 口径必须 False
     （vLLM 常驻），否则 camem.sleep 的 rtMallocHost 卡死会变成主导变量。

  反向思考：改共享基线脚本有风险 —— 8 机作业此刻正在执行同一个文件。
  但 bash 是顺序读取执行的，8 机日志已打出 python 侧
  `engram layers.14.engram.embed.weight: checkpoint rows [...]`，
  说明 bash 已经执行到最后的 python 启动行，不会再回头读这个文件。
  且两处改写都是 `X=${MY_ENV_...:-原字面量}`，未设环境变量时行为与原来完全一致；
  即使某个节点重新读取，行为也不变。

幂等：两个 sentinel 各自判断，已改过就跳过。
备份：<file>.bak.<TS>

用法：
  python3.11 1106_patch_yundao_base_1node_v1.py --apply --check
"""
from __future__ import annotations

import argparse
import hashlib
import py_compile
import shutil
import sys
import time

BASE = ("/efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/"
        "dsv41_grpo_yundao_math_dapo17k_8_nodes_final.sh")
TS = time.strftime("%Y%m%d_%H%M%S")

DATA_OLD = """TRAIN_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet
TEST_FILE=/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet"""
DATA_NEW = """# [1106-单机] 允许外层覆盖数据集：1 机对照 124 的 P1 用 dapo512，与 8 机 E1 的
# 全量 dapo-math-17k 区分开。不设置时路径与原来逐字相同，行为不变。
TRAIN_FILE=${MY_ENV_DSV41_TRAIN_FILE:-/efs_rl/m00899630/dsv41/rl_data/dapo_math_17k/dapo-math-17k.parquet}
TEST_FILE=${MY_ENV_DSV41_TEST_FILE:-/efs_rl/m00899630/dsv41/rl_data/dapo_math_val_64.parquet}"""

FCE_OLD = "FREE_CACHE_ENGINE=True"
FCE_NEW = """# [1106-单机] 做成开关：1 机对照 124 时用 False（vLLM 常驻，绕开 camem.sleep 的
# rtMallocHost 卡死）；默认 True 不变，8 机行为完全不变。
FREE_CACHE_ENGINE=${MY_ENV_DSV41_FREE_CACHE_ENGINE:-True}"""


def patch(src: str, sentinel: str, old: str, new: str, tag: str) -> tuple[str, bool]:
    if sentinel in src:
        print(f"  [skip] {tag}: already applied")
        return src, False
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ABORT: {tag}: pattern occurs {n}x (expect 1)")
    print(f"  [apply] {tag}")
    return src.replace(old, new, 1), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with open(BASE, "r", encoding="utf-8") as fh:
        src = fh.read()
    before_md5 = hashlib.md5(src.encode()).hexdigest()
    print(f"[base] {BASE}\n[base] md5_before={before_md5} lines={src.count(chr(10))}")

    if args.apply:
        bak = f"{BASE}.bak.{TS}"
        shutil.copy2(BASE, bak)
        print(f"[backup] {bak}")

        src, c1 = patch(src, "MY_ENV_DSV41_TRAIN_FILE", DATA_OLD, DATA_NEW, "dataset override")
        src, c2 = patch(src, "MY_ENV_DSV41_FREE_CACHE_ENGINE", FCE_OLD, FCE_NEW, "free_cache_engine override")

        if c1 or c2:
            with open(BASE, "w", encoding="utf-8") as fh:
                fh.write(src)
            print(f"[write] changed c1={c1} c2={c2}")
        else:
            print("[write] nothing to change")

    with open(BASE, "r", encoding="utf-8") as fh:
        body = fh.read()
    after_md5 = hashlib.md5(body.encode()).hexdigest()
    print(f"[base] md5_after={after_md5}")

    if args.check:
        print("[check] TRAIN_FILE override =", "MY_ENV_DSV41_TRAIN_FILE" in body)
        print("[check] TEST_FILE override  =", "MY_ENV_DSV41_TEST_FILE" in body)
        print("[check] FCE override       =", "MY_ENV_DSV41_FREE_CACHE_ENGINE" in body)
        # 默认值必须逐字仍在（否则改了默认行为）
        print("[check] default dapo17k kept =", "dapo_math_17k/dapo-math-17k.parquet}" in body)
        print("[check] default FCE True kept =", "MY_ENV_DSV41_FREE_CACHE_ENGINE:-True}" in body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
