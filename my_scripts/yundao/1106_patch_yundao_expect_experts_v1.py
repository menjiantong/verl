#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[1106-单机] 把 1103 预检里的 `n_routed_experts == 384` 改成可配，默认 384。

背景：单机对照要用 top32（32 专家 / 493.4 GiB），与 124 同一份模型；
但 1103 第 134 行硬断言 384，作业在预检阶段直接 AssertionError 退出
（2026-11-06 12:11 那次 ma-job-964c5ace 就是这样失败的）。

改法（默认行为逐字不变）：
    assert nexp == 384, "... need the full 384-expert checkpoint"
  →
    _EXPECT = int(os.environ.get("MY_ENV_DSV41_EXPECT_EXPERTS", "384"))
    assert nexp == _EXPECT, "... need the full {_EXPECT}-expert checkpoint"

反向思考：放宽断言会让"误用错权重"更容易发生，所以
  1) 不设环境变量时仍是 384，8 机路径零变化；
  2) 断言消息里打印实际要求值，日志里 `model_check_ok ... experts=` 也照旧打印实际值；
  3) 只在显式设置 MY_ENV_DSV41_EXPECT_EXPERTS 时才放宽。

用法：
  python3 1106_patch_yundao_expect_experts_v1.py --apply --check
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time

TARGET = ("/efs_rl/m00899630/workspace/dsv41-verl/verl/my_scripts/yundao/"
          "1103_run_dsv41_yundao_8node_engram_cp_v1.sh")
TS = time.strftime("%Y%m%d_%H%M%S")

OLD = '''nexp = text.get("n_routed_experts")
assert nexp == 384, f"{root}: n_routed_experts={nexp}, need the full 384-expert checkpoint"'''

NEW = '''nexp = text.get("n_routed_experts")
# [1106-单机] 专家数可配：单机对照 124 用 top32（32 专家），默认仍是 384。
# 不设环境变量时与原来逐字等价；断言消息与日志都会打印实际值，避免误用权重被静默放过。
import os
_EXPECT_EXPERTS = int(os.environ.get("MY_ENV_DSV41_EXPECT_EXPERTS", "384"))
assert nexp == _EXPECT_EXPERTS, (
    f"{root}: n_routed_experts={nexp}, need the full {_EXPECT_EXPERTS}-expert checkpoint")'''

SENTINEL = "MY_ENV_DSV41_EXPECT_EXPERTS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()
    before = hashlib.md5(src.encode()).hexdigest()
    print(f"[target] {TARGET}\n[target] md5_before={before}")

    if args.apply:
        if SENTINEL in src:
            print("  [skip] already applied")
        else:
            n = src.count(OLD)
            if n != 1:
                raise SystemExit(f"ABORT: pattern occurs {n}x (expect 1)")
            bak = f"{TARGET}.bak.{TS}"
            shutil.copy2(TARGET, bak)
            print(f"[backup] {bak}")
            with open(TARGET, "w", encoding="utf-8") as fh:
                fh.write(src.replace(OLD, NEW, 1))
            print("  [apply] expect_experts override")

    with open(TARGET, "r", encoding="utf-8") as fh:
        body = fh.read()
    print(f"[target] md5_after={hashlib.md5(body.encode()).hexdigest()}")
    if args.check:
        print("[check] override present =", SENTINEL in body)
        print("[check] default 384 kept =", 'os.environ.get("MY_ENV_DSV41_EXPECT_EXPERTS", "384")' in body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
