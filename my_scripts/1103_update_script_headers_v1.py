#!/usr/bin/env python3
# 1103 一次性脚本：把 CP response_from_nested 修复 commit 回填到脚本头
import io

FIX = "8c1518c8"

edits = [
    ("my_scripts/1103_test_cp_response_from_nested_v1.py",
     "# 验证 commit：verl c064ca65（脚本基线），修复 commit 见工作日志",
     "# 验证 commit：verl c064ca65（脚本基线）；修复 commit verl " + FIX),
    ("my_scripts/1103_run_cp_response_from_nested_test_v1.sh",
     "# 验证 commit：verl c064ca65（脚本基线）",
     "# 验证 commit：verl c064ca65（脚本基线）；修复 commit verl " + FIX),
    ("my_scripts/1103_run_dsv41_engram_rl_validate_v4.sh",
     "# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c",
     "# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c\n"
     "# 修复 commit（CP response_from_nested 切片，v1 trainer）：verl " + FIX),
    ("my_scripts/1103_run_dsv41_longrun_v1.sh",
     "# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c",
     "# 验证 commit（依赖基线）：verl 979dfa29、vllm 2f0dec710f、vllm-ascend 69de6ca8c\n"
     "# 修复 commit（CP response_from_nested 切片，v1 trainer）：verl " + FIX),
]

for path, old, new in edits:
    s = io.open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, s.count(old))
    io.open(path, "w", encoding="utf-8").write(s.replace(old, new))
print("HEADERS_OK " + FIX)
