# Copyright 2026
"""Reward for CountBenchQA rollouts (DeepSeek-V4.1 ``<think>`` / boxed format).

The dataset prompt asks the model to reason inside ``<think>...</think>`` and
to put the final integer answer in ``\\boxed{...}``.  Scoring therefore takes
the *last* boxed integer in the response.  If the model never emits a boxed
answer, we fall back to the last standalone integer after the final
``</think>`` (or the last line when no thinking block was closed) so an
occasionally format-broken but numerically correct answer still gets credit.

The returned dict is consumed by ``default_compute_score``: ``score`` is the
scalar reward, every other key is logged as reward-extra info.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(
    r"\\boxed\s*(?:\{\s*([+-]?\d+)\s*\}|\(\s*([+-]?\d+)\s*\)|([+-]?\d+))",
    re.IGNORECASE,
)
_STRICT_FINAL_RE = re.compile(r".*\\boxed\s*\{\s*[+-]?\d+\s*\}\s*$", re.DOTALL | re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
# Integers at the end of a sentence are common ("The final answer is 2."), so a
# trailing period must not disqualify the match; decimals ("2.5") still must.
_STANDALONE_INT_RE = re.compile(r"(?<![\d.])[+-]?\d+(?!\d|[A-Za-z]|\.\d)")


def _last_boxed_int(text: str) -> int | None:
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    groups = matches[-1].groups()
    value = next(group for group in groups if group is not None)
    return int(value)


def _last_int_after_think(text: str) -> int | None:
    closes = list(_THINK_END_RE.finditer(text))
    tail = text[closes[-1].end() :] if closes else text
    numbers = _STANDALONE_INT_RE.findall(tail)
    return int(numbers[-1]) if numbers else None


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.1,
    **kwargs,
) -> dict[str, float | int | bool | None]:
    """Score one CountBenchQA response.

    Args:
        solution_str: decoded model response (prompt excluded).
        ground_truth: dataset answer, a decimal integer string.
        format_score: weight of the "ends with a boxed integer" term.

    Returns:
        dict with ``score`` = (1-format_score)*accuracy + format_score*format.
    """
    del kwargs
    try:
        expected = int(str(ground_truth).strip())
    except (TypeError, ValueError):
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "predicted": None,
            "expected": None,
            "boxed": False,
        }

    response = str(solution_str)
    predicted = _last_boxed_int(response)
    boxed = predicted is not None
    if predicted is None:
        predicted = _last_int_after_think(response)

    accuracy = float(predicted == expected)
    valid_format = float(bool(_STRICT_FINAL_RE.fullmatch(response.strip())))
    score = (1.0 - format_score) * accuracy + format_score * valid_format
    return {
        "score": score,
        "acc": accuracy,
        "format": valid_format,
        "predicted": predicted,
        "expected": expected,
        "boxed": boxed,
    }


if __name__ == '__main__':
    import json
    # 替换为你的 jsonl 文件路径
    file_path = "infer_rollout_top16_se_32.jsonl"

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去除行尾的换行符并跳过空行
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
                # 你可以在这里直接处理单行数据，例如：print(item['name'])

    # 查看读取到的前 2 条数据
    print(f"共读取了 {len(data)} 条数据")
    for d in data:
        print('----------begin--------------')
        # print(d)
        
        # if (d['index'] == 22):
        print(d['index'])
        print(d['answer'])
        print(d['prediction'])
        print('---------output end---------------')
        print(compute_score(d['prediction'], d['answer']))
        # else:
        #     print(compute_score(d['prediction'], d['answer']))