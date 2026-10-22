#!/usr/bin/env python3
"""GSM8K eval against a running vLLM serve (before/after SFT comparison).

Prompt rendering matches the RL rollout path: the raw question goes through the
rl_tokenizer chat template -> "<|begin|><|User|>{q}<|Assistant|><think>".
Greedy decode so the before/after diff is model behaviour, not sampling noise.

Records per-sample: finish_reason (stop vs length), generated tokens, extracted
"Answer: N" (or boxed / ####), correctness vs ground_truth. A sample that hits
the token cap without ever producing an Answer line is the "never closes" case
that the loop diagnosis is about.

Usage:
  python scripts/eval_gsm8k_serve.py --port 20007 --n 32 \
      --out /mnt/share/m00899630/dsv41/sft_data/eval_before_top32.jsonl
"""

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request

# The container proxies hijack even localhost HTTP; bypass for vLLM calls.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm_num(s: str):
    if s is None:
        return None
    s = s.strip().rstrip(".")
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return s


def extract_answer(text: str):
    """Last 'Answer: N' wins; fall back to \\boxed{} and '#### N'."""
    for pat in (r"Answer:\s*([^\n]+)", r"\\boxed\{([^}]+)\}", r"####\s*([^\n]+)"):
        hits = re.findall(pat, text)
        if hits:
            return hits[-1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=20007)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tokenizer", default="/mnt/share/m00899630/dsv41/rl_tokenizer")
    ap.add_argument("--test-file", default="/mnt/share/m00899630/dataset/gsm8k_official/main/test-00000-of-00001.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-health", type=int, default=0, help="seconds to wait for /health before starting")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt-style", choices=["plain", "rl", "rl_endthink", "rl_nothink"], default="plain",
                    help="plain: raw GSM8K question. rl: wrap like the dapo RL dataset instructions.")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"

    if args.wait_health > 0:
        t0 = time.time()
        while time.time() - t0 < args.wait_health:
            try:
                with _OPENER.open(base + "/health", timeout=5) as r:
                    if r.status == 200 and b"ok" in r.read().lower():
                        break
            except Exception:
                pass
            time.sleep(15)
        print(f"[eval] server healthy after {time.time() - t0:.0f}s", flush=True)

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    df = pd.read_parquet(args.test_file).head(args.n)

    def run_one(row):
        idx, q, gt = row.name, str(row.question).strip(), str(row.answer)
        if args.prompt_style in ("rl", "rl_endthink", "rl_nothink"):
            q = (
                "Solve the following math problem step by step. The last line of your response should "
                "be of the form Answer: $Answer (without quotes) where $Answer is the answer to the "
                "problem.\n\n" + q + "\n\nRemember to put your answer on its own line after \"Answer:\"."
            )
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        if args.prompt_style == "rl_endthink":
            prompt = prompt.removesuffix("<think>") + "</think>"
        elif args.prompt_style == "rl_nothink":
            prompt = prompt.removesuffix("<think>")
        t0 = time.time()
        try:
            resp = post(
                base,
                "/v1/completions",
                {
                    "model": "auto",
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": 1.0,
                },
                timeout=1800,
            )
            ch = resp["choices"][0]
            text = ch.get("text", "")
            finish = ch.get("finish_reason")
        except Exception as e:  # keep the before/after files comparable even on error
            text, finish = f"<request error: {e}>", "error"
        pred = extract_answer(text)
        gt_num = norm_num(gt.rsplit("####", 1)[-1])
        return {
            "idx": int(idx),
            "question": q,
            "ground_truth": gt_num,
            "finish_reason": finish,
            "hit_cap": finish == "length",
            "has_answer": pred is not None,
            "pred": pred,
            "correct": norm_num(pred) == gt_num if pred is not None else False,
            "chars": len(text),
            "elapsed_s": round(time.time() - t0, 1),
            "text_tail": text[-600:],
        }

    rows = list(df.iterrows())
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda r: run_one(r[1]), rows))

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    stopped = sum(r["finish_reason"] == "stop" for r in results)
    has_ans = sum(r["has_answer"] for r in results)
    correct = sum(r["correct"] for r in results)
    print(f"[eval] n={n} closed(stop)={stopped} has_answer={has_ans} correct={correct}")
    print(f"[eval] saved -> {args.out}")
    for r in results[:3]:
        print(f"--- [{r['idx']}] finish={r['finish_reason']} pred={r['pred']} gt={r['ground_truth']}")
        print(f"    tail: {r['text_tail'][-220:]!r}")


if __name__ == "__main__":
    main()
