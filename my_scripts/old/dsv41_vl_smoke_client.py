#!/usr/bin/env python3
"""Query the standalone V4.1 VL server (vllm serve) and print three probes."""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

BASE = "http://80.5.25.118:20007"
DATA = "/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return json.load(resp)


def wait_health(timeout_s: int = 1800) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"[client] healthy after {time.time() - start:.0f}s", flush=True)
                    return
        except Exception:
            pass
        time.sleep(15)
    raise SystemExit("server never became healthy")


def main() -> None:
    wait_health()

    print("=" * 70)
    print("[1] raw completion '35+28='")
    r = post("/v1/completions", {"model": "auto", "prompt": "35+28=", "max_tokens": 32, "temperature": 0})
    print(repr(r["choices"][0]["text"]))

    print("=" * 70)
    print("[2] chat text, thinking")
    r = post(
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 35+28? Answer with only the number."}],
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": True},
        },
    )
    print(repr(r["choices"][0]["message"]["content"][:700]))

    print("=" * 70)
    print("[3] chat image, thinking")
    row = pq.read_table(DATA).slice(0, 1).to_pylist()[0]
    img = Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    question = row["prompt"][0]["content"].replace("<image>", "").strip()
    print("image:", img.size, "| gt:", row["reward_model"]["ground_truth"])
    r = post(
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": True},
        },
    )
    print(repr(r["choices"][0]["message"]["content"][:900]))
    print("usage:", r.get("usage"))


if __name__ == "__main__":
    main()
