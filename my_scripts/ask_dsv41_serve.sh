#!/usr/bin/env bash
# Ask the running V4.1 serve instance a fixed set of prompts, greedy, short.
#
# Why these four: the 10-08 symptom was a repetition loop on "What is 35+28?".
# An instruction-style prompt alone cannot tell a broken KV cache apart from a
# base checkpoint that was never aligned to instructions (this one has no
# chat_template in tokenizer_config.json), so the same run gets:
#   instruct   the user's own phrasing, for direct comparison with their run
#   raw-eq     "35+28=" -- how a base model is actually meant to be asked
#   factual    "The capital of France is" -- the shortest possible well-known continuation
#   long       >128 tokens, so the compressed index-K / long-KV planes span more
#              than one block and the marker-row path is exercised repeatedly
# All at temperature=0 so sampling cannot explain the result.
#
# Usage: bash scripts/ask_dsv41_serve.sh [port]     (waits for /health first)
set -uo pipefail

PORT=${PORT:-20007}
PORT=${1:-$PORT}
BASE="http://127.0.0.1:${PORT}"
TIMEOUT=${REQUEST_TIMEOUT:-600}

ask() {
    local label="$1" prompt="$2" max_tokens="${3:-48}"
    echo "----- ${label}"
    python3 - "$BASE" "$prompt" "$max_tokens" "$label" <<'PY'
import json, sys, urllib.request
base, prompt, max_tokens, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({
    "model": "auto", "prompt": prompt, "max_tokens": max_tokens,
    "temperature": 0, "n": 1,
}).encode()
req = urllib.request.Request(base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
except Exception as exc:  # keep the harness useful even on failure
    print(f"{label}: REQUEST FAILED {type(exc).__name__}: {str(exc)[:200]}")
    sys.exit(0)
choice = data["choices"][0]
text = choice["text"]
# Repetition is the symptom under test: report the longest run of a repeated
# word rather than eyeballing the string, so "looped" is a number, not a vibe.
words = text.split()
runs, best, best_word = {}, 0, ""
prev, streak = None, 0
for w in words:
    streak = streak + 1 if w == prev else 1
    prev = w
    if streak > best:
        best, best_word = streak, w
print(f"{label}: finish={choice.get('finish_reason')} tokens={data['usage']['completion_tokens']} "
      f"max_repeat={best} of '{best_word[:24]}'")
print(f"{label}: text={text[:400]!r}")
PY
}

echo "[ask] waiting for ${BASE}/health ..."
until curl -sf "${BASE}/health" >/dev/null 2>&1; do sleep 10; done
echo "[ask] healthy at $(date +%H:%M:%S)"

ask "instruct" "What is 35+28? Answer with only the number."
ask "raw-eq"   "35+28="
ask "factual"  "The capital of France is"
ask "long"     "$(python3 -c 'print("A math class writes: 2 + 2 = 4. " * 40)')" 64

echo "[ask] done at $(date +%H:%M:%S)"
