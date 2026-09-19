# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a tokenizer directory for DeepSeek-V4.1 RL runs.

The release checkpoints ship ``tokenizer.json`` / ``tokenizer_config.json`` **without a
chat template** — vLLM serves them through vllm-ascend's own encoder
(``vllm_ascend.patch.platform.patch_deepseek_v41_frontend.encoding``). verl's RL dataset
needs ``apply_chat_template`` at dataset-build time (overlong filtering, agent loop), so
this script writes a sibling tokenizer directory whose ``tokenizer_config.json`` carries
the same single-turn format the reference encoder produces:

    <bos> <｜User｜>{content} <｜Assistant｜><think>

Only a single user turn plus the generation prompt is templated; multi-turn / tool /
reasoning-effort rendering stays with vllm-ascend's encoder (use it directly for full
fidelity). Both actor and rollout consume the token ids built from this directory, so the
RL pipeline is unaffected by the simplification.

Usage:
    python3 scripts/make_dsv41_rl_tokenizer.py \
        --checkpoint /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random \
        --output /mnt/share/m00899630/dsv41/rl_tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<｜User｜>{{ message['content'] }}"
    "{% elif message['role'] == 'system' %}<｜System｜>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜><think>{% endif %}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "MODEL_PATH", "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random"
        ),
        help="Release checkpoint directory holding the tokenizer files.",
    )
    parser.add_argument("--output", required=True, help="Directory to write the RL tokenizer into.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = checkpoint / name
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; is --checkpoint a released checkpoint?")
        shutil.copy2(source, output / name)
        copied.append(name)

    config_path = output / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = CHAT_TEMPLATE
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"wrote {output} ({', '.join(copied)} + chat_template)")

    # Smoke-check the template through the same call verl's dataset makes.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(output))
    messages = [{"role": "user", "content": "Solve 1+1"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    expected = "<｜begin▁of▁sentence｜><｜User｜>Solve 1+1<｜Assistant｜><think>"
    assert prompt == expected, f"unexpected prompt: {prompt!r}"
    tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    token_ids = tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized
    assert len(token_ids) == 9, f"unexpected token count {len(token_ids)}: {token_ids}"
    print(f"prompt ok ({len(token_ids)} tokens): {prompt!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
