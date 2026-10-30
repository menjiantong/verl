# Copyright 2026
"""Processor glue for the DeepSeek-V4.1 vision-language checkpoints.

The released checkpoint ships image preprocessing as plain functions
(``image_processor.py``), not a HuggingFace ``ProcessorMixin``: transformers'
``AutoProcessor`` therefore has nothing to load and verl would create the
dataset with ``processor=None``, which cannot place image spans.

This module adapts FSDPTurbo's reference implementation (the same math vLLM's
``DeepseekV4VLMultiModalProcessor`` ports) behind the small interface verl
needs:

* ``apply_chat_template(messages, add_generation_prompt=True, tokenize=False)``
  renders the V4.1 role markers via :func:`verl.utils.tokenizer.deepseek.encode_messages`.
  Image content parts become ``<｜deepseek_image｜>`` placeholders.
* ``__call__(text=..., images=...)`` expands every placeholder into
  ``[IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h + [IMAGE_END]``
  and returns, per sample, the reference model's VL inputs:
  ``input_ids`` / ``attention_mask`` / ``token_types`` / ``images``.

The expansion is identical to what the vLLM rollout engine does, so the
trainer recomputes probabilities on exactly the token stream the engine
generated from.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import BatchFeature

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
TEXT_TYPE = -1


class _ImageProcessorProxy:
    """Minimal ``image_processor`` surface looked up by verl's dataset/agent loop."""

    def __init__(self, patch_size: int):
        self.patch_size = int(patch_size)


class VLBatchFeature(BatchFeature):
    """``BatchFeature`` whose ``images`` entry stays a list of per-sample lists.

    ``convert_to_tensors`` would try to stack the variable-shaped per-image
    records; they are not tensors and are consumed as objects by the FSDPTurbo
    adapter, so keep them out of the conversion.
    """

    def convert_to_tensors(self, tensor_type=None):
        images = self.data.pop("images", None)
        try:
            converted = super().convert_to_tensors(tensor_type=tensor_type)
        finally:
            if images is not None:
                self.data["images"] = images
        if images is not None:
            converted.data["images"] = images
        return converted


def _flatten_content(content: Any) -> str:
    """Render a message content (str or typed part list) as text with image markers.

    Matches vLLM's ``_normalize_messages``: content parts are joined with
    ``"\\n\\n"`` and image parts become ``<｜deepseek_image｜>``.  Byte-exact
    agreement with the reference encoder matters here: the RL prompt must be
    the same string the vLLM tokenizer-mode would render.
    """
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"image", "image_url", "input_image", "image_pil"}:
            pieces.append(IMAGE_PLACEHOLDER)
        elif part_type in {"text", "input_text", "output_text"}:
            pieces.append(part.get("text", ""))
    return "\n\n".join(pieces)


class DeepSeekV41VLProcessor:
    """Processor facade backed by FSDPTurbo's reference VL preprocessing."""

    def __init__(self, tokenizer, model_args, *, thinking: bool = False):
        from fsdp_turbo.models.deepseek_v41.image_processor import IMAGE_PLACEHOLDER as _PLACEHOLDER

        if _PLACEHOLDER != IMAGE_PLACEHOLDER:
            raise ValueError(f"Image placeholder mismatch: {_PLACEHOLDER!r} != {IMAGE_PLACEHOLDER!r}")
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.thinking = bool(thinking)
        self.image_token_id = int(model_args.image_token_id)
        self.image_token = IMAGE_PLACEHOLDER
        self.image_processor = _ImageProcessorProxy(model_args.vision_patch_size)
        # The tokenizer may carry a jinja template; this processor renders through
        # encode_messages instead, so expose it only for code that probes for one.
        self.chat_template = None

    # ------------------------------------------------------------------ chat
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        thinking: bool | None = None,
        drop_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools=None,
        **kwargs,
    ):
        """Render OpenAI-style messages with the vLLM reference V4.1 encoder.

        Uses ``vllm.tokenizers.deepseek_v41_encoding.encode_messages`` -- the
        same function the engine's ``tokenizer_mode="deepseek_v41"`` uses.  In
        thinking mode it prepends the official ``Reasoning Effort: N`` preamble;
        omitting it (the older verl port did) changes the prompt the checkpoint
        was trained on.
        """
        del kwargs
        from vllm.tokenizers.deepseek_v41_encoding import encode_messages

        rendered_messages = [
            {"role": message["role"], "content": _flatten_content(message.get("content", ""))}
            for message in messages
        ]
        if enable_thinking is None:
            enable_thinking = thinking if thinking is not None else self.thinking
        if reasoning_effort is None:
            reasoning_effort = "high"
        text = encode_messages(
            rendered_messages,
            thinking_mode="thinking" if enable_thinking else "chat",
            drop_thinking=bool(True if drop_thinking is None else drop_thinking),
            add_default_bos_token=True,
            reasoning_effort=reasoning_effort,
        )
        if tokenize:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return text

    # -------------------------------------------------------------- preprocess
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        return_tensors: str = "pt",
        padding: str = "longest",
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ) -> VLBatchFeature:
        """Expand image placeholders into the reference model's VL token stream."""
        del videos, audio, kwargs
        from fsdp_turbo.models.deepseek_v41.image_processor import prepare_vl_inputs

        if return_tensors not in (None, "pt"):
            raise ValueError("DeepSeekV41VLProcessor only supports return_tensors='pt'.")
        prompts = [text] if isinstance(text, str) else list(text or [])
        flat_images = list(images or [])
        image_offset = 0
        token_rows: list[list[int]] = []
        type_rows: list[list[int]] = []
        samples_images: list[list[Any]] = []

        for prompt in prompts:
            prompt = "" if prompt is None else str(prompt)
            image_count = prompt.count(IMAGE_PLACEHOLDER)
            prompt_images = flat_images[image_offset : image_offset + image_count]
            image_offset += image_count
            tokens, token_types, image_inputs = prepare_vl_inputs(
                prompt, prompt_images, self.tokenizer, self.model_args
            )
            if truncation and max_length is not None and len(tokens) > max_length:
                raise ValueError(
                    "Truncating a DeepSeek-V4.1 image span would corrupt token alignment; "
                    f"prompt has {len(tokens)} tokens > max_length={max_length}."
                )
            token_rows.append(list(tokens))
            type_rows.append(list(token_types))
            samples_images.append(list(image_inputs) if image_inputs else [])

        if image_offset != len(flat_images):
            raise ValueError(
                f"Prompts contain {image_offset} image placeholders but received {len(flat_images)} images."
            )
        if padding not in ("longest", "max_length", None):
            raise ValueError(f"Unsupported padding={padding!r}; use 'longest'.")

        max_len = max((len(row) for row in token_rows), default=0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids, attention_mask, token_types = [], [], []
        for tokens, types in zip(token_rows, type_rows):
            pad = max_len - len(tokens)
            input_ids.append(tokens + [pad_id] * pad)
            attention_mask.append([1] * len(tokens) + [0] * pad)
            token_types.append(types + [TEXT_TYPE] * pad)

        return VLBatchFeature(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_types": torch.tensor(token_types, dtype=torch.long),
                "images": samples_images,
            }
        )


def build_deepseek_v41_vl_processor(tokenizer, model_path: str, *, thinking: bool = False):
    """Build the processor from the checkpoint directory's ``config.json``."""
    from fsdp_turbo.models.deepseek_v41 import build_deepseek_v41_model_args

    model_args = build_deepseek_v41_model_args(model_path, include_vision=True)
    if not model_args.vision_enabled:
        raise ValueError(f"{model_path} has no vision tower; refusing to build a VL processor.")
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if token_id != model_args.image_token_id:
        raise ValueError(
            f"Tokenizer image token id {token_id} does not match model id {model_args.image_token_id}."
        )
    return DeepSeekV41VLProcessor(tokenizer, model_args, thinking=thinking)


__all__ = [
    "DeepSeekV41VLProcessor",
    "build_deepseek_v41_vl_processor",
    "IMAGE_PLACEHOLDER",
]
