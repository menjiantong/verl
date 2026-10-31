# Copyright 2026
"""Single-turn agent loop for DeepSeek-V4.1 vision-language RL.

The generic single-turn loop tokenizes the prompt and hands the ids straight to
vLLM.  For V4.1 vision that is not enough:

* vLLM's ``DeepseekV4VLMultiModalProcessor`` expects **one** image placeholder
  token per image and expands it into the full image span itself, so the
  rollout prompt must stay compact.
* The training model needs the **expanded** token stream plus the per-position
  ``token_types`` map and the per-image patch records, because it re-runs the
  ViT/aligner on the trainer side and overwrites the image positions.

This loop keeps both views: compact ids go to vLLM, the reference
``prepare_vl_inputs`` expansion becomes ``AgentLoopOutput.prompt_ids`` and
``multi_modal_inputs`` for the trainer.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("deepseek_v41_vl_single_turn")
class DeepSeekV41VLAgentLoop(SingleTurnAgentLoop):
    """Single-turn VLM loop: compact prompt to vLLM, expanded prompt to trainer."""

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract images (PIL) from the structured raw_prompt.
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
        self._assert_mm_supported(bool(multi_modal_data))

        # 2. compact prompt ids: one <｜deepseek_image｜> per image. The builder
        #    also hands back the rendered text for the trainer-side expansion.
        mm_inputs: dict[str, Any] = {}
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_inputs_out=mm_inputs,
        )
        rendered_text = mm_inputs.get("rendered_text")
        if rendered_text is None:
            raise RuntimeError(
                "DeepSeek-V4.1 VL agent loop requires the VL Continuous Token builder "
                "(name='deepseek_v41_vl_single_turn'); it did not provide rendered_text."
            )

        # 3. generate from the compact prompt.
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            output.token_ids,
            [],
            [] if output.log_probs else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        final_compact_prompt = (
            merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
            if response_mask
            else merge_result.token_ids
        )
        if final_compact_prompt != prompt_ids:
            raise RuntimeError(
                "DeepSeek-V4.1 VL single-turn loop saw a modified prompt during assistant merge; "
                "this loop only supports append-only single-turn rollouts."
            )

        # 4. expand image spans for the trainer. This is the reference
        #    preprocessing vLLM runs, so both sides see the same token stream.
        expanded_ids, token_types, image_inputs = self._expand_prompt(rendered_text, images, prompt_ids)
        prompt_length_limit = self.rollout_config.prompt_length
        if len(expanded_ids) > prompt_length_limit:
            raise ValueError(
                f"Expanded multimodal prompt is {len(expanded_ids)} tokens, exceeding "
                f"rollout.prompt_length={prompt_length_limit}. Increase data.max_prompt_length."
            )

        return AgentLoopOutput(
            prompt_ids=expanded_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(expanded_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            multi_modal_inputs={
                "token_types": torch.tensor(token_types, dtype=torch.long).unsqueeze(0),
                # One sample's image records; ``extract_multi_modal_inputs``
                # collects these into ``images[batch] = [ImageInput, ...]``.
                "images": image_inputs,
            },
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

    def _expand_prompt(self, rendered_text: str, images, compact_ids: list[int]):
        """Expand compact image placeholders with FSDPTurbo's reference preprocessing."""
        from fsdp_turbo.models.deepseek.deepseek_v41.image_processor import prepare_vl_inputs

        image_list = list(images or [])
        tokens, token_types, image_inputs = prepare_vl_inputs(
            rendered_text, image_list, self.tokenizer, self.processor.model_args
        )
        image_inputs = list(image_inputs or [])

        # Cross-check the compact ids against the expansion: the rollout engine
        # performs the same replacement, so any mismatch means the builder and
        # the reference processor rendered different prompts.
        rebuilt: list[int] = []
        image_iter = iter(image_inputs)
        image_token_id = self.processor.image_token_id
        for token_id in compact_ids:
            if token_id == image_token_id:
                try:
                    image = next(image_iter)
                except StopIteration as exc:
                    raise ValueError("Compact prompt has more image placeholders than processed images.") from exc
                rebuilt.extend([image_token_id] * int(image.types.numel()))
            else:
                rebuilt.append(token_id)
        try:
            next(image_iter)
            raise ValueError("Processed more images than the compact prompt has placeholders.")
        except StopIteration:
            pass
        if rebuilt != tokens:
            raise ValueError(
                "Compact prompt expansion does not match the reference VL processor output. "
                "The rollout prompt and trainer token stream would diverge."
            )
        return tokens, token_types, image_inputs

__all__ = ["DeepSeekV41VLAgentLoop"]
