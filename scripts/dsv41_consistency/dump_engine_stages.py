"""Engine-side dump: parameter fingerprints + staged activations + logprobs.

Loads the rollout checkpoint with vLLM/vllm-ascend only (no verl trainer, no weight sync)
and, in a single engine lifetime, produces everything the consistency comparison needs:

    <out-dir>/params.json                    weight fingerprints of the loaded engine
    <out-dir>/engine_len<L>.json             scored/argmax logprobs for the fixture
    <out-dir>/engine_stages_len<L>_rank*.pt  milestone activations of the prefill
                                             (via the Dsv41DumpExtension worker hooks)

Run through `run_dump_engine.sh` (Ascend + vLLM environment) so the engine sees the same
settings as the GRPO rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PARAM_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "layers.0.",
    "layers.1.",
    "layers.2.",
    "layers.3.",  # 0/2/3 were the structurally interesting ones (layer 2 owns the compressed
    #               KV + index, layer 3 reads its selection), but skipping layer 1 left it with
    #               only 4 of its ~25 params fingerprinted, and nothing in the 4-layer configs
    #               (engram off) justifies that. Existing dumps predate this line.
    "norm.weight",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32")
    ap.add_argument("--fixture", default="/mnt/share/m00899630/dsv41/dump/fixture.json")
    ap.add_argument("--out-dir", default="/mnt/share/m00899630/dsv41/dump/engine")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--mem-util", type=float, default=0.9,
                    help="gpu_memory_utilization; the vLLM worker refuses to start when the "
                         "free NPU memory is below this fraction of the device total, and every "
                         "card here carries a few GiB of other tenants' resident allocations, so "
                         "0.9 can fail by a few hundred MiB (see worklog §10).")
    ap.add_argument("--lengths", default="", help="Comma separated subset of fixture lengths.")
    ap.add_argument("--skip-stages", action="store_true", help="Only dump params + logprobs.")
    ap.add_argument("--skip-params", action="store_true")
    ap.add_argument("--worker-extension", default="dsv41_dump_extension.Dsv41DumpExtension")
    ap.add_argument("--batch-invariant", action="store_true",
                    help="Set rl_config.enable_batch_invariant (A/B for engine-internal consistency).")
    return ap.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    additional_config = {
        "mc2_comm_alg": "fullmesh_v2",
        "enable_engram": False,
        "engram_storage": "int8",
        "ascend_compilation_config": {"enable_npugraph_ex": False, "enable_static_kernel": False},
        # Same RL switch the GRPO script sets: weight_nz_mode=0, no expandable segments.
        "rl_config": {"enabled": True, "enable_batch_invariant": args.batch_invariant},
    }
    kwargs = {}
    if args.worker_extension:
        kwargs["worker_extension_cls"] = args.worker_extension

    llm = LLM(
        model=args.model_path,
        tokenizer_mode="deepseek_v41",
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        enable_expert_parallel=args.ep > 1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.mem_util,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_model_len,
        disable_log_stats=True,
        additional_config=additional_config,
        **kwargs,
    )
    print("[dump-engine] engine up", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_params:
        stats = llm.collective_rpc("dsv41_param_stats", args=(PARAM_PATTERNS,), timeout=600)
        # Keep the metrics from rank 0 (every rank holds the same dense weights; the routed
        # experts are EP-sharded, so layer records are merged across ranks by the compare step).
        with open(out_dir / "params.json", "w") as f:
            json.dump([{"rank": rank, "params": entry} for rank, entry in enumerate(stats)], f)
        print(f"[dump-engine] param fingerprints: {[len(s) for s in stats]} per rank", flush=True)
        for entry in stats[0][:12]:
            print(f"    {entry['name']:<60} {entry['shape']} {entry['dtype']} "
                  f"std={entry.get('std'):.5f} absmax={entry.get('absmax'):.5f}", flush=True)

    with open(args.fixture) as f:
        fixture = json.load(f)
    want = {int(x) for x in args.lengths.split(",") if x} or None
    samples = [s for s in fixture["samples"] if want is None or s["target_len"] in want]

    # temperature=0 keeps sampling out of the picture: we only read prompt logprobs, which
    # vLLM computes from the raw logits (`logprobs_mode=raw_logprobs`).
    sp = SamplingParams(max_tokens=1, prompt_logprobs=5, temperature=0.0)
    for sample in samples:
        ids = sample["input_ids"]
        length = sample["target_len"]
        stages = {}
        if not args.skip_stages:
            hooked = llm.collective_rpc("dsv41_hook_begin", timeout=600)
            print(f"[dump-engine] len={length}: hooks={hooked}", flush=True)
        result = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        if not args.skip_stages:
            written = llm.collective_rpc(
                "dsv41_hook_end",
                args=(str(out_dir), f"len{length}"),
                timeout=3600,
            )
            stages = written
            print(f"[dump-engine] len={length}: stage dumps {written[:2]}", flush=True)

        scored, argmax = [], []
        for position, entry in enumerate(result.prompt_logprobs):
            if entry is None:  # position 0 has no context
                continue
            scored_token = ids[position]
            if scored_token not in entry:
                raise RuntimeError(f"engine did not score token {scored_token} at position {position}")
            scored.append((int(scored_token), float(entry[scored_token].logprob)))
            best = max(entry.items(), key=lambda kv: kv[1].logprob)
            argmax.append((int(best[0]), float(best[1].logprob)))
        payload = {
            "target_len": length,
            "input_ids": ids,
            "scored": scored,   # (token_id, logprob of that token) for positions 1..S-1
            "argmax": argmax,   # (argmax token, its logprob) for the same positions
            "stages": stages,
        }
        with open(out_dir / f"engine_len{length}.json", "w") as f:
            json.dump(payload, f)

        # Decode-vs-prefill self-consistency, which is what a GRPO run actually compares:
        # the rollout logprobs come from token-by-token decode against the engine's KV cache
        # and dynamic batching, while the trainer recomputes everything in one padded
        # forward. Generating greedily and then re-scoring prompt+generated as a *prompt*
        # isolates the engine's own mode dependence, with no trainer involved.
        gen_sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=1, ignore_eos=True)
        gen = llm.generate([{"prompt_token_ids": ids}], gen_sp, use_tqdm=False)[0]
        generated = list(gen.outputs[0].token_ids)
        decode_logprobs = [
            (int(next(iter(entry))), float(entry[int(next(iter(entry)))].logprob))
            for entry in gen.outputs[0].logprobs
        ]
        rescored = llm.generate([{"prompt_token_ids": ids + generated}], sp, use_tqdm=False)[0]
        prefill_logprobs = []
        for position, entry in enumerate(rescored.prompt_logprobs):
            if entry is None or position < len(ids):
                continue
            token = generated[position - len(ids)]
            prefill_logprobs.append((int(token), float(entry[token].logprob)))
        payload = {
            "target_len": length,
            "prompt_len": len(ids),
            "generated": generated,
            "decode_logprobs": decode_logprobs,
            "prefill_logprobs": prefill_logprobs,
        }
        with open(out_dir / f"engine_decode_len{length}.json", "w") as f:
            json.dump(payload, f)
        print(f"[dump-engine] len={length}: decode-vs-prefill on generated tokens "
              f"{[(round(d - p, 4)) for (_, d), (_, p) in zip(decode_logprobs, prefill_logprobs)]}",
              flush=True)
        print(f"[dump-engine] len={length}: scored[1:4]={[round(p, 4) for _, p in scored[:3]]} "
              f"argmax[1:4]={[t for t, _ in argmax[:3]]}", flush=True)


if __name__ == "__main__":
    main()
