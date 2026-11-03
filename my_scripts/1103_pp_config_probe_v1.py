#!/usr/bin/env python3
# =============================================================================
# 1103 PP（流水线并行）支持度静态探针 v1
#
# 目的：用最小代价确认两个结论（不改代码、不占 NPU）：
#   1) verl RolloutConfig 对 vllm/sglang/trtllm 的 pipeline_model_parallel_size>1
#      是否直接拒绝（rollout.py:319-322）；
#   2) FSDPTurbo 的 DistributedConfig 是否存在 PP 字段（fsdp_turbo_config.py:308）。
#
# 用法：python3 my_scripts/1103_pp_config_probe_v1.py
# 依赖 commit：verl af78b8c5（探针本身无环境变量）
# =============================================================================


def probe(tag, fn):
    try:
        fn()
        print(f"MY_DEBUG_LOG pp_probe_{tag} result=OK")
    except Exception as e:  # noqa: BLE001 - 探针就是要打印异常类型
        print(f"MY_DEBUG_LOG pp_probe_{tag} result={type(e).__name__}: {e}")


def rollout_control():
    from verl.workers.config.rollout import RolloutConfig

    RolloutConfig(name="vllm")


def rollout_pp2():
    from verl.workers.config.rollout import RolloutConfig

    RolloutConfig(name="vllm", pipeline_model_parallel_size=2)


def turbo_pp1():
    from fsdp_turbo.fsdp_turbo_config import DistributedConfig

    DistributedConfig(fully_shard_parallel_size=16)


def turbo_pp2():
    from fsdp_turbo.fsdp_turbo_config import DistributedConfig

    DistributedConfig(pipeline_parallel_size=2)


probe("rollout_vllm_control", rollout_control)
probe("rollout_vllm_pp2", rollout_pp2)
probe("fsdpturbo_control", turbo_pp1)
probe("fsdpturbo_pp2", turbo_pp2)
