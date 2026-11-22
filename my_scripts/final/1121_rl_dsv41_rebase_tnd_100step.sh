#!/usr/bin/env bash
# [1120 P5c v3] RL TND 100 步长跑（交付脚本）：基线 = 已通过 3 步验证的
#   1121_rl_dsv41_rebase_tnd_3step_v6_noentropycompile.sh（worklog §33/§34），其余逐字相同；
#   仅两处非行为差异：
#   1) 步数默认 100（trainer.total_training_steps；MY_ENV_DSV41_STEPS 可覆盖，=3 即冒烟）；
#   2) MY_ENV_DSV41_RL_TND_DEBUG=0（不逐微批打印布局，避免 100 步刷屏；TND 断言仍在）。
# 关键修复（继承 v6）：actor/ref 的 fsdp_config.use_torch_compile=False。
#   TND 每 rank token 数不同 → 编译图按形状特化 → 每个新形状触发 triton autotune；
#   v4/v5 实测个别 config 慢 3~4 个数量级（>1500s），把 2 个 rank 卡在 do_bench、
#   其余 rank 卡在 post-forward AllReduce（engine_workers.py:199），最终 EI0006。
#   关闭后 eager chunked entropy 每 chunk ~10ms（微基准 §34.3），无步时损失。
# 状态：✅ 2026-11-20 19:28 → 23:53 UTC 于 7.150.8.255（16 die）跑满 100/100 步；
#   日志 logs/rl_rebase_tnd_100step_20261120_1930.out；步时 mean 148.3s / median 103.4s
#   （除 step1：mean 132.0s、median 103.4s）；末 10 步 entropy 0.103、score 0.914、resp_len 60.5；
#   收尾 DataLoader killed 为已知 teardown（dmesg oom_kill=0、无 EI0006），非训练失败。
#   运行期 md5 57027b7e64b9e2f270353396575ec7c0；本次仅回填本注释块。
# 代码基线：verl ece39d48（父 3a58c08a；含 padding_utils.py、countbenchqa.py、
#   fsdp_turbo_dsv41_impl.py 三文件 TND 补丁，已提交）+ FSDPTurbo 996893b @ dev_dsv41_tnd_1120。
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-100}
source "$here/env_dsv41_yundao_930.sh"

export CP_SIZE=1
export PYTHONPATH=/efs_rl/m00899630/workspace/dsv41-verl-refactor/mjt_workplace/task/1120_try_tnd/scratch/fsdp_rebase:${PYTHONPATH:-}
export MAX_PROMPT_LEN=1024 MAX_RESPONSE_LEN=2048
export MY_ENV_DSV41_GRAPH_MODE=1
export MY_ENV_DSV41_KV_CACHE_BYTES=3221225472
export MY_ENV_DSV41_CUDAGRAPH_CAPTURE_SIZES="[1,2,4,8,16,24,32]"
export MY_ENV_TRAIN_SPARSE_FLASH_ATTN=1
export FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV=3
export MY_ENV_DSV41_RL_SFA_LAYOUT=TND
export MY_ENV_DSV41_RL_TND_PAD_MULTIPLE=2
export MY_ENV_DSV41_RL_TND_DEBUG=0
# torch_npu 的 get_device_capability() 默认返回 None；dynamo 的 TMA 探测会与 (9,0) 比较而抛
# TypeError。8.0 < 9.0 ⇒ 不启用 CUDA TMA 分支（Ascend 无 TMA），仅影响编译特性判定，不影响数值。
export TORCH_NPU_DEVICE_CAPABILITY=8.0

exec bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  actor_rollout_ref.model.use_remove_padding=True \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS:-100}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  ++actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  ++actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  "$@"
