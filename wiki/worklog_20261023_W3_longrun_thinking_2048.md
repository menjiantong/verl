# worklog 2026-10-23 W3 长跑：2048 response + countbenchqa2 + thinking（120）

## 1. 要解决的问题与证据
- 基线：`my_scripts/_launch_W3_longrun_1022_log.txt`（110 step，512 response、旧 scorer、
  enable_thinking=false）。解析出的指标曲线（节选）：

  | step | reward | resp_len | clip | entropy | pg_loss | grad_norm | pearson | step_s |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 0.169 | 433.5 | 0.750 | 1.423 | 0.105 | 5.228 | 0.9989 | 630.7 |
  | 10 | 0.661 | 246.3 | 0.313 | 1.182 | 0.108 | 4.316 | 0.9980 | 356.5 |
  | 30 | 0.830 | 117.7 | 0.156 | 0.266 | -0.050 | 1.738 | 0.9842 | 336.9 |
  | 50 | 0.886 | 51.5 | 0.016 | 0.422 | 0.067 | 2.773 | 0.9741 | 362.2 |
  | 70 | 0.844 | 27.0 | 0.016 | 0.405 | 0.038 | 4.926 | 0.9698 | 249.0 |
  | 90 | 0.872 | 56.5 | 0.047 | 0.251 | 0.061 | 2.526 | 0.9618 | 325.9 |
  | 110 | 0.900 | 280.7 | 0.484 | 0.093 | 0.056 | 0.134 | 0.9751 | 328.2 |

- 现象：
  1) step1 clip_ratio=0.750：75% 的回答撞 512 上限（resp_len 433.5），推理被截断；
  2) step50–90 resp_len 掉到 27–56，entropy 从 1.42 降到 0.15–0.45，reward 反而升到 0.83–0.90；
  3) pg_loss 从 0.105 降到 ~0.04–0.07 后不再下降，grad_norm 大幅波动（0.13–4.9）；
  4) reward 只看最终整数，不校验 think/response 格式。
- 归因（代码证据）：
  - `verl/utils/tokenizer/deepseek_v41_vl.py` 的 `DeepSeekV41VLProcessor.apply_chat_template`
    在 enable_thinking=false 时用 chat 模式并追加 `</think>`，与数据 prompt 的“先 think 推理”冲突
    （见 `wiki/worklog_20261023_enable_thinking_effect.md`）；
  - `countbenchqa` 只取最终 boxed 整数，不奖励格式/思考，模型最短路径是直接给答案；
  - 512 上限让长推理被截断（step1 75% 撞顶）。
  → 三者叠加导致 response 塌缩、entropy 塌缩、reward 饱和，即“训练中途指标恶化”。

## 2. 方案设计（为什么这样改）
- 改 1：MAX_RESPONSE_LEN 512→2048（max_model_len=1152+2048=3200）。去掉截断，给 think 足够空间。
- 改 2：打分器换 `verl/utils/reward_score/countbenchqa2.py`（custom_reward_function 注入，不改
  __init__.py 全局路由）：只解析 `</think>` 之后的 response 通道，末尾必须严格 `\boxed{整数}`；
  格式分 0.1。→ 让“有 think + 正确格式”成为得分必要条件。
- 改 3：enable_thinking=true。→ 模板打开 `<think>`，与数据 prompt 一致。
- 三者必须同时改：只加长度 → 模型仍可短答；只加格式 reward + 512 上限 → 推理仍被截断；
  只开 thinking + 512 → 截断更严重（基线 75% 已撞顶）。
- 拓扑/资源：单机 16 卡（80.5.25.120），FSDP16/EP8，无性能优化开关；rollout util 0.6；
  预计 HBM 与 profiling 前基线相当（≤53 GB/卡）；120 根盘 97%（余 467G），/mnt/share 余 9.1T。
- 预期信号：resp_len 早期 >512 且 clip≈0；entropy 不出现基线式塌缩；reward 早期可能偏低
  （严格格式），但应随格式学会而上升；throughput 下降（响应变长）。

## 3. 启动与监控
（追加）

## 3. 启动与监控
- 12:15:38 UTC 在 80.5.25.120 启动（`setsid nohup`，launcher PID 2900124；main_ppo PID 2900523）：
  - outer log：`logs/W3_longrun_20261023_121538.outer.log`
  - inner log：`logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261023_121553.log`
  - experiment：`GRPO-DSV41-countbenchqa2-W3-longrun-20261023_121538`；total_training_steps=500
- 配置核对（/proc/2900523/cmdline 实测）：`data.max_prompt_length=1152`、
  `data.max_response_length=2048`、`rollout.max_model_len=3200`、
  `+data.apply_chat_template_kwargs.enable_thinking=true`、
  `reward.custom_reward_function.path=verl/utils/reward_score/countbenchqa2.py`、
  `reward.custom_reward_function.name=compute_score`。
- 资源监控：`my_scripts/1023_monitor_longrun_120.sh` 每 30s 采样 NPU HBM/AICore 与主机内存 →
  `logs/W3_longrun_120_1023_monitor.log`。

## 4. 首次 2048 长跑：step1 update OOM（12:37 UTC）
- 12:15:38 启动，配置核对无误；step1 rollout 正常（12:30–12:35），HBM 48.8 GB/卡。
- 12:35:42 `on_sample_end` → `sleep_replicas`；12:35:58 sleep 全部返回（engine sleep ~16 s）。
- 12:37:06 rank2 OOM：`50.40 GiB active / 137.87 MiB free / 50.91 GiB reserved`，总 61.27 GiB；
  位置 `FSDPTurbo/fsdp_turbo/distributed/expert_parallel/dispatcher.py:181`
  `experts_computation: gate = gate.clamp(max=act_limit)`。
- HBM 采样：12:36:23 23 GB → 12:36:55 52.8 GB → 12:37:27 65.4 GB（撞顶）→ 进程退出。
- 归因（evidence）：
  1. 训练峰值随 seq 线性增长：512 response（seq 1664）时 `max_memory_allocated=26.4 GiB`
     （reserved 48.2 GiB）；2048（seq 3200）时 forward 就到 50.4 GiB；
  2. 训练可用上限 = 61.27 − ~10.4（vLLM sleep(level=1) 后 persistent/其他）≈ 50.9 GiB；
  3. 历史结论（`wiki/run_and_improve_speed_20261020.md` §8.3/§8.6、
     `wiki/dsv41_memory_and_parallel_archive.md` E5.3）：util 0.5→0.4 不释放训练侧显存；
     整层 recompute 嵌套 checkpoint 崩；swap_activation 反向不释放（判死）；
     ffn recompute 与 attention chunk/recompute 已在本脚本开启。
  → 2048 在 16 卡单机上不可行（FSDP 只切参数、不切激活，加机器也无效）。

## 5. 处置：按用户要求测试 1024
- 新增 v1：`my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking_v1.sh`
  （仅覆盖 `MY_ENV_MAX_RESPONSE_LEN=1024`，其余完全继承 base）。
- base 兼容性修改：`export MY_ENV_MAX_RESPONSE_LEN=${MY_ENV_MAX_RESPONSE_LEN:-2048}`
  （不设置时保持原默认 2048；支持 v1 覆盖）。`bash -n` 通过。
- 预期：seq=1152+1024=2176；按 1664→26.4 GiB 线性外推峰值约 34–36 GiB，
  训练上限 ~51 GiB，余量约 15 GiB。
- 启动命令：`bash my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking_v1.sh`

## 6. 1024 长跑迁移到 118（15:58 UTC）
- 15:38 在 120 启动的 1024 长跑在 trainer 建模型阶段 OOM（15:45:03，NPU4 仅 6.89 GiB
  allocated / 6.45 MiB free）。**根因不是 1024 配置**：120 在 13:20–15:38 被外部租户断续占用
  （monitor 证据：13:52 10.8GB、14:13 12.6GB、14:55 61.5GB/100% AICore、15:17 20.8GB、
  15:27 58GB/100%），启动时租户仍占 58GB/卡，训练侧无法分配。
- 15:57 节点状态：118/119/120/122/124 均 16 卡全空闲；123 满；128 仅 8 卡空闲。
- 处置：新建 v2 脚本 `my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking_v2.sh`
  （MY_IP=118 + MY_ENV_MAX_RESPONSE_LEN=1024，其余继承 base），15:58:28 启动；
  配置核对：max_response_length=1024、max_model_len=2176、thinking=true、500 step。
- 同时按用户任务二在 122 启动 maxresp50 实验（见
  `wiki/worklog_20261023_task2_response_len_diffmean.md`）。

## 7. 1024 长跑 step1 结果（16:28 UTC，118）
- `critic/rewards/mean=0.0`、`response_length/mean=949.31`、`clip_ratio=0.8906`、
  `actor/entropy=0.9773`、`actor/pg_loss=0.0`、`grad_norm=0.0`、
  `rollout_probs_diff_mean=0.00624`、`pearson=0.9988`、`timing_s/step=636.5`、`throughput=7.70`。
- 关键问题：**reward 全 0**。countbenchqa2 要求 `</think>` 后的 response 通道且以
  `\boxed{整数}` 结尾；1024 上限下 89% 样本撞顶，think 未闭合 → score 0。
  pg_loss/grad_norm=0 → 本步无梯度、模型不更新。若后续步骤仍全 0，该实验无法产生学习信号。
- 注：update 阶段未 OOM（1024 的 seq 峰值在显存预算内），说明 1024 配置本身可运行。

## 8. 1024 长跑在第 4 步 OOM（10-23 16:43 UTC，118）
- 实际进度：step1/2/3 完成并落日志；第 4 步 actor update 时 OOM。
- 报错：`torch.OutOfMemoryError: NPU out of memory. Tried to allocate 2.47 GiB
  (NPU 2; 61.27 GiB total capacity; 47.57 GiB already allocated; 47.57 GiB current active;
  412.07 MiB free; 48.15 GiB reserved in total by PyTorch)`，
  位置 `FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py:1475`
  `logits = F.linear(x.to(self.compute_dtype), self.weight.to(self.compute_dtype))`。
- 显存时间线（monitor）：16:42:29 ~29.5GB/98% AICore（update 开始）→ 16:43:00 43–55GB →
  16:43:32 NPU2 64.4GB / NPU3 63.2GB（撞顶）→ OOM。
- 账：设备 61.27GiB；trainer reserved 48.15GiB；free 412MiB ⇒ 约 12.7GiB 在 trainer 之外
  （vLLM sleep 残留 + 分配器碎片）。vLLM 在 Ascend 只能 sleep level=1
  （`_resolve_sleep_level` 对 NPU 固定返回 1，注释 “vllm_ascend not support sleep_level now”），
  vllm-ascend `worker.sleep(level=1)` 仅 offload weights、释放 KV，DSA Hadamard/metadata 等
  persistent 分配保留（worker.py:245-265）。所以训练侧可用上限约 48–51GiB。
- 结论：1024 只是“比 2048 少一半激活”，不是安全配置；step1 峰值已 61.1GiB（99.7%），
  step4 的 lm_head 固定大块（2.47GiB）在余量不足时先失败。1536 不可行。
