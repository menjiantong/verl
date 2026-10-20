# DeepSeek-V4.1-Flash top32 bf16 noengram · CountBenchQA-lite VLM GRPO（2026-10-19）

> 任务：用新权重 + countbenchqa_lite 数据集跑 GRPO，先验证 reward 正常，再持续跑（用户叫停才停）。
> 验收：跑出第一个 reward、reward 曲线正常、持续跑；每 10 分钟巡检并记录显存/状态/日志证据。

## 1. 输入

| 项 | 路径 |
|---|---|
| 权重 | `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram` |
| 数据集（原始） | `/mnt/share/m00899630/workspace/mjt_workplace/datasets/countbenchqa_lite/train.parquet`（500 行，图文 VQA，images 列内嵌 JPEG） |
| tokenizer | `/mnt/share/m00899630/dsv41/rl_tokenizer`（tokenizer.json 与 checkpoint 同源，含 `<｜deepseek_image｜>` id=129264） |
| 启动脚本 | `verl/my_scripts/dsv41_grpo_countbenchqa_1node.sh`（单机 16 卡；GSM8K 的 `dsv41_grpo_2node_new.sh` 未改动） |

数据事实（CPU 预处理统计，thinking=True）：
- 500 行；prompt token 236–1064，均值 307；image span 182–1010 token（单图 1 张）；答案只有 2–10 共 9 类。
- 权重 config：40 层、32 专家、`engram_layer_ids=[]`、`vision_config` 32 层 ViT（hidden 1024, patch 14, downsample 3, max_image_tokens 1024）、`image_token_id=129264`。
- checkpoint 含 vision 权重 266 个（vision.blocks.* 259、aligner.* 4、image_start/end/newline 3），与 FSDPTurbo 训练模型参数名一一对应。

## 2. 代码改造（本任务新增/修改）

### 2.1 数据集预处理
- 新增 `verl/scripts/prepare_countbenchqa_lite.py`：按 seed=42 切 450 train / 50 val，schema 不变（`<image>` 保留，由 RLHFDataset 拆成 image content part）。
- 产物：
  - `/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet`（450）
  - `/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_val.parquet`（50）

### 2.2 打分器
- 新增 `verl/verl/utils/reward_score/countbenchqa.py`：
  - 取**最后一个** `\boxed{整数}`；没有 boxed 时回退到尾部 `</think>` 之后的最后一个独立整数。
  - `score = 0.9*acc + 0.1*format`（format = 结尾严格为 `\boxed{int}`）。
  - 返回 dict（score/acc/format/predicted/expected/boxed），extra 字段会被 reward loop 记录。
- `verl/verl/utils/reward_score/__init__.py` 注册 `data_source in {"countbenchqa_lite","countbenchqa"}`。
- 单测：正确 boxed=1.0、错答案 boxed=0.1、无 boxed 但答对=0.9、无数字=0.0。

### 2.3 视觉 processor（训练/数据集侧）
- 新增 `verl/verl/utils/tokenizer/deepseek_v41_vl.py`：
  - `apply_chat_template` 走 V4.1 参考 encoder（`encode_messages`），图片 part → `<｜deepseek_image｜>`；支持 `enable_thinking`。
  - `__call__` 用 FSDPTurbo `prepare_vl_inputs` 把 placeholder 展开为
    `[IMAGE_START] + ([IMAGE]*n_w + [IMAGE_NEW_LINE])*n_h + [IMAGE_END]`，
    返回 `input_ids / attention_mask / token_types / images`（每样本 image records）。
  - `VLBatchFeature.convert_to_tensors` 保留 `images` 为对象列表（不 stack）。
- `verl/verl/workers/config/model.py`：`HFModelConfig.__post_init__` 在
  `model_type=deepseek_v41` 且 `vision_n_layers>0` 时构建上述 processor
  （发行版 checkpoint 无 HF ProcessorMixin）。

### 2.4 rollout Continuous Token builder
- `verl/verl/utils/tokenizer/deepseek.py` 新增 `DeepSeekV41VLContinuousTokenBuilder`：
  - 渲染 V4.1 角色标记 + 每图 1 个 placeholder（紧凑），**不展开**（vLLM 会替换为完整 span）。
  - 通过 `mm_inputs_out["rendered_text"]` 把渲染文本交给 agent loop，供训练侧展开。
- `continuous_token_wiring.py`：新增 family `deepseekv41vl`；`deepseek_v41` 有 processor 时由文本 family 升级到 VL family。

### 2.5 agent loop
- 新增 `verl/verl/experimental/agent_loop/dsv41_vl_agent_loop.py`（注册名
  `deepseek_v41_vl_single_turn`）：
  - 紧凑 prompt → vLLM（`image_data=PIL list`）；
  - 生成后用 `prepare_vl_inputs` 展开 prompt，并把 expanded `prompt_ids`、
    `token_types`、`images` 放进 `AgentLoopOutput`；
  - 校验 compact→expanded 与参考 processor 输出逐 token 一致。
- `agent_loop.py::AgentLoopWorker._compute_multi_modal_inputs`：
  - 检测 loop 预置的 `token_types/images`，按 run 固定长度
    (`prompt_length+response_length`) 对齐（TQ 路径无 padding，padded 路径左对齐）；
  - 这样同 batch 的 token_types 形状一致，engine 再按 micro-batch 真实长度裁剪。
- `verl/verl/utils/model.py::extract_multi_modal_inputs`：
  - 非 tensor 值（`images` 的 per-sample ImageInput 列表）保持 list，不再强行 `torch.cat`。

### 2.6 训练 engine
- `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`：
  - `build_deepseek_v41_model(include_vision=VERL_DSV41_VISION==1)`；
  - vision/aligner/image_start/image_end/image_newline 参数 `requires_grad=False`（冻结）；
  - `prepare_model_inputs` 把 token_types 裁剪/补齐到 micro-batch `input_ids` 长度。
- `run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh`：
  - `VERL_DSV41_VISION=1` 时给 actor fsdp_plan 加
    `ignored_modules=['model.vision','model.aligner']`（冻结参数不参与 FSDP，保持副本）。

## 3. 启动

机器选择：126 → 118 → 122（用户指定顺序）。
- 15:13 检查：**126 已被租户占满**（16 卡各 ~60GB HBM、RAM 1.5T、load 78）→ 不可用。
- 15:14 检查：**118 全空**（HBM 每卡 ~3GB 基线、RAM 31G、load 33）→ 使用 118。

命令（118，`/mnt/share/m00899630/workspace/dsv41-verl/verl`）：
```bash
setsid nohup bash my_scripts/dsv41_grpo_countbenchqa_1node.sh 500 \
  > logs/countbenchqa_118_<timestamp>.nohup.log 2>&1 < /dev/null &
```
关键参数：单机 16 卡，FSDP16 / EP8 / EFSDP1 / TP1；引擎 TP8×DP2（EP16）；
`TRAIN_BSZ=8 × SAMPLE_N=8 = 64 序列/步`；`LR=5e-6`、`USE_KL=0`；
`MAX_PROMPT_LEN=1152`、`MAX_RESPONSE_LEN=1024`、`max_model_len=2176`；
`ROLLOUT_GPU_MEM_UTIL=0.6`（见 E01；首启 0.3 失败）；`VERL_DSV41_VISION=1`、`VERL_DSV41_VISION_FREEZE=1`；
`default_agent_loop=deepseek_v41_vl_single_turn`、`enable_thinking=true`。

## 4. 状态时间线

### 2026-10-19 15:14 启动（118）
- 启动命令与完整 hydra 参数见 nohup 日志；`ignored_modules` 已生效。
- 15:14 进入 TaskRunnerV1 配置展开阶段（无错误）。

（后续检查记录在下方持续追加）

### E01 2026-10-19 15:22 vLLM 引擎启动失败：No available memory for the cache blocks（118）
- **现象**：训练侧构建全部成功——`vision tower enabled: 266 parameters frozen`、
  `materializing 1184 parameters + 80 checkpoint buffers`、
  `checkpoint read: 1264 tensors slice-read directly (12.37 GiB this rank)`、
  `direct shard load verified ... bit-exact`；随后 vLLM `EngineCore_DP0` 在
  `_initialize_kv_caches → _check_enough_kv_cache_memory` 抛
  `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`。
- **根因**：`gpu_memory_utilization=0.3`（沿用纯文本的验证值）。文本引擎没有视觉塔；
  VLM 引擎的 profile run 额外跑 ViT/aligner + 1024-token dummy image，`non_kv_cache_memory`
  超过 `0.3*64GiB≈19.6GiB`，导致 `available_kv_cache_memory <= 0`。
- **证据**：日志 `countbenchqa_118_20261019_151451.nohup.log` 第 11243–11316 行
  （`vllm/v1/worker/gpu_worker.py:622`、`kv_cache_utils.py:879`）。
- **处置**：`ROLLOUT_GPU_MEM_UTIL=0.6`（≈39GiB/rank：引擎权重约 10–15GB + KV 余量；
  rollout 时 actor 参数已 CPU offload）。重启。

### 2026-10-19 15:25 重启（118，util=0.6）
- 新日志：`logs/countbenchqa_118_20261019_152615.nohup.log`，pid 1962998。
- 其余配置不变；等待 vLLM 引擎重新建模 + 基座权重同步。

### E02 2026-10-19 15:33–15:52 vLLM 引擎卡在 KV cache 重排（118，util=0.6 已过 KV 预算检查）
- **现象**：`gpu_memory_utilization=0.6` 后引擎越过了 `_check_enough_kv_cache_memory`，
  16 个 `VLLM::Worker_DP*` 各占 ~37.6GB HBM、持续 450% CPU，日志 18 分钟无新行。
- **定位**：`py-spy dump --pid <worker>` 栈：
  `_adjust_kv_layout (vllm_ascend/worker/model_runner_v1.py:5202)`
  → `_reshape_kv_cache_tensors` → `initialize_kv_cache_tensors` → `initialize_kv_cache`。
  该行 `stride = torch.empty(shape).stride()` 为读 stride 竟然**按整层 KV 大小分配临时张量**；
  40 层 × 大 KV 的 alloc/free 抖动让 worker 在驱动层自旋。
- **处置**：改为算术计算连续 stride（后缀累乘，语义等价、零分配）：
  `vllm-ascend/vllm_ascend/worker/model_runner_v1.py::_adjust_kv_layout`。
  本地验证 `(4743,128,1,576) → (73728,576,576,1)` 与 torch 一致。
- 重启。

### 2026-10-19 15:53 重启（118，KV 重排补丁）
- 日志：`logs/countbenchqa_118_20261019_155322.nohup.log`，pid 2053742。

### 2026-10-19 16:00:48 引擎初始化成功（KV 补丁生效）+ 进入 fit
- `sleep-debug: replica vLLMReplica: sleep returned` → `Training from scratch` →
  `all initialize finished, ready to fit`。
- 16:10 状态：HBM ~25.6GB/卡（引擎 asleep），AICore 0%；16 个 `WorkerDict` 各 ~120% CPU：
  `py-spy` 栈 = `actor_rollout_update_weights → async_send_weights → unfuse_moe_params →
  _export_param → wait_tensor(DTensor all-gather)`，即**首次全量权重同步**。
- 说明：`VERL_DSV41_LOCAL_EXPERT_EXPORT=1` 被 example 的守卫关闭（GEN_EP=16 != EP_SIZE=8，
  EFSDP=1 要求 GEN_EP==EP_SIZE），走通用 per-expert unfuse 导出；137GB bf16 预计 ~15–20 分钟。

### E03 2026-10-19 16:19 agent loop 输出校验失败（rollout 本身成功）
- **现象**：16:13 权重同步完成，8 个 prompt×8 采样开始生成；16:19 全部在
  `AgentLoopOutput(...)` 构造处失败：
  `pydantic ValidationError: mm_processor_output Input should be a valid list, input_value={'rendered_text': ...}`。
  随后 trainer 报 `Sync replay buffer selected terminal groups with no materializable trajectories`。
  生成本身（vLLM 16 卡，thinking，max 1024）约 6 分钟，无引擎错误。
- **根因**：`mm_processor_output` 字段类型是 `Optional[list[dict]]`（SGLang 视频载荷），
  我把 `mm_inputs`（含 rendered_text 的 dict）传了进去；另外基类 `AgentLoopOutput`
  **没有 `multi_modal_inputs` 字段**，pydantic 默认丢弃未声明字段 → 训练侧拿不到 token_types。
- **处置**：① 不再传 `mm_processor_output`；② 给 `AgentLoopOutput` 增加
  `multi_modal_inputs: Optional[dict[str, Any]]`；③ 启动参数增加
  `trainer.v1.sampler.sync_refill_failed_groups=True`；④ 开启 `VERL_AGENT_DUMP`
  落首批 prompt/response/reward 证据。本地 pydantic 构造成立。
- 重启。

### E04 2026-10-19 16:50 首个 rollout 完成但输出为乱码；训练 forward 报 `'dict' object has no attribute 'patches'`
- **rollout 证据**：`logs/agent_dump_countbenchqa.*` 首批 4 条：prompt tail 正确
  （`<｜User｜><｜deepseek_image｜>\nHow many ...<｜Assistant｜><think>`，plen=244），
  但 response 是随机中文/罕见 token、重复 `\\`，全部 rlen=1024、reward=0.0。
- **先排除权重同步**：`SYNC-PROFILE sender: 5024 tensors, 100.02 GiB, 203 buckets | export 707s`；
  用同一 config 做 meta 构建统计：训练模型 1184 params + 136 buffers，
  参数 101.4 GiB + buffers 0.07 GiB ⇒ **同步体积与全量模型一致，未漏发**。
- **前向报错**：`model.py:1732 img.patches` — `AgentLoopOutput.as_dict()`（pydantic
  model_dump）把 `ImageInput` dataclass 转成 dict，跨 TransferQueue 后训练侧拿到 dict。
- **乱码根因（强证据）**：`vllm/tokenizers/deepseek_v41_encoding.py` 的官方 encoder 在
  thinking 模式会在 prompt 开头插入 `<｜System｜>Reasoning Effort: 50 (range 1-100, ...)`；
  verl 旧 encoder 没有这段，模型收到的不是训练时格式。
- **处置**：
  ① `DeepSeekV41VLProcessor.apply_chat_template` 改用 vLLM 官方 `encode_messages`
  （thinking_mode/reasoning_effort=high，content parts 以 `\n\n` 连接）；
  ② `fsdp_turbo_dsv41_impl._normalize_dsv41_images` 在训练 forward 前把 dict 还原为
  `ImageInput`（patches 转 bf16、types 转 long）；
  ③ 独立 vLLM 冒烟脚本 `my_scripts/dsv41_vl_smoke.py`（load_format=auto、8 卡）验证
  模型本体 + 视觉链路。
- 新 prompt 长度：compact 85 / expanded 281（较旧版 +22/+37 token，仍 < 1152）。

### E05 2026-10-19 17:02 独立 vLLM 验证：base checkpoint + thinking 模式退化；改 chat 模式
- **方法**：`vllm serve`（8 卡 TP8+EP，load_format=auto，官方 `tokenizer_mode=deepseek_v41`）
  加载同一 checkpoint，绕过 verl 权重同步，直接验证模型本体。
- **结果（temperature=0）**：
  - 纯文本 `35+28=` → `'27=27=27=...'`（重复循环）；
  - chat 文本（thinking=True，官方 encoder）→ `'...35+28 = 35+28 = ...'` 循环；
  - 图片题 row0（“How many dogs”，gt=2）：
    - thinking=True → 384 token 全部重复“The output should be be the number of dogs”，未答；
    - **thinking=False → 完整推理并以 “The final answer is 2.” 正确作答**；
    - row1/row2（gt=7）两种模式仍循环到上限。
- **结论**：该权重是**未对齐指令的 base checkpoint**（与历史 `ask_dsv41_serve.sh` 注释一致），
  官方 thinking 前缀反而触发重复；**chat 模式至少能答对部分样本**。
- **处置**：RL 改用 `ENABLE_THINKING=false`（chat 模式），`MAX_RESPONSE_LEN=512`；
  reward 的兜底逻辑（无 boxed 时取 `</think>` 后/全文最后一个整数）可命中 “The final answer is 2”。
- 12 行贪心基线评估进行中（用于确认初始 reward 非零）。

### 2026-10-19 17:38 12 行贪心基线（chat 模式，官方 encoder，无 RL）
- 结果：1/12 直接判对；另 1 条（gt=2）回答 “The final answer is 2.” 因正则把句号当小数点误判为无答案。
- **修正**：`_STANDALONE_INT_RE` 改为允许句末句号、排除小数（`(?<![\d.])[+-]?\d+(?!\d|[A-Za-z]|\.\d)`），
  复测 “2.”→2、“2.5”→不匹配、“**38. **…5.**”→5。修正后基线约 **2/12（~17%）**。
- 结论：base 模型在 chat 模式下有非零正确率，可提供 GRPO 组内奖励方差；thinking 模式全退化，不采用。

### 2026-10-19 17:45 重启（118，chat 模式 + response 512 + 官方 encoder + images 还原）
- 日志：`logs/countbenchqa_118_20261019_174506.nohup.log`，pid 2336898。
- 生效配置确认：`enable_thinking=false`、`max_response_length=512`、
  `gpu_memory_utilization=0.6`、`default_agent_loop=deepseek_v41_vl_single_turn`；
  dataset 450/50，`Built DeepSeek-V4.1 VL processor` 正常。

### 2026-10-19 18:09 **第一个 reward 出现**（chat 模式首批 8 prompt × 8 采样）
- `VERL_AGENT_DUMP` 证据（`logs/agent_dump_countbenchqa.2417146/2417148/2417149`）：
  - uid `464a4dae…`（“How many sumo wrestlers”，gt=2）：
    - sid=3 **reward=0.9**（rlen 298，尾部 “The final answer is 2 and the boxed 2.”）
    - sid=4 **reward=0.9**（rlen 512）
    - sid=0/2 reward=0.0（重复循环）
    → **组内奖励有方差**，GRPO 有学习信号。
  - uid `d905291a…`（“How many dice”）：4 条均 0.0（全部循环到 512）。
- 训练侧：18:08 起进入 forward/backward（`torch.utils.checkpoint`、`_jagged_to_padded_dense_forward`
  CPU fallback 警告），说明 **images/token_types 丢失问题已修复**，视觉前向跑通。
- 观察：base 模型仍以重复循环为主，boxed 格式几乎不遵守；但 reward 兜底（最后一个整数）
  能识别 “The final answer is 2.”，给出 0.9。

### 2026-10-19 18:26 **step:1 完成（验收达成）**
```
step:1 - global_seqlen/mean:2843.8 - actor/entropy:1.5290
 - training/rollout_probs_diff_mean:0.007628 - training/rollout_probs_diff_max:0.3399
 - training/rollout_actor_probs_pearson_corr:0.99892
 - actor/pg_loss:0.035689 - actor/grad_norm:3.7277 - actor/lr:5e-06
 - actor/perf/max_memory_allocated_gb:26.06 - actor/perf/max_memory_reserved_gb:48.15
 - actor/perf/cpu_memory_used_gb:1105.3
 - critic/score/mean:0.21094  critic/score/max:0.9  critic/score/min:0.0
 - critic/rewards/mean:0.21094  critic/rewards/max:0.9  critic/rewards/min:0.0
 - critic/advantages/mean:-0.03569 max:1.6202 min:-1.6202
 - response_length/mean:460.95  max:512  min:90  clip_ratio:0.84375
 - timing_s/gen:213.2 old_log_prob:26.3 update_actor:220.0 update_weights:784.6 step:1265.5
 - perf/throughput:2.247 tok/s/?
```
- **首个 reward 均值 0.211**（组内有 0.9 满分样本），训推一致性 pearson 0.9989（视觉 token 流对齐 ✓）。
- 步时 21 分钟，其中权重同步 784s（通用 per-expert export，见 E04）；gen 213s、update 220s。
- 显存峰值 allocated 26.1GB / reserved 48.1GB（单卡 64GB），CPU 内存 1105GB（2.4T 机器内）。
- 继续跑，10 分钟巡检；除非用户叫停不停止。reward 曲线观察项：critic/rewards/mean、
  response_length/clip_ratio、pearson、grad_norm。

### 2026-10-19 18:53 step:2 完成（reward 上升）
```
step:2
actor/entropy:1.2989120483398438
training/rollout_actor_probs_pearson_corr:0.998635470867157
actor/pg_loss:0.08857891877414659
actor/grad_norm:2.783572196960449
actor/perf/max_memory_allocated_gb:26.45560884475708
actor/perf/max_memory_reserved_gb:48.3671875
training/global_step:2
critic/rewards/mean:0.25312501192092896
critic/rewards/max:0.8999999761581421
critic/rewards/min:0.0
response_length/mean:456.59375
response_length/max:512.0
response_length/min:24.0
response_length/clip_ratio:0.828125
timing_s/gen:180.89462574012578
timing_s/old_log_prob:24.23307002009824
timing_s/adv:0.16766424989327788
timing_s/update_actor:187.64672637009062
timing_s/update_weights:787.5064297500066
timing_s/step:1180.5173648700584
```
- reward 曲线：step1 0.2109 → **step2 0.2531**（+0.042）；pearson 0.9986；步时 1180s。

### 巡检 2026-10-19 18:55（node-118）
- 进程：pid 2336898（启动脚本）存活 1h10m；16 个 WorkerDict 存活；无 Traceback/OOM/ERROR。
- 日志：`countbenchqa_118_20261019_174506.nohup.log`，mtime 18:53，size 3.04MB。
- 指标：step1 reward **0.2109** / step2 **0.2531**（↑0.042）；
  response_length mean 461→457，clip_ratio 0.844→0.828；
  pearson 0.99892→0.99864；grad_norm 3.73→2.78；entropy 1.529→1.299；
  步时 1265s→1180s（其中 update_weights 785→788s）。
- 资源：NPU 16 卡 HBM 合计 used **470GB**（~29.4GB/卡），free 554GB（~34.6GB/卡）；
  该时刻 AICore 0%（处于权重同步 CPU 阶段）；主机内存 used 1214G / available 1282G。
- 状态：step3 已生成（18:46-18:47 running 8），正在 update/sync；继续跑。

### 巡检 2026-10-19 19:04（node-118）
- 进程 pid 2336898 存活 1h19m；16 WorkerDict 各 ~140% CPU；无 Traceback/OOM/ERROR。
- step1/2 指标同上；**step3 仍在权重同步**（py-spy 栈：update_weights → unfuse_moe_params →
  _export_param → wait_tensor，即通用 per-expert 导出，约 13 分钟）。
- NPU：16 卡 HBM used 470GB（~29.4GB/卡），free 554GB；AICore 0%（同步 CPU 阶段）。
- 主机内存 used 1214G / available 1282G；日志 mtime 18:53（同步完成时会刷新）。
- 结论：健康，等待 step3 完成。

### 巡检 2026-10-19 19:14（node-118）—— step3 完成，reward 继续上升
- 进程 pid 2336898 存活 1h29m；无 Traceback/OOM/ERROR。
- **step3**: critic/rewards/mean **0.3656**（max 0.9）；response_length/mean 396.1，
  clip_ratio 0.672（较 step2 的 0.828 明显下降 → 重复循环减少）；
  probs_diff_mean 0.0102，pearson 0.99742；grad_norm 2.127；entropy 1.130；
  步时 1168s（update_weights 793.9s）。
- **reward 曲线：0.2109 → 0.2531 → 0.3656（连续上升）**。
- 资源：HBM used 470.1GB / free 553.9GB；AICore 0%（同步阶段）；内存 1222G used / 1280G avail。
- step4 已开始（19:05-19:06 running 7-8）。

### 巡检 2026-10-19 19:24（node-118）
- pid 2336898 存活 1h39m；无 Traceback/OOM/ERROR。
- step4 尚未落指标：生成已完成（19:05-19:06 running 7-8），当前在 **update_weights**
  （py-spy: update_weights → unfuse_moe_params → _export_param，~140% CPU/rank）。
- 资源稳定：HBM used 470.1GB / free 553.9GB；AICore 0%（同步阶段）；
  内存 1221G used / 1280G avail；日志 mtime 19:12（step3 指标）。
- 预计 step4 指标 ~19:30 落盘。

### 巡检 2026-10-19 19:34（node-118）—— step4 完成，reward 明显上升
- **step4**: critic/rewards/mean **0.5625**（max 0.9）；response_length/mean **298.9**，
  clip_ratio **0.469**（step3 0.672 → step4 0.469，模型在学会更短作答）；
  probs_diff_mean 0.01016，pearson 0.99726；grad_norm 4.916；entropy 1.112；
  步时 1177s（update_weights 782.5s）。
- **reward 曲线：0.2109 → 0.2531 → 0.3656 → 0.5625（连续上升，+0.35 over 4 steps）**。
- 进程 pid 2336898 存活 1h49m；无 Traceback/OOM/ERROR；step5 已开始（19:25-19:26 running 8）。
- 资源：HBM used 470.1GB / free 553.9GB；AICore 0%（同步阶段）；内存 1234G used / 1270G avail。

### 巡检 2026-10-19 19:48（node-118）—— step5 完成，reward 高位持平
- **step5**: critic/rewards/mean **0.5484**（max 0.9）；response_length/mean **281.3**，
  clip_ratio **0.4375**；probs_diff_mean 0.01073，pearson 0.99586；
  grad_norm 3.113；**entropy 0.641**（step4 1.112，下降较快）；步时 1181s。
- **reward 曲线**：0.2109 → 0.2531 → 0.3656 → 0.5625 → **0.5484**（高位平台，单步波动属正常）。
- 响应长度持续下降（461→281），截断率 0.844→0.4375，模型输出更短更直接。
- 进程健康，无 error；step6 已开始（19:46 running 8）。
- 关注：entropy 0.641 下降较快，后续若继续骤降需留意策略坍缩（届时评估是否调
  entropy_coeff / 学习率）；当前 grad_norm/pearson 正常。

### 巡检 2026-10-19 19:54（node-118）
- pid 2336898 存活 2h09m；无 Traceback/OOM/ERROR。
- step6：生成完成（19:46），当前在 update_weights（py-spy 确认 unfuse_moe_params/_export_param，
  ~139% CPU/rank）；预计 ~20:03 落指标。
- 资源稳定：HBM used 470.1GB / free 553.9GB；AICore 0%（同步阶段）；
  内存 1245G used / 1278G avail。

### 巡检 2026-10-19 20:04（node-118）—— step6 完成，reward 创新高
- **step6**: critic/rewards/mean **0.7172**（max 0.9）；response_length/mean 328.2，
  clip_ratio 0.516；probs_diff_mean 0.01003，pearson 0.99623；
  grad_norm 1.891；entropy 0.6895；步时 1163s（update_weights 811.4s）。
- **reward 曲线**：0.2109 → 0.2531 → 0.3656 → 0.5625 → 0.5484 → **0.7172**（新高）。
- 资源快照（生成/更新活跃期）：HBM used **824.1GB** / free 199.9GB；
  AICore max 53% / avg 29.4%（引擎工作）；内存 1244G used / 1278G avail。
- 进程健康，无 error；step7 进行中。

### 巡检 2026-10-19 20:14（node-118）
- pid 2336898 存活 2h29m；无 Traceback/OOM/ERROR。
- step7：生成完成（20:05 起 running 8），当前处于 update/sync 阶段（HBM 470GB、AICore 0%）；
  预计 ~20:25 落指标。
- 资源：HBM used 470.0GB / free 554.0GB；内存 1248G used / 1279G avail。
- 已完成曲线：0.211 → 0.253 → 0.366 → 0.563 → 0.548 → 0.717（step6 新高）。

### 巡检 2026-10-19 20:24（node-118）—— step7 完成
- **step7**: critic/rewards/mean **0.6609**（max 0.9）；response_length/mean **250.8**，
  clip_ratio **0.3906**；probs_diff_mean 0.01182，pearson 0.99492；
  grad_norm 3.605；entropy 0.5972；步时 1125s（update_weights 812.5s）。
- **reward 曲线**：0.2109 → 0.2531 → 0.3656 → 0.5625 → 0.5484 → 0.7172 → **0.6609**
  （高位平台 0.55-0.72，单步 ±0.07）。
- 资源快照（生成活跃）：HBM used 824.8GB / free 199.2GB；AICore max **91%** / avg 53.9%；
  内存 1247G used / 1279G avail。
- 进程健康，无 error；step8 生成中。
- 关注 entropy 0.597（连续下降），暂未影响 reward / pearson。

### 巡检 2026-10-19 20:34（node-118）
- pid 2336898 存活 2h49m；无 Traceback/OOM/ERROR。
- step8：生成完成（20:24 running 8），当前 update/sync（HBM 470GB、AICore 0%）；
  预计 ~20:44 落指标。
- 资源：HBM used 470.1GB / free 553.9GB；内存 1250G used / 1279G avail。
- 已完成曲线：0.211/0.253/0.366/0.563/0.548/0.717/0.661。

### 巡检 2026-10-19 20:44（node-118）—— step8 完成
- **step8**: critic/rewards/mean **0.6891**（max 0.9）；response_length/mean 289.3，
  clip_ratio 0.422；probs_diff_mean 0.01124，pearson 0.99592；
  grad_norm 6.108；entropy 0.6781（较 step7 0.597 回升）；步时 1156s。
- **reward 曲线**：0.2109 → 0.2531 → 0.3656 → 0.5625 → 0.5484 → 0.7172 → 0.6609 → **0.6891**。
- 资源快照（生成活跃）：HBM used 824.1GB / free 199.9GB；AICore max 91% / avg 60.0%；
  内存 1251G used / 1276G avail。
- 进程健康，无 error；step9 生成中。

### 巡检 2026-10-19 21:05（node-118）—— step9 完成
- **step9**: critic/rewards/mean **0.6891**（max 0.9）；response_length/mean **186.4**，
  clip_ratio **0.2188**（step8 0.422 → 大幅下降，78% 响应自然结束）；
  probs_diff_mean 0.01489，pearson 0.99384；grad_norm 1.947；entropy 0.7535；步时 1227s。
- **reward 曲线**：… 0.7172 → 0.6609 → 0.6891 → **0.6891**（稳定平台 ~0.69）。
- 资源：HBM 470GB（同步阶段）；内存 1253G used / 1277G avail；进程健康，无 error。
- 观察：响应长度 461→186、截断率 0.84→0.22，模型已收敛到简短作答；
  reward 平台 0.69，max 0.9。

### 巡检 2026-10-19 21:05（node-118，step9 刚落盘）
- 进程 pid 2336898 存活 3h20m；无 Traceback/OOM/ERROR。
- step10：21:04 起生成中（running 8）；step9 指标见上一条。
- 资源：HBM used 824.0GB / free 200.0GB（引擎唤醒，KV 已分配）；
  AICore max 48% / avg 3.4%（生成刚开始）；内存 1252G used / 1277G avail。

### 巡检 2026-10-19 21:21（node-118）—— step10 完成（10 步里程碑）
- **step10**: critic/rewards/mean **0.703125**（max 0.9）；response_length/mean 339.6，
  clip_ratio 0.578；probs_diff_mean 0.01003，pearson 0.99548；
  grad_norm 0.941；entropy 0.5964；步时 1133s（update_weights 791s）。
- **10 步曲线**：0.211, 0.253, 0.366, 0.563, 0.548, 0.717, 0.661, 0.689, 0.689, **0.703**
  → 前 4 步快速上升，之后稳定在 **0.66-0.72** 平台；后 5 步均值 ~0.69。
- 资源：引擎唤醒时 HBM ~52.7GB/卡（free ~12.8GB/卡），AICore 生成中（快照 0-5%）；
  内存 1255G used / 1279G avail；进程 pid 2336898 存活 3h36m；无 error。
- 结论：reward 曲线正常（上升后稳定），持续跑。

### 巡检 2026-10-19 21:24（node-118）
- 进程 pid 2336898 存活 3h39m；无 Traceback/OOM/ERROR。
- step11：21:22 起生成中（running 8）；step10 = 0.7031（上文）。
- 资源：HBM used 387.4GB / free 636.6GB（生成早期）；AICore max 35% / avg 11.4%；
  内存 1262G used / 1272G avail。

### 巡检 2026-10-19 21:42（node-118）—— step11 完成，reward 创新高
- **step11**: critic/rewards/mean **0.7875**（max 0.9）；response_length/mean 283.1，
  clip_ratio 0.469；probs_diff_mean 0.01134，pearson 0.99465；
  grad_norm 1.737；entropy 0.6911；步时 1182s（update_weights 793.9s）。
- **11 步曲线**：0.211, 0.253, 0.366, 0.563, 0.548, 0.717, 0.661, 0.689, 0.689, 0.703, **0.7875**
  （新高，平台继续上移）。
- 进程 pid 2336898 存活 3h57m；无 Traceback/OOM/ERROR；内存 1256G used / 1278G avail。

### 巡检 2026-10-19 21:44（node-118）
- 进程 pid 2336898 存活 3h59m；无 Traceback/OOM/ERROR。
- step12：生成中（21:41-21:42 running 7、finished 1）；step11 = 0.7875（新高，上文）。
- 资源：HBM used 652.9GB / free 371.1GB（生成中）；AICore max 52% / avg 8.5%；
  内存 1262G used / 1272G avail。

### 巡检 2026-10-19 22:03（node-118）—— step12 完成
- **step12**: critic/rewards/mean **0.6188**（max 0.9）；response_length/mean 237.9，
  clip_ratio 0.328；probs_diff_mean 0.01265，pearson 0.99412；
  grad_norm 1.505；entropy 0.5817；步时 1135s（update_weights 796.5s）。
- **12 步曲线**：0.211, 0.253, 0.366, 0.563, 0.548, 0.717, 0.661, 0.689, 0.689, 0.703,
  0.7875, **0.6188**（单步波动 ±0.08-0.1；12 步均值 ≈0.567，后 6 步 ~0.70）。
- 进程 pid 2336898 存活 4h18m；无 Traceback/OOM/ERROR；内存 1265G used / 1269G avail。

### 巡检 2026-10-19 22:04（node-118）—— step13 生成中；HBM 瞬态偏紧
- 进程 pid 2336898 存活 4h19m；无 Traceback/OOM/ERROR。
- step13 生成中（22:00-22:01 running 8）；step12 = 0.6188（上文）。
- **HBM 明细**（npu-smi 进程表）：每卡只有我们两个进程 —
  训练 WorkerDict **41.7-48.4GB/卡** + vLLM worker **210MB/卡（asleep）**，无外部租户。
  生成期快照 HBM used 965.3GB（free 58.7GB，~3.7GB/卡），偏紧但未 OOM；
  生成结束后回落到 ~48-54GB/卡（引擎 sleep，training 占用为主）。
- 风险：生成期余量小，若后续 OOM 优先考虑下调 `gpu_memory_utilization`（需重启，
  当前无 ckpt，会丢进度）；暂时继续观察。内存 1265G used / 1268G avail。

### 巡检 2026-10-19 22:20（node-118）—— step13 完成
- **step13**: critic/rewards/mean **0.73125**（max 0.9）；response_length/mean 282.2，
  clip_ratio 0.469；probs_diff_mean 0.01163，pearson 0.99255；
  grad_norm 1.247；**entropy 0.456**（step12 0.582，继续下降）；步时 1163s（update_weights 831.9s）。
- **13 步曲线**：… 0.703, 0.7875, 0.6188, **0.73125**（平台 0.62-0.79）。
- 进程 pid 2336898 存活 4h35m；无 Traceback/OOM/ERROR；内存 1261G used / 1279G avail。
- 关注：entropy 0.456 已较低（1.53→0.46），grad_norm 1.25；若 entropy 继续骤降且
  reward 回落，再评估 entropy_coeff/重启；当前 reward 仍稳。

### 巡检 2026-10-19 22:24（node-118）
- 进程 pid 2336898 存活 4h39m；无 Traceback/OOM/ERROR。
- step14 生成中（22:20 起 running 8）；step13 = 0.73125（entropy 0.456，上文）。
- 资源：HBM used 876.0GB / free 148.0GB（~9.2GB/卡）；AICore max **100%** / avg 15.9%；
  内存 1268G used / 1271G avail。生成期余量较紧但正常。

### 巡检 2026-10-19 22:40（node-118）—— step14 完成（首个满分样本）
- **step14**: critic/rewards/mean **0.6203**；**max 1.0（首次出现满分：答案+格式都对）**；
  response_length/mean 247.1，clip_ratio 0.391；probs_diff_mean 0.01095，
  pearson 0.99481；grad_norm 3.224；entropy 0.4768；步时 1202s（update_weights 820.6s）。
- **14 步曲线**：… 0.703, 0.7875, 0.6188, 0.7313, **0.6203**（平台 0.62-0.79，单步噪声）。
- 进程 pid 2336898 存活 4h55m；无 Traceback/OOM/ERROR；内存 1263G used / 1277G avail。

### 巡检 2026-10-19 22:44（node-118）
- 进程 pid 2336898 存活 4h59m；无 Traceback/OOM/ERROR。
- step15 生成中（22:39-22:40 running 8）；step14 = 0.6203（max 1.0，上文）。
- 资源：HBM used 804.0GB / free 220.0GB；AICore max 96% / avg 27.3%（生成）；
  内存 1269G used / 1270G avail。

### 巡检 2026-10-19 23:00（node-118）—— step15 完成
- **step15**: critic/rewards/mean **0.7875**（max 0.9）；response_length/mean **195.1**，
  clip_ratio **0.297**；probs_diff_mean 0.01255，pearson 0.99137；
  grad_norm 1.550；**entropy 0.379**（继续下降）；步时 1228s（update_weights 805s）。
- **15 步曲线**：… 0.7313, 0.6203, **0.7875**（平台 0.62-0.79；15 步均值 ≈0.60）。
- 进程 pid 2336898 存活 5h15m；无 Traceback/OOM/ERROR；内存 1263G used / 1277G avail。
- 观察：entropy 0.379 偏低、response 185-283 波动、pearson 0.9914 仍 >0.99；
  reward 未见回落，继续跑并关注策略坍缩迹象。

### 巡检 2026-10-19 23:04（node-118）
- 进程 pid 2336898 存活 5h19m；无 Traceback/OOM/ERROR。
- step16 生成中（23:01 起 running 8）；step15 = 0.7875（上文）。
- 资源：HBM used 906.1GB / free 117.9GB（~7.4GB/卡，生成期偏紧）；AICore 快照 0%；
  内存 1270G used / 1270G avail。step15 batch 的 max seqlen 5584。

### 巡检 2026-10-19 23:20（node-118）—— step16 完成
- **step16**: critic/rewards/mean **0.6609**（max 0.9）；response_length/mean 223.5，
  clip_ratio 0.3125；probs_diff_mean 0.01254，pearson 0.99193；
  grad_norm 2.443；entropy 0.3874（stable low）；步时 1199s（update_weights 801s）。
- **16 步曲线**：… 0.7313, 0.6203, 0.7875, **0.6609**（平台 0.62-0.79）。
- 进程 pid 2336898 存活 5h35m；无 Traceback/OOM/ERROR；内存 1266G used / 1278G avail。
- 观察：entropy 0.387 稳定、pearson 0.9919；reward 平台波动正常。继续跑。

### 巡检 2026-10-19 23:24（node-118）
- 进程 pid 2336898 存活 5h39m；无 Traceback/OOM/ERROR。
- step17 生成中（23:21 起 running 8）；step16 = 0.6609（上文）。
- 资源：HBM used 868.0GB / free 156.0GB；AICore max 96% / avg 24.0%（生成）；
  内存 1272G used / 1270G avail。

### 巡检 2026-10-19 23:41（node-118）—— step17 完成，reward 新高
- **step17**: critic/rewards/mean **0.8453**（max **1.0**）；response_length/mean 210.4，
  clip_ratio 0.328；probs_diff_mean 0.01062，pearson 0.99359；
  grad_norm 1.036；entropy 0.4155；步时 1205s（update_weights 815.5s）。
- **17 步曲线**：… 0.7313, 0.6203, 0.7875, 0.6609, **0.8453**（新高，平台继续上移）。
- 进程 pid 2336898 存活 5h56m；无 Traceback/OOM/ERROR；内存 1267G used / 1279G avail。
- entropy 0.416 稳定、grad_norm 1.04；reward 未见退化，持续跑。

### 巡检 2026-10-19 23:44（node-118）
- 进程 pid 2336898 存活 5h59m；无 Traceback/OOM/ERROR。
- step18 生成中（23:41 running 5、finished 3）；step17 = 0.8453（新高，上文）。
- 资源：HBM used 812.0GB / free 212.0GB；AICore max 97% / avg 17.9%（生成）；
  内存 1274G used / 1269G avail。

### 巡检 2026-10-20 00:02（node-118）—— step18 完成
- **step18**: critic/rewards/mean **0.8016**（max 0.9）；response_length/mean **135.3**，
  clip_ratio **0.156**（仅 15.6% 顶到 512）；probs_diff_mean 0.01791，
  pearson 0.99069；grad_norm 1.701；entropy 0.5187；步时 1133s（update_weights 808.5s）。
- **18 步曲线**：… 0.7875, 0.6609, 0.8453, **0.8016**（近两步稳定在 ~0.80）。
- 进程 pid 2336898 存活 6h17m；无 Traceback/OOM/ERROR；内存 1274G used / 1271G avail。
- 观察：response 长度 135、clip 0.156，模型已收敛为简短直接作答；
  pearson 0.9907 仍 >0.99；reward 稳定高位。继续跑。

### 巡检 2026-10-20 00:04（node-118）
- 进程 pid 2336898 存活 6h19m；vLLM 引擎存活（vLLMHttpServer 2372107 + DPCoordinator
  + 2×EngineCore + 16×Worker_DP，当前 sleep）；无 Traceback/OOM/ERROR。
- step19 生成完成（00:00 running 7 / finished 1），当前 `actor_rollout_update_actor`。
- 资源：HBM used **275.5GB** / free 748.5GB（低占用更新阶段：训练 shard 11-13GB/卡 +
  vLLM sleep 保留 7-10GB/卡）；AICore 快照 0%；内存 1274G used / 1270G avail。
- step18 = 0.8016（上文），step19 指标预计 ~00:21。

### 巡检 2026-10-20 00:21（node-118）—— step19 完成
- **step19**: critic/rewards/mean **0.5922**（max 1.0）；response_length/mean 177.1，
  clip_ratio 0.219；probs_diff_mean 0.01448，pearson 0.99394；
  grad_norm 1.800；entropy 0.7304（回升）；步时 1153s（update_weights 799.3s）。
- **19 步曲线**：… 0.7875, 0.6609, 0.8453, 0.8016, **0.5922**（高位波动，单步 ±0.1-0.2）。
- 进程 pid 2336898 存活 6h36m；无 Traceback/OOM/ERROR；内存 1276G used / 1269G avail。
- 观察：长序列减少（resp 177、clip 0.219），entropy 回升 0.73，训练健康。继续跑。

### 巡检 2026-10-20 00:24（node-118）
- 进程 pid 2336898 存活 6h39m；无 Traceback/OOM/ERROR。
- step20 生成完成（00:19 running 8），当前 update/sync（HBM 470GB、AICore 0%）；
  预计 ~00:41 落指标。
- 资源：HBM used 470.1GB / free 553.9GB；内存 1272G used / 1274G avail。

### 巡检 2026-10-20 00:41（node-118）—— step20 完成（20 步里程碑）
- **step20**: critic/rewards/mean **0.7453**（max 0.9）；response_length/mean 308.2，
  clip_ratio 0.516；probs_diff_mean 0.00923，pearson 0.99432；
  grad_norm 1.203；entropy 0.3736；步时 1169s（update_weights 810.9s）。
- **20 步汇总**：
  - 全 20 步均值 **0.6295**；前 10 步均值 **0.540**；**后 10 步均值 0.719**（+0.179）。
  - 逐 step：0.211, 0.253, 0.366, 0.563, 0.548, 0.717, 0.661, 0.689, 0.689, 0.703,
    0.788, 0.619, 0.731, 0.620, 0.788, 0.661, 0.845, 0.802, 0.592, 0.745。
  - 后 10 步中 4 步 ≥0.78，多次出现 max=1.0；response 长度 461→~200-300，clip 0.84→0.16-0.52。
- 进程 pid 2336898 存活 6h56m；无 Traceback/OOM/ERROR；内存 1279G used / 1267G avail。
- 结论：reward 曲线正常上升并稳定在更高平台（后 10 步均值 0.72），继续持续跑。

### 巡检 2026-10-20 00:44（node-118）
- 进程 pid 2336898 存活 6h59m；无 Traceback/OOM/ERROR。
- step21 生成完成（00:38-00:39 running 7-8），当前 update/sync（HBM 470GB、AICore 0%）。
- step20 = 0.7453（20 步均值 0.630、后 10 步 0.719，上文）。
- 资源：HBM used 470.1GB / free 553.9GB；内存 1272G used / 1276G avail。

### 巡检 2026-10-20 01:01（node-118）—— step21 完成
- **step21**: critic/rewards/mean **0.7453**（max 0.9）；response_length/mean 222.3，
  clip_ratio 0.3125；probs_diff_mean 0.01091，pearson 0.99501；
  grad_norm 1.240；entropy 0.5993（回升）；步时 1158s（update_weights 809.5s）。
- **21 步曲线**：… 0.8453, 0.8016, 0.5922, 0.7453, **0.7453**（近两步持平 0.745）。
- 进程 pid 2336898 存活 7h16m；无 Traceback/OOM/ERROR；内存 1279G used / 1269G avail。
- 观察：entropy 0.599 回升、pearson 0.9950；训练稳定。继续跑。

### 巡检 2026-10-20 01:04（node-118）
- 进程 pid 2336898 存活 7h19m；无 Traceback/OOM/ERROR。
- step22 生成完成（00:58 running 6 / finished 2），当前 update/sync（HBM 470GB、AICore 0%）。
- step21 = 0.7453（近两步持平，上文）。
- 资源：HBM used 470.0GB / free 554.0GB；内存 1272G used / 1275G avail。

### 结束记录 2026-10-20 01:42（node-118）—— 用户指令暂停 / 杀进程
- 01:42 用户指令「暂停执行」「杀掉进程」；已删除巡检 cron（id `49db6732`），不再自动重启。
- 清理命令：`kill -9 -2336898`（启动脚本进程组）；`pkill -9 -f` 括号保护模式
  `V[L]LM::`、`ra[y]::`、`dsv41_grpo_countbenchq[a]`、`verl.trainer.main_pp[o]`、`vllm ser[v]e`；
  `ray stop --force`；`kill -9 2337166`。
- 01:47 复核：
  - 训练 / vLLM / EngineCore / raylet / GCS **活进程为 0**，无 D 状态进程。
  - 32 个 `ray::WorkerDict` 为 `<defunct>`（PPID 1、RSS=0，仅剩 task_struct，不占 HBM；
    容器内无法回收，随容器/ray 重启消失）。
  - `npu-smi` 16 chip 进程表为空、AICore 0%；容器 RAM used 1272G→840G，
    free 601G / available 1693G。
  - 注意：npu-smi HBM used 仍显示 470.1GB（16 chip），但进程表为空——系宿主机其他容器/租户
    占用（本容器进程已全部释放）；若后续重启训练，先复查 118 是否存在资源竞争，
    必要时按 126→118→122 换机。
- 最终训练进度：**完成 23 步**；第 24 步 generation 中（01:35:54 running 8）于 01:41-01:42 被杀，无指标。
  - step21 = 0.7453；step22 = 0.71875（max 1.0）；step23 = 0.590625（max 0.9），
    entropy 0.5659，pearson 0.99485，grad_norm 1.349，步时 1128.9s（update_weights 805.1s）。
  - **23 步均值 0.6368**；前 10 步 0.540；后 10 步（step14-23）**0.7108**。
  - 全程无 Traceback / OOM / ERROR；rollout 与训练 logprob 一致性 pearson ≈0.99。
- 关键产物：
  - 日志 `logs/countbenchqa_118_20261019_174506.nohup.log`（3.7MB，最后一条 TaskRunner 时间 01:35:54）
  - 启动脚本 `my_scripts/dsv41_grpo_countbenchqa_1node.sh`
  - 数据 `dsv41/rl_data/countbenchqa_lite_{train,val}.parquet`（450/50）
  - reward `verl/utils/reward_score/countbenchqa.py`；agent loop
    `verl/experimental/agent_loop/dsv41_vl_agent_loop.py`
- 状态：已停止，等待用户下一步指令（未自动重启）。
