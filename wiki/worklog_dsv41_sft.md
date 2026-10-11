# worklog：DeepSeek-V4.1 top32 单机 SFT 热启动（GSM8K）

> **任务定位（用户 2026-10-10 指示）**
> - **初步目标**：SFT 一个能正常回答计算题的 32 专家模型——当前 top32 权重回答数学题**陷入循环思考**（对应 RL 侧"never-EOS、reward 恒 -1"的模型侧根因）。
> - **最终目标**：接回昨天的 verl 全流程，**reward 有数值**。
> - **分工**：本文件记录 **SFT** 线；**verl/RL 线**继续记在 `worklog_dsv41_direct_shard_load.md`（§六 起）。
> - **纪律**：131 与 128 两台机器的代码/脚本/日志**保持一致**——每次改动后跑 `scripts/sync_to_128.sh`（tar over ssh，末尾 md5 对账即验收）。
> - **验收要求（用户）**：SFT 前后用数学题做**对比检验**（当前模型循环思考/答不对 → SFT 后应能收尾并给出答案）。

## 0. 机器与环境

| 项 | 说明 |
|---|---|
| 131（主） | `node-29-131`，16 卡（8×双芯 NPU），本会话所在 |
| 128（副） | `ssh -p 50001 root@80.5.25.128`，`node-29-128`，**当前 131 镜像的拷贝**（代码已 md5 对账一致） |
| 共享存储 | `/mnt/share` 两机共享（权重、数据、sft_data 都有） |
| 同步 | `scripts/sync_to_128.sh`（无 rsync，用 tar \| ssh；含 4 个热文件 md5 对账） |
| 16 卡利用 | vLLM 侧用 **DP2+TP8+dp_shared_memory**（worklog_dsv41_direct_shard_load.md §八 已验代码）；训练侧单机 FSDP16/EP8 |

## 1. 方案（定稿于 2026-10-10，依据：单机容量账 + fork 自带的引擎版 SFT）

**容量结论**：单机 16 卡跑 top32 SFT 宽裕——HBM 参数+梯度 ≈13GB/卡 + 激活（GSM8K 序列极短）≤15GB；主机 engram 366GB + 优化器 offload ~428GB ≈ 1.1TB < 2TB。RL 单机 16 卡当年死于 host（trainer+engine 两份 engram），**SFT 无 vLLM 引擎副本 → 天然可行**。

**框架**：fork 自带 `verl/trainer/sft_trainer_ray.py` + `sft_trainer_engine.yaml`（走 `engine_workers.TrainingWorker`，与 RL 同套引擎；默认配置就是 GSM8K 模板）。接线要点=把 `engine.strategy=fsdp_turbo` + turbo_config（FSDP16/EP8/ffn-recompute）+ 关动态批 + 模型路径/tokenizer 配进去（详见 §3 待办）。

## 2. 进度

### ✅ 阶段 ①：数据（2026-10-10 完成）
- 官方 GSM8K 就位：`/mnt/share/m00899630/dataset/gsm8k_official/{main,socratic}/{train,test}-*.parquet`（main train 7473 / test 1319 ✓ 已验）。
- 转换脚本：`scripts/prepare_gsm8k_sft.py`（官方 question/answer → verl `MultiTurnSFTDataset` 的 `messages` 列）：
  - 去 `<<expr=res>>` 标注；尾部 `#### N` → 末行 `Answer: N`（**与 RL 协议一致**）；
  - 助手轮结尾由 rl_tokenizer 模板自动带 `<｜end▁of▁sentence｜>`（**EOS 进训练目标 = 治 never-EOS 的药**）；
  - >max_length 过滤（本轮 0 条被丢）。
- 产物（本地 + 备份到共享）：`/mnt/share/m00899630/dsv41/sft_data/gsm8k_{main,socratic}_{train,test}_sft.parquet`
- 抽检原文：`…<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\nAnswer: 72<｜end▁of▁sentence｜>` ✓
- **长度实测**：main train token 数 mean 140 / p95 237 / **max 424**（远小于 1024 上限）→ 显存与吞吐都宽裕。

### ⏳ 阶段 ②：1 步冒烟（待办，门槛=所有新接线唯一验收点）
- 检查 `sft_trainer_ray.py` 的引擎接线（strategy=fsdp_turbo 是否直达 dsv41 引擎构建）、loss 路径、ckpt 保存。
- 起 16 卡 1 步：验收 loss 有限、显存/host 记录、**ckpt 试存一次**（dsv41+engram 的保存从未跑过）。

### ⏳ 阶段 ③：全量短跑（1 epoch = 7473 条）
### ⏳ 阶段 ④：评测（**前后对比**）
- **前**：baseline=当前 top32 原始权重，用 vLLM（DP2/TP8/16 卡，见 §0）对 GSM8K test 抽样生成，记录：是否收尾（EOS/finish_reason）、答案正确率、循环样例原文。
- **后**：SFT ckpt 导出 → 同题同采样 → 对比三项。
### ⏳ 阶段 ⑤：回 RL（verl 全流程，reward 有数值）— 记在 direct_shard_load 日志。

## 3. 关键配置（草稿，冒烟时定稿）

```bash
# 引擎（与 RL 同款接线，engine@engine: fsdp → fsdp.yaml）
engine.strategy=fsdp_turbo
engine.turbo_config.distributed.fully_shard_parallel_size=16
engine.turbo_config.distributed.expert_parallel_size=8
engine.turbo_config.memory.recompute=true
engine.turbo_config.memory.recompute_plan=["model.layers.{*}.ffn"]
# 模型
model.path=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16   # + tokenizer_path=rl_tokenizer
# 数据（dsv41 要求关 remove_padding/动态批）
data.train_files=/mnt/share/m00899630/dsv41/sft_data/gsm8k_main_train_sft.parquet
data.messages_key=messages  data.max_length=1024  data.use_dynamic_bsz=False
data.micro_batch_size_per_gpu=1  data.train_batch_size=16
# 优化；总步/epoch 冒烟后定
optim.lr=1e-5   checkpoint.save_contents=["model"]
```

## 4. 风险清单
1. `MultiTurnSFTDataset` 逐轮模板拼接 vs 整段渲染的差异（自带 `ignore_input_ids_mismatch` 兜底）→ 冒烟抽查逐 token；
2. ckpt 保存（dsv41/engram）从未验证 → 冒烟里试；engram 冻结可不存；
3. 吞吐未知 → 冒烟计时（GSM8K 短序列预计很快）；
4. SFT 后仍偶发长尾 → 加 epoch/加闭合样本，或用带 `<think>` 格式再训一版。

## 5. 更新（2026-10-10 傍晚）：冒烟通过 + 三个修复 + 全量启动

### ✅ 阶段② 冒烟结果（128 机器，1 步）
- `[fsdp_turbo_dsv41] parameters materialized in 30.5s (direct shard load)` → `Epoch 1/1: 0/467` → **step:1 完成**：
  - `train/loss: 1.8815118074417114`（有限、合理初值）`train/grad_norm: 15.391`（梯度在流）`train/lr: 1e-05` `train/global_tokens: 2287`（≈16 seq × 143 token，与数据吻合）
  - `perf/max_memory_allocated_gb: 28.62`（卡 61.3GiB——**余量巨大**）`max_memory_reserved_gb: 51.39` `cpu_memory_used_gb: 2068`（2.19TB 内，继续观察）
- **阻塞修复 1（策略白名单）**：SFT 直接从 yaml 构造 `FSDPEngineConfig` → 撞上 `workers/config/engine.py:306` 的 `assert strategy in ["fsdp","fsdp2","fsdp_turbo"]` 漏了 `fsdp_turbo_dsv41`（GRPO 侧靠 `ActorConfig.__post_init__` 里 `object.__setattr__` **在校验后**写入而绕开）。修：白名单补上 `fsdp_turbo_dsv41`（yaml:78 注释与注册表 `fsdp_turbo_dsv41_impl.py:804` 都早已认它）。
- **阻塞修复 2（ckpt 保存元数据段）**：权重分片本身**存成功**（`global_step_1/model_world_size_16_rank_*.pt` ×16 = **856GB**，其中约 786GB 是 per-rank engram 分片——冻结不变量，未来可做"保存时排除 engram、载入时从原始 ckpt 重读"的瘦身）；但 HF 元数据尾部崩：`AttributeError: 'FSDPDeepseekV41ForCausalLMAdapter' object has no attribute 'config'`。修：`fsdp_checkpoint_manager.py` 对 HF config 段加 `getattr(model,'config',None)` 守卫（dsv41 跳过、保留 tokenizer 保存、留显式日志行）。
- 附：**ckpt 体积现实**→ 全量 run 用 `max_ckpt_to_keep=1` 轮换 + `save_freq=200`（128 磁盘 3.4TB、清掉冒烟 ckpt 后 1.6TB 可用）。

### ✅ "前"评测（131，DP2 16 卡 serve）——循环问题实锤
- **16 卡 DP2 serve 配方落地成功**：`run_vllm_top32_dp2.sh`（TP8×DP2 + `dp_shared_memory:true` + `cpu_offload:true`），`/health 200`，两 DP 引擎 KV 池各 ~835K tokens，加载日志可见共享表卸载（`Engram table offloaded to registered host memory` 出现在 DP0/DP1 两侧）。
- 结果（32 题 GSM8K test、greedy、max 1024、rl_tokenizer 模板渲染=与 RL rollout 同款）：
  `[eval] n=32 closed(stop)=1 has_answer=2 correct=0`；31/32 顶满 1024 未收尾。
- 文件：`/mnt/share/m00899630/dsv41/sft_data/eval_before_top32_20261010_1753.jsonl`（SFT 后同脚本同参数跑"后"评测做 diff）。

### ⏳ 阶段③ 全量启动（18:19:53，128 机器）
`SAVE_FREQ=200 bash scripts/run_sft_dsv41_1node16.sh 467`（1 epoch=467 步，gbs 16，lr 1e-5）
- 日志：`/workspace-verl/verl/logs/sft_dsv41_steps467_20261010_181953.log`
- 预计：加载 ~15min → 每步数十秒~分钟级 → 数小时（下一步实测步时）。

### 阶段③ 调优（2026-10-10 夜）
- **首跑实测：144.07s/步 → epoch 外推 18h39m（太慢）**。归因：`param_offload=True` 的**逐步全参 host↔device 搬运**（= RL 里 250s/update 的同一固定开销成分；16 卡上 params+grads 仅 13GB/卡，根本不需要 offload）。
- **调优配置**：`PARAM_OFFLOAD=False`（param_offload 与 offload_policy 同关；optimizer 仍 offload）+ `GLOBAL_BSZ=64 / MICRO_BSZ=4`（116 步/epoch）+ `save_freq=60`（轮换保留 1 份，ckpt 856GB）。
- **连带修复**：`fsdp_turbo_dsv41_impl.py:1146` 的 checkpoint **buffer bit-exact 校验**在参数常驻 NPU 时 CPU/NPU 混比崩（`torch.equal ... other is on npu:0 ... on cpu`）——buffers 小，加 `.cpu()` 归一（`_verify_direct_shards` 早有同类守卫，仅此处漏）。
- 脚本新增 `PARAM_OFFLOAD` 开关（默认 True=旧 proven 配置）；重启后待实测步速验证（预期 ~2h/epoch）。

### 阶段③ 调优 v2（2026-10-10 夜，第二版）
- **param_offload=False 实测 OOM**：rank0 首个 forward 即 `E1010 19:14:46 NPU out of memory`（采样时卡 64885/65536MB——参数常驻 + per-layer 全参 all-gather + 编译工作区超限）；进程半死卡住。
- **改回 proven offload 配置 + 大 batch 摊薄固定开销**：144s/步的主成分是**每步固定**（优化器/offload 全参搬运往返），与 batch 大小弱相关 → `gbs 128 / micro 8`（58 步/epoch，预计 ~2.5h）。`PARAM_OFFLOAD` 开关保留（默认 True）。
- 二次实测中：`materialized in 5.2s`、buffer 校验修复生效（0 Traceback 到 OOM 前）。

### 阶段③ 调优 v2/v3 教训（batch 杠杆的两堵墙）
- **micro>4 撞 engram 缓存**：`ModelArgs.max_batch_size` 默认 4 管着 engram hash cache/压缩器状态（`engram.py:235` 报 `expanded size (4) vs (8)`）。**已修**：`max_batch_size` 从 engine 一路穿到 ModelArgs（`adapter.py` 三处 + `fsdp_turbo_dsv41_impl.py::_dsv41_max_batch_size()`，env `VERL_DSV41_MAX_BATCH_SIZE`，启动器自动= MICRO_BSZ；默认 4 不变）。
- **micro8 仍 OOM**（即使 param_offload=True）：v2（无 offload）与 v3（有 offload）都在 step1 OOM → batch 杠杆被 HBM 封顶。
- **关键反思**：v1 的"144s/步=18.7h"来自**首步含编译**的 tqdm 外推，**稳态步速从未测过**。v4 = 回 proven 形状（gbs16/micro1）+ 先测准 2-5 步稳态再定 batch/lr/步数。

### 阶段③ 主机泄漏确诊（2026-10-10 夜）
- 新工具 `/tmp/host_mem_sampler.sh`（15s 采样：node used/avail + driver/workers RSS + 最大 worker）→ `/tmp/host_mem_trace.csv`。
- **v4 死因定量**：工作进程 RSS **+~5GB/进程/步 × 16 = +80GB/步线性泄漏**（"358GB 跳变"= 4.5 步累积）；engram 787GB 是 CANN 注册内存（ps 不可见）叠在基线上。
- **否定项**：v6 关 `optimizer_offload` 后泄漏照旧（+108GB/2min）→ 与优化器 offload 无关；嫌疑收敛到 **param/offload_policy 的 FSDPTurbo 逐步机制**（泄漏量 ≈ 每 rank 参数分片 6.5GB/步）。
- **v7 试验**：全关 offload（PARAM+OPTIMIZER=False，微批 1）——看 ①泄漏是否消失 ②HBM 是否放得下（预算 ~53GB/61.3）。若 OOM → 下一步读 FSDPTurbo offload 机修泄漏点。
- 已定修复保留：`PARAM_OFFLOAD`/`OPTIMIZER_OFFLOAD` 开关（默认 True=proven）。

### 阶段③ 夜间收束（2026-10-10 24:00 前后）：no-offload 家族差 ~1-2GB，两条收尾路线已定
- **v7（全 no-offload）**：42s/步（提速 3×）、无 host 泄漏、host 基线 1305GB（省 760）——**唯一门槛：HBM alloc 57.4GB，死在后续 +182MB 分配**。
- v9（+熵关+桶512）：57.76 无改善（两项无效，保留无害）。v10（+prefetch 0）：**病态慢（>15min/步），prefetch 0 是毒药**。v8（混 offload）：同样病态慢（opt-offload 与常驻参数组合差）。v11（v7+expandable_segments）：步速 30s ✓ 但 57.53 仍差。
- **两条收尾路线**：① no-offload 再榨 1-2GB：剩候选=allocator 细参（max_split_size/gc_threshold）、`ep_plan.dispatcher` 变体、grad 驻留细节——逐项单测；② **proven 全 offload + 修 FSDPTurbo 逐步泄漏**（+5GB/进程/步的 host 副本；HBM 仅 28.5 最宽裕，但需源码定位泄漏点，且步速 ~125-144s）。
- 全部开关已固化（PARAM_OFFLOAD/OPTIMIZER_OFFLOAD/PAD_BUCKET/FWD_PREFETCH/BWD_PREFETCH/VERL_DSV41_MAX_{SEQ_LEN,BATCH_SIZE}），双机 md5 一致；`/tmp/host_mem_sampler.sh` 留档。

### 阶段③ 双机方案启动（2026-10-11 01:47，用户确认"双机都可以跑"）
- **转向**：单机 no-offload 差 1-2GB、单机 offload 有泄漏——**改双机 FSDP32**：每卡 states=params 3.25+grads 3.25+AdamW fp32 ~13+激活 ≈ 41GB/61.3 ✓，且绕开 offload 机器（无泄漏）。
- **新脚本 `scripts/run_sft_dsv41_2node.sh`**（master/slave，复刻 GRPO 双机的 set_env/ray 集群管理；SFT_ARGS 与单机同款 + nnodes=2 + fully_shard=32）。
- **新修复（clip 瞬态）**：`clip_grads.py::compute_grad_norm` 原实现把**每个梯度的 fp32 副本全部存 list 到最后**（52e9/16 rank ≈ **13GB/rank 瞬态**）——改为逐参数流式累加（数值逐位等价，per-param fp32 norm 相同、组内求和次序相同）。这很可能就是单机 no-offload 撞 57.4GB 的最后一脚。
- 附修：`pick_devices` 失败路径改为赋值式 `DEV=$(pick_devices)`（原经 `:-` 替换吞掉了 exit 1）。
- 日志：`verl/logs/sft_dsv41_2node_steps467_20261011_014711.log`；467 步（1 epoch）；save_freq=200/轮换 1。

### 阶段③ 双机 v2/v3（2026-10-11 02:10-02:55）：micro2 撞顶确诊 + opt-offload 无效 + micro1 方案
- **首跑（gbs64/micro2，无 offload）**：step1 全周期通过（`loss 1.8523 grad_norm 13.543 max_mem 55.147`，47s/步），**step2 backward 死**：traceback 落到 `transformer_impl.py:767 loss.backward()` → `FSDPTurbo/fsdp_turbo/ops/npu/grouped_matmul.py:60`（MoE 反向），`58.62GB +182MiB 请求`耗尽（torch 可用上限 ≈ 61.27 − 2.4 GB HCCL/驱动 ≈ 58.9）。
- **v3（同配置 + optimizer_offload=True）否定实验**：峰值 **逐位相同（55.146934509277344）**、死亡点相同（58.62+182MiB），但步速劣化 4×（188s/步，v8 同款病态）。→ **优化器态不是 step2 的 +3.5GB 来源；opt-offload 在本引擎不触达主导项，回滚**（两处默认已回 False）。
- **经验显存模型（3 数据点吻合）**：每卡峰值 ≈ **F≈21GB（固定：参数/梯度分片+通信缓冲+引擎缓冲） + ~20MB/每 token/卡（激活） + 优化器态（FSDP32≈12.9GB）**。
  - 单机 FSDP16 micro1：25.8（态）+31.6（512tok）=57.4 ✓；双机 FSDP32 micro2：12.9+42.3=55.2 ✓。
  - **预测双机 FSDP32 micro1 = 12.9+21+10.4 ≈ 44.5GB**（余量 ~14GB）✓。
- **v4 重发**：`GLOBAL_BSZ=64 / MICRO_BSZ=1`（峰值按单微批计，与 gbs 无关 → 保持 116 步不变），全 no-offload；脚本默认已改（MICRO_BSZ=1 / OPTIMIZER_OFFLOAD=False）并同步 128。日志：`/tmp/sft2_master4.log`（131）/ `/tmp/sft2_slave4.log`（128）。
- 附：`clip_grads.py:94` 的 "Multiple backends are registered" UserWarning 是 `_get_pg_default_device` 内省告警（无害，CLIP 相位在跑的证据）。

### ✅ 阶段③ 双机 v5（2026-10-11 03:05 起）：**`efsdp 1→2` 一举解决**（关键修复）
- **死因定案**：v2-v4 三次死亡签名逐位一致（step1 峰 ~55.1GB；step2 backward 在 `grouped_matmul.py:60` 要 +182MiB 时撞 ~58.8GB 顶）。checkpoint 实数：dense 14.07GiB + **experts 112.52GiB** + engram 366.8GiB(host)；EP8/efsdp=1 → **每卡专家 params ~14GiB + grads ~14GiB + 优化器态 ~15.5GiB ≈ 44GB 常驻**（占总峰 55GB 的 80%），micro/offload/FSDP32 都动不了它（RL 史证 ~60GB 平台与 token 无关）。
- **修复**：`engine.turbo_config.distributed.expert_fully_shard_parallel_size=1→2`（专家三件套每卡减半；mesh 自动 edp=2/efsdp=2/ep=8，**拓扑不变**、无 offload、无 micro 改动）。
- **实测（立竿见影）**：
  - 每卡直载 11.03→**5.75GiB**；融合专家 rank0 独立装配校验 **bit-exact** ✓（loader 的 `_assemble_fused_expert_window` 用模块本地分片元数据，拓扑无关）
  - **step1 峰值 55.15→30.42GB**（−24.7）；**step2 通过**（40.01GB，对比旧配置死点 58.6）；step3+ 稳态 **~13s/步**（对比 micro2 47s、opt-offload 188s）
  - loss 1.855→1.506（step2）→…，grad_norm 13.77→7.43，host 1234GB 稳定（无泄漏）
  - 116 步预计 ~25min + 末尾自动存 ckpt（trainer:373 `is_last_step` 触发 save；ckpt ~856GB，131 盘余 1.8TB ✓）
- 脚本已固化（efsdp=2）并同步 128；日志 `/tmp/sft2_master5.log`（完整 `verl/logs/sft_dsv41_2node_steps116_*.log`）。
- **推论**：RL 侧（若同用此引擎）此前 ~60GB 平台同源——EP8/efsdp=1 的专家三件套；后续 RL 走 SFT 热启动时同样配 efsdp=2 可根治显存顶（待验证）。

### ✅ 阶段③ 完成（2026-10-11 03:46）：116/116 步训练完毕 + ckpt 已存
- 全程 21 分钟训练 + ~3 分钟存盘；loss 1.855→（step106 ~0.7 区间），grad_norm 13.77→2.2；峰值 41.7GB/61.3；host 1.24TB 稳定。
- **ckpt**：`global_step_116/`，每卡 55.33GB（engram 45.8GiB + trainable ~7.9GiB），**全量 1.65TB**（双机 ×16 分片分别写本地盘：131=rank16-31、128=rank0-15——engram 每节点各存一份所致；单机 smoke 时 856GB）。
- HF config 段按守卫正常跳过（日志 "Model exposes no .config ... skipping"，tokenizer 已存）。
- 分片已在合并到共享盘 `/mnt/share/m00899630/dsv41/sft_ckpt_top32_gsm8k/global_step_116/`（供导出与后续 RL/vLLM 双机使用）。

### ✅ 导出工具（阶段④ 前置）：`scripts/export_sft_hf.py`（新增，离线 CPU 汇编）
- **原理**：SFT 存的是 sharded state dict（DTensor 带 DeviceMesh+placements）→ 用 torch 官方 `compute_local_shape_and_global_offset`（= loader `_plan_direct_shard_load` 同款）离线算 offset，逐 rank 拼回全量，**无需分布式**。
- **命名映射**（已对着原 ckpt index 验证）：`model.` 前缀剥离；fused `ffn.experts.gate_up_proj [ne,2i,h]` → 原 ckpt 的 `…experts.{E}.w1.weight`(gate 半) / `w3.weight`(up 半)，`down_proj` → `w2.weight`（loader `part_groups=(("w1",),("w3",))` 逐位一致）。
- **inspect 实测**（rank16 分片）：966 条目 = 924 DTensor + 42 plain；`model.embed.weight` global(129280,5120) loc(4040,5120) placements=(Replicate, Shard(0)) ✓（4040=129280/32 ✓）；engram 条目 8 个（含 `.ep_shard_000NN_of_00008` 后缀变体）；fused 专家 80 个（40 层 × 2）。
- **dry-map**：958 训练参数 → **958 mapped / 0 unmapped**；覆盖原 ckpt 6256 张量中的 4718；leftover 1538（engram 8 + 冻结/静态量，原值保留）。
- 输出布局：新 safetensors（vLLM 原命名）+ 含 engram/leftover 的原文件 symlink（混合文件则拷出所需张量）+ 重建 index.json + config/tokenizer symlink。
- 用法：`python3 scripts/export_sft_hf.py inspect|dry-map|export --ckpt-dir <dir> [--out-dir <dir>]`（输出至 `/mnt/share/m00899630/dsv41/weights/DeepSeek-V4.1-Flash-top32-sft116/`）。

### ✅ 导出完成（2026-10-11 04:52）
- 分片先合并到 `/mnt/share/m00899630/dsv41/sft_ckpt_top32_gsm8k/global_step_116/`（32/32，1.65TB；中途 ssh 超时断过一次，nohup 续传 rank4-9）。
- `export` 实测（mmap 生效：55GB/分片仅 ~6s，只触碰 ~8GiB 训练页）：assembly 4.2min → 958 缓冲（**99.2GiB**）→ 展开 4718 张 ckpt 命名张量 → 34 个新 shard（3GB/个）+ 4 个混合文件原地拷出（`model-sft-extra-*`，466 张 engram/冻结量）→ **index 6256 = 与原 ckpt 逐数一致**（4718 新 + 1538 原值：engram 8 + aligner/vision/`*.bias_vl` 等冻结量 ✓）。
- **探针**：`layers.0.attn.wq_b.weight` ≠ 原值（max_abs_diff 9.9e-4）✓ 训练已改变权重。
- 产物：`/mnt/share/m00899630/dsv41/weights/DeepSeek-V4.1-Flash-top32-sft116/`（494GB deref；config/tokenizer symlink 自原目录）。
- vLLM serve 已以该目录为 model_path 启动（`run_vllm_top32_dp2.sh` 加了 `MODEL_PATH` 环境覆盖）。

### ✅✅ 阶段④ 「后」评测完成（2026-10-11 05:07）——**SFT 热启动成功，循环问题治愈**

| 指标（GSM8K test 前 32 题、greedy、max 1024、同脚本同参数） | **前**（原 top32） | **后**（SFT-116） |
|---|---|---|
| 正常收尾 finish=stop | **1/32** | **29/32** |
| 产出 "Answer:" 行 | 2/32 | **29/32** |
| 答对 | **0/32** | **8/32（25%）** |

- 前：`eval_before_top32_20261010_1753.jsonl`；后：`eval_after_sft116_20261011_0501.jsonl`（均在 `/mnt/share/m00899630/dsv41/sft_data/`）。
- 样例题输出（后）：逐步推理行 + `Answer: N` + EOS 收尾（如 "2+1 = 3 bolts of fabric\nAnswer: 3"）。
- 结论：**"模型 never-EOS/循环思考"根因确系裁剪后行为退化，1 epoch GSM8K SFT 即治愈收尾问题**；25% 正确率仍有提升空间（本轮仅 main 7473 条 × 1 epoch、lr 1e-5、116 步——后续可加 socratic/多 epoch/调 lr）。
- 下一步（最终目标）：**verl GRPO 全流程**，actor 初始权重 = 本导出目录，目标 reward 有数值（非恒 -1）。

---

## 附录 A：关键执行命令（照抄可复现）

### A.1 双机 SFT（最终 v5，efsdp=2，全部成功的那次）
```bash
# 131（master）：
cd /workspace-verl
nohup bash scripts/run_sft_dsv41_2node.sh master 116 > /tmp/sft2_master5.log 2>&1 &
# 128（slave，等 master 打印 "waiting for the slave" 后）：
ssh -p 50001 root@80.5.25.128 'bash -lc "export PATH=/usr/local/python3.11.15/bin:\$PATH; cd /workspace-verl && \
  nohup bash scripts/run_sft_dsv41_2node.sh slave 80.5.25.131 > /tmp/sft2_slave5.log 2>&1 &"'
# 脚本默认即最终配置：MICRO_BSZ=1 GLOBAL_BSZ=64 PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=False
#   engine.turbo_config.distributed.expert_fully_shard_parallel_size=2   ← v5 关键修复
```
### A.2 ckpt 合并（双机各写本地盘 → 共享盘）
```bash
S=/mnt/share/m00899630/dsv41/sft_ckpt_top32_gsm8k/global_step_116
# 131 侧（rank16-31）：cp model_world_size_32_rank_{16..31}.pt $S/
# 128 侧（rank0-15）：ssh … 'cd /workspace-verl/ckpt_sft_dsv41/global_step_116 && cp … $S/'（曾因 ssh 超时断在 rank4-9，改用 nohup 续传）
```
### A.3 导出 vLLM 可加载目录
```bash
python3 scripts/export_sft_hf.py inspect  --ckpt-dir $S            # 类型/命名体检
python3 scripts/export_sft_hf.py dry-map  --ckpt-dir $S            # 映射验证：958 参数→0 未映射
python3 scripts/export_sft_hf.py export   --ckpt-dir $S --out-dir /mnt/share/m00899630/dsv41/weights/DeepSeek-V4.1-Flash-top32-sft116 \
        --verify-probe layers.0.attn.wq_b.weight   # 探针: 与原值 max_abs_diff=9.9e-4（已训练 ✓）
```
### A.4 serve + 评测（前后对比同配方）
```bash
MODEL_PATH=<导出目录> bash scripts/run_vllm_top32_dp2.sh        # DP2×TP8 单机 16 卡
python3 scripts/eval_gsm8k_serve.py --port 20007 --n 32 --out … # 前: eval_before_top32_20261010_1753.jsonl
                                                                # 后: eval_after_sft116_20261011_0501.jsonl
# 变体实验（4 种 prompt 风格 × 贪心/采样）：--prompt-style {plain,rl,rl_endthink,rl_nothink} --temperature {0,1}
```

## 附录 B：修复代码明细（含错误原文）

### B.1 `efsdp 1→2`（v5 根治 OOM）
- **错误日志原文**（三连复现，签名逐位相同）：
```
step:1 - perf/max_memory_allocated_gb:55.146934509277344 …
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 182.00 MiB
 (NPU 0; 61.27 GiB total capacity; 58.62 GiB already allocated; 58.63 GiB current active;
  153.08–216.46 MiB free; 58.80–58.88 GiB reserved)
   → traceback: transformer_impl.py:767 loss.backward() → ops/npu/grouped_matmul.py:60 (MoE 反向)
```
- **改动**：启动器一行 `engine.turbo_config.distributed.expert_fully_shard_parallel_size=1→2`；mesh 自动 edp=2/efsdp=2/ep=8。
- **原理**：EP8/efsdp=1 时每卡常驻 4/32 专家的 params+grads+优化器态 ≈44GB（占 55GB 峰的 80%）；efsdp=2 每卡 2 专家 → 峰 55.1→30.4GB（step1）→41.7（稳态）。融合专家 `_assemble_fused_expert_window` 用模块本地分片元数据，拓扑无关；rank0 独立装配 bit-exact 校验通过。

### B.2 `clip_grads.py` 流式化（v2 配套）
- 原实现把**每个梯度的 fp32 副本存入 list 直到函数尾**（52e9 params/16 rank ≈ 13GB/rank 瞬态）；改为逐参数 `g_f=local.float(); term=g_f.norm()**2` 立即释放、按 shard-group 累加标量（数值逐位等价）。

### B.3 `export_sft_hf.py`（阶段④前置，见文件头 docstring 含完整设计）
- 原理：SFT save = DTensor sharded state dict（`get_fsdp_state_ctx` 对 FSDP2 = nullcontext，直接 `model.state_dict()`）→ 用 torch `compute_local_shape_and_global_offset`（= loader 同款）离线算 offset 拼全量。
- 命名：`model.` 前缀剥离；`ffn.experts.gate_up_proj [ne,2i,h]` → 原 ckpt `{E}.w1.weight`(gate 前 half)/`{E}.w3.weight`(up 后 half)、`down_proj → w2`（loader `part_groups=(("w1",),("w3",))` 逐位核对）。
- engram/冻结量：原目录 symlink（纯）/拷出（混合）；index 重建 6256 张量与原数一致。
- **调试中修的两个 bug**：(1) `safe_open.get_dtype()` 返回字符串（"BF16"）需转 torch dtype（`TypeError: empty() received … dtype=str`）；(2) fused up 的缓冲形状应为 `(ne, 2i, h)`（w1 是 `(i,h)`）否则断言触发。
- mmap 加速：`torch.load(..., mmap=True)` → 55GB 分片仅读 ~8GiB 训练页（6s/分片）。

### B.4 其他
- `pick_devices` 赋值式（`DEV=$(pick_devices)`）防 exit 1 被 `:-` 吞；
- `run_vllm_top32_dp2.sh` model_path 改 `${MODEL_PATH:-原目录}` 环境覆盖；
- ckpt 存盘 HF 段守卫（dsv41 无 `.config`）夜间已验：日志 "Model exposes no .config … skipping"。
