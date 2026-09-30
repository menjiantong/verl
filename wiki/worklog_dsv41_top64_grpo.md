# DeepSeek-V4.1 **top64 真实权重**（40 层 / 64 专家 / 带 Engram）GRPO 复跑 — 工作记录

> 对象：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top64-bf16`（578 GB，40 层、
> 64 routed experts、Engram 层 1/14）。
> 任务（用户 2026-09-30 指示）：**用这个模型重跑之前的 GRPO 脚本**，关注**奖励曲线**与
> **训推一致**；先跑 1–2 步验证，数值正常则跑 ~15 步；问题/思路/改码原因/bug 全部记录。
> 状态：**进行中**（下文按小节随运行推进更新；每节标注是"已验证"还是"待验证"）。

---

## 0. 结论速览（随运行更新）

| 项 | 状态 |
|---|---|
| 训练侧 Engram 支持（此前完全缺失） | 已实现（§4），待运行验证 |
| 冒烟启动（1–2 步） | 进行中（§5） |
| 奖励曲线 | 待验证 |
| 训推一致（pearson / probs_diff） | 待验证 |
| 15 步 | 待验证 |

**一句话背景**：之前的 GRPO 一致性工作（`wiki/work_log/pearson_0999/`）用的是 4 层 32 专家
**无 Engram** 切片，一致性地板 0.9946；本次换成**完整 40 层 64 专家 + Engram**的真实发布
检查点，训练侧第一次要面对 Engram 表（393 GB 主机内存）与 64 专家路由。

---

## 1. 侦察：这个检查点与既有集成的事实

### 1.1 检查点结构（证据：`model.safetensors.index.json` + 分片头）

| 项 | 值 | 备注 |
|---|---|---|
| 层数 | 40 | `text_config.num_hidden_layers` |
| routed experts | **64**（top-6） | 原版 384（`config.json_bak` 仍是 384）；`expert_remap.json` 记录 384→64 的映射 |
| 张量命名 | **无前缀**（`layers.N.ffn.experts.E.w1.weight`、`embed.weight`） | 与 4 层真实切片一致，训练侧 `_MODEL_PREFIX="model."` 负责补前缀 |
| Engram | **有**：`engram_layer_ids=[1,14]`，表 `[384006168,256]` / `[384016682,256]` bf16 | 每表 ~196.6 GB，合计 **393 GB**（占 578 GB 的 68%） |
| MTP/DSpark | 有（`num_nextn_predict_layers=3`）但训练适配器强制关掉 | `_validate_training_args` |
| 分片 | 268 个 safetensors，最大分片 196 GB（就是 Engram 表） | 载入必须用切片读，不能整张读 |

### 1.2 引擎侧（vLLM / vllm-ascend）是**默认开 Engram** 的

* `vllm/config/engram.py`：`EngramConfig.cpu_offload` 默认取 `VLLM_PLE_CPU_OFFLOAD`，
  而该环境变量默认 `1`；表放在**注册过的主机内存**（`aclrtMallocHost` +
  `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`），NPU gather 内核通过 UVA 直接读，
  不占 HBM（`vllm_ascend/models/deepseek_v41/engram/npu.py`）。
* `vllm_ascend/platform.py:_validate_engram_config`：模型声明了 `engram_layer_ids` 时
  **自动构造 EngramConfig()**，无需（也没有）显式开关；要求 `TP∈{1,2,4,8}`、PP=PCP=DCP=1。
* 引擎的 Engram 参数是 `nn.Parameter(..., requires_grad=False)`
  （`AscendParallelEngramEmbedding.__init__`）——**推理侧本来就是冻结的**。
* 用户 09-28 合入的 engram 修复分支（`engram-node-local-edp`、`engram-bf16`）说明推理验证
  就是按 Engram ON 做的。

**推论**：训练侧如果不开 Engram，actor 的前向与引擎就不是同一个模型（Engram 在层 1/14 往
残差流里写入 gated 的 n-gram 特征），一致性指标会失去意义；反之训练侧必须要有 Engram 支持。

### 1.3 训练侧（verl + FSDPTurbo）此前**完全没有** Engram 支持（证据：改动前的代码）

* `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py` 里
  `engram_storage_backend="row_sharded"` 是硬编码，注释写着 "the checkpoint has no engram
  tables"——4 层切片确实没有 Engram，所以这条路径从没被走到过。
* 但 FSDPTurbo 上游已删除 `row_sharded`（`0769938`），`adapter.build_deepseek_v41_model`
  对**带 Engram 层的模型**会直接 `raise ValueError`。→ 换成这个检查点，**第一步就会炸**。
* `grep -rn engram verl/` 只有那一处（DCP 加载、优化器、权重导出都没有 Engram 概念）。

### 1.4 训练侧 Engram 的实现（FSDPTurbo，`models/deepseek_v41/host_offload_embedding.py`）

* `TorchHostOffloadEmbedding`（`host_offload` 后端）：`weight` 是 **EP 按行切分的主机 bf16
  Parameter**（`[384008192/8, 256]` ≈ 24.6 GB/rank/表），`_engram_host_offload=True`。
* 前向：`_TorchHostFetch` —— 纯 PyTorch/HCCL 的 EP all-to-all 路由 + CPU index_select；
  反向积累**稀疏梯度**到 `_pending_sparse_grad`（不是 `.grad`），由 FSDPTurbo 的
  `EngramOptimizer` 消费。
* FSDPTurbo 的 `apply_pre_fsdp_module_hooks` 会调用模块的 `pre_fsdp_hook`（= `shard_()`，
  在 FSDP 包装父层**之前**把 meta 表换成 owner-local 主机表），并把该模块加进
  `ignored_modules`——所以表**不归 FSDP 管**（不是 DTensor），也不会被 FSDP2 的 dtype 归一化
  或 all-gather 碰到。
* `clip_grad_norm`（FSDPTurbo）通过 `custom_clip_grad_norm` 钩子处理 Engram 稀疏梯度；
  表冻结时 `pending_sparse_grad` 恒为 `None`，钩子对 dense 行为无影响（已读码确认）。

### 1.5 内存账（这就是为什么必须去掉 ref 模型）

按 rank 估算（8 卡）：

| 项 | 每 rank | 全节点 |
|---|---|---|
| trainer actor Engram 表（2 张，bf16，pinned 主机） | 49.2 GB | 393 GB |
| trainer ref Engram 表（若保留） | 49.2 GB | 393 GB |
| trainer 专家参数（bf16，CPU offload；actor+ref） | 22.6×2 GB | 362 GB |
| AdamW 状态（仅专家等 92.5B 可训练参数，fp32×2） | ~90 GB | 740 GB |
| engine actor Engram 表（UVA 主机） | 49.2 GB | 393 GB |
| engine ref Engram 表（若保留） | 49.2 GB | 393 GB |
| **合计（保留 ref）** | | **≈2.7 TB > 2.4 TB 物理内存** |
| **合计（去掉 ref + KL）** | | **≈1.5–1.7 TB，可行** |

---

## 2. 设计决定（以及为什么）

| # | 决定 | 理由 |
|---|---|---|
| D1 | 训练侧用 `host_offload` 后端（替换 `row_sharded`） | `row_sharded` 已被上游删除且对带 Engram 的模型直接报错；`host_offload` 是唯一现存实现 |
| D2 | 训练侧 Engram **冻结**（`requires_grad=False`，且摘掉 `_grad_keepalive` 的 autograd 挂载） | ①引擎侧本来就是 `requires_grad=False`，冻结才与引擎语义一致（否则每步 393 GB 的表要同步）；②否则 AdamW 会为两张表分配 fp32 动量（~200 GB/rank）；③避免稀疏梯度在主机内存里堆积 |
| D3 | Engram 行**按 rank 切片**从 ckpt 读入（`safe_open(...).get_slice(name)[start:end]`） | 表 196 GB/张，整读会把 rank0 打爆；DCP 也不该碰它（不是 DTensor，形状对不上） |
| D4 | Engram 从 **DCP 加载列表 / 权重导出**中排除 | 同上；导出会每步搬 2×24.6 GB/rank 的无效流量 |
| D5 | **关闭 KL、不建 ref 模型**（`use_kl_in_reward=False` + `actor.use_kl_loss=False`） | 内存账（§1.5）不允许 ref 再带一份 Engram；`need_reference_policy()` 因此为假，ref worker/engine 根本不会建。数据是 DAPO，DAPO 的 GRPO 本来就去 KL |
| D6 | `LOCAL_EXPERT_EXPORT=1` | 40 层×64 专家的导出若走 all-gather，按旧测量（14 GB 模型 80 s/sync）外推 ~900 s/步；local 导出只发本 rank 的分片。脚本里本来就有这个开关与校验方法 |
| D7 | 保留 `VERL_DSV41_MAX_SEQ_LEN=2048`、`enforce_eager=True`、`rl_config.enabled=true` 等既有设置 | 与 p0999 campaign 完全同配置，便于横向比较 |

---

## 3. 存档：与推理侧的"同一性"依赖（运行时验证的判据）

1. **行编号**：两侧都是同一份 ckpt 的行号；训练侧 owner 划分是 EP 均匀 1/8（含尾部
   2024/1750 行 padding，永不寻址），引擎侧是按 hash 列分桶——划分不同**不影响**内容一致性，
   只要求"同一行号 = 同一内容"。
2. **token map**：训练侧 `NgramHashState` 用 tokenizer 构建压缩 token map，并校验
   `engram_compressed_vocab_size`（99092）；`rl_tokenizer/tokenizer.json` 与检查点目录里的
   **md5 完全相同**（`8a8245dc…`），所以两侧 map 一致。
3. **冻结一致性**：两侧表都来自同一 ckpt 且都冻结 → 权重同步不需要搬表；判据是
   `training/rollout_actor_probs_pearson_corr` 与 `training/rollout_probs_diff_mean`。

---

## 4. 代码改动（本轮）

### 4.1 `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`

| # | 位置 | 改动 | 为什么 |
|---|---|---|---|
| 1 | `_build_dsv41_module` | `engram_storage_backend="row_sharded"` → `"host_offload"` | 见 D1；旧值对该模型直接 `ValueError` |
| 2 | 模块级新增 `_ENGRAM_EMBED_SUFFIX`、`host_offload_engram_tables()`、`_engram_checkpoint_rows()` | 找出 host-offload 表并给出其检查点张量名；读表行数 | 表名取自 FSDP 包装前的模块树（`model.layers.N.engram.embed`），去掉适配器的 `model.` 前缀就是 ckpt 名 |
| 3 | `_build_fsdp_module` | 把 Engram 表从 `param_meta` 中剔除；把表对象列表传给 materialize | DCP 不该看见它们（D2/D4）；行分片要等 FSDPTurbo 的 `pre_fsdp_hook` 跑完才知道 |
| 4 | 新增 `_load_host_engram_tables()` | 逐 rank 用 safetensors **切片读**自己的行区间 → `copy_` 进主机表 → **冻结**；`all_gather_object` 校验 8 个 rank 的区间**恰好无缝覆盖** [0, logical)，并核对与 ckpt 行数一致；同时做"落位是否错位"的自检 | 见 D2/D3；区间不覆盖 = 某些行永远读到初始化值（静默错），重叠 = 两侧同 id 不同值 |
| 5 | `_export_param` | 名字以 `.engram.embed.weight` 结尾的参数**不导出** | 见 D4（每步会白搬 49 GB/rank） |
| 6 | `_materialize_dsv41_parameters` | 新增 `engram_tables` 形参，在 `refresh_dsv41_expert_metadata` 之后调用 4 | 顺序要求：表必须已由 `pre_fsdp_hook` 物化 |

**为什么冻结要连 `_grad_keepalive` 一起处理**：`TorchHostOffloadEmbedding` 故意把一个
`requires_grad=True` 的标量 `_grad_keepalive` 传给自定义 autograd Function，使反向**必然**
执行（elastic_buffer 路径需要它做实数据依赖）。表冻结后若仍让反向执行，每步会把
token×24 行的稀疏梯度累到主机内存且无人消费。把 keepalive 换成不带梯度的标量后，整个
`_TorchHostFetch` 不再建图，梯度只经残差流（x）流动。

### 4.2 `verl/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh`（新增）

在原脚本基础上：`MODEL_PATH` 指向 top64；`use_kl_loss=False` 且删掉整段 REF 配置（D5）；
`LOCAL_EXPERT_EXPORT` 默认 1（D6）；其余保持与 p0999 基线一致（便于横向比较）。

---

## 5. 运行记录（进行中）

（下节随运行补齐：启动 → 加载 → 第 1 步 → …）