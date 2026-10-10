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
| 冒烟启动（1–2 步） | 第一次启动 **失败**（DCP strict 加载，§5.1）；修复后训练侧通过（§5.4） |
| 推理侧单独自测 | **通过**（§5.5：重编 `_C_ascend` 后 vLLM serve 起服，问答实测正常） |
| 训练侧单独自测 | **通过**（§5.6：生产引擎一步 forward/backward + 裁剪，`grad_norm=2.58`；顺带修掉一个真 bug） |
| top64 GRPO 端到端 | **受限于主机内存**（§5.5b：基线 2.0 TB / 峰值 2.42 TB，权重同步窗口打爆 2.45 TB） |
| top32 GRPO 端到端（1 步） | 进行中（§5.7，2026-10-08 重启；上一轮 09:05 那次已跑到 `ready to fit`） |
| 奖励曲线 | 待验证（第 1 步跑完后取 `critic/rewards/mean` 等） |
| 训推一致（pearson / probs_diff） | 待验证（`rollout.calculate_log_probs=True` 已在配置里生效，指标键见 §5.7） |
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
| 7 | 模块级新增 `hide_engram_tables()` / `restore_modules()`；`set_model_state_dict` 调用被 try/finally 包住 | **修 DCP strict 加载崩溃**（§5.1/§5.2）：表在模块树里但不在 DCP 的 state dict 里 → 严格加载报 Missing key。加载期间把表从树上摘下（不是放松 strict） |
| 8 | `import torch.nn as nn` | 改动 7 需要 |

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

### 5.1 第一次启动（2026-09-30 02:29–02:38）：DCP 严格加载崩溃

**现象**：训练侧参数物化阶段立即退出，报 `Missing key: model.layers.1.engram.embed.weight`
（第 14 层同）。即本轮"Engram 表不进 DCP"的新设计与 DCP 的严格加载相冲突。

**取证的困难（先说清楚证据链的缺口）**：那次运行的容器随后被停掉（用户 03:00 告知换了
容器），`logs/` 下的运行日志与后台输出都随旧容器消失，只剩控制台 transcript
（`2026-09-30-022338-...txt`）里最后一句结论。因此下面把机理**重新推导到可复现**，
而不是照抄当时的判断。

**根因**（读本机 torch 2.10 源码确认，`torch/distributed/checkpoint/state_dict.py`）：

1. `set_model_state_dict` 默认 `StateDictOptions.strict=True`（该 dataclass 的 `strict` 字段）；
2. 它最终执行 `model.load_state_dict(state_dict, strict=info.strict, assign=...)`，而**模型侧的
   key 集合来自 `_iterate_valid_model_state()`——它是递归模块树得到的**；
3. 本轮改动把两张 Engram 表从 `param_meta` 剔除（改动 2/3），所以 `full_state` 里没有它们，
   但它们**仍然是模型树上的参数**：host_offload 后端只把它们放进 FSDP-Turbo 的
   `ignored_modules`（不做 DTensor 化），参数本身照样注册在树上；
4. → 严格加载把这两个 key 判为 missing。同源的第二处报错点在 `_load_model_state_dict`
   的循环里：`raise RuntimeError(f"Missing key: {fqn}.")`（只在 DDP/compile 前缀改写时进入）。
   两处都由 `info.strict` 门控，所以两者都指向同一个设计缺口。

### 5.2 修复：加载期间把 Engram 模块从模块树上摘下来（而不是放松 strict）

新增 `hide_engram_tables()` / `restore_modules()`（改动 7）：在 `set_model_state_dict`
之前把两张表所在的位置换成一个空 `nn.Module()`，`finally` 里换回原对象。

**为什么不用 `StateDictOptions(strict=False)`**（这正是 torch 报错信息里建议的写法）：

* `strict=False` 会把**所有**模型侧缺键都放过——缺一个专家分片、缺一个注意力权重同样表现为
  "Missing key"，那恰恰是最需要 fail-closed 的场景；
* 摘除只缩小 DCP 这一次调用的视野：表对象本身、`pre_fsdp_hook` 里 `shard_()` 的结果、
  FSDP-Turbo 的 ignored-module 记录都不动；`finally` 保证异常路径也复原。

**实现要点**：表按**对象身份**在模块树里定位（`named_modules(remove_duplicate=False)` + `id()`），
不按 checkpoint 名反推模块路径——FSDP 包装后暴露什么路径是它自己的事，推错就等于"悄悄换了
key 空间"，所以**定位不到必须报错而不是猜**。因此文件头新增 `import torch.nn as nn`
（原先只用 `torch`）。

### 5.3 修复前后的离线复核（廉价检查，先于重启做掉，避免用 30 分钟加载去试错）

| 检查 | 方法 | 结果 |
|---|---|---|
| 语法 / 导入 | `py_compile`，并在脚本的 PYTHONPATH 下 import 该模块 | 通过 |
| hide/restore 行为 | 玩具模块树：带 `model.` 前缀 / 不带前缀 / 无 Engram 模型 / 外来表对象 | 全绿；外来表 → 明确报错（不静默） |
| 行布局 vs 真实 ckpt | 读两个分片的 safetensors header + `text_config.engram_num_embeddings` | 见下表 |
| 冻结语义 | 读 `TorchHostOffloadEmbedding.forward` / `_TorchHostFetch` | 见下 |

**行布局**（EP=8，`_TABLE_ROW_ALIGNMENT=2048`，"物理行数"是训练侧对齐后的表长）：

| 表 | ckpt 行数（= logical） | 物理行数 | 每 rank 行数 | 末尾 rank 实际读 | padding |
|---|---|---|---|---|---|
| `layers.1.engram.embed.weight` | 384,006,168 | 384,008,192 | 48,001,024 | 47,999,000 | 2,024 |
| `layers.14.engram.embed.weight` | 384,016,682 | 384,018,432 | 48,002,304 | 48,000,554 | 1,750 |

→ `end = min(start + local_rows, logical)` 截断后 8 个 rank **恰好无缝覆盖** [0, logical)，
总行数 == ckpt 行数，`_load_host_engram_tables` 的覆盖性校验会通过，读取不越界。
padding 行（id ≥ logical）两侧都不会被哈希到：训练侧 forward 里有显式校验
`flat_ids >= table.logical_num_embeddings`（`_TorchHostFetch.forward`）。

**冻结语义**：`_TorchHostFetch.backward` 的稀疏梯度是**副作用**（`accumulate_sparse_gradient`
写入 `_pending_sparse_grad`），而它对 `weight` 本身返回 `None`；把 `_grad_keepalive` 换成
`requires_grad=False` 的标量后整个 Function 不建图 → 反向不执行 → `_pending_sparse_grad` 恒为
None → FSDP-Turbo 的 Engram 优化器与裁剪钩子无事可做，与引擎侧（`requires_grad=False`）一致。

### 5.4 第二次启动（本会话，2026-09-30 03:03）

容器被换过，环境逐项重验：

| 项 | 旧容器（记忆） | 本容器实测 |
|---|---|---|
| 主机名 / IP | `node-29-117`；`/etc/hosts` 里指向旧 IP `141.61.29.117` | `node-29-131` / `80.5.25.131`；`/etc/hosts` **没有该主机名的条目** |
| GLOO 网卡 | `enp48s3u1u1` | 实测 `enp48s3u1u1` → `80.5.25.131`（网卡名沿用） |
| NPU | 8 卡（共享） | 8 卡空闲（AICore 0%，HBM 各 ~3 GB 残留） |
| 主机内存 | 2.4 TB | total 2454 GB / available 2261 GB（Engram 需 ~1.5–1.7 TB，可行） |
| 模型与数据路径 | — | `/mnt/share/m00899630/{weights,dsv41}` 全部可达；`/mnt/share` 余 43 TB |
| 包解析 | 多份同名副本 | 脚本固定 `PYTHONPATH` 到 `/workspace-verl/{vllm,vllm-ascend-v41-private,FSDPTurbo}`，实测命中这三处 |

启动命令（1 步冒烟）与日志：

```bash
cd /workspace-verl/verl
MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=enp48s3u1u1 \
  bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps=1
# 日志：logs/DeepSeek-V4.1-Flash-top64-bf16-20260930_030337.log
```

`MASTER_ADDR` / `GLOO_SOCKET_IFNAME` 是记忆里的已知坑（`verl-dsv41-npu-grpo-launch-env`）：
主机名解析不到本机时，任何按主机名建 gloo/Ray process group 的路径都会死；显式指定后绕开解析。

**结果（训练侧物化，03:11–03:28）：成功，两处修复都生效**

取证位置提醒：**worker 的 `print` 不在驱动日志里**，Ray 把它收在
`/tmp/ray/session_latest/logs/worker-*.out`（驱动日志只有 Ray 转发的 warning/error）。
下次排查直接去那里，否则会误以为"卡住了没输出"。

rank 0（`worker-…-36566.out`）：

```
[fsdp_turbo_dsv41] materializing 924 parameters + 40 checkpoint buffers from /mnt/.../top64 (experts deferred to meta)
[fsdp_turbo_dsv41] checkpoint read: 964 tensors (1532 checkpoint tensors unused, e.g. vision tower)
[fsdp_turbo_dsv41] aligned 40 checkpoint buffers to cpu for the load (e.g. ['model.layers.0.ffn.gate.bias', ...])
[fsdp_turbo_dsv41] checkpoint buffers loaded: 40 (bit-exact against the checkpoint)
[fsdp_turbo_dsv41] parameters materialized in 669.0s
[fsdp_turbo_dsv41] moved 139 buffers to npu (e.g. ['model.engram_hash.primes', 'model.engram_hash.offsets'])
[fsdp_turbo_dsv41] engram layers.1.engram.embed.weight: checkpoint rows [0, 48001024) -> local rows [0, 48001024) of 48001024 (0 padding rows left as initialized); logical 384006168
[fsdp_turbo_dsv41] engram layers.14.engram.embed.weight: checkpoint rows [0, 48002304) -> local rows [0, 48002304) of 48002304 (0 padding rows left as initialized); logical 384016682
[fsdp_turbo_dsv41] host-offload Engram tables loaded: 2 table(s), 45.8 GiB per rank in 130.2s (frozen)
```

要点：

* **`924` 个待物化参数里不含 Engram 表**，说明改动 2/3（从 `param_meta` 剔除）生效；
  `Missing key` 没再出现，说明改动 7（加载期间摘除模块）生效——即 §5.1 的 bug 已修掉。
* 8 个 rank 实测行区间：rank 0–6 各 `[i·48001024, (i+1)·48001024)`；rank 7
  `[336007168, 384006168)` 且留 **2024** 行 padding。**与 §5.3 的离线推算逐位一致**，
  覆盖性校验通过（该检查在打印 "loaded" 之前，失败会抛错）。
* 表加载 130.2 s / 45.8 GiB per rank：NFS 上 8 路并发读同一个 196 GB 分片，约 360 MB/s/rank。
* 物化 669 s（rank 0 读 181 GB 专家张量 + 广播）；各 rank 峰值 HBM ~19.6 GB。

### 5.5 第二个故障：推理侧缺自定义算子 `npu_hc_pre_v3`（03:30，训练侧物化成功之后）

vLLM 引擎在 KV cache 之前的 profile run 里崩：

```
File ".../vllm_ascend/models/deepseek_v41/model.py", line 851, in hc_pre
    return torch.ops._C_ascend.npu_hc_pre_v3(
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_hc_pre_v3'
```

**根因：编译产物落后于源码。**

* `_C_ascend` 命名空间由本仓库 `csrc/torch_binding.cpp`
  （`TORCH_LIBRARY_EXPAND(_C_ascend, ops)`）注册，产物是
  `vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so`；
* 工作区那份 .so 是 **2026-09-18** 构建的，`strings … | grep -c npu_hc_pre_v3` = **0**；
  而当前源码（`local_run` 分支：上游 PR #16422 的 V4.1 稀疏注意力算子 + 09-24/09-28 的
  engram 修复）要求它有；
* V4.1 模型第一次 forward 就会调它，所以推理引擎必崩，与训练侧、与本轮 engram 改动都无关。

**为什么上次没暴露**：上一次（02:29）运行在**训练侧 DCP 加载**就退出了；vLLM 引擎是之后才建的，
根本没走到这一步。换容器不是病因本身，只是让"旧 .so / 新源码"的不匹配显性化。

**顺带查清的环境事实（新容器）**：

* `/workspace-verl/vllm` 是 git 仓库且 **HEAD 已在 `v0.30.0`**（detached），但 site-packages 的
  dist 元数据仍是 Sep 17 构建的 `0.28.1rc1.dev570`——editable finder 指向工作区 ✓，所以
  **纯 Python 代码是 v0.30.0**；脚本里 `VLLM_VERSION=0.30.0` 正是给这类分支判断兜底的。
* `vllm_ascend` 的 editable finder 指向 **`/workspace-verl/vllm-ascend`，该目录在本容器中不存在**；
  实际生效的是 `PYTHONPATH` 固定的 `/workspace-verl/vllm-ascend-v41-private`
  （`local_run` @ `c1e3442d0`，工作树干净）。

**处理（用户给出配方）**：重编两个包

```bash
cd /workspace-verl/vllm
git checkout v0.30.0
python use_existing_torch.py --prefix && pip install -r requirements/build/cuda.txt
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e .

# 配方里的 `vllm-ascend` 即本容器的工作区副本 vllm-ascend-v41-private（见上面 finder 事实）
cd /workspace-verl/vllm-ascend-v41-private
pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ \
  --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ \
  --trusted-host triton-ascend.osinfra.cn
```

校验清单（V4.1 模型用到的全部 `_C_ascend` 算子）：`npu_hc_pre_v3`、`npu_hc_post`、
`npu_quant_lightning_indexer_v3`、`inplace_partial_rotary_mul`、`npu_rms_norm_cast`。

**结果（03:53–04:22）：构建成功，推理侧已独立验证通过**

构建失败的真正原因（不是源码问题，是**陈旧的 CMake 缓存**）：

* `csrc/build/CMakeCache.txt`（Sep 18 那次构建留下的）里
  `AICPU_CUST_OBJ_TARGETS:INTERNAL=…;vllm_quant_lightning_indexer_metadata_obj` —— 这是**旧目标名**，
  当前源码只注册 `quant_lightning_indexer_v2_metadata`
  （`attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/CMakeLists.txt:24`）。
* 该变量是 `CACHE INTERNAL`（`cmake/func.cmake:822`：只增不减），CMake 重新配置时会复用旧列表，
  于是 `cmake/symbol.cmake:253` 的 `add_custom_command` 引用了一个**已不存在**的 target
  → `CMake Generate step failed`。
* 处理：`rm -rf csrc/build`（gitignore 的构建目录）后重编即通过。**教训：换源码分支/换容器后若沿用
  旧 build 目录，要警惕 CMake 缓存里的历史目标名。**

重编结果：

| 项 | 值 |
|---|---|
| vllm | `0.30.1.dev0+gced6857af.d20260930`（v0.30.0 checkout，`VLLM_TARGET_DEVICE=empty`） |
| vllm-ascend | `0.19.1rc2.dev2427+gc1e3442d0`（fork `local_run`，`COMPILE_CUSTOM_KERNELS=1`） |
| `_C_ascend` 算子校验 | `v4.1 ops ok : True missing: []`（`npu_hc_pre_v3`/`npu_hc_post`/`npu_quant_lightning_indexer_v3`/`inplace_partial_rotary_mul`/`npu_rms_cast`） |
| 新 .so | `vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` @ 04:14（`strings` 里已有 `npu_hc_pre_v3`） |

* 环境副作用（requirements.txt 触发）：numpy 2.4.6→**1.26.4**、transformers 5.10.4→**5.14.1**、
  fastapi/starlette 降级；torch 2.10.0 / torch-npu 2.10.0.post4 不变（import 正常）。
* 推理启动脚本（本容器适配版）：`scripts/run_vllm_top64.sh`（相对用户给的命令改了 3 处：
  路径 → `/workspace-verl`；`HCCL_IF_IP` → `80.5.25.131`；**补 `VLLM_VERSION=0.30.0`**——因为
  `vllm_version_is()` 比较 `Version(public)`，重编后的 vllm 报 `0.30.1.dev0`，不设该变量会落到错误分支；
  并去掉 `LD_PRELOAD=libjemalloc`，本容器无该库）。

**推理侧验证记录（04:15–04:22）**：

```
Engram table offloaded to registered host memory: 48002349 rows x 256, 22.89 GiB per rank   # 16 个 worker
Available KV cache memory: 33.54 GiB
INFO:     Application startup complete.        # ApiServer_0 / ApiServer_1 (DP=2)
```

`/v1/chat/completions` 实测：`7*8 → 56` ✓、`12+35 → 47` ✓（首答曾给出 42，重发同一问题即为 47，
说明是单次解码波动而非加载错误；该 ckpt 为 384→64 专家剪枝版，带思维链，质量本身就弱）。
日志：`logs/vllm_top64_20260930_041547.log`。

### 5.5b 第三次启动（GRPO 1 步）：主机内存打满，Ray 杀 worker

训练侧这次在 GRPO 里也一次性加载成功（`parameters materialized in 532.4s`、
`host-offload Engram tables loaded: 2 table(s), 45.8 GiB per rank in 24.8s (frozen)`），
随后在**建推理引擎**时失败：

```
ray.exceptions.OutOfMemoryError: 12 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 80.5.25.131) was 2422.66GB / 2454.43G
```

**内存账（实测，不是估算）**：

| 阶段 | 主机内存 | 构成 |
|---|---|---|
| 训练侧加载完成 | **1.34 TB** | Engram 表 393 GB（pinned）+ CPU-offload 专家 181 GB + AdamW 状态 ~395 GB + 尚未归还的加载瞬时 |
| 推理引擎初始化 | **2.35 TB（峰值）** | 再加引擎自己的 Engram 表 393 GB（16 worker 各 22.89 GiB）+ 建模/profile 瞬时 |
| Ray 默认杀线 | 2.33 TB | `RAY_memory_usage_threshold` 默认 0.95 × 2454 GB |

两个诊断结论：

1. ~~`top32` 换不出内存~~ —— **此结论作废（我读到的是重新生成前的旧目录状态）**。
   用户 08:42 刚用剪枝流水线重新生成了该 checkpoint；重新实测：

   | | top64 | top32（重新生成后） |
   |---|---|---|
   | `config.n_routed_experts` | 64 | **32** |
   | 每层专家张量数 | 192（64×w1/w3/w2） | **96（32×3）** |
   | `expert_remap.json` 的 `new_to_old` | 每层 64 项 | **每层 32 项** |
   | 总大小 | 578 GiB | **493.4 GiB** |

   即 top32 是**真正的 32 专家剪枝模型**（流水线见共享目录 `02_prune_moe_experts_dsv4.md`：
   `stat_moe.py` 统计路由频次 → `prune_moe_experts_dsv4.py` 物理删专家并重编号，同时改写
   `config.json` 的 `n_routed_experts`）。**教训：对"生成中/刚重新生成"的权重目录做结构判定前，
   先看文件 mtime**（我当时对着一份尚未完成重生成的目录下了结论）。
2. **被杀的真正推手里有"隐形内存"**：被 kill 的进程持有**设备注册主机内存**
   （`aclrtMallocHost`/registered），**不计入 `ps` RSS**。上一轮 serve/自测的残留进程在
   06:00 那次失败后仍占着 ~700 GB，导致下一次启动时 `free` 只有 ~1.4 TB 可用、峰值直冲
   2422 GB。清理后可用内存回到 2265 GB。→ GRPO 包装脚本 `run_grpo_top64.sh` 现在**启动前
   自动清扫** `VLLM::`/`ray::`/`vllm serve` 残留并打印 `free` 余量。

**处理**：把 `RAY_memory_usage_threshold` 提到 0.99（2.43 TB）让引擎初始化的瞬时峰值通过
（不改模型、不改训练语义），并每 10 s 采样内存观察峰值是否瞬时。若仍不行，剩下的杠杆是
训练侧的 AdamW 状态精度（~395 GB）或改多机 —— 两者都改训练语义/资源形态，需要用户拍板。

### 5.6 两侧独立自测脚本（新增到 `/workspace-verl/scripts/`）

用户要求"推理与训练各自能单独启动验证"，因此把两侧拆成独立脚本：

| 脚本 | 作用 | 关键点 |
|---|---|---|
| `run_vllm_top64.sh` | 只起推理（vLLM serve，16 卡 DP2×TP8） | §5.5 的 3 处适配；`VLLM_VERSION=0.30.0` 必设 |
| `run_train_engine_smoke.sh` + `train_engine_smoke.py` | 只测训练：torchrun 8 卡，直接驱动**生产引擎类** `FSDPTurboDSV41EngineWithLMHead`（与 actor worker 同一个类），流程 = build → DCP + Engram 行加载 → `train_mode` → 合成批次一次 forward/backward → `optimizer_step` → 导出载荷自检（确认 Engram 表**不在**同步载荷里） | 完全不含 vLLM 代码；拓扑与 GRPO 脚本一致（fsdp8/tp1/ep8/efsdp1 + 同样的 apply_modules/hook_modules/ep_plan + CPU offload） |

写这个驱动脚本时踩到的两个"环境/夹具"级问题（与引擎无关，但会让人误判）：

1. **`HFModelConfig` 会通过 vLLM 解析 ckpt 的 `config.json`**：`model_type` 必须是 vLLM 认识的名字。
   top64 发布件写的是 `deepseek_v41`（`get_config` → `DeepseekV41Config` ✓，且**不**需要 `import vllm_ascend`）；
   而老的 4 层夹具仍是 `deepseek_v4.1` → `ValueError: Transformers does not recognize this
   architecture`。所以"拿 4 层夹具先验证脚本"这条捷径走不通，训练自测必须直接用 top64。
2. 两个 ckpt 目录都**没有** `generation_config.json`：`get_generation_config()` 会先 OSError，
   再回退到 `get_huggingface_actor_config()`——即上面那条 vLLM 解析路径，所以问题 1 会在这里被暴露出来。

**训练自测脚本的开发过程（每轮报错都指向一处生产契约，逐条记录）**：

| 轮次 | 报错 | 结论 / 修法 |
|---|---|---|
| 1 | 用 4 层夹具 → `ValueError: Transformers does not recognize this architecture` | 夹具的 `model_type` 是旧名；改用 top64（§5.6 问题 1） |
| 2 | `FrozenInstanceError: Field 'strategy' is frozen` | `BaseConfig` 冻结字段；生产路径用 `object.__setattr__`（`actor.py:325`），照抄 |
| 3 | `TypeError: FSDPOptimizerConfig.__init__() got an unexpected keyword argument 'lr_scheduler'` | 字段名是 `lr_scheduler_type` |
| 4 | `TensorDict: batch dimension mismatch, got batch_size=[1] and value.shape=[]` | 标量不能直接塞进 `TensorDict(...)`；走 `tu.assign_non_tensor`（实测 `td["temperature"]` 会解包成 float） |
| 5 | `assign_non_tensor(... multi_modal_inputs=[])` → `NonTensorStack.from_list([])` 报错 | `assign_non_tensor` 对 **list** 走 stack 路径，空列表非法；引擎本来就 `.get(..., [])` 兜底，直接不传 |
| —（离线自测） | 加了一个 `--dry-run` 只建配置+构造引擎；又写了一次性单测在本地跑 `build_batch`（无需 NPU） | 把"每轮 16 分钟"降成"每轮 1 分钟"，上表 4/5 这两类错误以后者兜住 |

顺带确认的**引擎数据契约**（写脚本时必须踩对，来自源码而非猜测）：

* `use_remove_padding=False` 也要求 `input_ids`/`position_ids` 是 **jagged nested tensor**：
  该分支自己做 `to_padded_tensor` + `build_attention_mask_from_nested`（`transformer_impl.py`
  `prepare_model_inputs` 的 else 分支）。
* `prepare_model_outputs` **不产出 `logits`**，只给 `log_probs` / `entropy` / `sum_pi_squared`
  （均为按真实长度 jagged 的 nested tensor）——损失函数只能从这三个里取。
* 微批次切分 `prepare_micro_batches` 的 `use_dynamic_bsz` **默认 True**，会断言要求
  `max_token_len_per_gpu`；GRPO 配方是关的，所以脚本显式传 `use_dynamic_bsz=False`。
* `engine.train_mode()` 是**上下文管理器**（内含 CPU-offload 的进出场），不是普通调用。

**训练自测结果（05:36–06:00，第 7 轮）：通过**

```
[smoke] initialize() done in 974.8s
[smoke] parameters: total=123.41B trainable=98.83B        # 差的 24.58B 就是两张冻结的 Engram 表
[smoke] train_batch done in 310.1s metrics={'smoke/log_probs_abs_mean': [13.62], 'grad_norm': 2.5848}
[smoke] TRAIN-ENGINE SMOKE PASSED                          # exit 0
```

即：生产引擎的 build → DCP + Engram 行加载 → 一次 forward/backward（**真有梯度**）→ engram 感知的
梯度裁剪，全部跑通。`grad_norm=2.58` 说明梯度确实流过了网络（对比 p0999 的 `0.0`，那是 advantage
全 0 的 fixture）。

附带发现（不阻塞，已修脚本）：自测若**不设** `VERL_DSV41_LOCAL_EXPERT_EXPORT=1`，导出检查会走
all-gather 路径，对 CPU-offload 的 DTensor 分片做 `full_tensor()` → 同样报
`No backend type associated with device type cpu`。生产（GRPO 脚本默认 `LOCAL_EXPERT_EXPORT=1`）
走的是 local 导出（先把本地分片搬到设备），所以这条 all-gather 路径在 CPU-offload 下本来就不可用；
自测脚本已对齐生产设置。

**训练自测抓到的第一个真实 bug：CPU-offload 下梯度裁剪的 all_reduce 设备不匹配**

第 6 轮自测已经跑到 forward + backward 成功，挂在 `optimizer_step()` 里：

```
File ".../fsdp_turbo/training/clip_grads.py", line 161, in compute_grad_norm
    torch.distributed.all_reduce(local_sum, op=SUM, group=grp)
RuntimeError: No backend type associated with device type cpu
```

* 触发链：`fsdp_turbo_impl.optimizer_step` → FSDPTurbo `clip_grad_norm` → engram 专用处理器
  `_clip_host_engram_grad_norm`（`host_offload_embedding.py`）→ `compute_grad_norm(dense_parameters…)`
  → 对 **CPU 标量**调用 all_reduce，而 group 是 **hccl**（加速器）进程组。
* 根因：`compute_grad_norm` 取 `device = grads[0].device`，而本集成开着 `offload_policy=True`，
  FSDP2 的 CPUOffloadPolicy 让 **DTensor 的本地分片落在主机内存**；`_get_shard_groups_if_dtensor`
  又从 DTensor mesh 推出 hccl 组——两侧对不上。处理器后面虽然把标量搬到了
  `collective_device`，但**归约已经在那之前发生**，补不回来。
* **为什么以前没暴露**：p0999 系列跑的是"advantage 全 0"的 fixture —— 没有任何梯度，
  `grouped_grads` 为空、`compute_grad_norm` 提前 `return 0.0`，**根本走不到 all_reduce**
  （证据：那些日志里 `grad_norm:0.0` 且无该错误）。这条路径只有**真梯度**才会触发，
  而 top64 + 真实奖励的训练必然触发 —— 也就是说，没有这个训练自测，GRPO 会在第 1 步才炸。
* 修法（`fsdp_turbo/training/clip_grads.py`，fork 侧，纯 Python 无需重编）：新增
  `_reduction_device(group)`（用 `distributed_c10d._get_pg_default_device`）与
  `_all_reduce_on_group_device(tensor, group, op)`：标量先在**组能用的设备**上归约，再搬回原设备。
  inf-norm 与 p-norm 两处归约点都改用它；`group=None` 时行为不变。

### 5.7 第四次/第五次启动与 2026-10-08 重启（top32，1 步冒烟）

> 本节按用户要求分五类记录：**工作记录 / 执行记录 / 错误记录 / 修改记录 / 原因记录**。
> 后续每一轮运行都往 §5.7 的表格里追加，不覆盖旧事实（错的结论用删除线 + 作废说明保留）。

#### 5.7.1 执行记录（做了什么、命令、产物在哪）

| 时间 | 动作 | 命令 / 位置 |
|---|---|---|
| 09-30 08:44 | top32（重新生成前）GRPO 1 步 | 秒退，内存只到 147 GB → 见 §5.7.2 错误 E4 |
| 09-30 09:05 | top32（用户 08:42 重新生成后）GRPO 1 步，**清理残留后** | `MODEL_PATH=.../DeepSeek-V4.1-Flash-top32-bf16 EXPERIMENT_NAME=GRPO-DeepSeekV41-top32-8card bash scripts/run_grpo_top64.sh 1`；日志 `logs/grpo_top64_steps1_20260930_090452.log` + `logs/DeepSeek-V4.1-Flash-top32-bf16-20260930_090503.log` |
| 09-30 09:28 | 该轮跑到 **`all initialize finished, ready to fit`**（历史最远） | 训练侧 `parameters materialized in 281.4s`、`Engram tables loaded: 2 table(s), 45.8 GiB per rank in 127.7s (frozen)`；随后**会话被中断**（用户 `/export` 后 API 403，容器随后停），第 1 步从未产出 |
| 10-08 01:54 | 新会话状态恢复侦察 | 见 §5.7.3 环境复核 |
| 10-08 01:58 | **top32 GRPO 1 步重启**（同一命令） | 日志 `logs/grpo_top64_steps1_20261008_015837.log`、`logs/DeepSeek-V4.1-Flash-top32-bf16-20261008_015848.log`；旁路内存采样 `/tmp/mon_grpo.out`；02:22 死在 E8 |
| 10-08 02:41 | 第 3 次启动（已加 `_stage_on_device_for_ipc` 兜底） | 兜底**立刻抓到 E8 的真凶**：Engram 表根本没被排除在同步之外 → §5.7.7；日志 `logs/grpo_top64_steps1_20261008_024059.log` |
| 10-08 03:09 | 第 4 次启动（表真被排除后） | **历史最远**：权重同步完成 + 引擎起来 + 开始生成，随后 8 个 engine worker 死在 E9 → §5.7.8；日志 `logs/grpo_top64_steps1_20261008_030936.log`、`logs/DeepSeek-V4.1-Flash-top32-bf16-20261008_030948.log` |
| 10-08 04:39–05:01 | **E9 引擎独立复现 ×4**（dummy 权重、GRPO 配方的 argv/env、无 trainer/Ray/DCP） | `bash scripts/run_vllm_dsv41_rl_dummy.sh`；日志 `logs/vllm_rl_dummy_20261008_{043950,045206,045515,050000}.log`；04:55 与 05:00 是 `rl_config` 开/关 A/B（**两边都死** → 排除 RL 配置） |
| 10-08 06:26–06:52 | **E9 单卡微复现**：6 例矩阵（layout × marker × 两种修复） | `ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/test_scatter_nd_update_sk_skip.py --matrix`；结论与根因见 §5.7.8 |
| 10-08 08:44 | **方案 A 验证**（GRPO argv 原样、生产 worker extension、无探针） | 3 个请求全部返回、日志里 `MTE/vector core` 命中 0；日志 `logs/vllm_rl_dummy_20261008_084427.log` |
| 10-08 09:17–09:36 | **方案 C 重编**（`COMPILE_CUSTOM_KERNELS=1 pip install -v -e .`，16 min） | 装上了（`vllm_ascend-…d20261008`），但**单卡行为没变**→ 我的头文件改动没进 kernel 二进制；日志 `logs/build-vllm-ascend-20261008_091711.log`，排查清单见 §5.7.8 末 |
| 10-08 11:25–11:33 | **配置轴矩阵**（回答"为什么用户单独跑 vllm 不报错"）：R2=GRPO 形状 / R1=serve 形状 / R3=serve 形状但不带 expandable | 结论：**翻转结果是 sleep-mode**（它决定越界写"炸"还是"静默写脏"）；marker 在三组里都出现（8 / 288 / 288 次），落点算术完全相同。详见 §5.7.9 |
| 10-08 12:00–12:20 | **C 的重编（第二次，干净）**：`rm -rf csrc/build` 后重编；**单卡微复现三组全过**（`view+skip` FAULT→OK、`contig+skip` WRONG→OK、`view+no-skip` 仍 OK），运行时 `.o` 的 md5 与编译产物一致 | 日志 `logs/build-vllm-ascend-20261008_120004.log`；教训：**`.o` 文件名里的 hash 不是内容 hash**（按 tiling key/config 命名），"同名≠同一份二进制"，判据只能是 md5 + 行为 |
| 10-08 12:33–12:55 | **GRPO 1 步全流程尝试**（带 `DSA_SCATTER_DROP_MARKERS=0`，即只用 kernel 修复、关掉 python 兜底，用来证明 C 端到端有效） | **未走到 E9 那一步就被显存挤死**：engine 报 `Free memory on device (29.32/61.26 GiB) < gpu_memory_utilization 0.6`；日志 `logs/grpo_top64_steps1_20261008_123327.log`。训练侧一切正常（`parameters materialized in 462.0s`、`Engram tables loaded 45.8 GiB/rank in 30.8s`）。⇒ **C 目前只在单卡算子级验证过，还没有端到端验证** |
| 10-08 12:55–12:59 | 事后取证 + 清理我们的残留（ray/vllm 全部退出，本容器已无活进程） | 失败原因是**同机另一容器并发占用**：`npu-smi info -t proc-mem -i 0` 显示 `Process id 2644954 / 2644959, VLLM::Worker_DP, 55315 MB`，这两个 PID 不在我们命名空间（我们那次运行 PID 在 14–19 万段）；我 12:32 启动前查过，当时每卡只有 3 GB 残留。**新增启动前检查项：除了看 HBM 数字，还要看 `proc-mem` 的持有者 PID 是不是我们自己的** |
| 10-08 13:40 | **换机（node-29-119）前置验证**：单卡矩阵 **6/6 全 OK**（`view+skip` / `contig+skip` 均为 `3 rows written, guard intact`，对照组 `view+no-skip` 仍 OK）→ 携带过来的 `.o`（md5 `9a3f7946f898`/`2553f3b8c1a0`，与 12:00 重编产物一致）就是修复版，且在本机芯片（`Ascend910_9382`，与 `.o` 的 `ascend910_93` 目标码兼容）上行为正确，**无需重编** | `ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/test_scatter_nd_update_sk_skip.py --matrix` |
| 10-08 13:50 | **新机器首次 GRPO 1 步冒烟**（`DSA_SCATTER_DROP_MARKERS=0`，只验证 kernel 修复 C 的端到端）：因**另一容器正占着 chip 2**（`npu-smi info -t proc-mem -i 1` 显示 Process id 3960240、61865 MB、容器内 PID 280 —— 又一次"卡看着空闲其实是别人的"），可见设备改为 `0,1,3,4,5,6,7,8`；`HCCL_IF_IP` 换成本机 `80.5.25.119`（网卡名不变 `enp48s3u1u1`）。~~启动后 stage=01 RAY_UP、host 75 GB、8 芯片 HBM 均值 2.9 GB（正常早段）~~ → **13:52 死在 E10**（见下），此段作废 | `HCCL_IF_IP=80.5.25.119 ASCEND_RT_VISIBLE_DEVICES=0,1,3,4,5,6,7,8 DSA_SCATTER_DROP_MARKERS=0 MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16 EXPERIMENT_NAME=GRPO-DeepSeekV41-top32-8card bash scripts/run_grpo_top64.sh 1`；日志 `logs/grpo_top64_steps1_20261008_135055.log`；watcher `/tmp/watch_20261008_1351.log` |
| 10-08 13:52 | 首启用例失败于 **E10**：`WorkerDict_0:7` 创建即死（`set_device(8)` invalid），驱动 `ActorDiedError`、静默退出 | 日志同上一行（1140 行，停在 rank7 创建） |
| 10-08 13:55–14:00 | **E10 定位 + 修复**（机制见下面的 E10 细节；改动落在 `verl/single_controller/base/worker.py` NOSET 分支） | 探针：同环境 shell `set_device(8)` 复现失败、`set_device(2)` 成功且实占 phy-id 3（proc-mem） → 证明 `torch_npu.set_device` 收**逻辑下标**、映射是"第 i 个条目" |
| 10-08 14:01 | **修复后重启**：8/8 `WorkerDict` 创建成功、`[Gloo] Rank 4 is connected to 7 peer ranks...` 组网中；host 115 GB、HBM 均值 3.0 GB（早段正常，且已越过上次的崩溃点） | 日志 `logs/grpo_top64_steps1_20261008_140156.log`；watcher `/tmp/watch_20261008_1406.log`；监控 `/tmp/grpo_monitor_20261008b.out` |
| 10-08 14:25–14:27 | **历史首次全流程到 rollout**：`Engram tables loaded: 2 table(s), 45.8 GiB per rank`（14:25:03）→ **`all initialize finished, ready to fit`**（14:27:12，复现 09-30 的历史最远点）；引擎起服（`v=9`）、**rollout 实际跑了 ~5 分钟**（`running: 8` 持续、TQ 成功请求 159+、SimpleStorageUnit 落盘），**全程无 `vector core` 异常** → **方案 C 端到端行为证据成立**（marker INFO 行未出现——engine worker 的 INFO 被日志级别压掉；下次跑用 `/proc/<pid>/environ` 直查环境变量传播补上这半条证据） | 同上一行日志 |
| 10-08 14:33:59 | **E11：第 1 步 update 首次真跑 → 训练侧 NPU OOM**（rank2/rank5 等同时死，驱动 `ActorDiedError` 退出，训练工从未产出 step-1 指标） | traceback：`_update_actor → train_batch → forward_step → … → indexed_sparse_attention_torch → torch.gather` 要 6.94 GiB、卡上只剩 6.82（`47.71 already allocated`）。HBM 时间线：14:33:42 总 44→20.2 GB（引擎已睡、aicore 70%），17 s 后撞墙 |
| 10-08 14:45 | **E11 测量启动**：训练侧独立自测（top32、**seq 2048**、bsz 1、同 8 卡、无引擎），回答"update 在 2048 的真实峰值 / 单机够不够" | `/tmp/smoke2048.out` + `logs/train_engine_smoke_20261008_144928.log` |
| 10-08 15:05 | **E11 测量结论：单机 as-is 不够，且与别人占的 chip 2 无关** —— 训练侧单独跑（整卡 61.28 GB 全给它）在 seq 2048 也 OOM，8 rank 同死：`Tried to allocate 19.50 GiB`（21.9 allocated / 18.5 free）。同一算子、同一条线：`indexed_sparse_attention_torch` 的 `torch.gather`；19.5 GiB = 4-D int64 索引物化（b·s·topk·hd·8B，随 s 线性，与 GRPO 那次 6.94 GiB 之比 ≈ 2048/730 吻合） | `/tmp/smoke2048.out`；traceback 指向 `ops/cpu/sparse_attention.py:60` |
| 10-08 15:10 | **修复 E11：query 维分块**（`FSDPTurbo/fsdp_turbo/ops/cpu/sparse_attention.py`）。拆出 `_indexed_sparse_attention_queries`，外层按 `FSDP_TURBO_ATTN_CHUNK`（默认 256）沿 dim1 分块后 cat。每 query 在该算子里完全独立（gather/QK/softmax/PV）→ **数值严格等价**；单次索引物化 19.5 GiB → ~2.4 GiB/块 | 补丁见文件内注释；**CPU 等价性测试 8/8 组合全过**（fp32/bf16 × mask 开/关 × compressed 开/关，chunk 1/3/7/23/24/256 vs 不分块，bit-exact 或 ≤1e-6） |
| 10-08 15:12 | **分块后训练侧复测启动**（top32、seq 2048、同 8 卡） | `/tmp/smoke2048_chunked.out` |
| 10-08 15:23 | **分块复测结果：尖峰消失，但暴露真正的墙** —— 19.5 GiB 单次分配不再出现；新失败是 `Tried to allocate 1.69 GiB`（chunk 级）时 **`56.56 GiB 已活跃 / 仅 944 MB free`**（rank6）。解读：**每层 attention 为 backward 保留的中间量 ~5 GiB × 40 层 ≈ 200 GiB**，forward 在第 ~8 层撞满卡。交叉验证全部历史现象（4 层可训 ✓、seq 256 旧自测可过 ✓、GRPO ~730 token 样本死在 ~25 层 ✓）。**该墙与机器数量无关**（同 8 卡拓扑下每 rank 保留量不变；换机只省引擎 6.5 GB 残差） | `/tmp/smoke2048_chunked.out` |
| 10-08 15:26 | **修复 E11 第二步：attention 回退的 backward 重算**（`_AttentionRecomputeFn`，`FSDP_TURBO_ATTN_RECOMPUTE` 默认开）。chunk 在 backward 时重算，只在 forward/backward 之间保留"小输入"，每层保留从 ~5 GiB → O(1)；数值同一计算。CPU 等价性：fp32 差 1–2 ulp、bf16 前向 bit-identical、**`recompute` 与"仅分块"逐位相同（零额外误差）**，8/8 组合全过 | 同上文件；测试输出见本回合记录 |
| 10-08 15:28 | **重算版复测启动**（top32、seq 2048、同 8 卡） | `/tmp/smoke2048_recompute.out` |
| 10-08 15:40 | **重算版复测结果：仍 OOM，且平台几乎不变** —— `Tried to allocate 1.94 GiB` 时 **57.74 GiB 已活跃**（v2 同点是 56.56）。**但 traceback 证明重算路径确实生效**（栈：`_AttentionRecomputeFn.forward` → `_indexed_sparse_attention_queries`）⇒ **"per-layer attention 保留 ~5 GiB×40" 的假设不完整**：还有一类 ~1 GiB/层的累积在别处（v1: 21 GiB@首层 → v2/v3: ~57 GiB@崩点 的爬升曲线未被重算改变）。不再猜，上探针 | `/tmp/smoke2048_recompute.out` |
| 10-08 15:44 | **加内存曲线探针**（`FSDP_TURBO_ATTN_DEBUG=1`：每次 attention 调用打印 `alloc/reserved/free` + 形状），启动复测 | `/tmp/smoke2048_debug.out` |
| 10-08 15:52 | **用户决策：转双机，停止单机路径迭代**（单机测试进程已清、8 卡已释放） | — |
| 10-08 15:55 | **双机脚本就绪**：`scripts/run_grpo_2node.sh`（`master`/`slave` 两种模式）。拓扑默认 `NNODES=2`、**`FSDP_SIZE=16`、`EP_SIZE=8`**、`GEN_TP=GEN_EP=8`；每节点各自 pin `HCCL_IF_IP`（自动探测 NIC）、per-node `ASCEND_RT_VISIBLE_DEVICES`（master `0,1,3,4,5,6,7,8`）。已干跑验证：precheck、ray head 启动、等待循环（判据=活动节点数≥2）、`detect_ip`；`ray status` 资源行 `0.0/8.0 NPU`（带 env）——**坑：raylet 不带 `ASCEND_RT_VISIBLE_DEVICES` 启动会注册 16 NPU**，脚本已在 ray start 前 export | 脚本 + dry-run `/tmp/2node_master_dryrun.out` |

**双机拓扑依据（写下来防漂移）**：

- verl 的 Ray worker 分组用 **PACK**（`single_controller/ray/base.py:555-575`，按 node_ip 排序 PG、每 PG `local_world_size` 个 rank）⇒ rank 0-7 落 node0、8-15 落 node1，节点内成块（这是引擎"node-local DP"假设的前提）。
- 引擎侧：fork 的 rollout 适配器按 `rollout_world_size = TP×dp×pp = 8` 推导 `replica_rank = rank//8`（`vllm_rollout.py:81-88`）+ 上游 commit `8e1422db3`（node-local DP groups for multi-node serving）⇒ **16 rank / TP 8 = 2 个节点内副本**；不要设 `GEN_TP=16`（Engram 校验 TP ∈ {1,2,4,8}）。
- **`FSDP_SIZE=16` 是双机唯一能减少"每 rank 参数物化"的杠杆**（单机 40 层 update 的 ~55 GiB 平台里，非专家参数物化约占一半；FSDP16 把它减半）；`EP_SIZE` 必须保持 8（== GEN_EP，否则 `LOCAL_EXPERT_EXPORT` 自禁用、退回会死在 CPU-DTensor 的 all-gather 路径）。若 init 报拓扑不合法 → `FSDP_SIZE=8`（退化为两个与单机同构的副本，内存墙仍在，仅用于二分定位）。
- 未验证项：跨机 HCCL 实际带宽（默认走 `enp48s3u1u1`/80.5.25.x，与此前单机验证一致；若同步阶段过慢再评估 172.27.x RDMA 网）。

**取证位置提醒（沿用 §5.4 的教训）**：worker 的 `print` 不在驱动日志里，在
`/tmp/ray/session_latest/logs/worker-*.out`；驱动日志只有 Ray 转发的 warning/error。

**新机器要点（node-29-119，与旧机 node-29-131 的差异，已逐项实测）**：

| 项 | 实测 |
|---|---|
| 芯片 | `torch_npu.npu.get_device_name(0)` = `Ascend910_9382`；携带的 `ascend910_93` 目标 `.o` 直接可用（单卡矩阵 6/6 通过），**无需在新机重编** |
| 网卡 / IP | 网卡名不变 `enp48s3u1u1`，IP 是 `80.5.25.119`（旧机 .131）→ 启动必须覆盖 `HCCL_IF_IP`；`/etc/hosts` 里 `node-29-119 → 141.61.29.119`（非本机地址）→ 主机名解析照旧不可用，`GLOO_SOCKET_IFNAME` 兜底仍必要 |
| 设备占用 | chip 2（`/dev/davinci2`）被另一容器 VLLM 进程占 61.8 GB（启动前 `proc-mem` 已核 PID 非本容器）→ 可见设备 `0,1,3,4,5,6,7,8` |
| 工作区一致性 | 全量 /workspace-verl 于 13:31–13:33 拷入本机；vllm HEAD `ced6857af`（`0.30.1.dev0+gced6857af.d20260930`）、fork HEAD `c1e3442d0` + 两处工作区改动（`scatter_nd_update_hp.h` / `dsa_v41.py`）、`.o` md5 均与旧机一致 |
| top32 检查点 | `config.json` mtime 变成 **10-08 04:18**（另有 `config.json_bak`@09-20）——重新核过结构：`n_routed_experts=32`、40 层、96 专家张量/层、267 分片、493.4 GiB，与 09-30 版一致 ✓ |
| 监控工具 | `watch_grpo.py` 新增 `--device-ids`（本机设备集不连续，原"前 8 卡"会把别人的 chip 2 算进来）；本次用 `--device-ids 0,1,3,4,5,6,7,8` |

**E10 细节（换机首启，10-08 13:52；`set_device(8)` invalid）**：

- **现象**：`ray::Ht6Zx6WorkerDict_0:7:WorkerDict.__init__` 创建即死 —— `RuntimeError: ExchangeDevice ... error code is 107001` / `Invalid_Argument: ... value 8 for parameter userDevId is invalid. Expected value: [0, 8)`，驱动随后 `ActorDiedError`。
- **机制（三段证据）**：① `ray/_private/accelerators/npu.py:29-43`：Ray 的 NPU 加速器 id 就是 `ASCEND_RT_VISIBLE_DEVICES` 的**条目值**（直接 split 环境变量）→ 非连续集 `0,1,3,4,5,6,7,8` 下第 8 个 worker 拿到 id `"8"`；② 同环境 shell 复现：`set_device(8)` 失败、`set_device(2)` 成功且分配实占 **phy-id 3**（`proc-mem` 验证）→ `torch_npu.set_device` 收的是**逻辑下标**（0..n-1），映射是"第 i 个可见条目"；③ identity 列表 `0..7` 时"值 == 下标"，所以旧机（以及所有历史运行）从未暴露这条轴。
- **修复**（`verl/single_controller/base/worker.py` NOSET 分支，下方"修改记录"）：把 Ray id 映射回它在可见列表中的**位置**再 `set_device`；`LOCAL_RANK` 一并写这个位置（它在该函数里先于 WorkerDict 存 `_local_rank`，见 `:203`）。identity 列表下两者等价，**不改变旧行为**；非 identity 下 `LOCAL_RANK` 也不会再出现 "8" 这种越界rank。
- **引擎侧无需改**：vllm-ascend 的 `npu_ipc_engine.py:66-73` 明确按"逻辑下标 → 物理 chip"处理（注释里直接举了 `ASCEND_RT_VISIBLE_DEVICES=2,3` 的例子）；`platform.py:86-88` 也登记了 noset 环境变量。
- 教训（给以后换机/跳过坏卡用）：**"哪些卡可见"（环境变量）与"用第几张"（逻辑下标）在 NPU 栈里不是一回事**；条件和旧机不同时，要像这次一样用"同环境 shell 探针 + proc-mem 实证"把两者钉死，而不是照抄旧命令。

**判据（本轮要看什么）**：

| 目标 | 指标键 | 期望 | 出处 |
|---|---|---|---|
| 训推一致 | `training/rollout_actor_probs_pearson_corr` | ~0.99（4 层无 Engram 基线地板 0.9946） | §3；`verl/utils/debug/metrics.py:124` |
| 训推一致 | `training/rollout_probs_diff_mean` / `_max` / `_std` | mean ~1e-2 量级 | 配方脚本头注；`metrics.py:121-123` |
| 奖励曲线 | `critic/rewards/mean`（+ `critic/score/*`、`response_length/*`） | 非 0、随步数上升 | GRPO + DAPO 数学数据 |
| 同步代价 | 每步权重同步耗时 | local 导出（`LOCAL_EXPERT_EXPORT=1`）下远小于 all-gather 的 ~900 s 外推 | D6 |

* 触发条件已核实：`trainer/ppo/v1/trainer_base.py:1663` 只在
  `actor_rollout_ref.rollout.calculate_log_probs` 为真时算一致性指标；该键在
  `verl/trainer/config/rollout/rollout.yaml:237` 默认 `True`，并已在本轮 resolved config
  里确认（日志第 443 行 `'calculate_log_probs': True`）。**不需要额外加开关。**
* 该 fork 的 v1 训练器会经 `MetricsAggregator`（`trainer/ppo/v1/utils.py`）汇总这些键，
  `training/rollout_probs_diff_valid` 取 last。

#### 5.7.2 错误记录（本轮工作流的全部阻塞错误，按发生顺序）

| # | 现象 | 位置 | 根因 | 处置 |
|---|---|---|---|---|
| E1 | `Missing key: model.layers.{1,14}.engram.embed.weight` | 训练侧 DCP | 表在模块树上但不在 DCP state dict（严格加载） | 加载期间摘模块（§5.2） |
| E2 | `AttributeError: '_C_ascend' object has no attribute 'npu_hc_pre_v3'` | 推理引擎 profile run | `.so` 是 09-18 旧构建，落后于源码 | 重编（§5.5） |
| E3 | `CMake Generate step failed` 引用不存在的 target `vllm_quant_lightning_indexer_metadata_obj` | vllm-ascend 构建 | `csrc/build/CMakeCache.txt` 里 `AICPU_CUST_OBJ_TARGETS` 是 **CACHE INTERNAL，只增不减**，残留旧目标名 | `rm -rf csrc/build` 后重编（§5.5） |
| E4 | top32 启动**秒退**、日志无 traceback | 09-30 08:44 那轮 | 上一轮被 `TaskStop` 的 Ray 集群/worker 残留（含 03:03 那轮）与新 run 冲突；同时那次的 ckpt 目录正处于**用户重新生成过程中** | 彻底清进程 + 等重生成完成（§5.5b 教训：**对生成中的权重目录做结构判定前先看 mtime**） |
| E5 | `OutOfMemoryError: 12 worker(s) were killed`（2422/2454 GB） | top64 GRPO，引擎初始化/权重同步 | 物理内存不足（top64 基线 2.0 TB + 峰值 2.42 TB），且残留进程占着 **RSS 里看不见的设备注册主机内存** | 包装脚本启动前清扫 + `RAY_memory_usage_threshold=0.99`；**最终换 top32**（真 32 专家，493.4 GiB） |
| E6 | `RuntimeError: No backend type associated with device type cpu`（`optimizer_step`） | FSDPTurbo `clip_grads.compute_grad_norm` | CPU-offload 的 DTensor 梯度分片在主机，而进程组是 hccl | `_all_reduce_on_group_device`（§5.6） |
| E7 | 自测脚本连续 5 轮的配置/契约错误 | `scripts/train_engine_smoke.py` | 逐条对应生产契约 | §5.6 的轮次表 |
| E8 | `IndexError: list assignment index out of range`（`rebuild_ipc`）→ 8 worker 同死 → **驱动不退出、卡死** | 权重同步（`ready to fit` 后 2 min，10-08 02:22） | 主机张量（Engram 表）交给设备 IPC；表没被排除的真正原因是导出 key 带 `.ep_shard_*` 后缀，`endswith` 判断失效（§5.7.7） | §5.7.4 + §5.7.7 修复；自测检查改成独立证据路径 |
| E9 | `MTE accesses an invalid GM address` / vector core exception，**8 个 engine worker 在第一个真实 prefill 同时死**；同一 bug 在不带 sleep-mode 的配置下**不报错、只写脏**（§5.7.9） | 推理引擎 `indexer.py:109` → `npu_scatter_nd_update_sk` | `[-1,-1]` "跳过标记"被当成真实坐标：算子根本没有 skip 语义（`dsa_v41.py:207` 的 docstring 与 `device_op.py:46` 互相矛盾，实测后者对）；落点固定是 plane storage 之下 188 544 B，**可读与否由 `--enable-sleep-mode` 的分配方式决定** | §5.7.8 定性 + §5.7.9 配置轴矩阵；**方案 A 已验证生效**（`DSA_SCATTER_DROP_MARKERS` 可关），方案 C 源码已改但重编后行为未变（§5.7.8 末排查清单） |
| E10 | 换机首启：`WorkerDict_0:7` 创建即死 —— `RuntimeError: ExchangeDevice ... 107001` / `Invalid_Argument: ... value 8 for parameter userDevId is invalid. Expected value: [0, 8)`，驱动 `ActorDiedError`（13:52，启动后 ~1.5 min） | `verl/single_controller/base/worker.py:281` 的 `set_device(int(local_rank))` | **Ray 的加速器 id ≠ torch_npu 的逻辑下标**：Ray NPU manager 把 `ASCEND_RT_VISIBLE_DEVICES` 的条目值当 id（`npu.py:29-43`），本机为避开别人的 chip 2 用了 `0,1,3,4,5,6,7,8` → 第 8 个 worker 的 id 是 `"8"`；而 `set_device` 要逻辑下标（0..7）。identity 列表下两者相等，故旧机从未触发 | worker.py NOSET 分支加"值→位置"映射（`LOCAL_RANK` 同步用位置）；identity 时 no-op。探针实证见 E10 细节（shell 同环境 `set_device(8)` 失败 + `set_device(2)` 实占 phy-id 3） |
| E11 | 第 1 步 update：`torch.OutOfMemoryError: ... Tried to allocate 6.94 GiB (NPU 2; 61.28 GiB total; 47.71 already allocated; 6.82 free)`（14:33:59；rank2 差 ~0.12 GB、rank5 差 ~1.3 GB） | 训练侧 forward：`indexed_sparse_attention_torch` → `torch.gather`（`FSDPTurbo/ops/cpu/sparse_attention.py:60`；训练专用 torch 回退，fused MLA 是推理路径：`use_sparse_flash_attn=False`） | **贴边超限，与别人占的 chip 2 无关**（我们设备集不含 phy-id 2；OOM 的 "NPU 2" 是逻辑号=phy-id 3，那张卡整张是我们的）。colocate 下每卡可用 ≈ 54.75 GB（61.28 − 引擎睡眠残差+框架 6.53），update 峰值需求 ≈ 54.9+ GB。背景：**该操作点从未测过** —— §5.6 自测是 `seq_len=256`（激活 1/8），且 adapter 无激活重算（recipe: "V4.1 recompute is disabled inside the FSDPTurbo adapter"） | 先做训练侧独立测量（14:45 已启动，seq 2048）→ 按结果选：分块 attention gather（去掉 gather 的瞬时大头）／压缩共驻残差／（必要时）重算支持 |

#### 5.7.3 环境复核（10-08 新容器，逐项实测）

| 项 | 结果 |
|---|---|
| 主机 / IP | `node-29-131`（与 09-30 同一台），网卡 `enp48s3u1u1` → `80.5.25.131`，包装脚本已 pin |
| NPU | 16 张，AICore 0%、HBM 各 ~3 GB 残留 → **空闲**（共享机器，启动前必查） |
| 主机内存 | total 2454 GB / available 2418 GB；无残留进程占用 → 起点干净 |
| `_C_ascend` | `vllm_ascend_C...so` @ 09-30 04:14，`strings` 命中 `npu_hc_pre_v3`；5 个 V4.1 算子全在（`v4.1 ops ok: True missing: []`） |
| 包解析 | `PYTHONPATH=/workspace-verl/{vllm,vllm-ascend-v41-private,FSDPTurbo}`；vllm `0.30.1.dev0+gced6857af.d20260930`、numpy 1.26.4、transformers 5.14.1（重编副作用，已确认训练栈可用） |
| 代码改动仍在盘 | `verl/.../fsdp_turbo_dsv41_impl.py`（`hide_engram_tables`/`restore_modules` 在 165/207/597/601 行）+ `FSDPTurbo/.../clip_grads.py`（`_all_reduce_on_group_device` 在 99/176/195 行）；`git status`：这两个文件 + 本 worklog 为改动项，未提交 |
| top32 检查点 | `config.json` mtime 09-30 08:42（8 天未变）；`text_config.n_routed_experts=32`、40 层、topk=6、`engram_layer_ids=[1,14]`、表 `[384006168,256]/[384016682,256]`；275 个文件 |
| 数据/分词器 | `dapo_math_train_512.parquet`、`dapo_math_val_64.parquet`、`rl_tokenizer/{tokenizer.json,tokenizer_config.json}` 均在 |

**注意（非阻塞）**：容器里有**其他会话**的僵死进程（`git clone` vllm 在 `/test`、若干 curl），
不在本工作区路径、也不占 NPU/内存，**不动它们**；`/workspace-verl/vllm` 未被触碰。

#### 5.7.4 错误 E8 + 修复：权重同步把**主机张量**交给设备 IPC（10-08 第二次启动暴露）

**现象**（本轮 02:22，即 `ready to fit` 之后 2 分钟）：8 个 vLLM worker 同时抛

```
File ".../verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py", line 339, in receive_weights
    tensor = rebuild_ipc(handle, self.device.index)
File ".../bucketed_weight_transfer.py", line 51, in rebuild_ipc
    list_args[6] = device_id
IndexError: list assignment index out of range
```

随后**驱动不退出而是卡死**（发送端在 `self.socket.recv()` 等一个永远不会来的 ack）：日志 3 分钟
0 行增长、内存平、AICore 0%。这是 §5.8 监控表里"log +0 且 aicore=0"那条判据的实例。

**根因（全部有实测证据，不是推断）**：

| 证据 | 内容 |
|---|---|
| 探针 1 | `reduce_tensor(cpu_tensor)` → `func=rebuild_tensor, len(args)=3` |
| 探针 2 | `reduce_tensor(npu_tensor)` → `func=rebuild_npu_tensor, len(args)=15`，且**下标 6 就是设备号**（CUDA 约定一致，所以 `list_args[6]` 的写法本身没错） |
| 崩溃点 | 报错发生在 `receive_weights`（line 339，走 `bucket_meta[...]["handle"]`）而**不是** `_init_buffer`（line 385）→ 说明**桶缓冲区（设备张量）的 IPC 是好的**，坏的是"超桶大小张量"单独发的句柄 |
| 代码 | 只有 `_direct_send_large_weight` 会塞非 None handle；而 `transformer_impl.py:984` 的通用导出只对 **DTensor** 做 `.to(device).full_tensor()`，**非 DTensor 参数原样 yield**（在 `offload_policy=True` 下它们留在主机） |
| 尺寸表 | ~~top32 里超过 512 MiB 桶的只有 `embed.weight`/`head.weight`，**Engram 表已被排除**~~ → **此条判断错误**，见 §5.7.7：Engram 表其实**没被排除**，它才是那个超桶的主机张量 |

结论：**IPC handle 的 args 长度 = 3 ⇒ 那不是设备句柄**，`rebuild_ipc` 无条件写 `list_args[6]` 就炸。
（**触发它的具体张量判断错了**，机制判断对了——修复见 §5.7.7。）

**纠正一条旧结论**：`grep -l "IndexError: list assignment index out of range" logs/` 命中
`grpo_top64_steps1_20260930_071346.log`（09-30 07:53 的 **top64** 那轮）。所以 §5.5b 里"top64 卡在内存"
的判断**不完整**：那次同样是先死在这个 bug 上（内存是叠加的第二道墙）。作废记录保留在此，按用户要求不删。

**修法**（`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`，两处，纯 Python）：

| # | 改动 | 原因 |
|---|---|---|
| 1 | 新增 `_stage_on_device_for_ipc(weight, name)`，`_direct_send_large_weight` 先过它再 `reduce_tensor`；staged 张量绑定到 ack 之后才释放 | 主机张量在发送端做一次 H2D（接收端本来也要落到卡上），代价 = 一个张量的瞬时显存（1.26 GB），**比**"把桶调到 2 GB 避开 direct-send"**更省** HBM（桶是常驻的）；ack 前释放会把 block 还给本进程分配器而对端仍映射着 |
| 2 | `rebuild_ipc` 里 `len(list_args) < 7` 时抛 `RuntimeError("...not an accelerator IPC handle...")` | 失败要**说清是哪一侧错**。原 `IndexError` 把"CPU 张量"这个真因伪装成下标问题；另外若某个非设备句柄**恰好**有 ≥7 个参数，静默写下标 6 会变成"同步到错卡"这种更难查的错 |
| 3 | H2D 用**阻塞**拷贝 + `synchronize()` 后再发句柄 | `non_blocking=True` 只保证入队；跨进程读的是刚提交的拷贝目标，会和 ack 竞态 |

**回归测试**（先测再跑，省掉一次 30 分钟周期）：`tests/workers/rollout/test_bucketed_weight_transfer_ipc.py`
—— (a) 主机句柄必须被 `rebuild_ipc` **明确拒绝**；(b) staged 后的设备句柄经 `mp.Pipe` 传给**另一个进程**
rebuild，逐元素和一致。两卡即可跑：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 python -m pytest -q tests/workers/rollout/test_bucketed_weight_transfer_ipc.py
```

> 附带教训（**操作纪律**，不是代码）：我在同一行里写 `pkill -f "VLLM::|ray::" ... && 启动新脚本` 时，
> pkill 把**自己的 shell** 打死了（命令行里含同样字面量），exit 144、启动根本没发生；而 run 1 死后留下的
> `[VLLM::Worker_TP] <defunct>` 僵尸让 `ps|grep -c` 报了 9 个"engine 还活着"的假象。→ 监控脚本里
> `proc_count` 显式排除 `Z` 状态，且 kill 与启动分两条命令。

#### 5.7.7 E8 的真凶（第 3 次启动由兜底检查抓到）：Engram 表**根本没被排除在同步之外**

第 2 轮我按 §5.7.4 的假设（"超桶的只有 embed/head"）加了兜底：`_stage_on_device_for_ipc` 在把主机
张量搬上卡之前先检查大小，装不下就**报错并带上张量名**。第 3 次启动前它立刻兑现：

```
File ".../bucketed_weight_transfer.py", line 315, in _direct_send_large_weight
    staged = _stage_on_device_for_ipc(weight, name)
File ".../bucketed_weight_transfer.py", line 83, in _stage_on_device_for_ipc
    raise RuntimeError(
RuntimeError: Oversized weight model.layers.1.engram.embed.weight.ep_shard_00006_of_00008 (22.9 GiB)
  is on cpu and would not fit in the 36.1 GiB free on npu:6.
```

**根因**：导出用的 key **不叫** `...embed.weight`。FSDPTurbo 的
`TorchHostOffloadEmbedding._save_to_state_dict`（`host_offload_embedding.py:423-427`）把 key 改名成
`prefix + "weight" + _checkpoint_suffix()`，而 `_checkpoint_suffix()` =
`.ep_shard_{rank:05d}_{of_size:05d}`（`:341-344`）。我们的排除是

```python
if name.endswith(_ENGRAM_EMBED_SUFFIX):   # ".engram.embed.weight"
```

`endswith` 对 `...embed.weight.ep_shard_00006_of_00008` **永远不成立** → 每张表照旧导出
（每 rank 22.9 GiB 主机张量 × 2 张表 × 8 rank），既撞 IPC 假设（E8 的 IndexError），
又在算浪费：每步白搬 ~366 GiB（D4 想避免的正是这个）。

**两处同源错误（都要记）**：
1. 引擎侧排除用 `endswith` → 失效；
2. **我的自测检查也用同一个 `endswith`** → 09-30 那次打印 `engram tables in payload: []` 是
   **假绿**，检查器和被检查的代码犯了同一个命名假设错误，所以 bug 活到了运行时。
   教训：**检查必须用与被检对象不同的证据路径**（这里=对象身份/属性，而不是同一个字符串规则）。

**修复**：

| # | 位置 | 改动 | 理由 |
|---|---|---|---|
| 1 | `fsdp_turbo_dsv41_impl.py` 新增 `_is_host_engram_param(name, param)`，`_export_param` 用它 | **先看参数上的标记** `param._engram_host_offload`（FSDPTurbo 自己在 `:282` 打上的，它的优化器也用同一标记跳过表），名字里的 `.engram.embed.weight` **子串**只作兜底 | 不依赖 key 的拼法；`endswith`/`startswith` 这类"名字末尾/开头"的判断正是这次事故的形式 |
| 2 | `bucketed_weight_transfer.py::rebuild_ipc` | `len(args) < 7` → `RuntimeError`，说明"这不是加速器 IPC 句柄" | 把 `IndexError` 变成能说清方向的错误 |
| 3 | 同上，新增 `_stage_on_device_for_ipc` | 主机张量先阻塞 H2D + `synchronize()`；**超过设备可用内存一半就带名字报错** | 前者修 E8 的表现形式（任何主机张量），后者把"某个大表漏进同步"这类静默错误变成响亮错误——**它就是本轮抓到 E8 的钩子** |
| 4 | `scripts/train_engine_smoke.py` 导出检查 | 改子串匹配、**发现表就 raise**（不再只 print），并额外打印"超过 512 MiB 桶的张量：名字+体积+设备" | 假绿的直接修补；顺带把"哪些张量走 direct-send"变成自测的常规输出 |

回归：`_is_host_engram_param` 四条用例（带后缀+有标记 / 带后缀+丢标记 / 正常专家名 / gate bias）
离线全对；`tests/workers/rollout/test_bucketed_weight_transfer_ipc.py`（两卡跨进程 IPC）通过。

#### 5.7.8 E9：`[-1,-1]` 不是"跳过标记"——`npu_scatter_nd_update_sk` 把它当真实坐标（10-08 定位完成，修复待选）

**现象**：10-08 03:09 那轮（第 4 次启动，历史最远：权重同步完成、引擎起来、开始生成）8 个 engine worker
全部死在同一个设备 kernel 里，CANN dump 的 kernel 名是 `ScatterNdUpdateSk_38f6c0f4...`（该 `.o` 就是
09-30 04:13 那次重编的产物，`vllm_ascend/_cann_ops_custom/.../kernel/ascend910_93/scatter_nd_update_sk/`
里能对上号）：

    error code = 0x800000, errorStr: MTE accesses an invalid GM address or the cross-device memory
    access times out ... [Error]: The vector core execution is abnormal.

**定位链**（三段证据，每一段都比"再跑一次 GRPO"便宜）：

| # | 实验 | 结论 |
|---|---|---|
| 1 | `scripts/run_vllm_dsv41_rl_dummy.sh`：只跑引擎、dummy 权重、GRPO 配方原样的 argv/env（无 trainer、无 Ray、无 DCP）→ 10 分钟到第一个真实 prefill | **E9 与训练侧无关**。04:39 / 04:55 / 05:00 三次都在第一个真实请求的 prefill 里死。04:55 是 `rl_config={enabled:true, batch_invariant:true}`，05:00 是 `{false, false}` → **rl_config / batch-invariant 被排除**（这是当时唯一还没排除的"RL 特有配置"） |
| 2 | `scripts/dsa_scatter_probe.py`（`DSA_SCATTER_PROBE=1`，`--worker-extension-cls dsa_scatter_probe.ProbeWorkerExtension`）在主机侧检查每次 scatter 的坐标 vs plane shape | 坐标**全部在界内**（出事的 int8 plane：`rows[-1,1] cols[-1,1]`，`skipped=3/5`）→ 不是越界索引。失败那次 scatter 独有两个特征：**(a) 有 `[-1,-1]` 行**、**(b) plane 是 view**（`strides=(131072,128,1)`，连续期望 `64*128=8192`）。同轮的 bf16 scatter 是 contiguous + `skipped=0/5` → 活下来 |
| 3 | `scripts/test_scatter_nd_update_sk_skip.py --matrix`：单卡微复现，两个变量正交开，并在 plane 前后各放 16384 个元素的哨兵 | **marker 是因；view 只决定它静默还是响亮**（表见下） |

    layout  skip   fix    结果
    contig  no     -      OK
    contig  yes    -      算子"成功"，但哨兵抓到：plane 基址 -8320 元素被写了 128 字节（值=7）
    view    no     -      OK
    view    yes    -      FAULT: vector core exception / MTE invalid GM address  ← 就是 E9
    view    yes    drop   OK（3 行，marker 行不写）
    view    yes    sink   OK（3 行，marker 行重定向到保留页）

`-8320 = (-1*64 + -1*1) * 128`：kernel 先算 `linear = Σ coord[i]*mask[i]`（有符号），再把这个 `int`
直接 `static_cast<uint64_t>` 当元素偏移用。连续布局下偏移"绕回"到 buffer 前 8 KB（还在映射内 → 静默
改写邻居）；view 布局下走 `ComputeViewedRowOffset`，`i0 = linear/64` 无符号除法 ≈ 2.9e17 页 →
跑出映射外 → 设备异常。**同一个错误坐标，两种死法。**

**根因**：这个算子根本没有"跳过"语义。fork 自己两处文档互相矛盾，实测站在 `device_op.py` 那边：

| 位置 | 说法 | 实测 |
|---|---|---|
| `vllm_ascend/attention/dsa_v41.py:207` `scatter_cache_sk` docstring | "treats the builder's `[-1, -1]` coordinates as skipped rows, matching V4" | **错**。aclnn 包装 / tiling / 三个 kernel（`csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_{hp,large_index,common}.h`）没有任何一处处理负坐标，`grep -i skip` 只命中注释 |
| `vllm_ascend/device/device_op.py:46` | "this helper does not filter negative slots, **and neither fast operator is assumed to skip them**" | 对。它自己只喂 `slots[:tokens]` 前缀（V4.1 的 marker 是**交错**的，前缀切不出来，所以 V4.1 用不了这招） |

**marker 为什么必然出现**（不是偶发脏数据）：V4.1 的 index-K / long-KV 按 `compress_ratio=2` 存
"每 2 个 token 一个压缩组"，`dsa_v41.py:769-798` 先 `valid &= positions.remainder(2) == 1`，再把
invalid 行**两个坐标都**写成 -1（`_slot_mapping_2d` 预填 -1，`:597`）→ 每个真实 prefill 的每个压缩层
都有约一半行是 marker。`indexer.py:109-113` 一次连发两个 scatter（int8 K + fp16 scale），探针看到死在
第一个（第二个的打印从未出现）。

**为什么 09-30 之前没炸**：V4.1 支持 09-21 才落地（fork HEAD 09-29 "fix"），而 09-30 各轮 GRPO 全死在
E5/E8，从没跑到真实生成；启动期的 profile run 不走 `scatter_cache_sk`（探针第一次打印在
`Application startup complete` 之后，日志行 501→503 / 549→551）。E9 是"第一次真的开始生成"才暴露的
潜伏 bug —— 也解释了它为什么一上来就是"8 个 worker 同时死"而不是一次局部失败。

**影响面（比阻塞 GRPO 更大的一条，未证实）**：contiguous plane + marker 的组合是**静默改写 plane 前面
128 字节**（本次实测）。如果 long-KV plane（bf16 `(33357,128,1,512)`，stride0=65536 元素=连续）也吃到
marker，那它从 09-21 起每次压缩写入都在动邻居内存 —— 对 §real4 的"训推差距"取证是个干扰项。探针那次
没跑到 long-KV 的 scatter（层 0 的 index-K 先死），**待证实**：下一次冒烟把探针的 `_extent_check` +
`DSA_SCATTER_PROBE_TOUCH=1` 打开就能看到（代码已写，还没跑过）。

**修复选项**（矩阵已验证 drop / sink 都能让 faulting shape 变 OK）：

| 方案 | 改动 | 代价 | 备注 |
|---|---|---|---|
| **C（建议）改 kernel** | `arch22/scatter_nd_update_hp.h` 的 `DoBatch` / `ScatterTileSlice` 和 `large_index` 路径：`linearIndex < 0` 就跳过该行。builder 永远两坐标同为 -1 且 `mask[i]>=1` ⇒ `linear<0` 与"是 marker"严格等价 | 要重编自定义算子（§5.5 的路子；E3 提醒：动 target 列表需 `rm -rf csrc/build`） | 与 5 处调用点**已有的假设**一致，一次覆盖 view/contig 两种错法 |
| B sink 页 | 每个压缩 plane 多留 1 页，marker 坐标 `torch.where` 到那页 | 改 `cache_config.get_deepseek_v41_kv_cache_config` + 视图构造，并保证 block pool 永不分配该页 id | 无 sync；矩阵里 `fix=sink` 已验证 |
| A drop | `scatter_cache_sk` 里按 `(indices>=0).all(-1)` 过滤 | **每次 scatter 一次设备 sync**（每层 4 次），与 `_request_counts` 那种"故意不同步"的写法相反 | 不用重编，能立刻验证；正确性由 `fix=drop` 验证 |

**修复进展（10-08 08:44，方案 A 已验证；C 已落码待重编）**

方案 A（python 侧过滤 marker 行）打进了 `vllm_ascend/attention/dsa_v41.py:207-247`
（`scatter_cache_sk`，同时把那句错的 docstring 改成实测结论 + 注明同步代价）。然后**完全同一套
GRPO argv、同一台机、同一 dummy 权重**重跑复现脚本：

| 时间 | 配置 | 结果 |
|---|---|---|
| 04:39 / 04:55 / 05:00 | 未打补丁 | 3/3 在第一个真实 prefill 的 `indexer.py:109` scatter 上死（`MTE accesses an invalid GM address`） |
| 08:44 | **打了方案 A**，`WORKER_EXT=verl...vLLMColocateWorkerExtension DSA_SCATTER_PROBE=0`（即生产形态，不带探针） | **3 个请求全部返回，`finish_reason=length`，日志里 `MTE/vector core` 命中数 0** |

日志：`verl/logs/vllm_rl_dummy_20261008_084427.log`（08:44:27 起 → 08:46:51 三个请求跑完，
比未补丁版多活了整个 prefill + decode）。dummy 权重的输出文本是乱码，符合预期——**这一步证明的是
"不再炸"，不是"文本正确"**；文本正确性要用真实权重 + 用户那套 `run_vllm_top32.sh` 的 argv 再 A/B 一次。

方案 C 的落点（已改，未编译）：`csrc/moe/scatter_nd_update_sk/op_kernel/arch22/scatter_nd_update_hp.h`
的 `DoBatch`（:220-243，逐行 store 前判 `linearIndex < 0` 继续，UB 里已搬好的数据按 `slotElements_` 跳过）
与 `ScatterTileSlice`（:236-246，`for s` 之前就 continue，顺带保证 ping-pong 的 `MTE3_MTE2` flag 成对）。
**只需要这两处**：kernel 入口（`op_kernel/scatter_nd_update_sk.cpp:44-62`）只 dispatch 到 HP 和
LargeIndex 两条路，而 `LargeIndexKernel::ProcessOneBlock`（`scatter_nd_update_large_index.h:100-103`）
本来就有 `linearIndex >= startInt64_ && linearIndex < endInt64_` 的范围判断——负线性索引天然落在
范围外被跳过。这也解释了为什么这个 bug 一直没在 int64 索引的大缓存路径上露头。`no_sort.h` /
`scatter_nd_update.h` / `linear_index.h` 在这份 build 里不可达。

重编命令沿用 09-30 那次（`logs/build-vllm-ascend-20260930_035942.log` 开头）：
`COMPILE_CUSTOM_KERNELS=1 pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/`
，实测约 16 分钟。验证顺序：单卡 `--matrix --engine` 期望 FAULT→OK，再跑同一套 GRPO argv 期望不炸。

**09:17 重编跑完，但 C 目前无效（10-08 09:31 实测，别误判成"修好了"）**：`Successfully installed
vllm_ascend-0.19.1rc2.dev2427+gc1e3442d0.d20261008`，build log 里也确实调了
`build.sh --pkg --ops="scatter_nd_update_sk;…"`；可是单卡再跑一次 §5.7.8 的矩阵，行为**一点没变**
（`view+skip` 仍 `FAULT`，`contig+skip` 仍是 `plane_base-8320` 的静默改写）。二进制侧的线索也对得上：
`vllm_ascend/_cann_ops_custom/…/kernel/ascend910_93/scatter_nd_update_sk/*.o` 的 mtime 更新到 09:31，
**但 16 个 hash 名字全是 09-30 那批旧名，仓库里没有新 hash 的 .o** ⇒ 头文件改动没进 kernel 二进制。
下一步（按代价排序）：① `touch csrc/moe/scatter_nd_update_sk/op_kernel/scatter_nd_update_sk.cpp` 再编
（改的是 `arch22/*.h`，Ninja 很可能没把它当依赖）；② 不行就 `rm -rf csrc/build` 重编（E3 同族的缓存问题）；
③ 排除"运行时加载的不是这份 `_cann_ops_custom`"（`csrc/cmake/scripts/custom/install.sh` 可能把包装进了
toolkit 的 `opp/vendors/…`）。**判定标准只有一个：§5.3 的单卡矩阵行为变了没变**，文件时间戳不算证据。

**12:00 重编完成，C 已验证生效**：`rm -rf csrc/build` 后重编（16 min），运行时 `.o` 的 md5 与编译产物一致
（`9a3f7946f898` 对 `38f6c0f4`，`2553f3b8c1a0` 对 `0b353e2a`）。单卡矩阵三组全过：
`view+skip=yes` 从 FAULT 变 **OK: 3 rows written, guard intact**；
`contig+skip=yes` 从 WRONG RESULTS 变 **OK: 3 rows written, guard intact**；
`view+skip=no` 对照组仍 OK。kernel 里的 `linearIndex < 0` 跳过逻辑生效。
日志 `logs/build-vllm-ascend-20261008_120004.log`。

**关键教训**：`.o` 文件名里的 hash **不是内容 hash**（是按 tiling key/config 命名的），所以"同名 ≠ 同一个二进制"。
09:17 那次我据此判断"改动没进二进制"其实是巧合，真正的判据是 md5。

机制、代码逐个讲（每段的输入/输出/为什么错）、地址算术（`-65 → -8320 / -254080`）、复现工具与
"改了源码≠改了二进制"的完整记录，见
[`learn/02_dsv41_indexer_scatter_marker_e9.md`](learn/02_dsv41_indexer_scatter_marker_e9.md)。

**环境坑（本次新撞，记一次）**：`bash scripts/run_vllm_top32.sh` 若从 `/workspace-verl` 目录里起，
`import vllm` 会命中 `/workspace-verl/vllm` 这个**仓库目录本身**（namespace package，无 `__init__`），
报 `ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)` 秒退。要么换
cwd，要么像 GRPO 配方那样显式 `PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo`。

**复现与产物**：

| 项 | 位置 |
|---|---|
| 引擎独立复现 | `bash scripts/run_vllm_dsv41_rl_dummy.sh`（A/B 开关 `RL_ENABLED/BATCH_INVARIANT/BLOCK_SIZE/DO_SLEEP_WAKE/WORKER_EXT` 见脚本头） |
| 真实权重下问它会不会复读 | `bash scripts/ask_dsv41_serve.sh [port]`——4 个 greedy 提示（指令式 / `35+28=` 续写 / `The capital of France is` / >128 token 跨块），并把"最长重复词串"算成数字，避免靠肉眼判断复读 |
| 四次复现日志 | `verl/logs/vllm_rl_dummy_20261008_{043950,045206,045515,050000}.log` |
| 单卡矩阵 | `ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/test_scatter_nd_update_sk_skip.py --matrix`；`--engine` 用探针抓到的真实形状（33357×64×128 int8，stride 131072） |
| 跑单卡算子的前置 | 必须先 `bootstrap_custom_op_env(include_vendor_lib=True)`（脚本已内置），否则报 `aclnnScatterNdUpdateSk ... not in libopapi.so` —— 这是"算子没找到"，不是 E9 |

#### 5.7.9 为什么"单独跑 vllm 不报错"而 GRPO/复现脚本报了（10-08 11:25–11:33 配置轴矩阵）

用户的直接提问。答案不是"那个配置没触发 bug"，而是**触发了、但不报**。三组运行只差显存配置，
代码与权重（dummy）完全相同，且都关掉方案 A 的过滤（`DSA_SCATTER_DROP_MARKERS=0`）、都挂探针
（`DSA_SCATTER_PROBE=1 DSA_SCATTER_PROBE_STRICT=0`）：

| 组 | sleep-mode | expandable_segments | 其余参数 | marker 调用次数 | 设备异常命中 | int8 plane 的 storage 起点 | 起点是否 16 MB 对齐 |
|---|---|---|---|---|---|---|---|
| R2 = GRPO/复现形状 | **1** | 0 | mem 0.6 / batch 2048 / mlen 2048 / seqs 256 / prefix off | 8（只活到层 2） | **158** | `0x12d940000000` | **是** |
| R1 = 用户 serve 形状 | 0 | **1** | mem 0.9 / batch 8192 / mlen 8192 / seqs 16 / prefix on | **288** | **0** | `0x12d713550000` | 否 |
| R3 = serve 形状但不开 expandable | 0 | 0 | 同 R1 | **288** | **0** | `0x12d713550000` | 否 |

三组的 marker 落点算术**一模一样**（探针打印）：

```
marker_dest[view]=-254080B->0x…  plane_storage=[起点,终点)  below_storage=188544B
```

也就是每次都往"本 plane storage 起点之下 188 544 字节"写 128 字节（int8 K；fp16 scale 的落点是
`-262018B`，同一套公式换个 itemsize）。区别只在**那块地址可不可读**：

* 开 `--enable-sleep-mode` 时，KV 池张量由 CuMemAllocator 单独预留 VA，起点正好落在 16 MB 边界上
  → 它下面 188 KB **不属于本进程的任何映射** → MTE 越界 → 158 次设备异常、8 个 worker 同死；
* 不开 sleep-mode 时，KV 池是缓存分配器里一个几十 GB 的大块内部切片（起点只 4 KB 对齐）
  → 下面 188 KB 仍在本进程映射内 → **不报错，改成写脏邻居**。

**排除项**：R1 vs R3 只差 `expandable_segments`，结果相同 → **expandable 不是这条轴**；
`prefix_caching / mem_util / max_num_batched / max_num_seq` 也都不是（R1/R3 与 R2 同时差这些，
但 R1/R3 彼此只差 expandable 却同为"不炸"）。真正翻转结果的是 **sleep-mode**。

**结论口径（写下来防漂移）**：

1. "我的 vllm 起来了、能出词、没报错"**不是**引擎健康的证据；在这个 bug 下它等价于
   "每层每次压缩写入都在 KV arena 之下 188 KB 处写脏 128 B"（矩阵里 288 次）。
   用户观察到的**复读输出**与该静默形态相容，但**尚未证明是它的后果**——见下面待办 1。
2. 探针打印的 `in_mapped_block` 字段**不能当证据**：`torch.npu.memory.memory_snapshot()` 看不见
   CuMem/池化分配的块，R1/R3（不炸）也打印 `NO`。判据只有"有没有设备异常"这一条。
3. verl 侧为什么必然撞上：colocate 训练必须靠 sleep-mode 腾 HBM（`free_cache_engine`），
   且配方注释 `run_deepseek_v41_grpo_fsdp_turbo_npu.sh:113` 说明 sleep-mode 的池与
   expandable_segments 互斥。也就是说 **GRPO 走的是"会炸"的那一侧分配方式**，这不是偶然。

**本轮修改记录（都是诊断件，不含修复本体）**：

| 文件 | 改动 | 目的 |
|---|---|---|
| `scripts/dsa_scatter_probe.py` | 新增 `_marker_forecast()`：按 kernel 的两条分支复算 marker 落点，打印 `marker_dest/branch/delta/plane_storage/in_own_storage/in_mapped_block`；新增 `DSA_SCATTER_PROBE_STRICT=0` 让越界只警告不中断（否则探针自己先把实验停了，看不到算子的真实行为）；把 `[-1,-1]` 的注释从"是跳过标记所以不算违规"改成"marker 会被真写，计数而非豁免"；新增 `_mapped_ranges()`（结论：此路不可信，见上） | 把"为什么炸/为什么不炸"从叙述变成可打印的数 |
| `scripts/run_vllm_dsv41_rl_dummy.sh` | 新增 4 个镜像开关 `SLEEP_MODE / EXPANDABLE / PREFIX_CACHING / MEM_UTIL / MAX_NUM_BATCHED / MAX_SEQS`（默认仍是 GRPO 形状），`--enable-sleep-mode`/`--no-enable-prefix-caching` 从硬编码改为条件生成 | 能一行命令复刻用户的 serve argv |
| `vllm_ascend/attention/dsa_v41.py` | 方案 A 的过滤改为受 `DSA_SCATTER_DROP_MARKERS`（默认过滤）控制，env 读取上移为模块常量 | 让"带过滤/不带过滤"两种运行可以在同一份代码上对照，不必改两次文件 |

**产物**：`verl/logs/vllm_rl_dummy_20261008_{112503,112713,112936}.log`；驱动 `/tmp/e9_axis_matrix.sh`
（含三组参数与逐项统计：请求数 / 设备异常数 / marker 数 / 首个落点打印）。

**待办**：

1. 证"复读是否就是它"：R1 配置下 `DSA_SCATTER_DROP_MARKERS=0` 与 `=1` 各跑一次真实权重服务，
   同一 prompt 对比（`scripts/ask_dsv41_serve.sh` 已备好 4 个 greedy 提示与最长重复词串计数）。
2. ~~C 仍未生效~~ ⇒ **C 已在 12:00 那次干净重编后于单卡验证生效**（见 §5.7.1 12:00 行）；
   还缺的是**端到端**：12:33 的 GRPO 尝试被别的容器挤死，没走到 scatter。重跑命令（含关掉 python 兜底）：

   ```bash
   # 启动前先看持有者是不是我们自己的进程（同机另一容器会伪装成"卡空闲"）
   npu-smi info | grep -oE "[0-9]+ ?/ 65536" | head -8; npu-smi info -t proc-mem -i 0 | grep "Process id"
   DSA_SCATTER_DROP_MARKERS=0 \
   MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16 \
   EXPERIMENT_NAME=GRPO-DeepSeekV41-top32-8card bash scripts/run_grpo_top64.sh 1
   ```

   判据：日志里出现 `[dsa_v41] marker rows passed through to the op (DSA_SCATTER_DROP_MARKERS=0)`
   （证明开关真的传到了 worker），且 `Application startup complete` 之后的第一个真实 prefill
   不再出现 `vector core exception`。
3. 若 engine 起不来但确认不是别人占卡：`ROLLOUT_GPU_MEM_UTIL` 与训练侧物化后的 HBM 残留是硬约束
   （本次 29.3 GB free < 0.6×61.3 = 36.8 GB），先量出训练侧在 engine 构建时刻实际占多少，再决定是
   降 util 还是让 trainer 更彻底地 offload——**不要用降 util 去掩盖共存占用**。
4. C 端到端通过后应撤除 A（`dsa_v41.py` 的过滤 + 那个诊断开关），因为它每次 scatter 一次 DtoH 同步
   —— fork 的 `AGENTS.md:164-177` 明确反对热路径 `item()`。

#### 5.7.6 为什么继续用 top32（而不是再试 top64）

top64 在单机上撞到的是**物理内存上限**（E5），不是代码问题；top32 是真正的 32 专家剪枝版
（493.4 GiB vs 578 GiB），09-30 那轮实测内存显著更轻（同期 874 GB vs top64 的 1.34 TB），
并且是唯一一次跑到 `ready to fit` 的运行。Engram 表大小不变（两张 196.6 GB），所以
训练侧 Engram 路径仍然被完整覆盖 —— 换 top32 不牺牲本轮要验证的东西。

**结果（推理侧引擎 + 第 1 步）**：（随运行更新）

---

## 5.8 怎么监控这个运行（用户 2026-10-08 要求："每 5 s 输出日志/内存/显存，并说明含义与应有数值"）

工具：`/workspace-verl/scripts/watch_grpo.py`（本仓库外，属临时调试脚本区）。

```bash
cd /workspace-verl
setsid python3 scripts/watch_grpo.py \
    --log verl/logs/grpo_top64_steps1_<stamp>.log --interval 5 --worker-logs \
    > /tmp/watch.log 2>&1 < /dev/null & echo $! > /tmp/watch.pid
tail -f /tmp/watch.log          # 或 kill $(cat /tmp/watch.pid) 停掉
```

输出样例（每 5 s 一行）：

```
[03:01:00] stage=06 ENGRAM_LOAD | log +0 lines(1354) err=0 | host 1338/2454GB(55%) want 700-1400 |
           npu hbm 40.7/40.8GB want 15-25 aicore 0% | alive drv=1 w=8 v=9 |
           why=each rank slice-reads its 1/8 row range ... :: (vLLMHttpServer ...) NPUCachingAllocator...
```

| 字段 | 含义 | 怎么判读 |
|---|---|---|
| `stage` | 驱动日志 **和** Ray worker 日志里已出现的最靠后阶段标记（`STAGES` 表） | 阶段只应前进。worker 日志必须一起看：训练侧的 `materializing/checkpoint read/Engram tables loaded` 只进 worker `.out`，Ray 转发到驱动日志会**晚几分钟**，只看驱动会误判"卡在 RAY_UP" |
| `log +A lines(N) err=B` | A=本周期新增行数，N=总行数，B=错误模式累计命中 | **A=0 且 aicore=0 且 host 平** = 挂死（02:22 那次就是：engine worker 死了，sender 卡在 `socket.recv()` 等 ack）。`err` 突增看 §5.7.2/§5.7.4 的表对号 |
| `host` | 主机内存 used/total（`MemTotal-MemAvailable`，比 `free -g` 准） | 这是本模型的**约束资源**。见下面"内存账" |
| `npu hbm` | 前 8 卡的平均/最大 HBM 与 AICore | ~3 GB=空闲残留；训练物化期 ~19 GB；engine 权重+KV cache 后 ~40 GB（`gpu_memory_utilization=0.6`→每卡预留 ~38 GB）。**HBM 高但 AICore 0% 持续几分钟 = 集合通信死锁**，不是慢 |
| `alive drv/w/v` | driver / `ray::WorkerDict` / `VLLM::` 存活数（**已排除 `Z` 僵尸**） | `drv=1 w=8 v=0` 表示引擎还没起；`v>0` 但 `drv=0` 表示训练侧死了、只剩引擎。僵尸计数会骗人：02:22 失败后 `[VLLM::Worker_TP] <defunct>` 挂了 9 个，看起来"引擎还活着" |

### 内存账（top32，8 卡；解释 `want` 区间怎么来的）

| 项 | 公式 | 数值 |
|---|---|---|
| 训练侧 Engram 表（pinned host） | 2 表 × 384M 行 × 256 × 2 B ÷ 8 rank × 8 rank（每行只存一份 + 少量 padding） | **366 GB**（代码打印 45.8 GiB/rank） |
| 训练侧可训参数（bf16，CPU offload） | top32 ≈ 52.6 B 参数 × 2 B（top64 是 98.8 B，路由专家减半） | ~**105 GB** |
| AdamW fp32 状态（m+v） | 52.6 B × 4 B × 2 | ~**421 GB**，**第 1 步 optimizer_step 才分配** |
| 梯度（bf16，仅前向/反向期间） | ≈ 参数字节 | ~105 GB（瞬时） |
| 引擎侧 Engram 表 | 同训练侧（引擎自己也从同一 ckpt 读，8 worker × 22.89 GiB × 2） | ~**366 GB** |
| DCP/物化瞬时 | rank0 读 + broadcast 缓冲，未及时归还 | +200~400 GB（阶段性的，之后回落） |

→ 阶段期望（脚本里 `want` 区间的依据）：`06` 完成 ≈ 0.9~1.4 TB；`09 ready to fit` ≈ 1.4~1.6 TB；
`12` 第 1 步（AdamW+梯度到位）≈ 1.9~2.2 TB。**Ray 默认杀线 = 0.95×2454 = 2331 GB**，包装脚本已提到
0.99（2430 GB）；所以 `host` 一过 2.3 TB 就要当回事，过 2.4 TB 是物理极限。

实测锚点（同一模型同一脚本，用于校准）：09-30 那轮 874 GB（训练侧加载完）；10-08 run2：924 GB（训练侧完）
→ 1526 GB（ready to fit）→ 1710 GB（同步/崩溃时）。top64 同期分别是 1.34 TB / 2.42 TB（峰值）。

### 三条最有用的一次性命令

```bash
# 1) 训练侧真实进度（驱动日志没有的那些打印）
grep -rh "\[fsdp_turbo_dsv41\]" /tmp/ray/session_latest/logs/worker-*.out | tail

# 2) 引擎侧进度
sed 's/\x1b\[[0-9;]*m//g' verl/logs/grpo_*.log | grep -E "Engram table offloaded|Available KV cache|Application startup" | tail

# 3) 谁真的死了 / 僵尸（僵尸不算活着）
ps -eo stat,args | grep -E "[V]LLM::|[r]ay::WorkerDict" | sort | awk "{print \$1}" | uniq -c
```
