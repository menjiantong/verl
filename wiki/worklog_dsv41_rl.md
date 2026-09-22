# DeepSeek-V4.1 RL 实施工作记录（2026-09-17 → 2026-09-19）

> 目标：在昇腾 NPU 上用 **verl + FSDPTurbo + vllm/vllm-ascend** 对 **DeepSeek-V4.1（先 8 层、后 4 层减层权重）** 做 GRPO 强化学习。
> 本文件记录：做了什么、遇到什么问题、怎么修的、验证到什么程度、还剩什么。
> 相关文档：[方案](verl_fsdp_turbo_vllm_ascend_rl_plan.md) ｜ [改动盘点](README.md) ｜ [验证步骤](verification.md)
>
> **当前状态（2026-09-19）**：链路端到端跑通——**全尺寸（384 专家 / 114GB）GRPO 完整跑完 64 步**（1h13m，平均 68s/step）；权重同步**数值已验证一致**（全尺寸 pearson 0.90–0.92 / kl 0.38，见 4.4）；同步耗时从 920s 降到 **18s**（全尺寸，见 G13）；**尚未**在真实（非随机）权重上跑过。

---

## 0. 本地事实基准（以本地为准，方案里的版本描述已过时）

| 项 | 本地实况 | 来源 |
|---|---|---|
| 训练（参照） | `FSDPTurbo/examples/deepseek_v41/run.sh` + `config2.yaml`：FSDP=8 / TP=1 / EP=8，`engram_storage_backend: host_offload`，**随机初始化**，拓扑 `apply_modules: model.layers.{*}` 等 | 本地已跑通 |
| 推理（参照） | `scripts/run_vllm2.sh`：`vllm serve /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random`，TP=8 + expert parallel，`--max-model-len 8192`，`--enforce-eager`，`--tokenizer-mode deepseek_v41`，additional-config `{"mc2_comm_alg":"fullmesh_v2","enable_engram":false,...}` | 本地已跑通 |
| 模型权重 | `DeepSeek-V4.1-Flash-8layer-random`：**8 层 / 384 routed experts / 6 activated / 无 engram / compress_ratios=[0,0,2,2,2,2,2,2] / kv·index source layer = 2**，209 GB（19 shard），参数 ~111.4B（其中专家 108.7B） | `config.json` + index 实测 |
| 依赖解析 | `verl`、`vllm`、`vllm_ascend`、`fsdp_turbo` 均解析到 `/workspace-verl/` 下的本地副本（editable 安装），**机器上还有多份同名包副本**，见 §5 风险 | `python3 -c "import ..."` 实测 |

### 与方案（plan）的关键差异 —— 这些是本次实现的主要工作

| # | 方案里的说法 | 本地实况 | 结论 |
|---|---|---|---|
| 1 | 模型结构写死在 FSDPTurbo 内置 `config.json`（4-layer demo） | 推理用 8 层权重，结构与内置 demo 完全不同（层数/专家数/engram/compress_ratios 全不一样） | **训练侧必须支持外部 config**，否则训推结构不一致，RL 无法成立 |
| 2 | `_build_module` 只走"随机初始化 + FSDPTurbo 包装" | 原实现把随机初始化的 `state_dict()` 灌回去，**从不读 checkpoint** | actor/ref 起点与 rollout 不一致；**必须实现从 8 层权重加载** |
| 3 | 减层模型"先接受内置 config" | 108B 参数 / 209GB，**单卡 64GB 根本放不下整模型** | 原"每卡整模型再分片"路径必然 OOM；需要 meta 构造 + 分布式分片加载 |
| 4 | 权重流"原生支持，无需适配" | verl `unfuse_moe_params` 只认 `.mlp.experts.*`（Qwen 风格），训练模型是 `.ffn.experts.gate_up_proj`；vllm-ascend V4.1 `load_weights` 需要 **per-expert `w1/w3/w2`** 名字 | 必须补 unfuse 分支，否则 vLLM 侧 KeyError |

---

## 1. 实现总览

### 1.1 FSDPTurbo 侧（4 个文件，全部向后兼容）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `fsdp_turbo/models/deepseek_v41/adapter.py` | `build_deepseek_v41_model_args(config_path=...)` / `build_deepseek_v41_model(config_path=..., include_vision=..., max_seq_len=..., experts_meta_init=...)`；新增 `_model_args_kwargs_from_hf_config()`（HF `text_config`/`vision_config` → `ModelArgs`，并把推理专用的 DSpark/MTP 强制关掉） | 让训练结构来自 rollout checkpoint 的 `config.json`；不传 `config_path` 时行为与原来完全一致 |
| `fsdp_turbo/models/deepseek_v41/model.py` | `ModelArgs.experts_meta_init` 新字段；`MoE` 用它把 `DeepSeekV41Experts` 建到 meta | 专家占 108.7B/111.4B，只有延迟到加载时按分片物化才放得下 |
| `fsdp_turbo/models/deepseek_v41/experts.py` | `DeepSeekV41Experts(..., device=...)` | 同上，允许指定构造设备 |
| `fsdp_turbo/distributed/expert_parallel/expert_parallel.py` | 新增 `refresh_expert_parallel_metadata(modules, device)` | `expert_ids_per_ep_rank` 是**普通属性不是 buffer**，meta 构造会留下垃圾数据，加载后必须重建（见 §3 陷阱 2） |

### 1.2 verl 侧（5 个文件）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py` | **重写**：外部 config、meta 专家 + 分布式 checkpoint 加载（`set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)`，rank0 读盘、其余 rank 只保留分片）、buffer 落设备、EP 元数据重建、`prepare_model_inputs` 剥离 `position_ids` | 核心适配；详见 §2 |
| `verl/workers/engine/fsdp/utils.py` | `unfuse_moe_params` 增加 `.ffn.experts.{gate_up_proj,down_proj}` → per-expert `w1/w3/w2` 分支 | 训练侧融合 3D 参数 → vLLM 期望的 checkpoint 风格名字 |
| `verl/utils/model.py` | 新增 `get_auto_config_with_vllm_fallback()`；`get_huggingface_actor_config` 改用它 | transformers 5.10.4 不认 `deepseek_v4.1`；vllm-ascend 只把 config 类注册进 **vLLM 的** `_CONFIG_REGISTRY`，AutoConfig 路径会抛错 |
| `verl/workers/config/model.py` | `hf_config` 构造改走上面这个 fallback | 同上（原代码只兜底 `deepseek_v4`，不认 `deepseek_v4.1`） |
| `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | **重写**：对齐本地 8 层权重与 8 卡拓扑、修 hydra 语法 | 见 §2.4 |

### 1.3 新增脚本与数据

- `scripts/check_dsv41_offline.py`：**无需 NPU** 的离线自检 5 项（引擎注册 / HF config / 外部 config / 命名与融合 / build→prepare），一条命令回归本次全部离线结论。
- `scripts/check_dsv41_fsdp_turbo_build.py`：**不依赖 Ray/verl trainer** 的 8 卡冒烟（build → FSDPTurbo 包装 → 分布式加载 → 内存/设备分布报告 → 可选前向）。
- `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`：GRPO 启动脚本。
- 数据：从 `/mnt/share/m00899630/dapo-math-17k.parquet`（179 万条，标准 verl RL 格式：`data_source/prompt/ability/reward_model/extra_info`）切出
  - `/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet`（512 条）
  - `/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet`（64 条）
  - reward 用 verl 内置 `math_dapo`（`data_source=math_dapo` 自动路由），**无需自定义 reward 脚本**。

---

## 2. 关键实现细节（自查用）

### 2.1 模型构建

```python
build_deepseek_v41_model(
    tokenizer=..., engram_meta_init=True, engram_storage_backend="row_sharded",
    use_sparse_flash_attn=False, experts_meta_init=True,
    config_path=<MODEL_PATH>/config.json, max_seq_len=<VERL_DSV41_MAX_SEQ_LEN>,
)
```
- `config_path` 指向权重目录 → 结构与 rollout 完全一致（8 层 / 384 专家 / 无 engram）。
- 只有**专家张量**是 meta；其余参数（2.65B ≈ 5.3GB）和**全部 buffer**（`freqs_cis`、engram hash 表等）真实构造 —— 这是刻意避开整模型 meta 的地雷（§3 陷阱 1）。
- `max_seq_len` 只影响 RoPE 表与 attention scratch 的尺寸，由 `VERL_DSV41_MAX_SEQ_LEN` 控制（默认 = prompt+response）。

### 2.2 分布式权重加载

1. 包装前记录 `{name: (shape, dtype)}`（194 个参数）。
2. `FSDPTurbo(config, module).model` 完成 TP/EP/FSDP 分片（meta 参数分片后仍是 meta，零显存）。
3. rank0 读 checkpoint：
   - 直接映射 `checkpoint 名` → `model.<名>`（178 个）；
   - 专家按 `w1|w3 → gate_up_proj`（沿中间维拼接）、`w2 → down_proj` 融合（16 个，每层 2 个 3D 张量）；
   - 视觉/aligner/image_*/`gate.bias_vl`（274 个）跳过（纯文本 RL 不建 vision）。
4. `set_model_state_dict(..., StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=<offload_policy>))`：torch 逐 tensor 广播，meta 分片按 slice 物化（`assign=True`），FSDP2 的 `load_state_dict` post-hook 会自动 `reset_sharded_param()`。
5. buffer 搬到计算设备（参数走 offload 策略，buffer 跟随 `fsdp2_load_full_state_dict` 的做法）。
6. 重建 `expert_ids_per_ep_rank`；断言没有参数还留在 meta。

### 2.3 训练前向接口适配

| 引擎传入 | V4.1 参考模型 | 处理 |
|---|---|---|
| `input_ids` `[B,S]` | ✓ | 直接用（padded 路径，`use_remove_padding=False`） |
| `attention_mask` int32（1=有效） | ✓（`indexed_sparse_attention` 里 `.bool()` 后 True=attend） | 直接用 |
| `position_ids` | ✗ 不接受 | `prepare_model_inputs` 里 pop 掉（模型按绝对位置索引 RoPE 表） |
| `use_cache=False` | ✓ | 直接用 |
| 输出 | 需要 `output.logits` `[B,S,V]` | 适配器返回 `DeepseekV41CausalLMOutput.logits` ✓ |

### 2.4 启动脚本（8 卡）

- 训练：`actor/ref.strategy=fsdp_turbo_dsv41`，`fully_shard_parallel_size=8`、`tensor_parallel_size=1`、`expert_parallel_size=8`、`expert_fully_shard_parallel_size=1`、`ulysses=1`；`fsdp_plan.apply_modules={model.embed, model.layers.{*}, model.norm, model.head}`；`hook_modules=['model.layers.{*}']`；`cast_forward_inputs=False`、`output_dtype=null`（与本地 `config2.yaml` 一致，避免 HC 的 FP32 系数被压成 BF16）；`offload_policy/param_offload/optimizer_offload=True`。
- 无 recompute（V4.1 适配器明确不支持）。
- rollout：`name=vllm`、TP=8、EP=8、`enforce_eager=True`、`max_model_len=2048`、`max_num_batched_tokens=2048`、`free_cache_engine=True`、`additional_config.{mc2_comm_alg, enable_engram, engram_storage, ascend_compilation_config.*}`。

---

## 3. 遇到的问题与修复（问题 → 根因 → 修复）

### 3.1 已修复

| # | 问题/现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 训练结构与推理结构不一致（4 层 demo vs 8 层权重） | 原实现只读 FSDPTurbo 内置 `config.json` | `adapter.py` 支持 `config_path`：从权重目录 `config.json` 的 `text_config` 构造 `ModelArgs`（含 8 项 `compress_ratios`、384 experts、`engram_layer_ids=[]`、DSpark 强制关闭） |
| 2 | 单卡放不下整模型（209GB vs 64GB HBM） | 原路径先在每卡构造完整模型再分片 | 专家张量 meta 构造（`experts_meta_init`）+ DCP `broadcast_from_rank0` 分片物化：每 rank 只保留自己的分片（EP=8 时每卡约 27GB） |
| 3 | actor/ref 从不加载 checkpoint（随机初始化） | 原 `_build_fsdp_module` 只回灌自身随机 state_dict | 实现 checkpoint 读取 + 融合 + 分布式加载（§2.2） |
| 4 | vLLM 侧会 KeyError | 训练侧融合专家名 `...ffn.experts.gate_up_proj` 不在 vLLM 的映射表内（它只认 per-expert `w1/w3/w2` 或 `gate_proj/up_proj/down_proj`） | `unfuse_moe_params` 增加 `.ffn.experts.*` 分支，导出时拆成 per-expert `w1/w3/w2` |
| 5 | `AutoConfig.from_pretrained` 报 `Transformers does not recognize this architecture` | transformers 5.10.4 无 `deepseek_v4.1`；vllm-ascend 注册到的是 **vLLM 的** `_CONFIG_REGISTRY` | 新增 `get_auto_config_with_vllm_fallback()`：捕获 `KeyError('deepseek_v4.1')` → 确保导入 `vllm_ascend.patch.platform.patch_deepseek_v41_config` → 用 `vllm.transformers_utils.config.get_config` |
| 6 | 参考模型前向报 `Unsupported model inputs: position_ids` | V4.1 参考前向没有 `position_ids` 入参 | 引擎 `prepare_model_inputs` 覆盖：pop `position_ids`；同时**显式拒绝** packed（remove-padding）输入，避免拿错位置训练 |
| 7 | hydra 报 `no viable alternative at input '{"mc2_comm_alg"'` | hydra override 语法无法解析含 `{}` 的 JSON 字符串 | 改为逐键 override：`+...additional_config.mc2_comm_alg=fullmesh_v2` 等（5 条） |
| 8 | hydra 报 `Could not append to config ... output_dtype` | 默认树里 `fsdp_plan.output_dtype` 已存在（`bf16`），不能用 `+` | 改成普通覆盖 `output_dtype=null` |
| 9 | `expert_ids_per_ep_rank` 在 meta 下是垃圾数据（会导致 EP 派发静默错乱） | 它是**普通属性不是 buffer**，`load_state_dict`/物化都不会刷新它 | 新增 `refresh_expert_parallel_metadata()`，加载后重建（verl 引擎与独立冒烟脚本都会调用） |
| 10 | 「整模型 meta + `to_empty` 」会静默毁掉确定性 buffer | `to_empty` 用 `empty_like` 替换，`freqs_cis`/engram 表等 `persistent=False` buffer 不会被 `load_state_dict` 恢复 | 不做整模型 meta：只 meta 专家张量，其余照常真实构造；加载路径也不调用 `to_empty` |
| 13 | **`prepare_deepseek_v41_model_for_fsdp` 抛 `NotImplementedError: Cannot copy out of meta tensor; no data!`**（离线构建即复现，真机首次运行必崩） | 该函数逐模块 `module._apply(lambda t: t.to(device))`，而延迟到 meta 的专家张量不能 `.to()` 到真实设备（原代码只对 `RowShardedEmbedding`/`HostOffloadEmbedding` 特判跳过，不覆盖专家模块） | 改为通用规则：`tensor if tensor.is_meta else tensor.to(device=device)`，meta 张量原样保留（engram 与专家两种情况都覆盖）；已用 CPU 构建 → initialize → prepare 全流程验证：专家仍为 meta、其余参数与 buffer 正常迁移/转 bf16 |
| 14 | 权重同步会把 buffer 也发给 vLLM（潜在 KeyError） | 训练侧有 14 个 `persistent=False` buffer | 实测 `module.state_dict()` = 194 项且**不含任何 buffer** → 同步载荷只有参数，命名全部落在 vllm-ascend V4.1 `load_weights` 的映射表内 |

### 3.2 环境侧阻塞

| # | 问题 | 状态 |
|---|---|---|
| 11 | **NPU 被其他容器的任务占用**：0-3 卡仅剩 ~4GB、4-11 卡 ~31GB、12-15 卡 ~11GB（持有进程 PID 不在本容器命名空间内；两次采样显存完全不变，说明是真实驻留任务） | **未解决**，等用户协调释放。8 卡 RL 每卡峰值需求 ~46GB（专家分片 27GB + 广播临时 18GB + 稠密/激活），当前无法启动真机验证 |
| 12 | 多份同名包副本（`fsdp_turbo` / `vllm_ascend` 各有多个 editable 安装与源码副本） | 已确认当前解析全部指向 `/workspace-verl/` 下的目标副本；建议运行脚本显式 `PYTHONPATH`（§5） |

---

## 4. 已验证内容（离线，无需 NPU）

> 一条命令跑完全部离线检查：`cd /workspace-verl/verl && python3 scripts/check_dsv41_offline.py`（当前 5/5 通过）。
> 下面表格是逐项记录。

| 项 | 命令 | 结果 |
|---|---|---|
| 引擎注册 | `EngineRegistry.get_engine_cls('language_model','fsdp_turbo_dsv41')` | ✓ 返回 `FSDPTurboDSV41EngineWithLMHead` |
| 外部 config 转换 | `build_deepseek_v41_model_args(<权重目录>, max_seq_len=4096)` | ✓ `n_layers=8, n_routed_experts=384, engram=(), dspark=0, compress_ratios=(0,0,2,2,2,2,2,2)` |
| 权重目录可被 verl 读取 | `HFModelConfig(path=<权重目录>)` | ✓ `DeepseekV41Config`、`architectures=['DeepseekV41ForCausalLM']`、tokenizer 就绪 |
| 参数名映射覆盖 | 与 checkpoint index 对比 | ✓ 194 个参数全命中：178 直接映射 + 16 融合；274 个 checkpoint 张量按预期跳过（266 视觉 + 8 视觉路由 bias） |
| checkpoint 读取与融合 | `read_dsv41_checkpoint_state_dict()` 子集实测 | ✓ `gate_up_proj[E] = [w1 ; w3]`、`down_proj[E] = w2` **逐字节相等**（含第 0 与第 383 个专家）、`embed` 相等、形状/类型正确 |
| FSDPTurbo 单测 | `pytest tests/unit_tests/models/test_deepseek_v41.py` | ✓ 18 passed（`test_engram_parallel.py` 有 3 个失败，已用 `git stash` 确认是**改动前既有失败**，与本次无关） |
| hydra 配置组装 | 启动脚本 + `--cfg job` | ✓ exit 0；composed 配置里 strategy/拓扑/apply_modules/ep_plan/enforce_eager/TP·EP/additional_config 均正确 |
| DCP 分布式加载可行性 | 读 torch 源码（`state_dict.py` / `_state_dict_utils.py` / `_fsdp_param.py`） | ✓ 官方支持 "rank0 CPU full_state + meta 模型 + broadcast_from_rank0 → 按分片物化"；FSDP2 注册了 `load_state_dict` post-hook 自动 `reset_sharded_param()` |
| 构建 → initialize → prepare 全链路（CPU） | `build(experts_meta_init=True) → initialize → prepare(cpu)` | ✓ 专家保持 meta、其余参数转 bf16 并迁移、14 个 buffer 保持真实值（**由此发现并修复 §3.1-13**） |
| 权重同步载荷 | `module.state_dict()` 检查 | ✓ 194 项、无 buffer、命名均为 checkpoint 风格（`model.layers.N.ffn.experts.E.w1.weight` 等） |

**已尝试但不可行的验证方式**：想用 "2 rank CPU + gloo" 代理验证 DCP 的 meta 分片物化机制，实测不可行 —— FSDPTurbo 的并行状态与 mesh 绑定**加速器类型**（`init_parallel_state` 按 `torch_npu` 存在与否建 "npu" mesh），代理测试仍会向 NPU 分配（实测在 engram 表分片时要求单卡 91.55GB 而 OOM）。结论：这套机制只能在真机 NPU 上验证。

### 4.1 真机 8 卡冒烟（2026-09-17，NPU 空闲后执行，**全部通过**）

```
cd /workspace-verl/verl && torchrun --nproc_per_node=8 scripts/check_dsv41_fsdp_turbo_build.py --forward
```
| 阶段 | 实测结果 |
|---|---|
| 模型构建（专家 meta） | `model built in 43.1s (194 parameters)` |
| FSDPTurbo 包装（FSDP=8/EP=8） | `wrapped in 0.5s`（meta 参数分片零显存，符合预期） |
| checkpoint 物化（rank0 读 NFS + 广播） | **`parameters materialized in 275.3s`**（209GB / NFS 实测吞吐受共享存储影响） |
| 完整性 | `no meta parameters left`（无参数滞留 meta） |
| 每 rank 参数量 | **25.93 GiB**（其中 `layers` 25.62 GiB = 专家分片，与设计值 108.72B/8 ≈ 27GB 一致） |
| 设备分布 | 参数全在 **cpu**（`offload_policy` 生效），加速器 allocated **0.01 GiB** —— 为 colocated RL 留出显存 |
| 数值正确性 | `model.embed.weight absmax=1.0000`，与 checkpoint 文件实测 **absmax=1.0 / std=0.577** 一致 |
| **前向**（batch=1, seq=8） | **`forward ok: logits (1, 8, 129280) dtype=torch.float32 absmax=117.25`** —— EP 派发 + V4.1 注意力（eager）+ MoE 全链路打通 |

**遇到并修复的问题**：冒烟脚本最初显式传了 `position_ids=None` 给模型 → `ValueError: Unsupported model inputs: position_ids.`（真实引擎路径会在 `prepare_model_inputs` 剥离该入参，是脚本用法问题，非引擎缺陷）；同时把 rank 内存报告改为全 rank 打印，便于定位。

**尚待真机验证**：
1. `DEVICE=npu bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`（GRPO 端到端）
2. 权重同步一致性：rollout logprob 与 actor 重算 logprob 对齐

### 4.2 GRPO 端到端启动过程中遇到的问题（2026-09-17）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G1 | `ValueError: batch_size should be a positive integer value, but got batch_size=0` | 脚本没设 `data.val_batch_size`，验证 dataloader 用 0 建 `BatchSampler` | 脚本加 `data.val_batch_size=${VAL_BSZ}`（默认 8） |
| G2 | 数据集被全部过滤：`train dataset size: 0, val dataset size: 0`，日志首行 `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set` | **发行版 V4.1 权重的 tokenizer 不带 chat template**（vLLM 侧靠 vllm-ascend 自己的 encoder 渲染）；verl 的 RL 数据集在建集时（`filter_overlong_prompts`）必须调用 `apply_chat_template`，异常被吞掉后把每条样本判为超长 → 全部过滤 | 新增 `scripts/make_dsv41_rl_tokenizer.py`：复制 checkpoint 的 tokenizer 文件并补上单轮 chat template（`<bos><｜User｜>{content}<｜Assistant｜><think>`，与 vllm-ascend `encoding.py` 的参考编码一致），脚本用 `actor_rollout_ref.model.tokenizer_path` 指向它；实测 `train dataset size: 512, val dataset size: 64` |

> 说明：模板只覆盖"单 user 轮 + 生成提示"；多轮/工具/推理强度等仍以 vllm-ascend 的 encoder 为准（需要全保真时可直接复用该 encoder）。actor 与 rollout 消费的是同一份 token id，流水线正确性不受影响。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G3 | 引擎构建完成后，rollout 构造抛 `RuntimeError: Error checking IPC support: CANN toolkit info file does not exist: /usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info` | CANN 9.x 合并布局把安装信息文件改名为 `ascend_all_cann_install.info` / `ascend_ops_install.info`；**此前的同类修复在另一份 verl 副本里，本树（`dev-dsv41`）没有** | 在本树 `verl/utils/device.py::get_npu_versions()` 补回三种候选文件名的回退探测；验证 `get_npu_versions()` → `(A3, 26.0.rc1, 9.1.0)`、`is_support_ipc()` → `True` |
| G4 | 两个引擎的加载进度在 Ray worker 日志里完全看不到（`logger.info` 被 WARNING 级 root logger 吞掉），209GB 加载期间无任何反馈 | Ray worker 进程的 root logger 级别 | 引擎内改用 `_log_rank0()` 显式 `print`，现在能看到 `materializing 194 parameters… → checkpoint read: 194 tensors (274 unused) → parameters materialized in 535.5s` |

**实测启动耗时**（2026-09-17，共享 NFS/主机有其它任务）：模型构建 ~43s + 包装 ~0.5s + **每个引擎加载 ~9 分钟**（actor 535.5s；ref 同量级），即"双引擎 + vLLM"冷启动约 20 分钟。加载耗时主要来自 rank0 读 209GB（NFS，93% 容量占用）+ 逐 tensor 广播/分片/回 CPU。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G5 | 两个引擎加载完成、构造 rollout 时：`TypeError: ModelArchConfigConvertorBase.__init__() takes 3 positional arguments but 4 were given`（vllm-ascend `patch_deepseek_v4_vision.py:52` → `vllm/config/model.py:855 get_model_arch_config`） | vllm-ascend 的补丁按 vllm 版本分支：`vllm_version_is("0.27.1")` 走 2 参数基类，否则走 3 参数；`vllm_version_is` **优先读环境变量 `VLLM_VERSION`**，未设时用 `vllm.__version__` —— 本机是 `0.28.1rc1.dev570`（dev 版本连版本号都解析不了），于是走了 3 参数分支而本机 vllm 的基类只接受 2 个参数。参考脚本 `scripts/run_vllm2.sh` 正是靠 `export VLLM_VERSION=0.27.1` 规避 | GRPO 脚本 `DEVICE=npu` 分支加 `export VLLM_VERSION=${VLLM_VERSION:-0.27.1}`；**离线复现验证**：设该变量后 `EngineArgs(...).create_engine_config()` 成功，arch 归一化为 `DeepseekV41ForConditionalGeneration`（VL wrapper），不设则复现 TypeError |

> 该问题和方案里写的"版本风险"完全对应：**vllm-ascend 的补丁是按 0.27.1 的 API 写的**，本机 vllm 是 0.28.1rc1.dev，必须用 `VLLM_VERSION` 显式对齐。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G6 | rollout 的 vLLM engine core 崩溃：CANN 报 `Failed to execute operator MoeDistributeDispatchV2 … HCCL_BUFFSIZE_EP is too SMALL, maxBs = 32, h = 5120, epWorldSize = 8, localMoeExpertNum = 48` | EP 的 mc2 all-to-all（dispatch/combine）按 `HCCL_BUFFSIZE` 划分通信窗口，默认值装不下 `max_num_seqs × hidden × ep` 的需求；`scripts/run_vllm2.sh` 里正是设了 `HCCL_BUFFSIZE=1024` | 脚本补 `export HCCL_BUFFSIZE=1024`（并补齐 `ASCEND_CONNECT/TRANSFER_TIMEOUT`、`VLLM_USE_V2_MODEL_RUNNER=0`、`VLLM_ENGINE_READY_TIMEOUT_S` 等）＋ `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`。**独立验证**：用与本 RL 相同的推理参数直接 `vllm serve`，设 `HCCL_BUFFSIZE=1024` 后 `Application startup complete`，算子错误消失（每卡权重 28.62 GiB、KV cache 4.69 GiB） |
| G7 | 加上 G6 的环境变量后，vLLM worker 启动即 `AssertionError: Expandable segments are not compatible with memory pool` | 我照抄了 `run_vllm2.sh` 的 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，但**colocate 模式下 vLLM 的 sleep/内存池分配器与 expandable segments 互斥**（单机纯推理没有 sleep 模式所以不冲突）；verl 自己在权重同步前后用 `set_expandable_segments` 管理该设置 | 脚本中**不再设置** `PYTORCH_NPU_ALLOC_CONF`（其余环境变量保留） |

> 排查手法记录：vLLM engine core 的崩溃在 verl 日志里只显示 `Engine core initialization failed`，真正根因在 `(EngineCore pid=…)` / `(Worker_TPx_EPx pid=…)` 前缀的行里；用 `grep -nE "Failed to execute operator\|Error\|Assertion"` 能直接定位。另外**独立 `vllm serve` 探测**（约 5 分钟）比跑一整轮 GRPO（约 20 分钟）验证环境类修复快得多，本轮两个问题都是这样先验证再改脚本的。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G8 | vLLM 引擎起来后、权重同步的 `wake_up` 阶段报 `FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set weight_nz_mode=0 via --additional-config, or enable additional_config.rl_config (enabled: true)` | vllm-ascend 默认 `weight_nz_mode=1`（内部权重用 FRACTAL_NZ 布局）；RL 会从外部反复灌权重，NZ 布局会带来精度问题，所以框架在 RL 路径上主动报错要求关闭 | 脚本 `additional_config` 增加 `rl_config.enabled=true`——这是 vllm-ascend 官方的 RL 开关，它会自动把 `weight_nz_mode` 置 0、调用 `_disable_expandable_segments()`（顺带解决 G7 那类冲突），并可选开启 batch-invariant/训练一致性 |

> 至此 rollout 侧的启动路径（vLLM 引擎 → sleep → wake_up → 权重同步入口）已经打通到 `update_weights` 之前。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G9 | 首次权重同步（`update_weights` → `model.load_weights(param_updates)`）时 8 卡同时崩：`Invalid_Argument_Tensor_Input_Shape(EZ1007): ... The size of tensor self [2304, 4608] must match the size of tensor src [2304, 5120]`，栈底为 `routed_experts.py::_load_w13` 的 `expert_data.copy_(loaded_weight)` | 引擎初始化时 `process_weights_after_loading` 把 Ascend 侧的参数换成了**推理布局**：Ascend MoE 把融合专家权重转置（`w13_weight`: `[E, 2*inter, hidden]` → `[E, hidden, 2*inter]`，`w2_weight` 同理）。而 `RoutedExperts.weight_loader` 与训练侧张量都按 checkpoint 布局工作（单专家 `w1`/`w3` = `[inter, hidden]`）。在转置后的参数上，loader 对 `shard_dim=0`（此时是 `hidden=5120`）取 `shard_size=2560`，再被 `_narrow_expert_data_for_padding` 修剪到入参 dim0（2304），得到 `[2304, 4608]`（4608 = `2*inter`）——与入参 `[2304, 5120]` 永久不匹配（`4608` 正好等于 `2*moe_intermediate_size`，是定位该问题的指纹） | 同步前把融合专家权重**换回 checkpoint 布局**：新增 `verl/utils/vllm/npu_expert_layout.py::restore_expert_checkpoint_layout`（按 `w13.shape[1] == w2.shape[2] and w13.shape[2] == 2*w2.shape[1]` 识别推理布局，`transpose(1, 2)` 回退，幂等），接入 `verl/workers/rollout/vllm_rollout/utils.py` step 1（仅 NPU、非 LoRA）；step 3 原有的 `process_weights_after_loading` 会把转置再打回去，其它参数一律交给各层自己的 loader |

> G9 补充：回退必须用**视图**（`transpose(1, 2)`，不带 `.contiguous()`）。第一版写成 `transpose(1,2).contiguous()` 直接 OOM——`Tried to allocate 2.11 GiB ... 1.83 GiB free`：单个 `w13` 是 2.11 GiB，而此时 vLLM 的池子已经占满（权重 28.6GB + KV 4.7GB）。视图零分配，且 step 3 的 `transpose(1,2).contiguous()` 恰好把同一块存储转回去（原布局本就是 contiguous，`contiguous()` 是 no-op），往返都不产生拷贝。

> **弯路记录（避免重蹈）**：先按 vLLM 上游 CUDA IPC 引擎的做法改用官方的**分层重载**（同步前 `initialize_layerwise_reload`，收完后 `finalize_layerwise_reload` 代替 `process_weights_after_loading`；vllm-ascend 的 HCCL 权重传输引擎正是这么做的，而 NPU IPC 引擎的 start/finish 是 no-op）。专家权重确实装进去了（G9 的 shape 错误消失），但立刻卡在下一个参数：分层重载把每个参数恢复成**模型构造时**的形状（`record_metadata_for_reloading` 记录），而 Ascend 的线性层是在 **weight_loader 里就地换布局**的——例如 `vllm_ascend/ops/linear.py` 的 `wo_a`：checkpoint `[8192, 4096]` → 每 rank `[1024, 4096]` → `.view(n_local_groups, o_lora_rank, -1).transpose(2, 1)` → `[1, 4096, 1024]`，重放后与 kernel 张量（`[1024, 4096]`）维度不一致，报 `self [1024, 4096] must match src [1, 4096, 1024]`。**结论**：vllm-ascend 的线性层/注意力本来就按「推理布局下重载」设计（`wo_a` 的 else 分支注释即 "In RL update flows, wo_a can be loaded again after being transformed…"，`mla_v1` 的 `if not hasattr(self, "W_UV")` 也保证重复后处理是 copy 而非重建），真正装不下的只有融合 MoE 的转置，因此最终只回退 MoE。

> 附带修正（若日后仍要启用分层重载）：该机制会把一个层的张量缓存到该层收全为止（可能跨 bucket），而接收端 bucket buffer 与 direct-send 的 IPC 存储在回调返回后立即被复用（发送端等 ack 即覆写同一块 buffer），必须对入参做 `clone()`——与 LoRA 分支（#6454）同样的处理。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G10 | 权重同步**发送侧**（训练进程）OOM：`torch.OutOfMemoryError: Tried to allocate 8.44 GiB (NPU 0; 61.28 GiB total capacity; 18.46 GiB already allocated; 3.55 GiB free; 18.69 GiB reserved by PyTorch)`，栈顶为 `get_per_tensor_param` 的 `param.to(device).full_tensor()` | 导出走 `full_tensor()`：每个专家 DTensor 都要 all-gather 成完整的融合张量（`[384, 4608, 5120]` = **18.1 GiB/层**）再 `unfuse_moe_params` 拆成逐专家键；而 colocated 模式下**同一张卡上 vLLM 还占着 ~39GB**（权重 28.6 + KV 4.7 + 池开销），18GB 的瞬时要不下（8 层时首次出现；4 层权重减半后仍建议保留该改法） | 导出改为**按 EP 分片分块**流式进行：`_export_param` 钩子（`transformer_impl.py`，默认行为不变）+ DSV41 引擎覆写——专家参数按 `param.narrow(0, first, block).full_tensor()` 一块一块取（block = 本 rank 的本地专家数，8 卡下 48 个 ≈ 2.26 GiB），并带上**全局专家编号**（`split_fused_expert_tensor(name, tensor, first_expert_id=...)`，`unfuse_moe_params` 同源复用），峰值降到 1/8；同时把 `unfuse_moe_params` 里 mlp/ffn 两套拆分逻辑抽成同一个 helper |
| G11 | 8 层配置下权重同步**已通过**，但紧接着 `actor_rollout_ref_update_actor` OOM：`Tried to allocate 6.33 GiB ... 49.95 GiB already allocated`（各卡 50–51.5 GiB） | 8 层时每卡要同时装：vLLM 权重 ~28.6GB + 训练侧优化器状态/激活/梯度（专家分片 27GB + AdamW 一阶二阶动量）+ 临时缓冲；61GB 卡放不下。8 层 209GB 权重下这是**容量问题**，不是代码缺陷 | 用小层数权重把链路先跑通：新增 4 层权重（`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-random`，106GB，结构其余部分与 8 层一致：384 专家 / hidden 5120 / inter 2304 / `compress_ratios=[0,0,2,2]` / kv·index source layer=2），`MODEL_PATH=` 一行即可切换；层数、专家数全部由 `config.json` 驱动，训练/推理两侧自动跟随 |

> 权重同步**首次打通**（2026-09-18，8 层权重）：`update_weights` 全流程走完并进入训练步（`actor_rollout_ref_update_actor`），失败点从"同步加载 shape 错误"推进到"训练步显存不足" —— G9/G10 两个问题解决后，同步本身不再报错。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| G12 | 4 层权重下首个完整 step 跑通，但训练指标整片异常：`actor/entropy 0.102`、`training/rollout_probs_diff_{max,mean,std}: nan`、`rollout_corr/training_ppl: inf`（`training_log_ppl 325.5`）、`rollout_corr/{kl,k3_kl,rollout_log_ppl,...}: nan`、`actor/grad_norm: nan`（`WARN: grad_norm is not finite`）；`actor/kl_loss` 与 `actor/ppo_kl` 为 0.0（第 0 步 actor=ref 本就应为 0，符合预期） | **不是链路缺陷，是随机权重生成器的初始化尺度**。`mjt_workplace/deepseek41/inference/make_random_ckpt_dsv41_4layers.py` 对**每个**张量填 `torch.randn`（N(0,1)），连 RMSNorm 增益（应为 1.0）和 router 的 `gate.bias`（应为 0）都是 N(0,1)。于是线性层输出 RMS ≈ `sqrt(fan_in)·RMS(x)` ≈ `sqrt(5120)` ≈ 71，末层 logits 是 N(0,71) 量级：softmax 近似 one-hot（top-1 与次高间隔 ~165 nats），对「任意非 argmax token」的 NLL 正好 ≈ max logit ≈ 4.7σ ≈ **325** —— 与实测 `training_log_ppl 325.5` 和 `entropy 0.1` 完全吻合，`training_ppl = exp(325) = inf`。该量级下引擎侧 logprob 路径溢出（bf16 `exp(336)` → inf/nan）→ rollout 侧 logprob 全 nan、采样退化（不取 argmax 而返回任意 token）→ 训练侧给这些 token 的 logprob ≈ -325 → `rollout_probs_diff`（`|exp(actor)-exp(rollout)|`）与 pearson 全 nan，`grad_norm` 亦被污染 | 见下方 4.4：给生成器加 `--init scaled`（矩阵按 `1/sqrt(fan_in)`，norm 增益=1，bias/sink/HC 位移=0，`--head-gain` 控制 logits 尺度），生成 `DeepSeek-V4.1-Flash-4layer-scaled` 重跑做数值验证。**注意**：用 N(0,1) 权重的跑不能用来判断同步是否数值正确，所有 logprob/KL/PPL 指标在该权重下都无意义 |
| G13 | 同步**耗时**：`timing_s/update_weights` = 920s（step1）/ 934s（step2），占单步 95%（gen 22–55s、update_actor 25s）。日志时间线：01:37:35 `sleep`（释放 38.18 GiB）→ 01:38:10 `wake_up(['weights'])` → 之后引擎侧安静 15 分钟（只有 `No available shared memory broadcast block found in 60 seconds`，说明 worker 都在忙）→ 01:53:27 `wake_up(['kv_cache'])` 结束 | **已定位：瓶颈在发送侧的导出，不在传输也不在加载**。profiling 数据（scaled32 权重 14GB，`VERL_SYNC_PROFILE=1`）：每 rank **478 个张量 / 12.14 GiB / 21 个 bucket**，其中 `export`（生成器内耗时：`full_tensor()` all-gather + 逐专家拆分）**80–83s**，而 `flush`（等接收端加载完的 ack，含引擎侧 load）只有 **1.9–2.3s**。即传输+加载 ≈ 2s，导出 ≈ 80s。导出慢的原因：`offload_policy=True` 时参数在 CPU 上，`param.narrow(...).to(device).full_tensor()` 的 all-gather 走的是 CPU（gloo/host）路径；而且每个 expert 参数被切成 8 个 block 各做一次 all-gather，**每 rank 实际搬运 ≈ 126 GiB**（= 8 卡全量冗余），只为送达本 rank 引擎真正需要的 ≈ 9 GiB | 新增**只导出本 rank 专家**的路径（`VERL_DSV41_LOCAL_EXPERT_EXPORT=1`，`fsdp_turbo_dsv41_impl.py::_export_local_experts`）：colocate 下 trainer rank i 的 ZMQ 对端就是 engine rank i，两侧专家维都按 rank 连续切分，因此直接从本地分片按全局专家号发出即可，跳过 all-gather 与 7/8 的冗余传输。已在 scaled32（32 专家，每 rank 4 个）上验证：导出 `80s → 0.048s`、每 rank 传输 `12.14 GiB → 4.76 GiB`、bucket `21 → 6`、`update_weights 86.7s → 6.2–6.5s`、`step 162s → 25–74s`，而三项一致性指标**不降反略升**（`diff_mean` 0.0122→0.0103/0.0116/0.0118、`pearson` 0.967→0.979/0.978/0.968、`kl` 0.123→0.111/0.107/0.101）——即专家配对与全局编号都正确。失败模式是「响亮」的：配对错了这三项会跳到 O(1)/~0.5。当前默认仍走全量导出（env 开关 opt-in），待 384 专家全尺寸复验后再决定是否设为默认 |

### 4.3 4 层权重端到端（2026-09-18 → 2026-09-19，**已跑通一个完整 step**）

命令：`MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-random bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`
（4 层 / 384 专家 / 106GB；冒烟用快速配置：`train_batch_size=8 / n=2 / max_response_length=128`，日志名带权重目录名）

| 阶段 | 结果 |
|---|---|
| 模型构建 + 加载（actor/ref） | ✓（层数与专家数全部由 `config.json` 驱动，未改代码） |
| vLLM 引擎 + `all initialize finished, ready to fit` | ✓ |
| **首次权重同步（G9+G10 修复）** | ✓ 全流程走完，无 shape 错误、无 OOM；**但耗时 920s**（G13） |
| rollout 生成（`AgentLoopWorkerTQ` + TransferQueue） | ✓ 8 prompt × n=2，每步 22–55s（`enforce_eager` 逐 token 解码） |
| `old_log_prob` / `ref` / `adv` / `update_actor` | ✓ 3.3s / 2.7s / 0.2s / 25s；峰值分配 40.4GB/卡、保留 51.7GB（G11 的 8 层 OOM 在 4 层下消失） |
| **完整 step（step:1 与 step:2）** | ✓ **端到端首次跑通**：gen → old_log_prob → ref → adv → update_actor → update_weights → 回到 rollout，`Training Progress: 2/64` |
| 指标数值 | ✗ 整片 nan / inf（G12：随机权重是 N(0,1) 所致，非链路缺陷） |
| 速度 | ✗ 每步 986–1025s，其中 920–934s 是权重同步（G13） |

> 关键结论：**链路本身已通**——权重同步（G9 的布局回退 + G10 的分块导出）能把训练侧张量装进引擎，训练步能在 4 层权重下同时容纳 vLLM 与训练侧状态，rollout/训练/同步三步循环连续跑了两步。剩余的是「数值可信」与「速度」两件事，分别见 4.4 与 G13。

### 4.4 数值验证（进行中）：scaled 随机权重

N(0,1) 的随机权重下「同步是否数值正确」无法判断（所有 logprob 指标都是 nan，见 G12），因此需要一份**数值上像样**的随机权重：

```bash
cd /workspace-verl/mjt_workplace/deepseek41/inference
python3 make_random_ckpt_dsv41_4layers.py --config config.dsv41_4layer.bf16.json \
    --init scaled --head-gain 4 \
    --out /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled
```

`--init scaled`（新加，默认 `randn` 保持原行为不变）：矩阵按 `N(0, 1/sqrt(fan_in))`（每层输出 RMS ≈ 输入 RMS），RMSNorm 增益 = 1，bias / `attn_sink` / HC 的 `base|scale` = 0，`head.weight` 额外乘 `--head-gain`。`--head-gain 4` 让 logits ≈ N(0,4)（top-1 概率 ~0.4，接近真实模型），这样：

- 采样不再退化、logprob 路径不溢出（bf16 下 exp(19) 远未溢出）；
- 同步若正确，rollout 侧与训练侧对同一批 token 的 logprob 只差 bf16 数值噪声（量级 1e-2）；
- 同步若错位（例如专家错绑、布局没回退），采样 token 在训练侧会得到 O(1)～O(10) 的 logprob 落差。

**判读标准**（跑一步看这几项）：

| 指标 | 同步正确 | 同步有问题 |
|---|---|---|
| `training/rollout_probs_diff_mean` | ~0（<0.05） | O(1) 以上 |
| `training/rollout_actor_probs_pearson_corr` | ~1（>0.99） | 明显 < 1 |
| `rollout_corr/kl`、`k3_kl` | ~0 | 明显正 |
| `training_log_ppl` vs `rollout_log_ppl` | 接近 | 差距大 |
| `actor/grad_norm` | 有限 | nan |

> ⚠️ **阈值要按"权重/专家数"重标定（2026-09-22 修正）**：上表的"~0 / >0.99"是在 G12 那批权重上得出的，**不能当通用阈值**。后续一致性专项（见 [work_log/worklog_dsv41_module_input_probe.md](work_log/worklog_dsv41_module_input_probe.md)）证明：这两个指标主要是**两套栈的固有内核数值差（每模块 ~0.5% ≈ bf16 的 1 ULP）**经 MoE 路由离散翻转逐层放大的结果，**随专家数与 head-gain 显著变化**——32 专家 scaled32 是 `0.010/0.979`，384 专家 `scaled` 是 `kl 0.38 / pearson 0.79`，二者都是"同步正确"的正常表现；真实（fp8、head 正常尺度）权重预计回到 kl ≈ 0.01 量级。
> **判据改为看量级断层**：真错位（专家错绑/布局没回退）会是 **O(1) 的 logprob 落差 / pearson ~0.5**，与"专家数放大"有数量级差异；同时用**直接量**做健康检查——权重指纹（`check_engine_params.py`）、模块 rel_err / Δlogp σ（`probe_module_inputs.py` + `summarize_probe.py`）。
> 另：默认配置下 `old_log_probs` 由训练侧重算（`trainer_base.py::_compute_old_log_prob`；本脚本未设 `algorithm.rollout_correction`），**这个差进入的是 off-policy 程度与监控指标，不进入 PG ratio**。

**结果（2026-09-19，scaled32 权重，第一步）**：

| 指标 | N(0,1) 权重（G12） | scaled 权重 |
|---|---|---|
| `actor/entropy` | 0.102 | **5.02** |
| `training/rollout_probs_diff_mean` | nan | **0.0122** |
| `training/rollout_probs_diff_max` | nan | 0.428 |
| `training/rollout_actor_probs_pearson_corr` | nan | **0.967** |
| `rollout_corr/training_log_ppl` / `rollout_log_ppl` | inf / nan | **5.21 / 5.09** |
| `rollout_corr/kl` / `k3_kl` | nan | **0.123 / 0.246** |
| `rollout_corr/ppl_ratio` | nan | 1.13 |
| `actor/grad_norm` | nan | **0.0（有限）** |
| `timing_s/update_weights` | 920s | **86.7s** |
| `timing_s/step` | 1025s | **162s** |

> 结论：**权重同步数值正确**——引擎侧与训练侧对同一批 rollout token 的 logprob 只差 bf16 数值噪声（mean |Δp| = 0.012、KL = 0.12 nats、pearson 0.967）。若专家错位/布局没回退（G9 那类问题），这三项会给出 O(1) 的落差。`grad_norm = 0.0` 是因为当前是随机权重、所有 reward 都 -1（GRPO 组内优势全 0），没有梯度信号，属正常。
>
> 顺带确认：`/workspace-verl/verl/scripts/check_dsv41_fsdp_turbo_build.py --forward` 在 scaled32 上给出 `logits (1,8,129280) absmax=18.56`，与 `N(0, head_gain=4)` 的理论值（4×4.84 ≈ 19.4）一致，说明 init 尺度符合预期。
>
> 若日后判读为「同步有问题」，下一步做**权重级校验**：在引擎侧加只读 RPC（对指名参数算 checksum/absmax）与训练侧同名参数对比，直接定位是传输、布局回退还是 loader 的问题。

**同步耗时优化（G13）在 scaled32 上的对照**（同一权重、同一配置，只切 `VERL_DSV41_LOCAL_EXPERT_EXPORT`）：

| 指标 | 全量导出（默认） | 本地专家导出 |
|---|---|---|
| 发送侧 `export` | 80–83s | **0.048s** |
| 发送侧 `flush`（传输+引擎加载） | 1.9–2.3s | 1.6–2.2s |
| 每 rank 传输量 / bucket 数 | 12.14 GiB / 21 | **4.76 GiB / 6** |
| `timing_s/update_weights` | 86.7s | **6.2–6.5s** |
| `timing_s/step` | 162s | **24.8–74.1s** |
| `rollout_probs_diff_mean` / `pearson` | 0.0122 / 0.967 | 0.0103–0.0118 / 0.968–0.979 |

> 结论：慢的从来不是「传输」也不是「引擎加载」，而是**训练侧在 CPU-offload 参数上做 all-gather**。把不需要的那 7/8 直接从导出里去掉后，同步只剩传输+加载的秒级开销，且数值一致性指标不变（略好）。

**全尺寸对照（384 专家 / 114GB 权重）**：

| 侧 | 指标 | 全量导出（04:19） | 只导出本 rank 专家（04:44） |
|---|---|---|---|
| 发送侧（训练） | profile | `4702 tensors / 104.96 GiB / 213 buckets`，**export 928.9s**，flush 13.3s | `670 tensors / 16.37 GiB / 30 buckets`，**export 0.1s**，flush 11.5s |
| 接收侧（引擎） | profile | `215 buckets / 4702 tensors`，views 0.1s，**load 11.2–11.5s** | `32 buckets / 670 tensors`，views 0.0s，**load 10.0–10.2s**，sync 0.5s |
| step1/2/3 | — | — | revert 0.0s + receive/load 12.4–14.1s + post-load 0.0s |

> 生产尺度上「导出 : 传输+加载 ≈ 930s : 13s」，与 scaled32 的小尺度结论完全一致——**瓶颈始终是训练侧那个 8× 冗余的 all-gather**，与传输/加载无关。改用本地专家导出后 `timing_s/update_weights` 从 920s 降到 **18.0 / 18.5 / 18.9s**（约 50×）。

**全尺寸 64 步 GRPO（2026-09-19 04:34 → 05:59，1h13m，平均 68.1s/step，`LOCAL_EXPERT_EXPORT=1`）**：

| 指标 | step 1 | step 2 | step 3 | 判读 |
|---|---|---|---|---|
| `training/rollout_probs_diff_mean` | 0.0250 | 0.0254 | 0.0235 | 同步正确（专家错位会 O(1)） |
| `training/rollout_actor_probs_pearson_corr` | 0.917 | 0.902 | 0.900 | 同步正确（错位会掉到 ~0.5） |
| `rollout_corr/kl` | 0.382 | 0.372 | 0.379 | 仅为 bf16/内核差异 |
| `actor/entropy` | 4.94 | 4.95 | 4.98 | 正常（scaled 权重） |
| `actor/grad_norm` | 0.0 | 0.0 | 0.0 | 随机权重下 reward 全 -1 → 优势全 0 → 无梯度信号，正常 |
| `timing_s/update_weights` | 18.0s | 18.5s | 18.9s | 对比修复前 920s |
| `timing_s/step` | 125.7s | 65.7s | 70.7s | 对比修复前 986–1025s |

> `Training Progress: 100%|██████████| 64/64 [1:13:18, 68.07s/it]` —— **全尺寸（384 专家 / 114GB）下 GRPO 完整跑完 64 步**，权重同步数值一致性在大规模上同样成立（48 专家/rank 的编号与槽位对得上）。收尾时报 `RuntimeError: DataLoader worker (pid …) is killed by signal: Killed.`（dataloader 子进程被 OOM 杀，属主机内存压力，不影响 64 步结果）。
>
> 注意开关有两个写法，二者等价：脚本层的 `LOCAL_EXPERT_EXPORT=1` 或引擎层的 `VERL_DSV41_LOCAL_EXPERT_EXPORT=1`（脚本已做 fallback；脚本层变量优先，只设其一即可）。


## 4.5 训推一致性专项（2026-09-20，详见 [work_log/worklog_dsv41_train_infer_consistency.md](work_log/worklog_dsv41_train_infer_consistency.md)）

对「`rollout_actor_probs_pearson_corr` 只有 0.95–0.98、`kl 0.10–0.19`」做了逐阶段数值分桶（新夹具 + 双侧 dump + 引擎 worker extension），结论：

- **权重无问题**：引擎持有的参数与 checkpoint 逐张量一致（672 项抽查 664 项精确匹配，其余是未覆盖的 vision 塔）。
- **差异从第 0 层注意力核内部开始**（embedding / attn_norm / q 投影完全一致，注意力输出差 0.34%，逐 token 均匀），随后被未训练模型逐层放大约 40 倍，最终隐状态相对误差 14% → 逐 token logprob 噪声 σ≈0.32–0.53 nats、均值≈0、argmax 一致率约 79%。该 σ 用 `KL≈0.5σ²` 正好解释线上 `kl 0.10–0.19`，`probs_diff_max 0.46–0.77` 对应约 21% 的 argmax 翻转位置。
- **引擎自身 decode↔prefill 一致性很好**（0.05–0.14 nats），开启 `rl_config.enable_batch_invariant=true` 后 ≤0.006 nats（已写入启动脚本）。
- **两条"对齐内核"路线均不可行**：训练侧 eager 注意力改 fp32 无端到端收益；训练侧启用融合 SparseFlashMla 被 A2/A3 稠密布局约束堵死（`cmp_mask_mode`/`cmp_ratio=4`）。
- 另发现训练侧 padded/rmpad 路径的 **label roll 会跨样本回绕**（每序列 1 个响应 token 的 logprob 错，±0.03–0.10 nats/序列），建议单独修复。
- 新增工具链 `scripts/dsv41_consistency/`（夹具/dump/参数校验/四种对比）、`VERL_DSV41_DUMP_BATCH` 落盘开关。

## 4.6 真实权重复跑（2026-09-22）：训推差距的主因是一个**可修的 dtype 不对称**

用真实 DeepSeek-V4.1-Flash 切出 4 层（384 专家）重跑 §4.5 的整套 harness
（**完整工作记录：[work_log/worklog_dsv41_real_weights_4layer.md](work_log/worklog_dsv41_real_weights_4layer.md)**；
另见 [work_log/worklog_dsv41_module_input_probe.md](work_log/worklog_dsv41_module_input_probe.md) §8），三条结论会影响这里的判读标准：

1. **随机权重把因果搞反了**：随机权重下"把模块输入钉成引擎的值"能让 σ 降 22–35×（噪声几乎全部来自输入差被 MoE 放大）；
   真实权重下只降 1.2–1.4×。真实权重的主因是：**checkpoint 里 fp32 的 MoE 路由器纠偏 bias
   （`layers.<i>.ffn.gate.bias`）被训练侧的 `prepare_deepseek_v41_model_for_fsdp` 统一转成 bf16（`adapter.py:395-398`），
   而引擎保持 fp32**（架构本身声明 fp32：`model.py:936`；bf16 与 **W8A8** 两套真实权重实测都是 F32）。
   该 bias 只进专家**选择**不进权重，所以它的舍入是**纯离散**的：喂逐位相同的输入，top-6 集合仍在 **20–62%** 的 token 上不同。
2. **量化贡献**（4 组 A/B，`VERL_DSV41_KEEP_FP32_PARAMS=1|gate|continuous`，len=200）：
   baseline σ **0.2573 → 0.0967（只还原 gate.bias）→ 0.0668（全还原）**；输入钉死后的 σ **0.1882 → 0.0202 → 0.0191**；
   MoE 地板 0.0115–0.0449 → **0.0034–0.0042**；argmax 一致率 0.82 → **0.97**。只还原 `hc_*`/`attn_sink` 则**毫无变化**。
3. **§4.5 的健康指标判读再修正一次**：真实权重下这个不对称是**可修项**（训练侧保 fp32 / 引擎侧对齐 bf16，见 plan §7.4），
   修完再看是否需要 TIS；`rollout_corr/kl` 等指标在修之前**不能当作"同步是否健康"**的依据（它们现在混进了 10× 的参数级离散种子）。

## 5. 剩余风险与待办

| 风险/待办 | 说明 | 建议 |
|---|---|---|
| 显存紧（每卡 ~46GB 峰值） | 专家分片 27GB + 广播临时 18GB（单个 3D 专家张量）+ 稠密参数/激活 | 真机先跑冒烟；不足时考虑 EFSDP>1（需 `EP*EFSDP*EDP=8`）、或分块加载 |
| 若加载阶段 OOM：降峰值的三个办法 | ① 把加载拆成 **逐层调用** `set_model_state_dict`（每次只传该层参数，广播峰值从 18GB 降到 18GB/层内不变——单层专家仍是 18GB，收益有限）；② 让每 rank **直接读自己的分片**（自行按 `Shard(0)` 切 safetensors，峰值降到 ~2.3GB/层，代价是要写分片切片逻辑）；③ 临时降低 EP 并把 `expert_fully_shard_parallel_size` 用起来（**注意**：8 卡下专家总切分数固定 = EP×EFSDP = 8，每卡仍是 27GB，除非减少参与训练的设备） | 先按当前实现跑；OOM 时优先做 ② |

**显存定量**（8 卡、全卡空闲时）：
- 专家：108.72B 参数 / 8 = 13.6B ≈ **27GB/卡**（EP/EFSDP 怎么拆都一样，8 卡下总切分数固定为 8；EDP 复制只会更差）。
- 稠密：2.65B / 8 ≈ 0.33B ≈ 0.7GB/卡（FSDP 分片，可 CPU offload）。
- 加载广播：单个 3D 专家张量 18.1GB（`384×4608×5120` bf16）**每卡瞬时**，DCP 逐 tensor 广播，同一时刻只有一个。
- 因此加载峰值 ≈ **45–50GB/卡**；训练/rollout 阶段靠 `offload_policy` + vLLM sleep 与 27GB 专家常驻共存。

| 风险/待办 | 说明 | 建议 |
|---|---|---|
| 权重同步峰值 | `get_per_tensor_param` 会对每个专家 DTensor 做 `full_tensor()`（18GB/次上卡），再 unfuse | 观察真机显存；必要时用 bucketed 传输/降低并发 |
| 训练吞吐 | V4.1 参考注意力在 NPU 上走 `indexed_sparse_attention_torch`（eager，非 fused DSA）；`use_remove_padding=False` 走 padded | 先正确后快；后续评估 `use_sparse_flash_attn=True` / packed 路径 |
| 稠密参数导出仍走 offload/all-gather 路径 | 专家之外（embed/head/attention/mlp，约 1GB/rank）仍按 `_export_param` 默认实现 `param.to(device).full_tensor()`，在 CPU-offloaded DTensor 上做 all-gather；scaled32 实测这部分的传输量约 3 GiB/rank、折合约 3–5s | 若日后仍是瓶颈：需要按**单个参数**对齐 FSDP 分片与引擎 TP 分片（embedding/head 两侧都是 dim0 连续切分，但 loader 还会再 narrow，不能直接送分片）；或按 G13 的思路让引擎回报所需分片 |
| `LOCAL_EXPERT_EXPORT` 默认关闭（已验证） | 该路径依赖「colocate 1:1 rank 配对 + 两侧专家按 rank 连续切分 + 专家数能被 EP 整除」（脚本里已加 `GEN_EP == EP_SIZE` 的守卫；引擎侧 placement 默认就是 `linear` 连续切分，见 `expert_map_manager.py`）。**已在全尺寸 48 专家/rank 上验证**：`update_weights` 18s、`diff_mean` 0.024、`pearson` 0.90–0.92 | 想设为默认：需把「引擎侧实际专家集」在同步时回报给训练侧（RPC 返回值即可，代价是首次同步仍走全量），或至少加一个引擎侧覆盖率断言。当前 opt-in 的代价只是脚本里多一个 `LOCAL_EXPERT_EXPORT=1` |
| 真实（非随机）权重尚未验证 | 目前所有验证都在**随机权重**上：结构、显存、同步一致性、速度都已覆盖，但真实 checkpoint 的数值行为（fp8 反量化、engram/dspark 未启用分支等）没跑过 | 拿到真实 V4.1-Flash 权重后，先 `MODEL_PATH=` 直接切；注意真实权重是 fp8 专家 + `quantization_config`，本仓库的 bf16 scaled 随机权重不覆盖那条路径 |
| **停作业必须连 Ray worker 一起杀** | `kill` 掉 `main_ppo` / 启动脚本后，Ray 的 `WorkerDict` 进程会**变成孤儿继续占主机内存**（每个约 20–30GB，8 个一批；多次启停累积过 ~1TB）。主存被吃满后，下一次作业会在启动阶段**静默死掉**（日志停在 `all initialize finished, ready to fit` 之后，无任何 traceback；dmesg 里往往也看不到 OOM 记录），排查时极易误判成代码问题 | 重新启动前先清干净：`pgrep -af "ray::WorkerDict"`；`pgrep -f "VLLM::Worker_TP"`；必要时 `pkill -9 -f ray::WorkerDict`。启动前 `free -g` 确认 available 有 1.5TB 以上（114GB 权重 × 多份 + offload 常驻） |
| **随机权重的 init 尺度会决定指标可读性** | 见 G12：N(0,1) 填充会让 logits ~N(0,71)、softmax 近 one-hot，所有 logprob/KL/PPL 指标变 nan；必须用 `--init scaled` 版本才能做数值判读 | 生成随机权重一律加 `--init scaled --head-gain 4`；`--head-gain` 决定采样分布尖锐度（4 ≈ 真实模型的 top-1 概率 ~0.4） |
| 包副本歧义 | 机器上存在多份 `fsdp_turbo` / `vllm_ascend` 副本，`PYTHONPATH` 决定加载哪份 | 运行脚本显式导出 `PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo` |
| ref 模型二次加载 | actor 与 ref 各自读一遍 209GB checkpoint（rank0 读盘） | 冒烟阶段可接受；后续可加缓存/共享初始化 |
| save/load checkpoint | 当前脚本 `save_freq=-1`（先跑通） | 跑通后开启并验证 FSDPTurbo DCP 保存/恢复 |
| 全层模型迁移 | 本实现已与 checkpoint 结构对齐，理论上换全层 config 即可 | 全层需重新评估显存与 EP/EFSDP 拓扑 |

---

## 6. 复现命令速查

```bash
# 离线自检（无需 NPU）
cd /workspace-verl/verl
python3 -c "from verl.workers.engine import EngineRegistry; print(EngineRegistry.get_engine_cls('language_model','fsdp_turbo_dsv41'))"
env DEVICE=npu bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh --cfg job | tail -5
python3 -m pytest /workspace-verl/FSDPTurbo/tests/unit_tests/models/test_deepseek_v41.py -q
# G9 修复的 CPU 回归（专家权重回退 checkpoint 布局后再灌）
PYTHONPATH=/workspace-verl/vllm python3 -m pytest tests/utils/test_npu_expert_layout_on_cpu.py -q

# 真机 8 卡冒烟（NPU 空闲时）
cd /workspace-verl/verl
PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
HCCL_CONNECT_TIMEOUT=1500 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=59511 scripts/check_dsv41_fsdp_turbo_build.py --forward

# GRPO 端到端
cd /workspace-verl/verl
DEVICE=npu PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh

# 数值可用的随机权重（矩阵按 1/sqrt(fan_in)、norm=1、bias=0、head ×4），
# 以及小专家版（32 专家，14GB，用于快速迭代；其余结构一致）
cd /workspace-verl/mjt_workplace/deepseek41/inference
python3 make_random_ckpt_dsv41_4layers.py --config config.dsv41_4layer.bf16.json --init scaled \
    --head-gain 4 --out /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled
python3 make_random_ckpt_dsv41_4layers.py --config config.dsv41_4layer_e32.bf16.json --init scaled \
    --head-gain 4 --out /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32

# 快速冒烟（14GB 权重 + 128 token 响应 + 每 prompt 2 样本），带同步分段计时
cd /workspace-verl/verl
MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled32 \
VERL_SYNC_PROFILE=1 VERL_DSV41_LOCAL_EXPERT_EXPORT=1 DEVICE=npu \
MAX_RESPONSE_LEN=128 SAMPLE_N=2 TRAIN_BSZ=8 VAL_BSZ=8 \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
```

> 两个 env 开关（都只影响 NPU/DSV41 路径，默认关闭）：
> - `VERL_SYNC_PROFILE=1`：打印权重同步的分段耗时（发送侧 export/flush、接收侧 views/load/sync、step1/2/3 汇总）。worker 侧 `VERL_LOGGING_LEVEL` 默认 WARN，故这些行用 `print` 而非 `logger.info`。
> - `VERL_DSV41_LOCAL_EXPERT_EXPORT=1`：只导出本 rank 的专家（见 G13 与 4.4）。前提是 colocate 的 1:1 rank 配对与两侧专家连续切分；配对错误会在 `rollout_probs_diff_mean`/`pearson` 上暴露。

> `VERL_DSV41_RANDOM_INIT=1` 可跳过 checkpoint IO（专家用 N(0,0.02)），用于快速验证链路；`VERL_DSV41_MAX_SEQ_LEN` 控制 RoPE/scratch 尺寸。
