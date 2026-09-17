# DeepSeek-V4.1 RL 实施工作记录（2026-09-17）

> 目标：在昇腾 NPU 上用 **verl + FSDPTurbo + vllm/vllm-ascend** 对 **DeepSeek-V4.1（8 层减层权重）** 做 GRPO 强化学习。
> 本文件记录：做了什么、遇到什么问题、怎么修的、验证到什么程度、还剩什么。
> 相关文档：[方案](verl_fsdp_turbo_vllm_ascend_rl_plan.md) ｜ [改动盘点](README.md) ｜ [验证步骤](verification.md)

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

**尚待真机验证**（NPU 空闲后按顺序执行）：
1. `torchrun --nproc_per_node=8 scripts/check_dsv41_fsdp_turbo_build.py --forward`（构建 + 加载 + 前向 + 显存报告）
2. `DEVICE=npu bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`（GRPO 端到端）
3. 权重同步一致性：rollout logprob 与 actor 重算 logprob 对齐

---

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
| 权重同步峰值 | `get_per_tensor_param` 会对每个专家 DTensor 做 `full_tensor()`（18GB/次上卡），再 unfuse | 观察真机显存；必要时用 bucketed 传输/降低并发 |
| 训练吞吐 | V4.1 参考注意力在 NPU 上走 `indexed_sparse_attention_torch`（eager，非 fused DSA）；`use_remove_padding=False` 走 padded | 先正确后快；后续评估 `use_sparse_flash_attn=True` / packed 路径 |
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

# 真机 8 卡冒烟（NPU 空闲时）
cd /workspace-verl/verl
PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
HCCL_CONNECT_TIMEOUT=1500 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=59511 scripts/check_dsv41_fsdp_turbo_build.py --forward

# GRPO 端到端
cd /workspace-verl/verl
DEVICE=npu PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
```

> `VERL_DSV41_RANDOM_INIT=1` 可跳过 checkpoint IO（专家用 N(0,0.02)），用于快速验证链路；`VERL_DSV41_MAX_SEQ_LEN` 控制 RoPE/scratch 尺寸。
