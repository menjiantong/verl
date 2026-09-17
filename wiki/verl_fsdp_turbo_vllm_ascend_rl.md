# verl + FSDPTurbo + vllm-ascend 昇腾 RL 训练集成方案

> 目标：在华为昇腾 NPU（Atlas 800T A2/A3，单机 8 卡起步）上，利用 **verl**（RL 编排）+ **FSDPTurbo**（训练引擎）+ **vLLM / vllm-ascend**（rollout 推理后端），对 **DeepSeek-V4.1 系列** 做 RL 训练（GRPO 类）。
> 阶段目标：先纯文本 → 后多模态；先屏蔽 V4.1 新结构（Engram/CSA2/稀疏索引）跑通基础链路 → 再逐步开启 V4.1 特性。

---

## 一、现状盘点（已核实）

### 1.1 工作区仓库
| 仓库 | 来源 | 当前 HEAD / 版本 | 说明 |
|---|---|---|---|
| `verl/` | github.com/verl-project/verl | `main`（cb21203a 等，含 Ascend 近期提交） | **已内置 `fsdp_turbo` 策略引擎适配**（`verl/workers/engine/fsdp/fsdp_turbo_impl.py`，华为 2026 版权，注册 `backend="fsdp_turbo", device=["cuda","npu"]`），`turbo_config` 全量 YAML 已预留（`verl/trainer/config/engine/fsdp.yaml`） |
| `FSDPTurbo/` | gitcode.com/guihaowen666/FSDPTurbo | 5af9d68（deepseek-v41 BF16 host Engram / sparse flash MLA） | 自带 `fsdp_turbo/models/deepseek_v41/`（V4.1 训练模型适配：adapter/engram/experts/model），V4.1 图文 SFT 示例独立于 verl 运行 |
| `vllm/` | github.com/vllm-project/vllm | **用户已 pin `a97dacb710`（≈ v0.28.1rc0-570，dev）** | 高于官方示例 pin 的 0.18.0；用于训练→推理权重同步的宿主 |
| `vllm-ascend/` | github.com/GDzhu01/vllm-ascend（本地定制分支） | e67ab6495 | **含 DeepSeek-V4.1 eager 推理定制**：`models/deepseek_v41/`（model/compressor/indexer/dspark）、`attention/dsa_v41.py`（V4.1 专用 cache 布局）、`core/deepseek_v41.py`（hybrid cache specs）、`ascend_config.py` 等 |

### 1.2 已确认的官方集成路径
- **verl 官方 NPU 示例**：`verl/examples/grpo_trainer/run_qwen3_5_27b_fsdp_turbo.sh` 已给出 **NPU 全链路参数**（DEVICE=npu 自动切 16 设备/fsdp_size=16、HCCL/RAY_ASCEND 环境变量、turbo_config 组装、vllm rollout 引擎参数）。
- **verl 插件架构**：`verl/docs/hardware/multi_chip_support.rst` 描述了 Platform/Engine 双注册表，`fsdp_turbo` 已注册进 EngineRegistry。
- **FSDPTurbo 模型接入 verl**：verl 引擎 `_build_module` 使用 `get_hf_auto_model_class(hf_config)` + `from_pretrained(trust_remote_code=...)` 构造模型。官方示例用的 Qwen3.5 是 transformers 注册类；**DeepSeek-V4.1 训练类在 `fsdp_turbo/models/deepseek_v41/`（未注册 transformers AutoModel）**——这是本方案最大适配点。
- **权重同步**：`BaseEngine.get_per_tensor_param` / `get_per_tensor_param_shard` 是统一流；FSDPTurbo 引擎继承 `FSDPEngineWithLMHead`，未 override 权重流（依赖 DTensor 导出）。

### 1.3 版本风险
- verl 官方示例 pin：`vllm==0.18.0 + vllm-ascend@54879467 + transformers@cc7ab9be`。
- 你的基线：`vllm@a97dacb（v0.28.x dev） + vllm-ascend（本地定制深度分支）`。
- **结论**：由于 vllm-ascend 的 V4.1 eager 推理是定制依赖（vllm ≥0.28 才有对应 V4.1 支持），官方 0.18.0 基线用不上。方案以你的本地 vllm + vllm-ascend 为准，但 **FSDPTurbo 训练侧对 vllm 版本不敏感**（权重流只依赖 `get_per_tensor_param` 与 HF 名），风险可控。

---

## 二、目标架构

```
┌────────────────────────── 昇腾 NPU 节点（8×Atlas 800T A2/A3）──────────────────────────┐
│                                                                                           │
│                                 verl（Ray 分布式）                                        │
│   ┌───────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐      │
│   │ Actor Engine  │   │  Ref Engine   │   │   Rollout      │   │  Reward / Critic │      │
│   │ (FSDPTurbo)   │   │ (FSDPTurbo)   │   │  (vLLM+ascend) │   │  (verl reward)   │      │
│   │ strategy=     │   │ strategy=     │   │  name=vllm     │   │  (先简单规则)      │      │
│   │   fsdp_turbo  │   │   fsdp_turbo  │   │  vllm-ascend   │   │                  │      │
│   └──────┬────────┘   └──────┬────────┘   └───────┬────────┘   └──────────────────┘      │
│          └───────── 权重同步 get_per_tensor_param ─┘ (HF 名流)                            │
│                          (unfuse MoE 等)                                                  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
        │                                  │
        │ FSDPTurbo: FSDP2 + TP + EP + CP  │ vllm-ascend: V4.1 eager 推理 (compressor/indexer/cache)
        │ Engram row-sharding（后续）       │ （paged KV / 长上下文）
        └──────────────────────────────────┴───────────────────────────────────────────────
```

- **训练引擎**：Actor / Ref 都用 `strategy=fsdp_turbo`，`turbo_config.distributed` 配 FSDP/TP/EP 拓扑。
- **rollout（推理）**：`rollout.name=vllm` + 安装后的 `vllm_ascend` 自动插件（`is_torch_npu_available` 自动走 NPU 路径）。
- **权重同步**：verl 的 actor→vllm 权重新加载，经 `BaseEngine` 流的 `get_per_tensor_param`（HF 名）→ vLLM `load_weights`，原生支持，无需额外适配（前提：vllm 能注册 V4.1 模型类）。

---

## 三、实施步骤

### Phase 0：环境与版本锁定（1–2 天）

目标：确定能跑的软件栈，形成可复现的构建（Dockerfile 或 conda）。

1. **CANN / driver / firmware**：按昇腾版本安装（A2/A3，先确认 atlas driver 与 CANN 版本）。
2. **Python + PyTorch**：`python>=3.10`，`torch>=2.9`（FSDPTurbo 要求），配套 `torch_npu>=2.9`（A2/A3 对应包）。
3. **torch/transformers 对齐**：用 `transformers@cc7ab9be`（官方示例 pin，V4.1 需要 transformers 支持 `deepseek_v41` 的 config/processor）。注意 **vllm 的 transformers 兼容范围**（v0.28.x 对 transformers 版本有要求，需确认）。
4. **vllm + vllm-ascend**：
   - vllm：`VLLM_TARGET_DEVICE=empty pip install -e vllm/`（或直接到 v0.28.x）
   - vllm-ascend：`pip install -e vllm-ascend/`（本地定制分支）
5. **FSDPTurbo**：`pip install -e FSDPTurbo/`（含 `npu` extra：`torch_npu, fla-core, flash-linear-attention`）
6. **verl**：`pip install -e verl/`（`requirements-npu.txt` 给出 NPU 依赖：`triton-ascend==3.2.2`、`TransferQueue @ git+...ascend/TransferQueue` 等）。
7. **验证**：各自 import 通过 + 单卡 SFT 冒烟（verl `run_sft_engine.sh` NPU 途径）确认 FSDPTurbo/verl/vllm-ascend 三方可协同。

> 里程碑 M0：`python -c "import verl, fsdp_turbo, vllm, vllm_ascend"` 全通过；最小 SFT 单卡跑通。

### Phase 1：DeepSeek-V4.1 训练侧接入 verl（核心适配，3–5 天）

**问题**：FSDPTurbo 的 V4.1 模型在 `fsdp_turbo/models/deepseek_v41`（自定义类，未注册 transformers AutoModel），verl 的 `get_hf_auto_model_class` 无法直接构造。

**方案 A（推荐）**：在 verl 侧加一个最小的 **external 模型 wrapper**，让 V4.1 成为可通过 `AutoConfig/AutoModel` 构造的 HF 类。
- 位置：`verl/workers/engine/fsdp/` 下新增 V4.1 适配（或放在 `verl/utils/model.py` 的 automodel 扩展点），`_build_module` 通过 `model_type=deepseek_v41` 分发到 FSDPTurbo 的 `initialize_deepseek_v41_model` / `prepare_deepseek_v41_model_for_fsdp`（FSDPTurbo README 明确提供这两个入口）。
- 或更小侵入：**先不改 verl 核心**，编写一个 **轻量 HF wrapper 模块**（`deepseek_v41_modeling.py`），内部调用 `fsdp_turbo.models.deepseek_v41` 构造，注册 `AutoModelForCausalLM`/`AutoConfig`（`model_type="deepseek_v41"`），再用 `actor_rollout_ref.model.trust_remote_code` / `external_lib` 注入。verl 侧零改动。

> ⚠️ 需要在 plan 实现阶段确认 FSDPTurbo V4.1 `model.py` 的构造签名，以及它依赖的 `config.json`（官方 `DeepSeek-V4.1-Flash` config）能否被 transformers `AutoConfig` 读入。FSDPTurbo 示例已能独立跑，说明模型可构造——只需把它的 `build_model` 流程搬到 verl 引擎能触发的路径下。

**并行/拓扑**：
- 先屏蔽 V4.1 新结构，用**减层**（few-layer）普通 Transformer/mHA 代替，极小模型验证链路。
- 8 卡默认拓扑：`FSDP=4, TP=2, EP=4, expert DP=2`（官方示例默认），`ulysses=1`；后续全层再调整。

### Phase 2：rollout 侧 vllm-ascend 接入与验证（2–3 天）

1. **确认 vllm 能加载 DeepSeek-V4.1**：vllm-ascend 注册 `DeepseekV4ForCausalLM` 等，但需要确认 vllm 0.28.x 侧的 `AutoModelForCausalLM`/checkpoint 读取对 V4.1（含 config.json 中 V4.1 新结构字段）支持。若 vllm core 不支持 V4.1 config，在 vllm-ascend 内做 config → 模型类映射（已部分存在）。
2. **rollout 配置**（对齐官方脚本）：
   ```
   actor_rollout_ref.rollout.name=vllm
   actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}   # 如 2/4
   actor_rollout_ref.rollout.gpu_memory_utilization=0.4~0.5
   actor_rollout_ref.rollout.n=5                                    # GRPO
   actor_rollout_ref.rollout.enable_chunked_prefill=True
   actor_rollout_ref.rollout.max_num_batched_tokens=8192
   actor_rollout_ref.rollout.free_cache_engine=True
   +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
   +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0   # NPU (多模态时需要)
   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
   ```
3. **验证点**：单步（非 RL）—— 用 SFT 检查点 → vllm rollout 生成 → 权重从 actor→vllm 同步后 logprob 一致性。

### Phase 3：RL（GRPO）端到端（2–3 天）

1. 数据：**先纯文本**（geo3k/gsm8k 风格 parquet，`messages+images` 若无图则 images 可为空）。
2. reward：先用简单规则/格式 reward（后续接自定义）。
3. 启动：参考 `run_qwen3_5_27b_fsdp_turbo.sh` 改造为 **`run_deepseek_v41_grpo_fsdp_turbo_npu.sh`**：
   - DEVICE=npu 分支环境变量（HCCL_CONNECT_TIMEOUT、HCCL_HOST/NPU_SOCKET、RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES、ASCEND_RT_VISIBLE_DEVICES）
   - actor/ref 的 `turbo_config`、apply_modules、recompute 计划
   - `strategy=fsdp_turbo`、`entropy_checkpointing`、`offload_policy` 等
4. 里程碑 M2：8 卡上 GRPO 跑通，`train.log` 看到 rollout→train→loss 循环，权重更新无卡死。

### Phase 4：多模态 + V4.1 完整结构（后续）

- 多模态：数据 `messages+images`，`data.image_key=images`，启用 vllm-ascend image processor（V4.1 processor 已在 `fsdp_turbo/models/deepseek_v41/processor.py` 与 vllm-ascend `mm_preprocess.py`）。
- V4.1 新结构（Engram/CSA2/稀疏索引）：FSDPTurbo 训练侧已支持（Engram BF16 row-shard、host offload、sparse flash MLA），但 **vLLM eager 推理侧是 bring-up 状态**（compressor/indexer 已实现但非全 fused DSA），RL 性能/内存需专项验证。建议在基础链路稳定后单独立项。

---

## 四、关键风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| FSDPTurbo V4.1 模型未接入 verl AutoModel 路径 | 训练侧最大阻塞 | Phase 1 的 wrapper / dispatcher 方案；FSDPTurbo 独立 SFT 已通，说明模型可构造 |
| verl 官方 pin 是 vllm 0.18，你的 vllm 是 0.28 | rollout 权重同步 API 偏移 | 权重流 `get_per_tensor_param`+`load_weights` 是稳定接口；先用最小 SFT+rollout 冒烟验证 |
| DeepSeek-V4.1 官方权重为 FP8/FP4，训练需 BF16 | 无法直接加载官方权重 | 你已计划"减层权重"，从零/转换 BF16 初始化（FSDPTurbo 示例即 BF16 随机初始化） |
| vllm rolloutee是否需要 V4.1 的 KV-cache/compressor 全路径 | 大上下文 RL rollout 性能 | Phase 2 用**减层**权重验证 eager 路径正确性；性能优化（fused DSA）在后续阶段 |
| 多卡 Ray + HCCL 环境复杂 | 环境排障时间长 | 严格复用官方 16 卡/8 卡环境变量模板，先单机 8 卡 |
| Engram 表极大（~366GiB/2 表） | 8 卡内存不足 | 先屏蔽 Engram（Phase 1–3）；开启时用 `engram_meta_init=True` + row sharding + host offload |

---

## 五、可交付物

1. 环境构建脚本（Dockerfile 或 conda env yaml），锁定 CANN/torch/transformers/vllm/vllm-ascend/FSDPTurbo/verl 版本。
2. **DeepSeek-V4.1 模型接入 verl 的最小 wrapper**（`deepseek_v41` → FSDPTurbo 构造，注册 AutoModel/AutoConfig）。
3. **`run_deepseek_v41_grpo_fsdp_turbo_npu.sh`** GRPO 启动脚本（单机 8 卡，纯文本起步）。
4. 验证清单：SFT 冒烟 → 权重同步一致性 → GRPO 端到端。
5. (后可交付) 多模态 + V4.1 完整结构 RL 配置。

---

## 六、完成定义（DoD）

- [ ] `verl + FSDPTurbo` 在 8×NPU 上构建模型、FSDP 训练无错。
- [ ] `verl → vllm-ascend` 权重同步后 rollout 生成 + logprob 一致。
- [ ] 纯文本 GRPO 在 8×NPU 上跑通 ≥1 个完整 epoch，loss/rollout 正常、可保存/恢复 checkpoint。
- [ ] 减层 DeepSeek-V4.1 权重验证完毕；给出全层迁移需要解的问题清单。

---

## 附：与官方示例的差异对照（为什么不能直接跑 run_qwen3_5_27b_fsdp_turbo.sh）

| 差异项 | 官方示例 | 本方案 |
|---|---|---|
| 模型 | Qwen3.5-27B（transformers 注册类） | DeepSeek-V4.1（FSDPTurbo 自定义类，需接入） |
| 推理后端 | vllm 0.18 | vllm 0.28 + vllm-ascend 定制（V4.1 eager） |
| 并行 | 16 卡/8 卡统一示例 | 8 卡起步，FSDP/TP/EP 按 V4.1 减层调整 |
| V4.1 新结构 | 无 | 先屏蔽，后续开启 |
