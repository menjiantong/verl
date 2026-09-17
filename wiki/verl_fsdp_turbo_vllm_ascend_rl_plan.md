# verl + FSDPTurbo + vllm-ascend 昇腾 RL 训练集成方案

> 状态：**方案确认版**（2026-09-17）
> 目标：在华为昇腾 NPU（单机 8 卡起步）上，用 **verl**（RL 编排）+ **FSDPTurbo**（训练引擎）+ **vLLM / vllm-ascend**（rollout 推理）对 **DeepSeek-V4.1 系列** 做 RL 训练（GRPO 类）。
> 阶段：先纯文本 → 后多模态；先屏蔽 V4.1 新结构（Engram/CSA2/稀疏索引）跑通基础链路 → 再逐步开启。

---

## 一、现状盘点（已核实，2026-09-17）

### 1.1 工作区仓库
| 仓库 | 当前 HEAD / 版本 | 说明 |
|---|---|---|
| `verl/` | main（含 Ascend 近期提交） | **已内置 `fsdp_turbo` 策略引擎适配**（`verl/workers/engine/fsdp/fsdp_turbo_impl.py`，注册 `backend="fsdp_turbo", device=["cuda","npu"]`），`turbo_config` YAML 已预留（`verl/trainer/config/engine/fsdp.yaml`） |
| `FSDPTurbo/` | 5af9d68（deepseek-v41 BF16 host Engram / sparse flash MLA） | 自带 `fsdp_turbo/models/deepseek_v41/`（V4.1 训练适配：adapter/engram/experts/model），V4.1 图文 SFT 示例独立于 verl |
| `vllm/` | **用户 pin `a97dacb710`（≈ v0.28.1rc0-570）** | 高于官方示例 pin 0.18.0；rollout 宿主 |
| `vllm-ascend/` | 本地定制分支 e67ab6495 | **含 DeepSeek-V4.1 eager 推理定制**：`models/deepseek_v41/`（model/compressor/indexer/dspark）、`attention/dsa_v41.py`、`core/deepseek_v41.py` |

### 1.2 官方集成路径（已确认）
- **verl 官方 NPU 示例**：`verl/examples/grpo_trainer/run_qwen3_5_27b_fsdp_turbo.sh` 给出 NPU 全链路参数（DEVICE=npu 自动切 16 设备/fsdp_size=16、HCCL/RAY_ASCEND 环境变量、turbo_config 组装、vllm rollout 引擎参数）。
- **verl 插件架构**：Platform/Engine 双注册表（`verl/docs/hardware/multi_chip_support.rst`），`fsdp_turbo` 已注册进 `EngineRegistry`。
- **权重同步**：`BaseEngine.get_per_tensor_param` / `get_per_tensor_param_shard` 统一流；FSDPTurbo 引擎继承 `FSDPEngineWithLMHead`，未 override（依赖 DTensor 导出）。

### 1.3 关键技术结论（决定实现方式）
1. **verl 模型加载是 HF AutoModel 路径**：`get_hf_auto_model_class` + `from_pretrained(trust_remote_code=...)`。官方 Qwen3.5 案例用 `module_patches`（FSDPTurbo 内部 `apply_module_patches`）把 transformers 类替换成 FSDPTurbo 自研实现。
2. **DeepSeek-V4.1 训练模型不在 transformers 里**：是 FSDPTurbo 独立 `nn.Module`（`DeepseekV41ForCausalLMAdapter` 包装 + `reference_model.ModelArgs`），`build_deepseek_v41_model()` 从 **FSDPTurbo 包内 `config.json`**（4 层 demo 配置）读结构，**与 verl `model.path` 无关**。
3. **因此不能走 HF AutoModel 路径**，需要新引擎子类直接调用 FSDPTurbo V4.1 构建流程（源码级 dispatch），哲学与 Qwen3.5 `module_patches` 一致。

### 1.4 版本风险
- verl 官方 pin：`vllm==0.18.0 + vllm-ascend@54879467 + transformers@cc7ab9be`。
- 用户基线：`vllm@a97dacb（v0.28.x dev） + vllm-ascend（本地定制分支）`。
- **结论**：vllm-ascend V4.1 eager 推理依赖 vllm ≥0.28，官方 0.18.0 用不上。方案以本地 vllm + vllm-ascend 为准；FSDPTurbo 训练侧对 vllm 版本不敏感（权重流只依赖 `get_per_tensor_param` + HF 名），风险可控。

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

---

## 三、实施步骤（分阶段，每步可验证）

### Phase 0：环境与版本锁定
1. CANN/driver/firmware（A2/A3）+ Python ≥3.10 + `torch≥2.9` + `torch_npu≥2.9`。
2. `transformers@cc7ab9be`（含 deepseek_v41 config/processor 支持）。
3. vllm：`VLLM_TARGET_DEVICE=empty pip install -e vllm/`（或 v0.28.x）。
4. vllm-ascend：`pip install -e vllm-ascend/`（本地定制分支）。
5. FSDPTurbo：`pip install -e FSDPTurbo/`（含 `npu` extra：`torch_npu, fla-core, flash-linear-attention`）。
6. verl：`pip install -e verl/`（`requirements-npu.txt`：`triton-ascend==3.2.2`、`TransferQueue`）。
7. **验证**：`python -c "import verl, fsdp_turbo, vllm, vllm_ascend"` 全通过；最小 SFT 单卡冒烟。

### Phase 1：DeepSeek-V4.1 训练侧接入 verl（核心适配）
- 新引擎子类 `FSDPTurboDSV41EngineWithLMHead`（override `_build_module`，调用 FSDPTurbo V4.1 构建流程）——详见"改动清单"。
- 模型结构暂由 FSDPTurbo 内置 `config.json`（4-layer demo）决定；后续支持外部 config。
- 8 卡默认拓扑：`FSDP=4, TP=2, EP=4, expert DP=2`（先屏蔽 Engram 时 EP=1），`ulysses=1`。

### Phase 2：rollout 侧 vllm-ascend 接入与验证
- 确认 vllm 0.28 + vllm-ascend 能加载 V4.1（config→模型类映射已在 vllm-ascend 部分存在）。
- rollout 配置对齐官方脚本（见改动清单 E）。

### Phase 3：RL（GRPO）端到端
- 纯文本数据（geo3k/gsm8k 风格 parquet），简单规则 reward。
- 启动脚本 `run_deepseek_v41_grpo_fsdp_turbo_npu.sh`。

### Phase 4（后续）：多模态 + V4.1 完整结构
- 多模态：`messages+images`，vllm-ascend image processor（`mm_preprocess.py`）。
- Engram/CSA2/稀疏索引：FSDPTurbo 训练侧支持，vLLM eager 推理侧 bring-up 状态，需专项验证。

---

## 四、改动清单（精确到文件）

### A. 模型接入核心（必改，3 个文件）

**① 新增 `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`**（约 150–250 行）

继承 `FSDPTurboEngineWithLMHead`，仅 override `_build_module`（不走 HF AutoModel，直接走 FSDPTurbo V4.1 构建流程）：

```python
# 从 FSDPTurbo 引入 V4.1 构建三件套
from fsdp_turbo.models.deepseek_v41 import (
    build_deepseek_v41_model,
    initialize_deepseek_v41_model,
    prepare_deepseek_v41_model_for_fsdp,
)

@EngineRegistry.register(
    model_type="language_model",
    backend="fsdp_turbo_dsv41",
    device=["npu", "cuda"],
)
class FSDPTurboDSV41EngineWithLMHead(FSDPTurboEngineWithLMHead):
    def _build_module(self):  # 完全 override
        # 1. 处理 CP/Ulysses 冲突（继承 Turbo 逻辑）
        # 2. build_deepseek_v41_model(tokenizer=..., engram_meta_init=True, ...)
        # 3. initialize_deepseek_v41_model(training_model)
        # 4. prepare_deepseek_v41_model_for_fsdp(device=..., parameter_dtype=bf16)
        # 5. FSDPTurbo(config, training_model) 包装
        return <FSDPTurbo 包装后的 model>
```

- 复用 `fsdp_turbo_impl.py` 的 `_init_parallel_state` / `optimizer_step` / CP/Ulysses 冲突处理 / offload / QAT（全部继承）。
- `engram_meta_init` 先设为 True（屏蔽 Engram 的延迟分配）或 False（先跑通），以 FSDPTurbo 内置 config 为准。

**② 修改 `verl/verl/workers/engine/fsdp/__init__.py`**
```python
from .fsdp_turbo_dsv41_impl import FSDPTurboDSV41EngineWithLMHead
__all__ += ["FSDPTurboDSV41EngineWithLMHead"]
```

**③ 修改 `verl/verl/workers/engine/__init__.py`**
```python
from .fsdp import FSDPEngine, FSDPEngineWithLMHead, FSDPTurboEngineWithLMHead, FSDPTurboDSV41EngineWithLMHead
__all__ += ["FSDPTurboDSV41EngineWithLMHead"]
```

### B. 配置层（1 个文件）

**④ 修改 `verl/verl/trainer/config/engine/fsdp.yaml`**
- `strategy` 注释加上 `fsdp_turbo_dsv41`，默认值按 8 卡 V4.1 起步调整。
- `turbo_config` 默认：`fully_shard_parallel_size: 4, tensor_parallel_size: 2, expert_parallel_size: 1, ulysses_parallel_size: 1`（8 卡，屏蔽 MoE/Engram 时）。

### C. rollout（vllm-ascend）侧（验证为主）

**⑤ 验证 vllm-ascend V4.1 模型可加载**（`models/__init__.py:32` 已注册 `DeepseekV4ForCausalLM`，`models/deepseek_v41/` 完整）。若 config 含 vllm 未识别字段报错 → 在 vllm-ascend 内做 config→class 映射（已有雏形）。

### D. 数据/奖励（先纯文本）

**⑥ 数据格式**：verl 多模态 runner 支持 `messages + images`（官方示例 `data.image_key=images`）。纯文本时 images 空即可。**无需改代码**。
**⑦ reward**：先用 verl 内置 rule-based reward。**无需改代码**。

### E. 启动脚本

**⑧ 新增 `verl/examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`**（约 250 行，仿 `run_qwen3_5_27b_fsdp_turbo.sh`）

```bash
# DEVICE=npu 分支环境变量照抄官方脚本（HCCL_CONNECT_TIMEOUT、HCCL_SOCKET、RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES...）

# 模型后端选用新引擎
actor_rollout_ref.actor.strategy=fsdp_turbo_dsv41
actor_rollout_ref.ref.strategy=fsdp_turbo_dsv41

# turbo_config（8 卡，屏蔽 Engram）
+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=4
+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tensor_parallel_size=2
+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.expert_parallel_size=1
...（apply_modules / recompute 结构照 Qwen3.5，模块名按 deepseek_v41 config 调整）

# rollout vllm-ascend
actor_rollout_ref.rollout.name=vllm
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
actor_rollout_ref.rollout.tensor_model_parallel_size=2
...
```

### F. 验证脚本（可选）

**⑨ 新增 `scripts/check_fsdp_turbo_v41_sync.py`**（权重同步 logprob 一致性）。

---

## 五、风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| FSDPTurbo V4.1 模型未接入 verl AutoModel 路径 | 训练侧最大阻塞 | 新引擎子类源码级 dispatch（方案 A）；FSDPTurbo 独立 SFT 已通说明模型可构造 |
| 官方 pin 是 vllm 0.18，用户 vllm 是 0.28 | rollout 权重同步 API 偏移 | 权重流 `get_per_tensor_param`+`load_weights` 稳定接口；最小 SFT+rollout 冒烟验证 |
| 官方 V4.1 权重为 FP8/FP4，训练需 BF16 | 无法直接加载官方权重 | 用户计划"减层权重"，从零/转换 BF16 初始化 |
| vllm rolloutee是否需要 V4.1 KV-cache/compressor 全路径 | 大上下文 RL rollout 性能 | 减层权重验证 eager 正确性；fused DSA 优化后续 |
| 多卡 Ray + HCCL 环境复杂 | 排障耗时长 | 严格复用官方 16 卡/8 卡环境变量模板，先单机 8 卡 |
| Engram 表极大（~366GiB/2 表） | 8 卡内存不足 | 先屏蔽 Engram；开启时用 `engram_meta_init=True` + row sharding + host offload |
| **模型结构写死在 FSDPTurbo 包内 config.json（4-layer demo）** | 无法通过 `model.path` 指定层数 | 先接受内置 config；后续让 `build_deepseek_v41_model_args` 支持外部 config |

---

## 六、完成定义（DoD）

- [ ] `verl + FSDPTurbo` 在 8×NPU 上构建模型、FSDP 训练无错。
- [ ] `verl → vllm-ascend` 权重同步后 rollout 生成 + logprob 一致。
- [ ] 纯文本 GRPO 在 8×NPU 上跑通 ≥1 个完整 epoch。
- [ ] 减层 V4.1 权重验证完毕；给出全层迁移问题清单。
