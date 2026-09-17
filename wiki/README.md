# verl × FSDPTurbo × vllm-ascend — DeepSeek-V4.1 RL 实施改动盘点

> 状态：**代码已实施**（2026-09-17）
> 方案详细文档见 [verl_fsdp_turbo_vllm_ascend_rl_plan.md](verl_fsdp_turbo_vllm_ascend_rl_plan.md)
> 验证步骤见 [verification.md](verification.md)

---

## 一、改动总览

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py` | **新增** | DeepSeek-V4.1 专用 FSDPTurbo 引擎子类（核心） |
| 2 | `verl/verl/workers/engine/fsdp/__init__.py` | 修改 | 导入并导出新引擎 |
| 3 | `verl/verl/workers/engine/__init__.py` | 修改 | 顶层导出新引擎 |
| 4 | `verl/verl/trainer/config/engine/fsdp.yaml` | 修改 | strategy 注释 + turbo_config 8 卡默认值 |
| 5 | `verl/examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | **新增** | DeepSeek-V4.1 纯文本 GRPO 启动脚本（8 卡 NPU） |
| 6 | `wiki/verl_fsdp_turbo_vllm_ascend_rl_plan.md` | **新增** | 完整方案（背景/架构/风险/DoD） |
| 7 | `wiki/verification.md` | **新增** | 本文件配套的验证步骤 |
| 8 | `wiki/README.md` | **新增** | 本次改动盘点（本文件） |

---

## 二、核心代码改动细节

### 2.1 新增引擎子类 `fsdp_turbo_dsv41_impl.py`

**设计目标**：让 verl 能用 FSDPTurbo 构建 DeepSeek-V4.1 训练模型（该模型是 FSDPTurbo 独立 `nn.Module`，**不在 transformers AutoModel 体系内**，verl 通用 `from_pretrained` 路径无法构造）。

**关键实现**（完整代码见文件）：

```python
@EngineRegistry.register(
    model_type="language_model",
    backend="fsdp_turbo_dsv41",
    device=["npu", "cuda"],
)
class FSDPTurboDSV41EngineWithLMHead(FSDPTurboEngineWithLMHead):
    def _build_module(self):
        # 复用父类 FSDPTurboEngine 的 Qwen VLM monkey-patch guard
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return self._build_dsv41_module()
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def _build_dsv41_module(self):
        from fsdp_turbo.models.deepseek_v41 import (
            build_deepseek_v41_model,
            initialize_deepseek_v41_model,
            prepare_deepseek_v41_model_for_fsdp,
        )
        training_model = build_deepseek_v41_model(
            tokenizer=self.model_config.tokenizer,
            engram_meta_init=True,          # Engram 表延迟分配（meta），屏蔽巨大表内存
            engram_storage_backend="row_sharded",
            use_sparse_flash_attn=False,    # 先屏蔽 V4.1 稀疏注意力
        )
        initialize_deepseek_v41_model(training_model)
        training_model = prepare_deepseek_v41_model_for_fsdp(
            training_model,
            device=torch.accelerator.current_accelerator(),
            parameter_dtype=torch.bfloat16,
        )
        return training_model
```

**为什么这样设计**：
- 继承 `FSDPTurboEngineWithLMHead`（已有 `_init_parallel_state` / `_build_fsdp_module` / `optimizer_step` / CP-Ulysses 冲突处理 / offload / QAT 全套）。
- 只 override `_build_module`：返回 FSDPTurbo V4.1 adapter，**父类 `_build_fsdp_module` 会自动做 FSDPTurbo 包装 + state_dict 加载**（`module.state_dict()` → `FSDPTurbo(config, module)` → `fsdp2_load_full_state_dict`），对齐 FSDPTurbo 官方示例流程。
- `_build_module` 不走 HF `from_pretrained`，因此不需要 verl 的 meta init context——FSDPTurbo V4.1 用 `torch.set_default_dtype(bf16)` 自建参数。

**路由**：`engine_workers.py:138` 用 `backend=self.engine_config.strategy` 查找 → `actor.strategy=fsdp_turbo_dsv41` 命中 `EngineRegistry["language_model"]["fsdp_turbo_dsv41"]`（device=npu 时 key="npu"）。

### 2.2 导出修改（2 个 __init__.py）

- `fsdp/__init__.py`：`from .fsdp_turbo_dsv41_impl import FSDPTurboDSV41EngineWithLMHead`，加入 `__all__`。
- `engine/__init__.py`：从 `.fsdp` 批量导入并加入 `__all__`。

### 2.3 配置默认值（fsdp.yaml）

```yaml
# fsdp or fsdp2 or fsdp_turbo or fsdp_turbo_dsv41
strategy: fsdp        # 默认仍为 fsdp；要在 GRPO 脚本里显式指定 fsdp_turbo_dsv41

turbo_config:
  distributed:
    fully_shard_parallel_size: 4   # 8 卡: FSDP=4, TP=2 → 2×DP
    tensor_parallel_size: 2
    expert_parallel_size: 1        # 屏蔽 MoE/Engram 时保持 1
    expert_fully_shard_parallel_size: 1
    ulysses_parallel_size: 1
    fsdp_plan:
      param_dtype: bf16
      reduce_dtype: fp32
      output_dtype: bf16
      fsdp_implementation: native
      num_to_forward_prefetch: 1
      num_to_backward_prefetch: 1
```

### 2.4 GRPO 启动脚本（run_deepseek_v41_grpo_fsdp_turbo_npu.sh）

参照官方 `run_qwen3_5_27b_fsdp_turbo.sh` 改造，`DEVICE=npu` 分支完整保留 HCCL/RAY_ASCEND 环境变量，关键差异：

| 项 | 官方 Qwen3.5 | 本脚本 V4.1 |
|---|---|---|
| strategy | `fsdp_turbo` | `fsdp_turbo_dsv41` |
| apply_modules | `model.visual/*`, `model.language_model.*` | `model.model.embed/layers/head/norm/vision` |
| 拓扑 | fsdp_size=16（NPU 分支） | fsdp_size=4, tp=2（8 卡） |
| rollout TP | GEN_TP=4 | GEN_TP=2 |
| 多模态 | image_key=images | 纯文本起步（images 字段可先在建数据时省略） |

---

## 三、未改动但需要知晓的约束

1. **模型结构由 FSDPTurbo 包内 `config.json` 决定**（`fsdp_turbo/models/deepseek_v41/config.json`，4-layer demo）。`actor_rollout_ref.model.path` **不决定层数**，只提供 tokenizer。→ 后续如需外部 config 扩展，改 `build_deepseek_v41_model_args()`。
2. **vllm/vllm-ascend 代码零改动**（rollout 侧验证为主，见 verification.md）。
3. **reward / data 零改动**（用 verl 内置 rule-based + 标准 parquet 格式）。
