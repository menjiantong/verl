# 验证步骤 — DeepSeek-V4.1 RL（verl + FSDPTurbo + vllm-ascend）

> 配套改动见 [README.md](README.md)，方案见 [verl_fsdp_turbo_vllm_ascend_rl_plan.md](verl_fsdp_turbo_vllm_ascend_rl_plan.md)
> 按顺序执行，每步有明确通过标准。

---

## Phase 0：环境与依赖（先行，约 1–2 天）

### 0.1 安装清单
| 组件 | 版本/来源 | 命令 |
|---|---|---|
| Python | ≥ 3.10（推荐 3.11） | — |
| PyTorch | ≥ 2.9 | pip install torch |A2/A3 对应 wheel|
| torch_npu | ≥ 2.9（匹配 CANN） | 昇腾安装指南 |
| transformers | @cc7ab9be 附近（含 deepseek_v41 config） | pip install git+...@cc7ab9be |
| vllm | 本地 `a97dacb`（≈v0.28.x） | `VLLM_TARGET_DEVICE=empty pip install -e vllm/` |
| vllm-ascend | 本地定制分支 | `pip install -e vllm-ascend/` |
| FSDPTurbo | gitcode 仓库，deepseek_v41 版 | `pip install -e FSDPTurbo/[npu]` |
| verl | 本地（含本次改动） | `pip install -e verl/` + `requirements-npu.txt` |

### 0.2 通过标准
```bash
python -c "import verl, fsdp_turbo, vllm, vllm_ascend, torch_npu"
# 全部成功（无 MissingDependency）
```

---

## Phase 1：训练侧验证（FSDPTurbo × verl，核心）

### 1.1 先跑 FSDPTurbo 自带 SFT（隔离 verl，验证模型可训练）

```bash
cd FSDPTurbo/examples/deepseek_v41
# 改 config.yaml: data.dataset_path 指向真实纯文本数据
bash run.sh    # 单机 8 卡
```

**通过标准**：loss 正常下降，无 OOM，checkpoint 可保存。

### 1.2 verl 引擎导入冒烟（验证新引擎注册）

```bash
cd verl
python -c "
from verl.workers.engine import EngineRegistry
import verl.workers.engine  # 触发注册
cls = EngineRegistry.get_engine_cls('language_model', 'fsdp_turbo_dsv41')
print('engine OK:', cls)
"
```

**通过标准**：打印 `FSDPTurboDSV41EngineWithLMHead`，不报 Unknown backend。

### 1.3 verl FSDP SFT 冒烟（真机 8 卡，NPU）

用 verl 的 SFT 引擎测试或手工构造（参考 `tests/special_e2e/sft/run_sft_engine.sh` 的 NPU 变体），关键参数：

```bash
# 最小验证: 纯文本、减层、小 batch
MODEL_PATH=<tokenizer路径>
python3 -m verl.trainer.main_sft \
    data.train_files=<train.parquet> \
    data.val_files=<test.parquet> \
    data.train_batch_size=2 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp_turbo_dsv41 \
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=4 \
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tensor_parallel_size=2 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['console']
```

**通过标准**：actor 能 build 模型、FSDP wrapper 完成、backward 无错。
**若失败排查**：
- `Unknown backend: fsdp_turbo_dsv41` → 引擎注册未生效，检查两个 `__init__.py` 导入路径。
- 模块名不匹配（apply_modules 报错）→ 先去掉 apply_modules/recompute，或按日志修正为 `model.model.*`。
- tokenizer 兼容 → 确认 HF tokenizer 与 `DeepSeekV41Processor` 兼容（参考 FSDPTurbo 示例用 `PreTrainedTokenizerFast`）。

---

## Phase 2：rollout 侧验证（vllm-ascend 加载 V4.1）

### 2.1 vllm-ascend 离线加载 V4.1

```bash
python -c "
from vllm import LLM
# 用减层权重目录 (含 config.json + safetensors)
llm = LLM(model='<减层V4.1权重目录>',
          tensor_parallel_size=2,
          gpu_memory_utilization=0.45)
out = llm.generate(['Hello'])
print(out[0].outputs[0].text)
"
```

**通过标准**：能构图、生成输出。报错则进 vllm-ascend 侧查 config→模型类映射。

### 2.2 verl 权重同步一致性（可选但强烈建议）

```bash
# 用 1.3 产出的 SFT checkpoint 或随机初始化，跑一个最小 GRPO step
# 观察 actor→vllm 权重同步后 rollout 的 logprob 不再随 step 变化（已对齐）
```
可复用 verl 现有单测思路（对比同步前后 `input_ids` 对应的 `log_prob`）。

---

## Phase 3：GRPO 端到端（8 卡 NPU，纯文本）

### 3.1 准备数据
- 用 geo3k/gsm8k 风格 parquet（`prompt` + 参考答案），转成 verl RL 格式：
  ```json
  {"messages":[{"role":"user","content":"..."}], "images": null}
  ```
  （纯文本时 `images` 可省略）

### 3.2 运行启动脚本

```bash
cd verl
DEVICE=npu \
MODEL_PATH=<tokenizer> \
TRAIN_FILE=<train.parquet> \
TEST_FILE=<test.parquet> \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
```

### 3.3 通过标准
1. Ray 集群带 8×NPU 资源就绪。
2. Actor/Ref Engine 构建成功（日志出现 `FSDPTurboDSV41EngineWithLMHead`）。
3. Rollout 生成完成，reward 计算、GRPO loss 正常，无卡死/超时。
4. 权重同步后 rollout logprob 稳定。
5. `save_freq` 设为 >0 时 checkpoint 可保存/恢复。

### 3.4 常见问题
| 现象 | 排查 |
|---|---|
| HCCL 连接超时 | 确认 HCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME、800T 网卡、`HCCL_CONNECT_TIMEOUT` |
| Ray 找不到 NPU | `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` + `ASCEND_RT_VISIBLE_DEVICES` |
| rollout OOM | 降 `gpu_memory_utilization`、升 `max_num_batched_tokens` 调优 |
| FSDPTurbo module 名不匹配 | 日志打印实际 module 名，改 `FSDP_APPLY_MODULES`/`RECOMPUTE_PLAN` |
| engram/CSA2 报错 | 确认 `use_sparse_flash_attn=False`、`engram_meta_init=True`，且 FSDPTurbo 内置 config 是 demo 层数 |

---

## Phase 4（后续）：多模态 + V4.1 完整结构

- 多模态：数据加 `images` 字段，`data.image_key=images`，vllm-ascend `mm_processor_cache_gb=0`；验证 FSDPTurbo `vision.blocks.*` 并行。
- Engram/CSA2/稀疏索引：FSDPTurbo 训练侧能力开启（`use_sparse_flash_attn=True`、`engram_storage_backend`），vLLM eager 推理侧专项对齐（compressor/indexer/cache 布局）。
- 全层模型：先解决 FSDPTurbo 内置 config 只能出 4-layer 的问题（扩展 `build_deepseek_v41_model_args` 支持外部 config，或引入全层 config）。

---

## 附加：本次改动的静态自检（已在本机通过）

```bash
python -m py_compile verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py \
                  verl/workers/engine/fsdp/__init__.py \
                  verl/workers/engine/__init__.py
# 全部通过（无语法错误）
```

真机运行时再补：`EngineRegistry.get_engine_cls('language_model', 'fsdp_turbo_dsv41')` 冒烟。
