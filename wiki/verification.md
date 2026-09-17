# 验证步骤 — DeepSeek-V4.1 RL（verl + FSDPTurbo + vllm-ascend）

> 配套：[改动盘点](README.md) ｜ [方案](verl_fsdp_turbo_vllm_ascend_rl_plan.md) ｜ [工作记录](worklog_dsv41_rl.md)
> 模型：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random`（8 层 / 384 experts / 无 engram，209GB）

---

## Phase 0：环境（已就绪，本地实测）

| 组件 | 本地路径 | 解析结果 |
|---|---|---|
| verl | `/workspace-verl/verl` | editable ✓ |
| vllm | `/workspace-verl/vllm` | editable（v0.28.1rc1.dev570+ga97dacb71）✓ |
| vllm-ascend | `/workspace-verl/vllm-ascend-v41-private` | editable（含 V4.1 eager 定制）✓ |
| FSDPTurbo | `/workspace-verl/FSDPTurbo` | editable（fsdp-turbo 0.1.0）✓ |
| transformers | 5.10.4 | **不认 `deepseek_v4.1`** → 走 vLLM config fallback（已实现） |

```bash
python3 -c "import verl, vllm, vllm_ascend, fsdp_turbo, torch_npu; print('ok')"
# 建议显式指定，避免同名副本歧义：
# PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo
```

---

## Phase 1：离线自检（无需 NPU，**已通过**）

一条命令跑完全部离线检查（5 项，实测 `all offline checks passed`）：

```bash
cd /workspace-verl/verl
python3 scripts/check_dsv41_offline.py
# [ OK ] engine registration — FSDPTurboDSV41EngineWithLMHead
# [ OK ] hf config (vLLM registry fallback) — DeepseekV41Config, tokenizer vocab=128000
# [ OK ] external config -> ModelArgs — n_layers=8, experts=384, engram=(), dspark=0
# [ OK ] checkpoint names + expert fusion — 194 model params: 178 direct + expert-fused, 274 unused (vision); byte-exact
# [ OK ] build -> prepare (meta experts) — 16 experts deferred to meta, 14 buffers real
```

以下为等价的分步命令（排查时用）：

```bash
cd /workspace-verl/verl

# 1. 引擎注册
python3 -c "from verl.workers.engine import EngineRegistry; print(EngineRegistry.get_engine_cls('language_model','fsdp_turbo_dsv41'))"
# → FSDPTurboDSV41EngineWithLMHead

# 2. 模型 config 可被 verl 读取
python3 -c "from verl.workers.config.model import HFModelConfig; c=HFModelConfig(path='/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random', trust_remote_code=True); print(type(c.hf_config).__name__, c.architectures)"
# → DeepseekV41Config ['DeepseekV41ForCausalLM']

# 3. 外部 config → ModelArgs
python3 -c "from fsdp_turbo.models.deepseek_v41 import build_deepseek_v41_model_args as f; a=f('/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random', max_seq_len=4096); print(a.n_layers, a.n_routed_experts, a.engram_layer_ids, a.compress_ratios)"
# → 8 384 () (0, 0, 2, 2, 2, 2, 2, 2)

# 4. HTTP 语法/配置组装
env DEVICE=npu bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh --cfg job | tail -3
# → exit 0，composed 配置含 strategy=fsdp_turbo_dsv41 / EP=8 / additional_config.*

# 5. FSDPTurbo 单测（本次改动回归）
python3 -m pytest /workspace-verl/FSDPTurbo/tests/unit_tests/models/test_deepseek_v41.py -q
# → 18 passed（test_engram_parallel.py 的 3 个失败为改动前既有）
```

另已验证（脚本内含）：checkpoint 名称映射 194/194 命中、专家融合 `[w1 ; w3]`/`w2` 逐字节一致、hydra 覆盖全部生效。

---

## Phase 2：真机 8 卡冒烟（**待 NPU 空闲**）

```bash
cd /workspace-verl/verl
PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
HCCL_CONNECT_TIMEOUT=1500 HCCL_HOST_SOCKET_PORT_RANGE=60000-60050 HCCL_NPU_SOCKET_PORT_RANGE=61000-61050 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TASK_QUEUE_ENABLE=1 \
torchrun --nproc_per_node=8 --master_port=59511 scripts/check_dsv41_fsdp_turbo_build.py --forward
```

**通过标准**
1. `building model (routed experts deferred to meta)` → `model built ... (194 parameters)`
2. `wrapping with FSDPTurbo` 完成
3. `parameters materialized in <N>s`（读盘 209GB，预计数分钟）
4. `no meta parameters left`
5. 每 rank 的显存/设备报告：专家参数在加速器上（EP 分片），稠密参数按 offload 策略
6. `--forward` 时打印 `forward ok: logits (1, 8, 128000)`

**排查**
| 现象 | 方向 |
|---|---|
| 设备初始化失败（内存不足） | NPU 被占用；`npu-smi info` 看空闲 |
| HCCL 超时 | `HCCL_CONNECT_TIMEOUT` / socket 端口范围 / 网卡 |
| `parameters left on meta` | checkpoint 缺张量 → 看 rank0 的 missing 列表（引擎会直接报错） |
| `Checkpoint ... has shape ... expects ...` | config 与 checkpoint 不匹配（换成对应层数的权重目录即可） |
| EP 派发异常/结果乱 | 确认 `refresh_dsv41_expert_metadata` 已执行（构建日志无 `no meta parameters left` 之后的报错） |

---

## Phase 3：GRPO 端到端（**待 Phase 2 通过**）

```bash
cd /workspace-verl/verl
DEVICE=npu PYTHONPATH=/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo \
bash examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
```

可调：`TRAIN_BSZ`（默认 8）、`SAMPLE_N`（默认 5）、`MAX_PROMPT_LEN`/`MAX_RESPONSE_LEN`（默认 1024/1024，同时决定 `VERL_DSV41_MAX_SEQ_LEN`）、`ROLLOUT_GPU_MEM_UTIL`（默认 0.6）、`EP_SIZE`/`EFSDP_SIZE`。

**通过标准**
1. Ray 起 8 卡资源，actor/ref 两个引擎都完成模型构建（日志出现 checkpoint 加载条数与耗时）。
2. rollout 生成完成（vLLM 加载 8 层权重 + `additional_config` 生效）。
3. `update_weights` 后 rollout 与 actor 的 logprob 对齐（不随 step 漂移）。
4. GRPO loss/kl/reward 正常打印，无卡死。
5. `trainer.save_freq>0` 时 checkpoint 可存/可取。

**已知调优点**：显存（专家分片 27GB/卡 + 权重同步时 18GB 单张量峰值）、`use_remove_padding=False` 的吞吐代价、V4.1 注意力在 NPU 上走 eager（非 fused DSA）。

---

## Phase 4（后续）：多模态 / 完整结构 / 全层

- 多模态：构建时 `include_vision=True`（`adapter.py` 已支持 `vision_config` 映射），数据加 `images`，`data.image_key=images`。
- V4.1 特性：`use_sparse_flash_attn=True`（NPU `SparseFlashMla`）、Engram（`engram_meta_init=True` + 存储后端）。
- 全层权重：换 `MODEL_PATH` 即可（结构随 config 走），需重新评估显存与 EP/EFSDP 拓扑。

---

## 附：本次新增的诊断/冒烟脚本

- `scripts/check_dsv41_fsdp_turbo_build.py`：绕开 Ray/verl trainer 的三步验证（构建 / 分片加载 / 前向）+ 每类参数的显存与设备分布报告；`--random-init` 可完全跳过 209GB 读盘。
- 环境变量：`VERL_DSV41_MAX_SEQ_LEN`（RoPE/scratch 尺寸）、`VERL_DSV41_RANDOM_INIT=1`（跳过 checkpoint）。
