# verl × FSDPTurbo × vllm-ascend — DeepSeek-V4.1 RL 实施改动盘点

> 状态：**代码已实施 + 离线验证通过；真机 8 卡验证待 NPU 空闲**（2026-09-17）
> 工作记录（问题/修复/验证）：[worklog_dsv41_rl.md](worklog_dsv41_rl.md)
> 方案：[verl_fsdp_turbo_vllm_ascend_rl_plan.md](verl_fsdp_turbo_vllm_ascend_rl_plan.md) ｜ 验证步骤：[verification.md](verification.md)

---

## 一、本地基准（以本地目录为准）

- 训练参照：`FSDPTurbo/examples/deepseek_v41/run.sh` + `config2.yaml`（FSDP=8 / TP=1 / EP=8，`apply_modules: model.layers.{*}`，无 recompute）
- 推理参照：`scripts/run_vllm2.sh`（`vllm serve /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-8layer-random`，TP=8 + EP，eager，max-model-len 8192）
- 模型：8 层 / 384 experts / 无 engram 的减层权重，209GB（~111.4B 参数）

---

## 二、改动总览

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py` | **重写** | 外部 config + meta 专家构造 + 分布式 checkpoint 加载 + 前向入参适配（核心） |
| 2 | `verl/workers/engine/fsdp/utils.py` | 修改 | `unfuse_moe_params` 增加 `.ffn.experts.*` → per-expert `w1/w3/w2` |
| 3 | `verl/utils/model.py` | 修改 | 新增 `get_auto_config_with_vllm_fallback()`（支持 `deepseek_v4` / `deepseek_v4.1`） |
| 4 | `verl/workers/config/model.py` | 修改 | `hf_config` 构造改走上述 fallback |
| 5 | `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | **重写** | 8 层权重 + 8 卡拓扑 + 修 hydra 语法 |
| 6 | `scripts/check_dsv41_offline.py` | **新增** | 无需 NPU 的离线自检（5 项，当前全通过） |
| 6b | `scripts/check_dsv41_fsdp_turbo_build.py` | **新增** | 不依赖 Ray 的 8 卡构建/加载/前向冒烟 + 显存报告 |
| 7 | `FSDPTurbo/fsdp_turbo/models/deepseek_v41/adapter.py` | 修改 | `config_path` / `include_vision` / `max_seq_len` / `experts_meta_init` 参数（默认行为不变） |
| 8 | `FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py` | 修改 | `ModelArgs.experts_meta_init`；`MoE` 据此把专家建到 meta |
| 9 | `FSDPTurbo/fsdp_turbo/models/deepseek_v41/experts.py` | 修改 | 支持 `device=` |
| 10 | `FSDPTurbo/fsdp_turbo/distributed/expert_parallel/expert_parallel.py` | 修改 | 新增 `refresh_expert_parallel_metadata()` |
| 11 | `wiki/worklog_dsv41_rl.md` | **新增** | 本次工作记录（问题/修复/验证） |
| 12 | `wiki/verification.md` | 更新 | 补离线已验证项与真机待办命令 |

数据（已就绪）：`/mnt/share/m00899630/dsv41/rl_data/dapo_math_{train_512,val_64}.parquet`（由 `dapo-math-17k.parquet` 切分，reward 走 verl 内置 `math_dapo`）。

---

## 三、核心实现要点

### 3.1 结构来自 rollout checkpoint

```python
build_deepseek_v41_model(
    tokenizer=..., engram_meta_init=True, use_sparse_flash_attn=False,
    experts_meta_init=True,
    config_path=<MODEL_PATH>/config.json,          # 8 层 / 384 专家 / 无 engram
    max_seq_len=int(os.environ.get("VERL_DSV41_MAX_SEQ_LEN", 8192)),
)
```
`build_deepseek_v41_model_args()` 不传 `config_path` 时仍读内置 4 层 demo，**旧行为不受影响**。

### 3.2 108B 专家参数：meta 构造 + 分片加载

- 专家张量 meta 构造（`experts_meta_init=True`），其余参数与 **全部 buffer 真实构造**（避开整模型 meta 会毁掉 `freqs_cis` 等确定性 buffer 的坑）。
- `FSDPTurbo(config, module)` 分片后，rank0 读 safetensors 并融合专家（`w1|w3 → gate_up_proj`、`w2 → down_proj`），
  `set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=offload_policy)` 逐 tensor 广播，各 rank 只保留自己的分片。
- 加载后：buffer 落设备 → 重建 `expert_ids_per_ep_rank` → 断言无 meta 残留。

### 3.3 权重同步（actor → vLLM）

- 名字链路：`module.state_dict()` → `convert_weight_keys`（本环境 no-op）→ `full_tensor()` → **`unfuse_moe_params`（新增 `.ffn.experts.*` 分支）** → bucketed 传输 → `vllm load_weights`。
- vllm-ascend V4.1 `load_weights` 期望 **checkpoint 风格名**：`layers.N.ffn.experts.E.{w1,w3,w2}.weight`（可带 `model.` 前缀），内部再映射到 `...mlp.experts.routed_experts.w13_weight`。

### 3.4 前向接口

| 引擎传入 | 处理 |
|---|---|
| `position_ids` | 剥离（参考模型按绝对位置索引 RoPE，不接受该入参） |
| `attention_mask`（int32，1=有效） | 直接用（`.bool()` 后 True=attend） |
| packed / remove-padding 输入 | **显式报错**（脚本设 `use_remove_padding=False`，走 padded 路径） |

---

## 四、未改动但需知晓

1. vllm / vllm-ascend / 数据 / reward **零改动**。
2. 拓扑与 `config2.yaml` 对齐：`FSDP=8, TP=1, EP=8, EFSDP=1, ulysses=1`；`cast_forward_inputs=False`、`output_dtype=null`（保留 HC 的 FP32 系数）。
3. V4.1 适配器不支持 recompute → 脚本不设 recompute plan。
4. 依赖解析：`verl` / `vllm` / `vllm_ascend` / `fsdp_turbo` 均指向 `/workspace-verl/` 本地副本；机器上有多份同名副本，运行脚本应显式设置 `PYTHONPATH`。
