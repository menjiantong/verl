# DeepSeek-V4.1 训推一致性（train–inference consistency）专项工作记录

> 目标：解释并改善 GRPO 运行中 `rollout_actor_probs_pearson_corr` / `rollout_corr/kl` 等指标反映的「训推一致不够好」问题。
> 触发点：`wiki/work_log/DeepSeek-V4.1-Flash-4layer-scaled32-20260919_025432.log`（scaled32 权重，step1：`pearson 0.9793`、`kl 0.1108`、`rollout_probs_diff_mean 0.0103`）。
> 前置文档：[worklog_dsv41_rl.md](../worklog_dsv41_rl.md)（链路打通/加载/同步优化）。
> 日期：2026-09-20。所有实验脚本在 `scripts/dsv41_consistency/`，dump 产物在 `/mnt/share/m00899630/dsv41/dump/`。

---

## 0. 结论摘要（先看这里）

1. **权重没问题**：引擎（vLLM/vllm-ascend，从 checkpoint 直接加载）持有的参数与 checkpoint **逐张量一致**（672 个抽查参数中 664 个精确匹配，8 个未匹配项是我映射表未覆盖的 vision 塔参数）。
2. **差异从第 0 层注意力核内部产生**：同一输入下，embedding、第 0 层 `attn_norm`、第 0 层的 q 投影 **完全一致**（rel_err 0 / 1e-6），但第 0 层注意力**输出**差 0.34%（所有 token 均匀分布，非缩放差异）。
3. **随后被未训练模型逐层放大**：0.34% →（MoE）6.4% → 13% → 19% → 22%，最终 `model.norm` 相对误差 **14%**，导致逐 token logprob 噪声 **σ≈0.32（len64）/ 0.53（len200）nats、均值≈0、argmax 一致率 79%**。
4. **该噪声量级可以定量解释线上指标**：KL(π_rollout‖π_train) ≈ 0.5σ² ≈ 0.05–0.14，与实测 `kl 0.10–0.19` 同量级；`rollout_probs_diff_max 0.46–0.77` 对应极少数 argmax 翻转位置（两侧 top-1 不同 → 概率差可达 0.7）。**不是**权重错位/同步错误（那会是 O(1) 量级）。
5. **引擎自身 decode↔prefill 的一致性很好**（同 token 重算差 0.001–0.14 nats），不是主因；开启 `rl_config.enable_batch_invariant=true` 后可压到 ≤0.006 nats（末位一个 −0.10 离群），已写入启动脚本。
6. **两条"对齐内核"的路都被堵死**（详见 §4）：训练侧 eager 注意力改 fp32 **没有**收益（端到端噪声反而略升）；训练侧启用融合 SparseFlashMla 因 **A2/A3 硬件/布局限制**不可行（`cmp_mask_mode`/`cmp_ratio` 约束）。
7. **head 尺度放大了这一切**：该随机 checkpoint 用 `--head-gain 4`。用实测 Δh 重算可知：gain 4 → σ(Δlogp)=0.56、KL≈0.157；gain 1（正常量级）→ σ=0.14、KL≈0.010。即**当前"看起来糟"的 kl 有约 16 倍来自刻意的 head 增益**（§3.3）。
8. 曾怀疑的 **label roll 跨样本回绕**在真实运行中**未生效**（逐位置实测末位 Δ 正常，§5），故本次指标与它无关。
9. **真实运行逐位置实测**（`VERL_DSV41_DUMP_BATCH`）：Δ(trainer−rollout) 均值 **−0.1127**（与 `kl=+0.1127` 精确对应）、std 0.54、max 7.2 nats；\>2 nats 的 token 占 1.5%，位置分散无聚集（§6.2）。
10. **对照实验**：开启 `rl_config.enable_batch_invariant=true` 后跑 5 个 step，`pearson/kl` 与历史运行同区间波动（无显著变化）——印证主项不是引擎内部批组成差异（§6.1）。

> 一句话：**这是两套内核数值近似的固有差异（种子在注意力，放大在未训练模型 + head-gain 4），不是链路 bug。** 想进一步逼近必须让训练侧用上与引擎相同的算子（受 A2/A3 布局限制，见 §4.2），否则只能在 RL 侧用 IS/裁剪吸收这一量级。

---

## 1. 方法与工具（本次新建）

| 工具 | 作用 |
|---|---|
| `scripts/dsv41_consistency/make_fixture.py` | 生成**公共输入夹具**：从 DAPO 数据取真实 prompt、按 chat template 渲染、拼成 64/200/700/1500 token 四档（超过 `sliding_window=128`、`index_topk=512`、压缩边界），保证两侧喂**完全相同的 token id** |
| `scripts/dsv41_consistency/dump_trainer_stages.py` | 训练侧（FSDPTurbo，8 卡 EP=8）：构建/加载路径与真实引擎一致，挂钩子导出**每层激活**（embed / 每层 Block / attn / attn_norm / ffn / gate / experts / shared_experts / indexer / compressor）、`indexed_sparse_attention` 算子的**输入输出**（q/kv/sink/topk_indices）、两种 logprob 约定、top-16 |
| `scripts/dsv41_consistency/dump_engine_stages.py` + `run_dump_engine.sh` | 引擎侧（vLLM+EP=8，**只加载 checkpoint，不经过 verl 同步**）：参数指纹 + `prompt_logprobs` + 挂钩子导出每层激活 + **decode-vs-prefill 自一致性**测量 |
| `scripts/dsv41_consistency/dsv41_dump_extension.py` | vLLM **worker extension**（`worker_extension_cls` 注入）：`dsv41_param_stats`（参数指纹）、`dsv41_hook_begin/end`（模块钩子 + 猴子补丁：eager 注意力实现、`DeepseekV41DecoderLayer.hc_pre/hc_post/rms_norm_cast`、Ascend MoE router） |
| `scripts/dsv41_consistency/check_engine_params.py` | **权重级校验**：把引擎参数（含 TP/EP 分片、NZ/转置等布局变换）映射回 checkpoint 张量并比较 `std/absmax/mean` |
| `scripts/dsv41_consistency/compare_stages.py` / `compare_attention.py` / `compare_logprobs.py` / `compare_routing.py` | 训练侧 vs 引擎侧逐阶段/逐算子/logprob/路由对比 |

**与真实 RL 的对应关系**：夹具走**同一段 token**、同一权重、同一拓扑（TP=8/EP=8），因此可以直接对比「训练侧重算」与「引擎侧前向」。唯一差别是 RL 用的 rollout logprob 来自**decode**（我在引擎侧另做了 decode-vs-prefill 自一致性测量来覆盖这一点，见 §3.4）。

---

## 2. 权重级校验（先排除"同步/加载错误"）

`check_engine_params.py` 把引擎侧 8 个 rank 的参数逐个映射回 checkpoint：

- 覆盖：`embed/head`（TP 切分）、`wq_a/wq_b/wkv/wo_a/wo_b`（含 `wo_a` 的 `[groups, o_lora, in]` 转置布局）、compressor/indexer、`mlp.gate.*`、shared experts（w1/w3 融合）、**routed experts 的 `w13_weight`/`w2_weight`（EP 分片 + Ascend 转置布局）**、`hc_*`。
- 结果：**664/672 匹配**；8 个 "FAIL" 全部是 `vision.norm.weight`（我的映射表没写 vision 塔，非引擎问题）。
- 结论：**引擎确实持有与 checkpoint 相同的权重**。所以线上指标里的 0.1 nats 不可能来自"权重没同步对"。（权重错位/专家错绑的失败模式是 O(1) 量级，与实测不符。）

---

## 3. 逐阶段数值对比（核心证据）

### 3.1 每层激活的相对误差（len=64，训练侧 bf16 eager 为基准）

| 阶段 | rel_err | cos |
|---|---|---|
| `model.embed` | **0.0000** | 1.0000 |
| `layers.0.attn_norm` | **0.0000** | 1.0000 |
| `layers.0.attn`（注意力输出，含 wo_a/wo_b） | **0.0057** | 0.9999 |
| `layers.0.ffn`（MoE 输出） | 0.0644 | 0.9979 |
| `layers.1.attn` / `.ffn` | 0.0445 / 0.1311 | 0.9990 / 0.9914 |
| `layers.2.attn` / `.ffn` | 0.0668 / 0.1870 | 0.9978 / 0.9825 |
| `layers.3.attn` / `.ffn` | 0.0865 / 0.2168 | 0.9962 / 0.9765 |
| `model.norm` | **0.1434** | 0.9897 |

即：**分歧从第 0 层注意力开始，之后每块约 ×1.5–2 累积**（未训练模型对这种误差没有"结构性抗性"）。

### 3.2 注意力算子级定位（len=64）

- 第 0 层 `q`（训练侧 [64 heads] 的 0–7 头 vs 引擎侧 rank0 的 8 头）：**rel_err = 1e-6**，即投影 + RoPE 两侧一致。
- 同一输入进 `indexed_sparse_attention`：
  - 训练侧（eager torch）：QK 用 **bf16** einsum 后 `.float()`，softmax 后概率 **cast 回 bf16** 再做 PV；
  - 引擎侧（`npu_sparse_flash_mla` 融合核，`dsa_v41.py::_native_attention`）：核内高精度、分页 KV。
  - 输出差异：**rel_err 0.00339**，且**逐 token 高度均匀**（p10 0.56% / p50 0.58% / p90 0.59%），不是少数翻转 → 指向**算子内部数值**而非选择错误。
- 顺带确认：两侧的窗口/压缩选择语义一致（`sliding_window=128` ↔ 引擎 `ori_win_left=127`；`index_topk=512`；层 2 产出压缩 KV、层 3 复用）。

### 3.3 logprob 端到端统计（引擎 prefill vs 训练侧，正确对齐后）

| | len=64 | len=200 |
|---|---|---|
| mean Δ(engine−trainer) | −0.0352 | +0.0105 |
| std Δ | 0.3207 | 0.5293 |
| median \|Δ\| | 0.1081 | 0.1183 |
| p99 \|Δ\| | 0.9758 | 1.9451 |
| argmax 一致率 | 79.4% | 78.8% |

**与线上指标的关系**：KL(π_rollout‖π_train) ≈ E[Δlogp] + 0.5·Var(Δ) ≈ 0.5σ² → σ≈0.32/0.53 给出 **0.05–0.14 nats**，正好覆盖线上 `kl 0.10–0.19`。`rollout_probs_diff_max` 0.46–0.77 则由 21% 的 argmax 翻转位置贡献（两侧 top-1 不同时，概率差可达 0.7）。

> 注意：`kl` 用**采样自 rollout 的 token** 做期望，数学上恒 ≥0（是 KL 的无偏估计），所以"均值恒正"**不是**同步出错的证据。早期我按"系统性偏差"去找原因，属于误判，此处更正。

**放大机制的定量验证**（把实测的隐状态差 Δh 直接过 head 权重，预测 logprob 差）：

| | len=64 | len=200 |
|---|---|---|
| 实测 Δlogp：mean / std | −0.0350 / 0.3181 | +0.0134 / 0.5596 |
| 预测 Δlogp（Δh→head，`log_softmax(z_e)−log_softmax(z_t)`） | −0.0355 / 0.3183 | +0.0138 / 0.5596 |
| 相关系数 | **0.999** | **1.000** |
| `‖Δh‖/‖h‖` | 0.1434 | 0.1294 |

即：**训练-推理之间 logprob 的全部差异，都是最终隐状态差异（13–14%）经 head 传播的结果**，logprob 路径本身没有额外的坑。

**head 尺度的影响（对真实权重的预估）**：把 head 权重按比例缩到 gain 2 / gain 1，用同一份实测 Δh 重算：

| head gain | σ(Δlogp) | KL ≈ 0.5σ² |
|---|---|---|
| 4（当前 scaled32 权重） | 0.5596 | **0.157** |
| 2 | 0.2846 | 0.041 |
| 1（正常量级模型） | 0.1423 | **0.010** |

→ 当前"看起来很糟"的 kl 0.10–0.19 有相当一部分是**该随机 checkpoint 刻意用的 `--head-gain 4`** 把 logit 噪声放大了 4 倍（KL 放大 ~16 倍）。同等的逐层数值差在正常 head 尺度下预期 kl ≈ 0.01（仍需真实权重验证）。

### 3.4 引擎自身 decode ↔ prefill（同一 token 重算）

对同一序列先生成 8 个 token，再把 prompt+生成 整体作为 prompt 重新打分：

| 配置 | 8 个 token 的 Δ（decode − prefill） |
|---|---|
| 默认（`enable_batch_invariant=false`） | +0.0144, +0.0684, +0.0012, +0.0915, +0.1355, −0.0092, +0.0028, +0.0975 |
| `enable_batch_invariant=true` | **0.0, +0.0013, +0.0062, +0.0003, +0.0046, +0.0062, −0.0049, −0.1002** |

→ 引擎内部模式差异约 0.05–0.14 nats，开启 batch-invariant 后除末位外全部 ≤0.006。（末位 −0.10 疑似 KV cache 末位/采样相关，待查。）

---

## 4. 被排除/失败的修复路线（都做了实验）

### 4.1 训练侧注意力改 fp32（负结果）

在 `dump_trainer_stages.py --fp32-attention` 里把 eager 注意力的 QK/PV 从 bf16 改成 fp32：

| | bf16 eager | fp32 eager |
|---|---|---|
| 层 0 注意力输出 rel_err | 0.00339 | **0.00191** |
| `model.norm` rel_err | 0.1434 | 0.1469 |
| logprob std Δ | 0.3207 | 0.4352 |
| argmax 一致率 | 79.4% | 81.0% |

→ 只把"种子"减半，**端到端没有收益**。说明差异是**分布式**的：引擎融合核（attention/MoE/HC/norm-cast 各自的近似与累加顺序）与训练侧 eager 实现之间处处有 ~0.1–0.5% 的数值差，单独改一处不可能对齐。

### 4.2 训练侧启用融合 SparseFlashMla（被硬件/布局限制堵死）

`--sparse-flash-attn` 的两次尝试与报错：

1. `ModuleNotFoundError: No module named 'cann_ops_transformer'` —— 根因：**`torch_npu` 导入后该包不可再导入**（实测先导入再导入 torch_npu 就失败）；且我的运行脚本用 `export PYTHONPATH=...` **覆盖**了基础环境里含 CANN site-packages 的 PYTHONPATH。→ 修复：脚本改为追加 `${PYTHONPATH:-}`，并在 `torch_npu` 之前预导入 `cann_ops_transformer`。
2. `Failed to execute tiling function: cmpMaskMode should be 3 on A2/A3, but got 0` —— FSDPTurbo 对无压缩 KV 的层（`compress_ratio=0`，即 0/1 层）传 `cmp_mask_mode=0`。
3. 强行传 3 后：`cmp_ratio should be 4 on A2/A3 when cmp_topk is non-zero (CSA with cmp_sparse_indices), but got 2`。

→ **结论**：A2/A3 上**稠密 BSND 布局**的 SparseFlashMla 要求 `cmp_ratio=4`，而本模型是 `compress_ratios=[0,0,2,2]`；引擎能用是因为它走**分页布局（PA_BBND）**。因此"训练侧复用引擎注意力核"在当前实现下不可行（要做得让 FSDPTurbo 支持分页布局，工程量远超本次范围）。**这是已知硬阻塞，记录在此避免重复踩坑。**

### 4.3 其它已排除项

- **权重同步错位**：见 §2（权重级校验通过）。
- **引擎 batch 组成依赖**：见 §3.4（量级远小于主因，且已开启 batch-invariant 进一步压低）。
- **label roll 工件**：见 §5（真实存在但不是主因，符号随机）。

---

## 5. 训练侧 logprob 标签构造的核查（曾怀疑 roll 工件，实测未生效）

`verl/workers/engine/fsdp/transformer_impl.py:1208` 用整条平铺流做 label 位移：

```python
input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
```

`input_ids_rmpad` 是 micro-batch 的 token 平铺流；若它承载多样本或 padding，`roll(-1)` 会让每个样本**最后一个位置**的标签取到「下一个样本的首 token / padding token」。用夹具两端 logprob 估算，**若发生**该回绕，该位置误差为 −4.3～−12.6 nats（符号随机），对序列均值贡献 ±0.03–0.10 nats。

**但本轮真实运行实测并未出现**（`VERL_DSV41_DUMP_BATCH` 落盘，step0，16 序列 × 128 token，response_mask 右对齐且满长）：

| 位置 | Δ(trainer−rollout) mean | std | max \|Δ\| |
|---|---|---|---|
| 最后 1 个响应 token | +0.021（step0）/ −0.008（step1） | 0.278 / 0.098 | 3.02 |
| 其余响应 token | −0.114 / −0.117 | 0.538 / 0.576 | 7.20 |

即当前 `use_remove_padding=False` 的 RL 路径下末位无异常。**结论：该 roll 逻辑未影响本次指标**（很可能是 padded 路径另走位移构造）；仅当切到 rmpad 路径时需要重新核查。

---

## 6. 已实施的修复与验证

| # | 修改 | 文件 | 状态 |
|---|---|---|---|
| 1 | 引擎侧开启 `rl_config.enable_batch_invariant=true`：消除引擎自身的批组成依赖（decode vs prefill 实测 ≤0.006 nats） | `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | ✅ 已改；**验证中**（见 §6.1） |
| 2 | 运行脚本 PYTHONPATH 改为**追加**基础环境（否则 CANN 的 python site-packages 丢失，任何 `cann_ops_transformer` 路径都不可用）；`dump_trainer_stages.py` 在 `torch_npu` 之前预导入 `cann_ops_transformer` | `scripts/dsv41_consistency/*` | ✅（本轮踩坑修复） |
| 3 | 新增 `VERL_DSV41_DUMP_BATCH=<dir>`：在 debug 指标处落盘 `rollout_log_probs/old_log_probs/responses/response_mask/...`，用于离线逐位置分析 | `verl/utils/debug/metrics.py` | ✅（env 门控，默认关闭） |
| 4 | 一致性对比工具链（夹具/双侧 dump/worker extension/4 个 compare 脚本） | `scripts/dsv41_consistency/` | ✅ |

### 6.1 真实 GRPO 验证（batch-invariant 开关对照）

配置与历史运行对齐（scaled32 / `TRAIN_BSZ=8` / `n=2` / `MAX_RESPONSE_LEN=128` / `LOCAL_EXPERT_EXPORT=1`）。

| 指标（step1） | 历史（2026-09-19 02:54，无 batch-invariant） | 本次（开 batch-invariant） |
|---|---|---|
| `training/rollout_probs_diff_mean` | 0.01032 | 0.01308 |
| `training/rollout_actor_probs_pearson_corr` | 0.9793 | 0.9662 |
| `rollout_corr/kl` | 0.1108 | 0.1127 |
| `rollout_corr/log_ppl_diff_min` | 0.0466 | 0.0483（同量级） |
| `timing_s/step` | 74.1s | 62.5s（step5） |

本次运行 5 个 step（开关生效）对照历史 6 个 step：

| step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pearson（本次） | 0.9662 | 0.9654 | 0.9620 | 0.9591 | 0.9718 | — |
| pearson（历史） | 0.9793 | 0.9779 | 0.9676 | 0.9506 | 0.9705 | 0.9738 |
| kl（本次） | 0.1127 | 0.1158 | 0.1151 | 0.1015 | 0.1214 | — |
| kl（历史） | 0.1108 | 0.1068 | 0.1010 | 0.1906 | 0.1150 | 0.1130 |

**结论：无明显变化**（两条曲线重叠，逐 step 波动同量级）。与预期一致——batch-invariant 只消除「引擎自身模式差异」这一小项（≈0.05 nats），而逐 token 噪声主项（σ≈0.5）来自 §3–§4 的算子差异。**该开关保留**（无副作用、消除一类不确定性），但**不能当作解药**。

### 6.2 真实运行逐位置分析（`VERL_DSV41_DUMP_BATCH` 落盘，step0/1）

| 量 | step0 | step1 |
|---|---|---|
| 响应 token 数 | 2048（16×128） | 2048 |
| Δ(trainer−rollout) mean | **−0.1127** | −0.1158 |
| Δ std / med\|Δ\| | 0.5366 / 0.1026 | 0.5741 / 0.1074 |
| Δ p99 / max | 2.48 / **7.20** | 2.79 / 5.71 |
| \|Δ\|>2 nats 的 token | 31/2048（1.5%） | — |
| 逐序列均值 Δ 范围 | −0.029 ～ −0.145 | −0.042 ～ −0.180 |

- **均值 −0.11 与 `rollout_corr/kl = +0.1127` 精确对应**（kl 定义为 E[rollout−trainer]，且期望取自 rollout 采样，故必然 ≥0；这里等于 0.11 即噪声驱动）。
- 大偏差（\|Δ\|>2，最多 7.2 nats）**分散在各位置**（相对位置均值 0.66，无末位聚集），是 argmax 翻转类事件的尾部。
- 与夹具结论一致：**逐 token 噪声 σ≈0.5 nats 是全部差距的来源**。

---

## 7. 遗留问题与建议

1. **注意力算子的数值对齐**（最大的单项）：训练侧 eager torch（bf16 分数/bf16 概率）vs 引擎融合核（分页、核内高精度）。可行的下一步：
   - 让 FSDPTurbo 的 SparseFlashMla 支持**分页 KV（PA_BBND）**布局 —— 这样 `cmp_ratio=2` 等约束可满足（当前稠密 BSND 在 A2/A3 上要求 `cmp_ratio=4`）；
   - 或在训练侧把 eager 实现改成**与引擎核完全一致的数学**（需要引擎核的 tiling/累加细节，代价大）。
2. **MoE/HC/norm-cast 的数值对齐**：即使注意力对齐，MoE（`npu_grouped_matmul` vs 引擎 gmm/量化路径）、HC（`npu_hc_pre_v2` 融合核 vs eager sinkhorn）、`rms_norm_cast` 仍贡献同量级差异，需要一并评估。
3. **label roll 工件**（§5）：建议单独修复。
4. **随机权重放大了结论的悲观程度**：本 checkpoint 是未训练模型 + `head-gain 4`，对隐状态误差没有结构性抗性（14% 隐状态误差 → 0.5 nats）。真实训练权重下同等 0.1–0.2% 的逐层数值差预计产生**更小**的 logprob 差异。建议拿到真实权重后重跑本工具链（`MODEL_PATH=` 一行切换）再定阈值。
5. **RL 侧吸收**：`rollout_correction`（IS/TIS）已是 verl 现成能力，0.1 nats 级别的不一致在 GRPO 里通常是可接受并用裁剪/IS 吸收的；本工作把它的来源定量化，便于决定是否需要更强的纠正。

---

## 8. 复现命令

```bash
# 0) 夹具（无需 NPU）
cd /workspace-verl/verl && python3 scripts/dsv41_consistency/make_fixture.py

# 1) 训练侧逐阶段 dump（8 卡）
bash /tmp/run_dump_trainer.sh                # 等价：torchrun --nproc_per_node=8 scripts/dsv41_consistency/dump_trainer_stages.py \
                                             #   --model-path .../DeepSeek-V4.1-Flash-4layer-scaled32 --out-dir .../dump/trainer
#   变体：--fp32-attention（fp32 注意力 A/B）、--sparse-flash-attn（融合核，受 §4.2 限制）

# 2) 引擎侧 dump（含参数指纹/阶段/decode-vs-prefill），一次引擎启动全出
bash scripts/dsv41_consistency/run_dump_engine.sh --lengths 64,200
bash scripts/dsv41_consistency/run_dump_engine.sh --lengths 200 --skip-params --batch-invariant   # A/B

# 3) 对比（离线）
python3 scripts/dsv41_consistency/check_engine_params.py
python3 scripts/dsv41_consistency/compare_stages.py    --len 64
python3 scripts/dsv41_consistency/compare_attention.py --len 64
python3 scripts/dsv41_consistency/compare_logprobs.py  --len 64        # --convention next|roll
python3 scripts/dsv41_consistency/compare_routing.py   --len 64

# 4) 真实 GRPO（开 batch-invariant + batch 落盘）
cat /tmp/run_grpo_bi.sh
```

**环境要点（本轮踩到的坑）**：
- 运行脚本**不要覆盖** `PYTHONPATH`：基础环境里的 `/usr/local/Ascend/cann-9.1.0/python/site-packages` 是 `cann_ops_transformer` 的唯一来源。
- `HCCL_HOST_SOCKET_PORT_RANGE=60000-60050` / `HCCL_NPU_SOCKET_PORT_RANGE=61000-61050` 必须显式设置，否则与机器上其它任务争默认端口（`Bind_IP_Port ... port 16666 already bound`）。
- `vLLM` 离线 `LLM()` 需要 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（fork 子进程里 `torch.set_num_threads` 会崩：`pool INTERNAL ASSERT FAILED ... Invalid thread pool`）。
- 后台任务用 `setsid` 启动，避免前台命令超时被杀时连带 SIGTERM。
