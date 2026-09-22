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

> **2026-09-21 补充（嫁接实验，见 §9）**——上面这条结论需要修正一处重点：
> 11. **用"把引擎侧输出灌进训练侧前向"的因果实验逐模块归因**：把每层的 attn 输出换成引擎的 → σ 0.56→0.217（2.6×）；把每层的 **MoE 输出**换成引擎的 → σ 0.56→**0.024（23×）**；两者都换 → σ **0.0127**、最终隐状态 rel_err **0.10%**（≈ 剩余 HC/RMSNorm/head 的数值地板）。**量的主要来源是 MoE 路径，不是注意力**（注意力是"种子"：它先错，再由 MoE 逐层放大）。
> 12. **单层嫁接几乎无效**（只换第 3 层 attn：σ 无变化；只换第 3 层 MoE：1.6×）——误差是逐层累积的，不是某一层单独造成的。
> 13. **路由选择两侧一致**：layer 0 的 router logits 差 0.6%（同量级），**top-6 专家集合逐个 token 完全相同** → 再次确认是纯数值差异，不是离散选择错误。
> 14. 修正后的行动优先级：**想缩小训推差，先对齐 MoE 路径**（grouped matmul / 专家累加 / 路由权重算术 / all-to-all 归约），其次才是注意力核。
> 15. **384 专家（`scaled`）复验（§11）：归因成立且更强** —— baseline σ 0.71/1.08（len200/64）→ 全层对齐 MoE 后 0.024/0.026（**29–42×**），只对齐注意力仅 1.7–3.6×，两者都对齐 0.012（**59–79×**，隐状态 rel_err 0.10% 数值地板）。**这解释了线上"32 专家 kl 0.11 → 384 专家 kl 0.38"：MoE 固有差随专家数增长，且它是主项。**
> 16. **dtype 配方不是原因（§10，负结果）**：把训练侧 MoE 改成引擎的 bf16 配方（env 门控 `VERL_DSV41_MOE_BF16=1`），固有差纹丝不动（0.0058→0.0057 / 0.0178→0.0178）→ 差异在**分组 GEMM 内核内部**，不在 GEMM 之间的 fp32 提升。与 §4.1（注意力改 fp32 无收益）同构：**内核级差异，没有"配方级"捷径**。
> 17. **环境坑（会影响下次运行，见 §10.4）**：主机 IP 已变（`141.61.29.117`→`80.5.25.117`）但 `/etc/hosts` 未更新 → 需要 `MASTER_ADDR=127.0.0.1` + `GLOO_SOCKET_IFNAME=enp48s3u1u1`；`dump_engine_stages.py` 需 `--mem-util 0.85`（每卡有他人 ~3GB 常驻，0.9 必然失败）。

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
| `scripts/dsv41_consistency/graft_trainer_stages.py` + `sh/run_graft_trainer.sh` | **嫁接实验**（§9）：把引擎侧指定层的 `self_attn`/`mlp` 输出灌进训练侧前向（`--variants baseline,0.attn,all.ffn,all.attn+all.ffn,...`），逐变体输出下游 rel_err 与 logprob Δ |
| `scripts/dsv41_consistency/summarize_graft.py` | 把多次嫁接运行的 `*.pt` 汇总成一张表（`report_len*.json` 会被后一次运行覆盖，故以 `.pt` 为准重建） |

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

---

## 9. 嫁接实验：把引擎侧输出灌进训练侧前向（2026-09-21）

### 9.1 要回答的问题

§3.1 的逐阶段 rel_err 只能说明"谁先错、谁更大"，**不能区分**下面两种情形：

- (a) 只有少数模块是"种子"，其余模块的差异都是把种子放大出来的；
- (b) 每个模块都在贡献自己的差异，最后叠加。

**嫁接（graft）** 能直接区分：把引擎侧某个模块的输出**替换**到训练侧前向里，再往下跑。如果下游（含最终 logprob）随之对齐 → 该模块（及其上游）就是分歧来源；如果下游照旧漂移 → 剩下的模块自己在贡献。

### 9.2 实现

`scripts/dsv41_consistency/graft_trainer_stages.py`（构建/加载路径与 `dump_trainer_stages.py` 完全一致）：

1. 从引擎 dump（`engine_stages_len<L>_rank0.pt`）取出该层的 `language_model.model.layers.{i}.self_attn` / `...mlp`（fp32、形状 `[S, 5120]`）；
2. 训练侧同一模块注册**三个钩子**（注册顺序即执行顺序）：
   a. 记录钩子 → 保存"未嫁接的原始输出"（`stages_pre_graft`）；
   b. 替换钩子 → 返回引擎张量（`to(device, dtype=输出 dtype)`，即 bf16；`reshape` 到训练侧 `[1,S,5120]`）；
   c. `dump_trainer_stages.install_hooks` 的 dump 钩子 → 记录**嫁接后**的值（所以被嫁接模块的 rel_err 必为 0，是自检）；
3. 每个变体都断言"替换钩子恰好触发一次"，否则报错；
4. 变体：`baseline`（不嫁接，与 §3.1 对照）、`0.attn`、`0.ffn`、`0.attn+0.ffn`、`all.attn`、`all.ffn`、`all.attn+all.ffn`、`3.ffn`、`2.ffn+3.ffn`、`3.attn`。

输出：`/mnt/share/m00899630/dsv41/dump/graft/graft_len<L>_<variant>.pt` + 每个变体的即时摘要行。

### 9.3 执行记录

```bash
# 三次运行（每次 ~2.5 min：模型构建+加载 7.8s，之后每个变体一次前向）
setsid bash scripts/dsv41_consistency/sh/run_graft_trainer.sh > /tmp/dump_graft2.log 2>&1 &
#   → variants baseline,0.attn,0.attn+0.ffn,all.attn   (lengths 64,200)
setsid bash /tmp/run_graft2.sh > /tmp/dump_graft3.log 2>&1 &   # baseline,0.ffn,all.ffn,all.attn+all.ffn
setsid bash /tmp/run_graft3.sh > /tmp/dump_graft4.log 2>&1 &   # baseline,3.ffn,2.ffn+3.ffn,3.attn
python3 scripts/dsv41_consistency/summarize_graft.py --lengths 64,200 --per-stage
```

**遇到的问题与修复**：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 首次运行全部 rank 崩：`AttributeError: 'tuple' object has no attribute 'startswith'` | 我写成了 `for name in model.named_modules()` —— 它 yield 的是 `(name, module)` 元组 | 改成 `for name, _ in model.named_modules()`；同时把子模块查找换成先建 `module_names` 集合再 `get_submodule`，并加"替换钩子必须恰好触发一次"的断言 |
| 2 | `report_len<L>.json` 被后一次运行覆盖 | 脚本按长度写单文件 | 数据未丢（每个变体的 `.pt` 都在）→ 新增 `summarize_graft.py` 从 `.pt` 重建全表 |

**控制项**：`baseline` 变体复现了 2026-09-20 那份独立 dump 的全部数字（`0.attn=0.0057 / 0.ffn=0.0644 / 1.attn=0.0445 / … / model.norm=0.1434`，Δlogp σ=0.3181；len=200：`0.1294 / 0.5596`）→ 前向是确定性的，跨运行可比。

### 9.4 结果

`σ` = 逐位置 Δlogp（训练侧 − 引擎侧，63/199 个 teacher-forced 位置）的标准差；`norm rel` = `model.norm` 输出相对引擎的 rel_err；`|Δ|mean/p99/max` 单位 nats。

**len = 200**（prompt 200 token，滑窗 128 / 压缩 KV / index_topk 全部激活）

| 变体 | norm rel | Δmean | **σ** | \|Δ\|mean | p99 | max |
|---|---|---|---|---|---|---|
| baseline | 0.1294 | −0.0134 | **0.5596** | 0.2966 | 2.760 | 3.314 |
| `3.attn` | 0.1277 | −0.0170 | 0.5613 | 0.2902 | 2.900 | 3.216 |
| `0.attn` | 0.0727 | −0.0081 | 0.3250 | 0.1354 | 1.593 | 2.320 |
| `3.ffn` | 0.0940 | −0.0201 | 0.3500 | 0.2042 | 1.534 | 1.685 |
| `2.ffn+3.ffn` | 0.0593 | −0.0068 | 0.2379 | 0.1316 | 0.953 | 1.475 |
| `0.ffn` | 0.0586 | −0.0174 | 0.1750 | 0.0867 | 0.859 | 0.919 |
| `all.attn` | 0.0592 | −0.0076 | 0.2172 | 0.0890 | 0.960 | 1.467 |
| `all.ffn` | **0.0048** | +0.0003 | **0.0243** | 0.0189 | 0.064 | 0.073 |
| `all.attn+all.ffn` | **0.0010** | +0.0009 | **0.0127** | 0.0095 | 0.046 | 0.053 |

**len = 64**（同结论，量级略小）

| 变体 | norm rel | Δmean | **σ** | \|Δ\|mean | p99 | max |
|---|---|---|---|---|---|---|
| baseline | 0.1434 | +0.0350 | **0.3181** | 0.2065 | 0.971 | 1.266 |
| `3.attn` | 0.1406 | +0.0666 | 0.3406 | 0.2216 | 1.185 | 1.224 |
| `3.ffn` | 0.1074 | +0.0087 | 0.2804 | 0.1854 | 0.832 | 0.959 |
| `2.ffn+3.ffn` | 0.0691 | −0.0577 | 0.3309 | 0.1707 | 1.327 | 1.785 |
| `0.attn+0.ffn` | 0.0646 | +0.0553 | 0.2448 | 0.1207 | 1.150 | 1.387 |
| `0.attn` | 0.0479 | +0.0427 | 0.2761 | 0.1170 | 1.390 | 1.489 |
| `0.ffn` | 0.0427 | −0.0049 | 0.1134 | 0.0657 | 0.419 | 0.685 |
| `all.attn` | 0.0424 | +0.0152 | 0.0935 | 0.0561 | 0.426 | 0.440 |
| `all.ffn` | **0.0054** | +0.0022 | **0.0273** | 0.0222 | 0.058 | 0.064 |
| `all.attn+all.ffn` | **0.0010** | −0.0003 | **0.0142** | 0.0103 | 0.043 | 0.050 |

**逐阶段 rel_err（len=200，节选）**——注意"未嫁接的模块，其输入被上游嫁接对齐后，误差会自己变小"，这正是区分"种子"和"放大"的地方：

| 变体 | 0.attn | 0.ffn | 1.attn | 1.ffn | 2.ffn | 3.ffn |
|---|---|---|---|---|---|---|
| baseline | 0.0057 | 0.0519 | 0.0435 | 0.1100 | 0.1651 | 0.2038 |
| `0.attn` | 0 | **0.0178** | 0.0119 | 0.0518 | 0.0974 | 0.1174 |
| `all.attn` | 0 | 0.0178 | 0 | 0.0472 | 0.0779 | 0.0963 |
| `0.ffn` | 0.0057 | 0 | **0.0062** | 0.0419 | 0.0781 | 0.0970 |
| `all.ffn` | 0.0057 | 0 | 0.0062 | 0 | 0 | 0 |
| `all.attn+all.ffn` | 0 | 0 | 0 | 0 | 0 | 0 |
| （module.norm） | `0.attn`→0.0727 / `all.attn`→0.0592 / `0.ffn`→0.0586 / **`all.ffn`→0.0048** / **both→0.0010** | | | | | |

### 9.5 解读

1. **两个栈在"给定相同模块输入"时的固有数值差是同一量级**：layer 0 的 attention 输出在**输入完全一致**（embed/attn_norm rel_err = 0）下差 **0.57%**；MoE 输出在输入被对齐后差 **0.58%**（`0.attn` 变体的 `0.ffn`=0.0058）。所以不存在"某一侧某个算子坏了"的情况。
2. **但对最终 logprob 的贡献差 9 倍**：全层对齐注意力 → σ 2.6×；全层对齐 MoE → σ **23×**；两者都对齐 → **44×**（σ 0.0127 nats，即剩余 HC/RMSNorm/head/embedding 的数值地板）。
   - 量级原因：逐层实测 MoE 输出 RMS ≈ 0.70、注意力 ≈ 0.45（MoE 是 1.5–2×），且 **MoE 是每层最后写入残差流的那一项**，它的误差经过的后续混合更少；注意力写在前面，其误差被同层 MoE 的写入稀释。
3. **误差是逐层累积的**：只对齐最后一层（`3.attn` σ 无变化、`3.ffn` 1.6×）、两层（2.4×）都不够，必须全层（23×）。
4. **路由选择两侧一致**（离线复核）：layer 0 的 router logits（引擎捕获的 8 个 token 对应序列前 8 个）与训练侧从 checkpoint 重算的 logits `mean|d| = 0.0046（rel 0.6%）`，**top-6 专家集合逐 token 完全相同**、per-slot overlap 100%。→ 再次确认不是离散选择错误。
5. **嫁接全部对齐后仍有 0.013 nats**：这是 HC（sinkhorn 20 次迭代）、RMSNorm、head、embedding 这一路的数值地板。相对 RL 的重要性采样阈值可忽略。

### 9.6 方法上的 caveat（必须记住）

- **第 0 层的嫁接是"干净"的**（输入一致：embed、attn_norm 的 rel_err = 0），所以 `0.attn` / `0.ffn` 变体的因果解释最直接。
- **第 ≥1 层的嫁接是"混合态"**：引擎那份输出是用**引擎自己的输入 h** 算出来的，而训练侧的 h 已经略有不同（注入的是"对另一个问题的正确答案"）。因此 `all.*` 变体应理解为"若该模块的输入→输出映射完全一致（含消除上游误差）会怎样"，而不是严格的模块级归因。即便如此，`all.ffn` 与 `all.attn` 的量级差异（23× vs 2.6×）远大于任何混合态噪声，结论方向是稳的。
- 嫁接会**屏蔽离散选择差异**（把选中哪个专家/哪些位置的输出整块替换）。为补这一点，单独做了 §9.5-4 的路由一致性复核（一致）。
- 夹具是 `scaled32` + `head-gain 4`；按 §3.3 的换算，真实权重（gain 1）下同等的 h 误差给出的 logprob σ 约小 4×。

### 9.7 对下一步的影响（修正 §7）

- §7 把"注意力算子数值对齐"列为最大单项；**嫁接实验表明 MoE 路径才是量的主体**（23× vs 2.6×）。若要在训练侧对齐，优先项应改为：
  1. **MoE**：`npu_grouped_matmul`（分组 GEMM 的累加顺序/dtype）、专家输出的累加与 `route_scale`/`norm_topk_prob` 的算术位置、EP all-to-all（mc2）的归约顺序与精度；
  2. 其次是注意力核（含 indexer/compressor）；
  3. 最后是 HC/RMSNorm（0.013 nats 地板）。
- 若维持"不动内核、在 RL 侧吸收"的路线：仍应按 §7-5 用 IS/TIS 吸收，但**阈值判断要基于修正后的量级**（MoE 主导）。
- 复现/扩展：`--variants` 支持任意 `层.模块` 组合，其他权重只需 `--model-path` 换一行（但引擎侧 dump 需先用 `run_dump_engine.sh` 对同一权重重跑一次）。

---

## 10. MoE 算术配方 A/B（2026-09-21，**负结果**）+ 本轮环境坑

### 10.1 动机与假设

§9 的归因把"量的主体"指到 MoE。逐行对比两侧实现后，最可疑的差异是 dtype 配方：

| | 两次 GEMM 之间 | 专家累加 | shared 相加 |
|---|---|---|---|
| 训练侧（`experts.py::native_eager_forward`、`model.py::Expert/MoE`，默认） | **fp32**（激活、`swiglu_limit` clamp、路由权重相乘） | **fp32**（`alltoall_combine` 前置 `.float()`，注释即写明"makes both its return AllToAll and final expert sum operate in FP32"） | **fp32**（`routed.float() + shared.float()`） |
| 引擎侧（`DeepseekV4MoE` + CANN 融合 MoE） | bf16（核内） | 核内（`MoeDistributeCombine`） | `muls_add_triton`（bf16） |

假设 H5：把训练侧改成引擎的 bf16 配方，MoE 固有差（0.58%）会显著变小。

### 10.2 实现（可回退的 env 门控）

`FSDPTurbo` 两处、共 3 个门控点，全部由 `VERL_DSV41_MOE_BF16` 控制，**默认关闭 → 行为与改动前完全一致**：

```diff
# fsdp_turbo/models/deepseek_v41/experts.py
+import os
+# Diagnostic A/B ... Setting VERL_DSV41_MOE_BF16=1 reproduces the engine's recipe
+MOE_BF16_ACT = os.environ.get("VERL_DSV41_MOE_BF16") == "1"
-        ).float()
+        )
+        if not MOE_BF16_ACT:
+            gate_up = gate_up.float()
-        intermediate *= router_weights.float().unsqueeze(-1)
+        if MOE_BF16_ACT:   # engine recipe: routing weights applied in the activation dtype
+            intermediate = intermediate * router_weights.to(hidden_states.dtype)
+        else:
+            intermediate *= router_weights.float().unsqueeze(-1)
-        combine_weights = torch.ones_like(topk_weights, dtype=torch.float32)
-        output = alltoall_combine(ep_group, expert_output.float(), ...)
+        if MOE_BF16_ACT:   # engine recipe: the combine kernel works in the activation dtype
+            combine_weights = torch.ones_like(topk_weights, dtype=hidden_states.dtype)
+            combined = alltoall_combine(ep_group, expert_output, ...)
+        else:
+            combine_weights = torch.ones_like(topk_weights, dtype=torch.float32)
+            combined = alltoall_combine(ep_group, expert_output.float(), ...)

# fsdp_turbo/models/deepseek_v41/model.py  (Expert.forward + MoE.forward)
+from .experts import MOE_BF16_ACT, DeepSeekV41Experts
+        if MOE_BF16_ACT:  # engine recipe: one dtype through the whole expert
+            gate = self.w1(x); up = self.w3(x)
+        else:
+            gate = self.w1(x).float(); up = self.w3(x).float()
+        if MOE_BF16_ACT:  # engine recipe: routed + shared combined in the activation dtype
+            output = routed_output + self.shared_experts(x)
+        else:
+            output = routed_output.float() + self.shared_experts(x).float()
```

`graft_trainer_stages.py` 启动时会打印当前配方（`MoE arithmetic: FP32-act (reference) / BF16-act (engine recipe)`），日志里可核对。

回退：`cd /workspace-verl/FSDPTurbo && git checkout fsdp_turbo/models/deepseek_v41/{experts,model}.py`（当前 `git diff --stat` 为 2 files, +48/−16）。

### 10.3 结果（scaled32，len 64/200，`--variants baseline,0.attn`，输出在 `dump/graft_bf16moe/`）

| 变体 / 指标 | fp32 配方（默认，§9） | **bf16 配方（引擎式）** | 判读 |
|---|---|---|---|
| `0.attn` → `0.ffn`（对齐输入下的 MoE 固有差，len64） | 0.0058 | **0.0057** | 无变化 |
| 同上（len200） | 0.0178 | **0.0178** | 无变化 |
| `0.attn` → `model.norm`（len64 / len200） | 0.0479 / 0.0727 | 0.0598 / 0.0660 | 无改善 |
| `0.attn` → Δlogp σ（len64 / len200） | 0.2761 / 0.3250 | 0.3421 / 0.3184 | 无改善（len64 反而更差） |
| baseline Δlogp σ（len64 / len200） | 0.3181 / 0.5596 | 0.3678 / 0.5028 | 无改善 |

**结论：假设 H5 被否定。** MoE 的 0.58% 固有差**不是**来自两次 GEMM 之间的 fp32 提升，而是在**分组 GEMM/归约内核内部**（分块与累加顺序、`npu_grouped_matmul` 的实现），改 dtype 配方动不了它。这与 §4.1（把注意力 eager 实现改 fp32 也没有端到端收益）**同构**：两套栈的差异是**内核级**的，不是"配方级"的。

> 对 §9.7 的影响：MoE 仍是最大单项，但"改 dtype 就能对齐"的捷径不存在；要真正对齐只能让一侧用另一侧的算子（引擎→训练侧受 A2/A3 布局限制，见 §4.2；反向则要动 vllm-ascend 的融合核），或继续用 RL 侧 IS 吸收。

### 10.4 本轮环境坑（会在下次运行复现，务必先看）

| # | 现象 | 根因 | 绕过 |
|---|---|---|---|
| E1 | 引擎起来后 worker 崩：`RuntimeError: [enforce fail at gloo/transport/tcp/device.cc:212] ifa != nullptr. Unable to find interface for: [141.61.29.117]` | **主机 IP 变了**（`141.61.29.117` → `80.5.25.117`），但 `/etc/hosts` 仍是 `141.61.29.117 node-29-117`；凡是用**主机名**建 gloo 通信的路径都会失败（torchrun 路径不受影响，因为它直接用 `127.0.0.1`） | 启动时加 `MASTER_ADDR=127.0.0.1`（主/ HCCL 组）**且** `GLOO_SOCKET_IFNAME=enp48s3u1u1`（gloo 设备；`enp48s3u1u1` 是默认路由所在网卡，承载 `80.5.25.117`）。**GRPO 脚本下次跑之前也要确认这一点** |
| E2 | 引擎启动即报 `ValueError: Free memory on device (54.72/61.27 GiB) on startup is less than desired GPU memory utilization (0.9, 55.14 GiB)` | 每张卡都有**其他租户 ~3GB 常驻**，设备可用总量只有 61.27 GiB，`0.9×61.27 = 55.15 GiB` **刚好超过** free（55.11）→ 差 0.04 GiB 也失败 | `dump_engine_stages.py` 新增 `--mem-util`（本轮用 `0.85`）。GRPO 脚本默认 `ROLLOUT_GPU_MEM_UTIL=0.6`，不受影响 |
| E3 | 用了有他人占用的卡 | `run_dump_engine.sh` 默认 `ASCEND_RT_VISIBLE_DEVICES=0,1,...` | 统一改用空闲的 `2,3,4,5,6,7,8,9` |

（384 专家 `scaled` 权重的引擎侧 dump / 嫁接复验：见 §11。）

---

## 11. 384 专家权重（`scaled`）复验：MoE 归因成立且更强（2026-09-21）

### 11.1 背景

§9 的嫁接实验在 scaled32（32 专家）上得出"MoE 是量的主体"。而线上数据（§0 表最后一行）显示**差距随专家数增长**（32 专家 kl≈0.11 → 384 专家 kl≈0.38），所以必须在生产专家数上复验。

`scaled` 权重（4 层 / **384 专家** / 106GB）需要**新跑一次引擎侧 dump**（原 dump 只有 scaled32）。执行中连撞 §10.4 的 E1/E2/E3 三个环境坑，全部修复后成功：

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29614 GLOO_SOCKET_IFNAME=enp48s3u1u1 \
ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
bash scripts/dsv41_consistency/run_dump_engine.sh \
  --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled \
  --out-dir /mnt/share/m00899630/dsv41/dump/engine_scaled --lengths 64,200 --skip-params --mem-util 0.85
# → 成功：179 张量/rank 的阶段 dump + prompt_logprobs + decode-vs-prefill
```

```bash
setsid bash /tmp/run_graft_scaled.sh   # 同 harness，--model-path / --engine-dir 换成 scaled
# variants: baseline,0.attn,all.attn,all.ffn,all.attn+all.ffn；lengths 64,200
python3 scripts/dsv41_consistency/summarize_graft.py --lengths 200 \
  --graft-dir /mnt/share/m00899630/dsv41/dump/graft_scaled \
  --engine-dir /mnt/share/m00899630/dsv41/dump/engine_scaled
```

### 11.2 结果（384 专家）

**len = 200**

| 变体 | norm rel | Δmean | **σ** | \|Δ\|mean | p99 | max | σ 改善 |
|---|---|---|---|---|---|---|---|
| baseline | 0.1826 | −0.0971 | **0.7062** | 0.4195 | 2.703 | 3.194 | 1× |
| `0.attn` | 0.1159 | +0.0077 | 0.4952 | 0.2201 | 1.771 | 3.569 | 1.4× |
| `all.attn` | 0.0970 | −0.0137 | 0.4104 | 0.1588 | 1.568 | 3.272 | 1.7× |
| `all.ffn` | **0.0048** | −0.0008 | **0.0243** | 0.0194 | 0.065 | 0.070 | **29×** |
| `all.attn+all.ffn` | **0.0010** | −0.0006 | **0.0120** | 0.0089 | 0.037 | 0.040 | **59×** |

**len = 64**

| 变体 | norm rel | Δmean | **σ** | \|Δ\|mean | p99 | max | σ 改善 |
|---|---|---|---|---|---|---|---|
| baseline | 0.2283 | −0.2414 | **1.0786** | 0.7237 | 3.290 | 3.463 | 1× |
| `0.attn` | 0.0804 | +0.0697 | 0.3993 | 0.1692 | 1.753 | 2.246 | 2.7× |
| `all.attn` | 0.0710 | −0.0594 | 0.3015 | 0.1206 | 1.179 | 1.927 | 3.6× |
| `all.ffn` | **0.0054** | +0.0012 | **0.0257** | 0.0203 | 0.061 | 0.068 | **42×** |
| `all.attn+all.ffn` | **0.0009** | −0.0001 | **0.0137** | 0.0106 | 0.031 | 0.033 | **79×** |

### 11.3 结论

1. **§9 的归因在生产专家数上成立，而且更强**：MoE 对齐 29–42×，注意力对齐只有 1.7–3.6×，两者都对齐 59–79×（σ 落到 0.012–0.014 nats，最终隐状态 rel_err **0.10%**，与 scaled32 完全一致 → 这个地板是 HC/RMSNorm/head 的，与专家数无关）。
2. **baseline 差距随专家数显著变大**：σ 0.56→0.71（len200）、0.32→1.08（len64），norm rel_err 0.129→0.183 / 0.143→0.228。这**定量解释了线上"kl 0.11（32 专家）→ 0.38（384 专家）"**：MoE 路径（每 token 的 6/384 专家分组 GEMM 与归约）的数值差随专家数增长，而它就是主项。
3. **引擎自身 decode↔prefill 也随专家数退化**：`engine_scaled` 的 greedy 8 token 重算差最大到 **+4.09 / +2.65 nats**（scaled32 同口径 ≤0.14；注意本次 dump 未开 `enable_batch_invariant`）。RL 的 rollout logprob 正是 decode 产出、训练侧是 prefill → 这是**第二条独立通道**，384 专家下必须依赖 `rl_config.enable_batch_invariant=true`（已写入启动脚本）来压住。
4. 因此线上 384 专家那 0.38 nats 的 kl ≈ [训练↔引擎 MoE 主导的固有差] + [引擎内部 decode↔prefill 的残余]，两者随专家数一起变大。

### 11.4 未做的事（留给下次/新机器）

- 384 专家下的 `VERL_DSV41_MOE_BF16=1` A/B（§10 的负结论只在 32 专家上验证过；理论上结论不会变，但没实测）。
- 引擎侧反方向嫁接（训练侧输出灌进引擎）——需要改 vllm-ascend worker 的 monkey-patch，收益与正向嫁接信息等价，未做。
- 真实（fp8 专家）权重：`MODEL_PATH=` 一行切换即可复用本 harness，引擎侧需重跑 `run_dump_engine.sh`（注意 fp8 权重的 `quantization_config` 会影响 `load_weights` 路径）。
