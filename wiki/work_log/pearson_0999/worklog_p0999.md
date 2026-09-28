# 把 `rollout_actor_probs_pearson_corr` 从 0.994 推向 0.999 — 工作记录（2026-09-27）

> 对象：`logs/DeepSeek-V4.1-Flash-4layer-real-20260923_143811.log`（真实 4 层切片 / 384 专家 /
> GRPO，脚本 `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh`）。
> 历史结论见 `wiki/worklog_dsv41_real4_prod_consistency.md`、`wiki/worklog_dsv41_gate_bias_fp32_fix.md`、
> `wiki/worklog_dsv41_module_input_probe.md`。
> 状态：**已完成**（2026-09-27）。
>
> **一句话结论**：**0.999 在这个 recipe 上不可达 —— 但它不可达的原因已经被定量钉死**：
> 训推 logprob 差距是"两套独立 bf16 内核栈"的固有差（每模块 ~1 个 bf16 ULP，4 层累积
> ≈0.5% 隐状态差 ≈0.23 nats logprob 噪声），**要 0.999 必须把总差距压到 0.4×**（§0 的
> 噪声敏感度曲线），而这等价于"两侧共用同一套内核"。本轮用 **7 个真实权重 8 卡 GRPO A/B**
> 把"训练侧有多余舍入点"这个前提**证伪**了：读码找到的 4 处差异（attention 分数/概率、
> MoE 路由权重作用位置、MoE 路由器输入精度、lm_head 精度）逐个打开后指标不动或变差
> （§5.2–5.5），其中"把引擎 lm_head 改成 fp32"这一条**明确让一致性变差 2.5×**
> ——反而证明了实验装置对"舍入点错位"是灵敏的，所以"零"是真零。
> 落地建议见 §6.3/§7：**用 RL 侧的 TIS 消化这一量级（既有工作已实现并验证），
> 不要用 pearson 当验收指标。**

---

## 0. 起点：目标是多少？（先量化，再动手）

用 `/tmp/dsv41_batch_real4_tis/batch_step{0,1}.pt`（生产 dump，rank0，两步）做「噪声缩放敏感度」实验：

```
p_actor(a) = exp(rollout_logp + a * (actor_logp - rollout_logp))
```

| a（= 总差距缩小的倍数） | step0 pearson | step1 pearson |
|---|---|---|
| 1.00（实测基线） | 0.99457 | 0.99285 |
| 0.75 | 0.99690 | 0.99571 |
| 0.50 | 0.99856 | 0.99792 |
| **0.40** | **0.99906** | 0.99861 |
| 0.33 | 0.99935 | 0.99902 |
| 0.20 | 0.99975 | 0.99962 |
| 0.00 | 1.00000 | 1.00000 |

**结论：要到 0.999，训推 logprob 总差距必须缩到现在的 ~0.4 倍（≈2.5×）。**
这是一个硬指标，后面所有改动都按它来判。

## 1. 差距的形状：不是"均匀噪声"，是"窄核 + 重尾"

`/tmp/p0999/an1_shape.py`（对生产 dump 逐 token 分解）：

```
dlogp: mean=-0.0275  std=0.2318  median|d|=0.0502  p99=1.04  max=2.20
50.0% 的 token |dlogp|>0.05；5.1% >0.5；1.17% >1.0
```

最关键的一条自证：**KL = 0.0275，而 0.5·σ² = 0.0269** —— 完全相等。
即 dlogp 是**零均值高斯样噪声**，没有任何系统性偏置（不存在"某一侧算错公式"这类 bug），
**唯一的路就是把噪声幅度压下来**。

方差结构（probs 空间协方差贡献）：

| p_actor 桶 | token 占比 | 协方差贡献占比 |
|---|---|---|
| > 0.6 | 3.2% | **61.4%** |
| 0.3 – 0.6 | 4.6% | **22.7%** |
| 0.1 – 0.3 | 8.3% | 4.4% |
| < 0.1 | 84% | 11.5% |

而 dlogp 的方差本身：|dlogp| < 0.5 的 95% token 只贡献 ~5% 的方差，
**95% 的方差在 |dlogp| > 0.5 的 5%"重尾"token 上** —— 即离散翻转类事件。

## 2. 差距可归约到哪里：**"对齐舍入点"**

关键观察（既有工作已经证明过）：**两侧 Q 投影的相对差 <1e-4**，也就是"几乎逐位一致"。
这说明：同一个算子、同样的 bf16 输出舍入点、同样的输入 → 结果逐位相同。
差异 **不是** 来自 GEMM 的 fp32 累加次序（那只有 1e-7），而是来自
**两侧在结构上不同的"中间张量被舍入成 bf16"的位置**。

设某模块前后向的精确值差为 Δ、bf16 的相对 ULP 为 U：

- **不对齐**：|round(a) − b| ≈ Δ + U/4；
- **对齐**：|round(a) − round(a+Δ)| ≈ min(Δ, U)。

本模型里 Δ（上游核差累积）≈0.5%，U≈0.4–0.8% —— 二者同量级，
**所以"把训练侧多余的舍入点去掉/搬到引擎的位置"是有理论收益的（约 1.5–2×）**，
而"换 dtype 配方"（历史上试过、阴性）不会有收益，因为它没有移动舍入点。

### 2.1 逐模块点出**多余的舍入点**（读码，训练侧 vs 引擎侧）

| # | 位置 | 引擎（decode 路径） | 训练侧（原） | 差别 |
|---|---|---|---|---|
| 1 | attention 分数 | `npu_sparse_flash_mla` 核内 fp32 | `einsum(q,k)` → **bf16** → `.float()` | **多一次 bf16 舍入** |
| 2 | attention 概率 | 核内 fp32 | `softmax` fp32 → `.to(bf16)` 再做 PV | **多一次 bf16 舍入** |
| 3 | MoE 路由器输入 | `rms_norm_cast` 的 **fp32** 输出（`hidden_states_fp32=x_fp32`） | `ffn_norm` 只返回 bf16，Gate 吃 `x.float()` | **输入差一个 bf16 舍入** |
| 4 | MoE 路由权重位置 | combine 核内（down 之后）`y=Σw_i·down_i(·)` | `x = w·act` **再** `.to(bf16)` 进 down GEMM | **多一次 bf16 舍入，且值不同** |
| 5 | head logits | `head_dtype=None` → lm_head 在模型 dtype → **bf16 logits** | `ParallelHead` fp32 权重 + `x.float()` → **fp32 logits** | **输出精度不同** |
| 6 | logprob 归一化 | `logits.log_softmax(dim=-1, dtype=fp32)` | `torch_npu.npu_cross_entropy_loss` | 核不同（在相同 logits 下只差 1e-7，可忽略；#5 修好后影响更小） |

### 2.2 #5 是 **vLLM 官方文档点名要为 RL 一致性打开的开关**

`vllm/config/model.py` 的 `ModelConfig.head_dtype` docstring 原文：

> "- Generation models default to the model dtype; set
>   `--hf-overrides '{"head_dtype": "float32"}'` to run the lm_head in
>   fp32, **which is required for RL training-inference consistency
>   (the trainer computes logits in fp32)**."

本 recipe 没有设置它（`LogitsProcessor.__init__` 读到的 `model_config.head_dtype` 是 None →
`_get_head_dtype` 对生成模型返回 `dtype` = bf16）。**所以引擎的 logits 是被 bf16 舍入过的，
而训练侧是 fp32 —— 这正是官方文档指出的、要显式修掉的那一类不一致。**

（1–5 已实现为 env 门控补丁，见 §4。）

### 2.3 顺带排除的项（读码 + 既有证据）

- **采样温度/惩罚**：resolved config 是 `temperature 1.0 / top_k -1 / top_p 1 / repetition_penalty 1.0`，
  且 Ascend 的 `AscendSampler` 把 `logprobs_mode` **钉死在 `raw_logprobs`**（`vllm_ascend/sample/sampler.py:16`），
  返回的就是原始 logits 的 `log_softmax`，没有任何 processor 参与 → **不是**"采样参数污染 logprob"。
- **RoPE**：训练侧 `apply_rotary_emb` 用 `view_as_complex(unflatten(-1,(-1,2)))`，即 **相邻元素配对（interleave）**，
  与引擎 `inplace_partial_rotary_mul(..., rotary_mode="interleave", partial_slice=[nope_head_dim, head_dim])`
  **约定一致**；两侧都是写回 bf16 原地张量（同一次舍入）。既有测量 q 投影 <1e-4 也印证了这一点。
- **o_proj 的调用形态**：既有 §4.7 微基准（einsum vs `npu_transpose_batchmatmul`，99.98% 逐位一致）已证伪；
  引擎 TP=8 时 `n_local_groups = 8 // 8 = 1`，每个 rank 用本 rank 的 heads 算**完整**的
  `wo_a`（4096→1024）再 `wo_b`（1024→5120）得到 partial，最后由调用方 all-reduce；
  训练侧是世界大小 1（`world_size = 1` 硬写在 `Transformer.__init__`），一次 GEMM 跨全部 8 个 group。
  **结构性差异确实存在，但按 partial 的 bf16 舍入估算是 ~0.02% 量级，不足以解释 §4.4 那个 ~0.45% 的"尾巴"**
  —— 说明 §4.4 的尾巴是**残差估计**（`eps_module² − eps_core²`），当 core 与 tail 的误差相关时不可靠。
  这条列为"未证实的候选"，不优先。

### 2.4 重尾的形状与"按位置否定 channel-2（引擎 decode↔prefill）"

生产 dump（`/tmp/p0999/batch_baseline/`，16 seq × 128 tok，SAMPLE_N=2）的分桶：

| \|dlogp\| 区间 | 占比 | 方差占比 | mean p_roll | mean p_actor |
|---|---|---|---|---|
| [0, 0.05) | 50.0% | **0.5%** | 0.0963 | 0.0965 |
| [0.05, 0.2) | 35.1% | 6.2% | 0.0656 | 0.0639 |
| [0.2, 0.5) | 9.9% | 21.1% | 0.0560 | 0.0489 |
| [0.5, 1.0) | 4.1% | **35.4%** | 0.0272 | 0.0253 |
| [1.0, 2.0) | 1.0% | 29.2% | 0.0058 | 0.0067 |
| ≥2.0 | 0.1% | 7.7% | 0.0260 | 0.0034 |

**→ 5.2% 的 token（|dlogp|>0.5）承载 72% 的方差；81% 只有 0.5%。** 高 p token 的相对误差：

| p_actor 阈值 | 中位 \|dlogp\| | 中位相对误差 \|Δp\|/p |
|---|---|---|
| > 0.7 | 0.0049 | **0.49%** |
| > 0.5 | 0.0105 | 1.06% |
| > 0.3 | 0.0159 | 1.58% |
| > 0.1 | 0.0396 | 3.96% |

**"0.49% ≈ 1 个 bf16 ULP"** 这个数字很重要：它说明**要 0.999 就必须把高 p token 的相对误差压到 ~0.19%
（≈0.4 ULP）**，即"两侧必须落在同一个 bf16 格点上"，而不是"各自更准一点"。

**按位置否定 channel-2**：rollout 的第一个 response token 是**引擎 prefill** 的产物（`max_num_batched_tokens=448`
覆盖全部 prompt），位置 1..127 才是 **decode**。若"引擎自身 decode↔prefill 不一致"是主要来源，
位置 0 应当显著更干净。实测（三个 dump 一致）：

```
pos 0 的 med|dlogp| = 0.0634 / 0.0725 / 0.0745   ← 比中段更大，不是更小
pos 8..127 的 med|dlogp| 平均 = 0.0417 / 0.0414 / 0.0455
|dlogp|>0.5 的事件在位置 0..11 上均匀分布（[1,1,0,0,1,0,3,2,0,0,0,1] 等）
```

→ **channel-2（引擎内部 decode/prefill 模式差）不是主项**，从候选里划掉。
这把差距完全归到"训练前向 vs 引擎前向的内核差"这一条通道上。

### 2.5 已证伪：**"训练侧被 left-pad 到 prompt_length，RoPE 位置整体偏移"**

动机：`agent_loop.py` 的 `_postprocess` 用 `padding_side="left"` 把 prompt 补齐到
`rollout_config.prompt_length`（resolved = 512），而
`FSDPTurboDSV41EngineWithLMHead.prepare_model_inputs` 又**主动丢掉 `position_ids`**
（docstring: *"The V4.1 reference forward indexes its RoPE tables by absolute position"*）。
若真是这样，训练侧会把**同一批 token 放在 RoPE 位置 `pad + i`** 上（而不是引擎的 `i`），
而 compressor（`compress_ratio=2`）按 `pos // ratio` 分组 → 会产生一个**系统性**的错位。
这也解释得通"生产 σ 0.23 是 harness σ 0.0967 的 2.4 倍"（harness 是单条未 pad 的序列）。

**证伪证据（两条，互相独立）**：

1. **直接测量**（新探针 `dump_trainer_stages.py --padding-probe`，真实 4 层权重，8 卡）：
   同一串 token 跑两遍 —— 一遍不 pad（引擎布局），一遍左 pad + attention_mask（trainer 布局）：

   | pad | len | 中位 \|Δlogp\| | std | p99 | max | \|Δ\|>0.5 |
   |---|---|---|---|---|---|---|
   | 440 | 64 | **1.76** | 2.48 | 6.93 | 7.74 | 85.7% |
   | 441 | 64 | 1.73 | 2.50 | 7.02 | 7.59 | 87.3% |
   | 442 | 64 | 1.75 | 2.49 | 6.94 | 7.73 | 85.7% |
   | 440 | 200 | 1.52 | 2.50 | 8.01 | 8.42 | 83.4% |
   | 441 | 200 | 1.58 | 2.54 | 7.97 | 8.36 | 80.9% |
   | 442 | 200 | 1.55 | 2.50 | 7.98 | 8.41 | 83.4% |

   → 左 pad **确实**把 logprob 打掉 1.5 nats 量级（且**与 pad 的奇偶无关** → 不是 compressor 分组，
   更像是 pad 的 compressed-KV 污染：`indexed_sparse_attention_torch` 的掩码分支条件
   `key_value.size(1) == attention_mask.size(1)` 在 `compress_ratio>0` 的层（2/3）恒为 False，
   **compressed 部分从不施加 attention mask**）。
   **但如果生产真的这样跑，pearson 会是 ~0.5 而不是 0.9946。**

2. **读码**：`transformer_impl.py:1274-1292`（padded 分支）：
   ```python
   max_seq_len = int(input_ids.offsets().diff().max().item())
   input_ids = torch.nested.to_padded_tensor(input_ids, padding=pad_token_id,
                                             output_size=(batch_size, max_seq_len))
   ```
   微批是 **nested tensor**，只 pad 到**微批内的最大有效长度**；而
   `ppo_micro_batch_size_per_gpu=1` → 每个微批只有 1 条样本 → `max_seq_len` 就是它自己的长度
   → **实际是 [1, real_len] 的紧凑张量，根本没有 pad**。
   所以训练侧的 RoPE 位置就是 `0..N-1`，与引擎一致。**这条假设关闭。**

   （顺带记录：这个探针本身证明"把 trainer 的输入左 pad"会让一致性瞬间崩塌到 1.5 nats，
   是有用的负面知识——以后谁想动 padding 布局可以直接引用它。）

## 3. 基线与方差控制

- 数据 `data.shuffle=False`、advantage 全 0（reward 全 −1）→ **权重从不变化**，
  每次运行的唯一随机性是采样出的 response。
- 同配置历史三次 step1：0.99505 / 0.99415 / 0.99457 → **run-to-run 波动 ≈ ±0.0005**，
  足以分辨 2.5× 级别的改动。
- 驱动脚本：`scripts/dsv41_consistency/sh/run_p0999_ab.sh`（`TAG=` / `ALIGN=` 控制开关，
  每跑 2 步、日志写 `logs/p0999-<TAG>.log`、dump 写 `/tmp/p0999/batch_<TAG>/`）。

## 4. 代码改动（全部 env 门控，默认关闭 = 逐位等价于改动前）

| 文件 | 门控 | 内容 | 默认关时 |
|---|---|---|---|
| `FSDPTurbo/fsdp_turbo/ops/cpu/sparse_attention.py` | `VERL_DSV41_ALIGN_ATTN=1` | 分数与概率都留在 fp32（#1 #2） | 逐位等价 |
| `FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py` | `VERL_DSV41_ALIGN_ROUTER=1` | `RMSNorm` 额外保留 fp32 输出，`Block` 把它传给 `MoE`，Gate 用它路由（#3） | 逐位等价 |
| `FSDPTurbo/fsdp_turbo/models/deepseek_v41/experts.py` | `VERL_DSV41_ALIGN_MOE=1` | 路由权重搬到 combine（down 之后），与引擎同序（#4） | 逐位等价 |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `VERL_DSV41_ALIGN_HEAD=1` | 给引擎加 `hf_overrides["head_dtype"]="float32"`（vLLM 官方为 RL 一致性提供的开关） | 不注入 |
| `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | `VERL_DSV41_ALIGN_NOBI=1` | 把 `rl_config.enable_batch_invariant` 改成 `false`（脚本里原来硬编码 `true`） | `true`（原行为） |

新增诊断工具：

| 文件 | 内容 |
|---|---|
| `scripts/dsv41_consistency/dump_trainer_stages.py --padding-probe` | 同一串 token 跑"不 pad"与"左 pad + mask"两遍，直接量出 padding 布局对 logprob 的影响（§2.5 用它证伪了位置偏移假设；也是以后动 padding 前的预警工具） |
| `scripts/dsv41_consistency/sh/run_padding_probe.sh` | 上面那个探针的一键驱动（真实 4 层权重） |
| `scripts/dsv41_consistency/sh/run_p0999_ab.sh` | 本轮 A/B 驱动：`TAG=<名> ALIGN=<attn,router,moe,head\|am\|all> [STEPS=] [SAMPLE_N=] [VERL_DSV41_ALIGN_NOBI=1]`，一条命令跑一次 2 步 GRPO 并把 dump 落到 `/tmp/p0999/batch_<TAG>/` |
| `/tmp/p0999/pool.py`（工作副本） | 把多次 dump 池化成统计量（pooled pearson / med\|Δlogp\| / 尾部占比 / 高 p 相对误差），是本轮做小效应判读的主要工具 |

回退：

```bash
cd /workspace-verl/FSDPTurbo && git checkout \
  fsdp_turbo/ops/cpu/sparse_attention.py fsdp_turbo/models/deepseek_v41/model.py \
  fsdp_turbo/models/deepseek_v41/experts.py
cd /workspace-verl/verl && git checkout \
  verl/workers/rollout/vllm_rollout/vllm_async_server.py \
  examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh
```
（`scripts/dsv41_consistency/` 下的新工具建议保留。）

## 5. A/B 结果

### 5.0 度量与统计口径（先说清楚，否则会误判）

- **指标的噪声水平**：同一配置下 10 个 step 的 `pearson` 实测
  `0.99505 / 0.99518 / 0.99187 / 0.99440 / 0.99458 / 0.99328 / 0.99464 / 0.99731 / 0.99513 / 0.99431`
  → **单步 σ ≈ 0.0014，均值 0.99458**。所以**单步对比分辨不了 0.001 量级的改动**，必须靠
  "把多步 dump 池化"（`/tmp/p0999/pool.py`）。
- 池化后的基线（4 次独立运行的 2~10 步 dump）：

  | 运行 | tokens | pooled pearson | med\|d\| | std | >0.5% | >1.0% | 高 p 相对误差 |
  |---|---|---|---|---|---|---|---|
  | real4(默认 sync) | 8192 | 0.99449 | 0.0508 | 0.2417 | 5.57 | 1.31 | 0.0054 |
  | real4_fast | 8192 | 0.99491 | 0.0506 | 0.2429 | 5.52 | 1.11 | 0.0063 |
  | real4_tis | 8192 | 0.99367 | 0.0501 | 0.2290 | 5.15 | 1.06 | 0.0060 |
  | p0999 baseline | 20480 | 0.99464 | 0.0494 | 0.2315 | 5.26 | 1.07 | **0.0054** |

  → **"高 p token 相对误差"这个统计量在 4 次独立运行上稳定在 0.0054–0.0063**，是比 pearson 灵敏得多的读数。

### 5.1 必要的公式复核：pearson 到底由什么决定？

对生产 dump 直接算：
`var(p)=2.84e-2`、`var(Δp)=3.13e-4` → `var(Δp)/(2·var(p)) = 0.0055`，
而实测 `1 − pearson = 0.0054`。**逐位吻合**，即

```
1 − pearson ≈ Var(Δp) / (2·Var(p)),   Δp = p_actor − p_rollout,  Var(Δp) = E[p²·(Δlogp)²]
```

**这条式子推翻了一个我先前用过的直觉**（"高 p token 的相对误差决定 pearson"）。
`p²` 权重的确是偏向高 p，但**平滑的 0.5% 相对误差贡献极小**：p=0.7、Δlogp=0.009 的项是
`0.49·8e-5 ≈ 4e-5`；真正主导的是**中 p + 巨大 Δlogp 的"翻转"token**
（p=0.3、Δlogp=1.0 的项是 `0.09`，大 2000 倍）。
→ **结论：要提 pearson 就必须降"翻转率"，而降翻转率的唯一办法是降噪声幅度**（阈值型响应）。

### 5.2 A/B-4：引擎侧 fp32 lm_head（`VERL_DSV41_ALIGN_HEAD`）——**负结果，且是重要的一条**

动机见 §2.2（vLLM 官方文档点名 "required for RL training-inference consistency"）。
实测（每步独立读出"高 p 相对误差"）：

| | step0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中位 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0106 | 0.0056 | 0.0040 | 0.0050 | 0.0049 | 0.0033 | 0.0034 | 0.0059 | 0.0042 | 0.0088 | **0.0049** |
| **head=fp32** | 0.0125 | 0.0130 | 0.0127 | 0.0133 | 0.0145 | 0.0109 | 0.0113 | 0.0125 | 0.0149 | 0.0112 | **0.0126** |

**把引擎的 head 换成 fp32 让高 p token 的相对误差一致地变差 2.5×**（p10 步全在同侧，不是噪声；
resolved config diff = 0，唯一差别就是这个开关，且引擎参数 dump 里确实有 `'{"head_dtype": "float32"}'`）。

**这条负结果的解释**：`prepare_deepseek_v41_model_for_fsdp` 会把**所有**浮点参数（含
`ParallelHead.weight`）转成 bf16，所以训练侧的 head 权重是 bf16 值；而引擎默认的
`head_dtype=None → dtype=bf16` 走的是 bf16 GEMM（**输出被舍入成 bf16**），
**两侧本来就"舍入在同一点"**。把引擎改成 fp32 等于**单方面把舍入点搬走**，反而制造了不一致。
（vLLM 文档那句话的隐含前提是"训练侧用 fp32 算 logits"，本 recipe 不满足这个前提。）

**同时这条负结果反过来说明了两件事**：
1. **"对齐舍入点"这个原则是定量的、有效的** —— 一个算子的舍入点错位就能让高 p 相对误差摆动 2.5×；
   那么把**现存的**错位点修好，同样量级的收益是可期待的。
2. head 这一项**不是** pearson 的杠杆：head 相对误差变差 2.5×，pearson 只从 0.99458 → 0.99405
   （§5.1 的公式预测它只该动 ~10%，实测正是这样）。**head 分支关闭。**

### 5.3 A/B-1..3：训练侧"舍入点对齐"三个补丁 —— **全部为零**

| 运行 | 门控 | 步数 | pooled pearson | 单步均值 | med\|d\| | std | >0.5% | 高 p 相对误差 |
|---|---|---|---|---|---|---|---|---|
| **baseline** | — | 10 | **0.99464** | 0.99458 | 0.0494 | 0.2315 | 5.26 | 0.0054 |
| attn | `ALIGN_ATTN`（分数/概率留 fp32） | 2 | **0.99486** | 0.99486 | 0.0455 | 0.2251 | 4.66 | 0.0067 |
| moe | `ALIGN_MOE`（路由权重搬到 combine） | 2 | **0.99462** | 0.99461 | 0.0488 | 0.2341 | 5.44 | 0.0065 |
| head | `ALIGN_HEAD`（引擎 lm_head 转 fp32） | 10 | **0.99403** | 0.99405 | 0.0422 | 0.2398 | 5.61 | **0.0127** |
| nobi | 引擎 batch-invariant 关 | 2 | **0.99402** | 0.99393 | 0.0498 | 0.2411 | 5.49 | 0.0057 |
| all | attn+router+moe+head | 2 | **0.99642** | 0.99640 | 0.0398 | 0.2168 | 5.13 | 0.0134 |
| all2（`all` 复跑） | 同上 | 2 | **0.99493** | 0.99481 | 0.0393 | 0.2209 | 4.98 | 0.0104 |
| am | attn+router+moe（不含 head） | 2 | **0.99435** | 0.99436 | 0.0471 | 0.2297 | 5.66 | 0.0128 |

**读数（逐条）**

- `attn` +0.0002、`moe` −0.0000、`nobi` −0.0006、`am` −0.0003 ——
  **全部在 run-to-run 噪声里**（单步 σ=0.0014；2 步均值的标准误 ≈0.0010；pooled ±0.0005）。
  注意 `attn` 的 `update_actor` 从 39.4s 涨到 55.1s（fp32 einsum 更慢）→ **补丁确实生效了**，
  只是**没有收益**。
- `all` 第一次读到 0.99642（看起来 +0.0018），**复跑 `all2` 只有 0.99493** → **是抽样运气**，
  两次 4 步合起来单步均值 0.99560（≈ +0.0010 ± 0.0008，约 1.2σ，不显著）。
- `all`/`all2` 的 `med|d|` 一致降到 0.039（−20%），但**这个降幅跟着 `head` 开关走**：
  不含 head 的 `am` 是 0.0471（≈baseline），而 `head` 单独跑也是 0.0422。
  **中位误差下降 ≠ pearson 改善**——`head` 正是"中位变好、尾部变得极差（高 p 相对误差 2.5×）
  所以 pearson 变差"的活例子。§5.1 的公式解释了为什么：pearson 由尾部决定。

**这五条合起来给出一个重要的（也是负面的）结论**：
**训练侧并没有"比引擎多出来的 bf16 舍入点"。** 我读码能找出的四处"多余舍入点"
（attention 分数、attention 概率、路由权重作用位置、lm_head 精度）**动完之后指标都不动或变差** ⇒
**引擎侧的融合核（`npu_sparse_flash_mla` / Ascend 融合 MoE）在这些位置同样工作在 bf16 上，
两侧的舍入点本来就是对齐的**。`head` 那条是反向验证：把它**调歪**能立刻看到 2.5× 的摆动，
说明这个实验装置对"舍入点错位"是灵敏的——**它的"零"是真零，不是测不出来。**

### 5.4 A/B-5：关掉引擎的 batch-invariant 核（`VERL_DSV41_ALIGN_NOBI`）

动机：离线 harness 的 `engine_real4` dump 是**关着** batch-invariant 取的
（`dump_engine_stages.py` 的默认），而所有生产跑都是**开着**的；生产 σ(0.23) 又是
harness 自然输入 σ(0.0967) 的 2.4 倍 —— "batch-invariant 核（Triton/AscendC 的另一套实现）
离训练侧 eager 算子更远" 是当时的头号候选。

结果（`'enable_batch_invariant': False` 已在 resolved config 里核实）：
**0.99475 / 0.99312（pooled 0.99402）→ 没有改善。假设不成立。**
"引擎自身的核选择"这条支路关闭。
（遗留：生产 σ(0.23) 与 harness σ(0.0967) 的 2.4 倍差距**最终没有找到解释**；
已排除的因素有：批大小（n=2 vs n=4 的 dump 一致：0.2315 vs 0.2317–0.2429）、
decode/prefill（§2.4 位置 0）、batch-invariant（本节）、padding/位置（§2.5）。
剩下的候选只有"数据总体不同：harness 用 prompt 自然文本，生产用模型自己生成的 response"。）

### 5.5 A/B-6：组合（`all` = attn+router+moe+head；`am` = 去掉 head）

见 §5.3 的表。**`all` 的 +0.0018 未能复现（`all2` 0.99493）；`am` 0.99435。组合没有收益。**

### 5.6 最后一个候选：**logprob 公式本身的核差** —— 也关闭

动机：训推对比的**最后一步不是模型核，而是 logprob 公式**。引擎用
`logits.log_softmax(dim=-1, dtype=torch.float32)`（`vllm/v1/sample/sampler.py::compute_logprobs`）；
训练侧走 `verl/utils/torch_functional.logprobs_from_logits`，在 NPU 上（无 flash-attn）
分派到 `logprobs_from_logits_torch_npu` = **`torch_npu.npu_cross_entropy_loss`**
（在 129280 维上做归约的另一个融合核）。这一项若不等价，会**不被任何下游平均掉**地
直接进入 dlogp —— 与逐层核差的性质完全不同，所以值得单独测。

新探针 `dump_trainer_stages.py --logprob-probe`（真实 4 层权重，8 卡，与 §2.5 同一次运行）：

```
len=64  npu_cross_entropy vs log_softmax: mean=+0.00000 std=0.00000 med|d|=0.00000 max|d|=0.00000 bits_eq=44.4%
len=64  v2(torch)         vs log_softmax: mean=+0.00000 std=0.00000 med|d|=0.00000 max|d|=0.00000 bits_eq=44.4%
len=200 npu_cross_entropy vs log_softmax: mean=+0.00000 std=0.00000 med|d|=0.00000 max|d|=0.00000 bits_eq=59.3%
len=200 v2(torch)         vs log_softmax: mean=+0.00000 std=0.00000 med|d|=0.00000 max|d|=0.00000 bits_eq=58.8%
```

→ **两条训练侧路径与引擎公式的差都在 1e-5 nats 以下（实际 ~1e-6，只有末位比特不同）**。
`bits_eq` 只有 44–59% 说明确实不是逐位相同，但量级完全不重要。**这条候选关闭。**

（同一次运行也复算了 §2.5 的 padding 探针，数字与上一次**逐位重合**
（1.4679/1.7597/85.7% 与 1.3407/1.5223/83.4%）——顺带证明了这套 harness 的前向是确定的。）

## 6. 结论

### 6.1 目标本身是"可量化但不可达"的

- **要 0.999，必须把训推 logprob 总差距压到 0.4×（§0 的 α 曲线）。**
- 差距的形态：`1 − pearson ≈ Var(Δp)/(2·Var(p))`，而 `Var(Δp) = E[p²·(Δlogp)²]`。
  对生产 dump 逐 token 分解：**p ∈ [0.3,1] 的 9.5% token 占 74.4% 的 Var(Δp)**，
  它们的中位 |Δlogp| 是 0.039（p≈0.38）和 0.011（p≈0.72）——
  **即"中高概率 token 上约 1 个 bf16 ULP 的 logit 差"（0.002–0.004 nats/|logit|≈10）。**
- **1 个 bf16 ULP 是这两套栈的"地板"**：每一层、每个模块的两个独立 bf16 实现，
  只要中间张量的舍入点不完全重合，输出就差 ~1 ULP；4 层累积后 ≈0.5% 隐状态差
  ≈0.23 nats 的 logprob 噪声。**要做 2.5×，需要两侧在"半个 ULP 以内"一致 —— 等价于要求
  两侧用同一个内核。**

### 6.2 本轮最有价值的（负面）结论：**训练侧并不存在"多余的舍入点"**

**本轮一共关掉了 9 条候选通道**（每条都有实测或读码证据）：

| # | 候选 | 处置 | 证据 |
|---|---|---|---|
| 1 | 采样参数污染 rollout logprob | 关闭 | temperature 1.0 / top_k −1 / top_p 1 / rep 1.0；Ascend `AscendSampler` 把 `logprobs_mode` 钉死在 `raw_logprobs` |
| 2 | 权重同步 / expert 导出错位 | 关闭 | 既有工作（§2 权重级校验 664/672）+ fast/default sync 指标同带 |
| 3 | 引擎自身 decode↔prefill 不一致 | 关闭 | §2.4：位置 0（prefill 产物）不比其他位置干净 |
| 4 | 训练侧被 left-pad → RoPE 位置偏移 | **证伪** | §2.5：微批是 `[1, real_len]` 紧凑张量；且探针显示真左 pad 会掉 1.5 nats |
| 5 | attention 分数/概率多两次 bf16 舍入 | 关闭 | §5.3：`0.99486` vs baseline `0.99464`（噪声内） |
| 6 | MoE 路由权重作用位置不同 | 关闭 | §5.3：`0.99462`（±0） |
| 7 | MoE 路由器输入 bf16 vs fp32 | 关闭 | §5.3：只以组合形式跑（`am 0.99435`） |
| 8 | 引擎 batch-invariant 核"不像训练侧" | 关闭 | §5.4：关掉 `0.99402`（噪声内） |
| 9 | **logprob 公式的核差**（`npu_cross_entropy_loss` vs `log_softmax`） | 关闭 | §5.6：差 <1e-5 nats |
| ★ | 引擎 lm_head 精度（vLLM 文档点名） | **反向确认** | §5.2：改成 fp32 让高 p 相对误差一致变差 **2.5×** |

我按"对齐舍入点"原则找出并实现了 4 处差异，逐个 A/B（§5.2–5.5，全部 8 卡真实权重 GRPO）：

| 补丁 | 预期 | 实测（pooled pearson，baseline 0.99464） |
|---|---|---|
| attention 分数/概率留 fp32（去掉 2 次 bf16 舍入） | 让训练侧靠近引擎融合核 | **0.99486**（+0.0002，噪声内） |
| MoE 路由权重搬到 combine（与引擎同序） | 同上 | **0.99462**（±0） |
| MoE 路由器输入改用 fp32 norm 输出 | 与引擎 `rms_norm_cast` 的 fp32 分量对齐 | 只以 `router` 名义并进 `all`/`am`，未单独成跑 |
| 引擎 lm_head 转 fp32（vLLM 文档点名要开的开关） | 让引擎靠近训练侧 fp32 头 | **0.99403**（**明确变差**，高 p 相对误差 2.5×） |
| 引擎 batch-invariant 关闭 | 换回"更像训练侧"的默认核 | **0.99402**（−0.0006，噪声内） |
| 组合 attn+router+moe（`am`） | 上述之和 | **0.99435**（±0） |
| 组合 attn+router+moe+head（`all` / `all2`） | — | 0.99642 → 复跑 0.99493（**不可复现**） |

**"变差"那条反而是最有信息量的**：它证明了**该开关改变的是舍入点**，而且证明
**引擎的默认 bf16 lm_head 与训练侧本来就是对齐的**（`prepare_deepseek_v41_model_for_fsdp`
把所有浮点参数（含 `ParallelHead.weight`）统一转 bf16）。既然把一个算子调歪能立刻看到
2.5×的摆动，那么"三个补丁全部为零"就只能有一个解释：

> **引擎侧的融合核（`npu_sparse_flash_mla`、Ascend 融合 MoE）在我改的那些位置上也工作在
> bf16 上；两侧的中间张量舍入点本来就重合。剩下的差异不是"舍入点错位"，而是
> "同一个 bf16 张量在两套栈里由不同的 fp32 累加路径算出来"——量级 1e-7，再被 bf16 输出
> 舍入放大成 ~1 ULP 的概率≈δ/ULP。**

### 6.3 想要真正到 0.999，只剩三条路（按可行性排序）

1. **让两侧共用内核**（唯一能到 0.999 的路）。既有工作已经试过并被硬件/布局堵死：
   训练侧启用融合 `SparseFlashMla` 在 A2/A3 上要求 `cmp_mask_mode=3` + `cmp_ratio=4`，
   而本模型是 `compress_ratios=[0,0,2,2]`；引擎能用是因为它走**分页布局 PA_BBND**
   （`wiki/worklog_dsv41_train_infer_consistency.md` §4.2）。要做就得让 FSDPTurbo 支持分页 KV
   —— 工程量远超本次范围。**本轮的 §5.3/5.4 进一步说明：不共用内核，任何"对齐"都拿不到收益。**
2. **两侧同时提高精度**（比如都到 fp32）—— 引擎侧不可行；而且**只把一侧提精度会变差**
   （§5.2 已实测，这是本轮最反直觉的一条）。
3. **RL 侧消化（推荐）**：`algorithm.rollout_correction`（TIS）把这一不一致以重要性权重
   正确计入 PG loss。它**不改变 pearson**，但把"差距对训练的影响"归零。
   既有工作已在 dsv41 recipe 上落地并验证（`wiki/worklog_dsv41_real4_prod_consistency.md` §5.8：
   `rollout_is_ratio_fraction_high 0.68%/0.93%` 与本轮 §0 的"~1% 高置信翻转 token"独立互证）。

### 6.4 与既有结论的关系

本轮**修正**了既有工作的一处判断：`wiki/...real4_prod_consistency.md` §4.7 把注意力尾部的
差异归因于"引擎 TP 头切 + HCCL all-reduce 的并行结构差"，并据此把三块核差升级为
"结构差、不可单点消除"。本轮实测支持"不可单点消除"**这个结论**，但**否定了"训练侧有多余
舍入点"这一前提**——真正的原因更朴素：两套 bf16 核的 fp32 累加路径不同，
在 bf16 输出舍入处被放大到 1 ULP。这也解释了为什么 §4.7 的 o_proj 微基准
（einsum vs `npu_transpose_batchmatmul` 99.98% 逐位一致）会得到"形态无罪"的结果：
**不是形态问题，是"两个不同的 fp32 求和"问题。**

## 7. 后续建议

1. **不要把 `rollout_actor_probs_pearson_corr` 当验收指标**。它是"两套内核数值一致性"的
   原始读数，其天花板由两侧实现决定（本模型实测 ~0.995），改动它需要动内核而不是动配置。
   验收指标建议改为 TIS 生效后的 `rollout_is_ratio_fraction_high` / `rollout_is_eff_sample_size`
   与训练稳定性（grad norm / reward 曲线）。
2. **保留 `head_dtype` 的默认值**（不要按 vLLM 文档那句话去开 fp32）：文档的隐含前提是
   "训练侧用 fp32 算 logits"，本 recipe 的 FSDP 预处理把所有参数转 bf16，前提不成立。
   这条值得回写给 vLLM/verl。
3. **`enable_batch_invariant` 保持开启**（脚本默认）：关掉没有收益（§5.4），而它消除的是
   引擎自身的批组成依赖（另一类风险）。
4. **如果将来 40 层复跑**：预期 pearson 不低于本 4 层切片（层数更多 → 累积核差更多），
   所以 0.999 的目标在 40 层上更不可达；建议直接用 TIS。
5. **本轮的 4 个 env 门控默认全部关闭**，行为与改动前逐位一致；它们保留为诊断工具
   （`VERL_DSV41_ALIGN_{ATTN,ROUTER,MOE,HEAD}` + `VERL_DSV41_ALIGN_NOBI`），
   `run_p0999_ab.sh` 是现成的 A/B 驱动。

---

## 8. 复现命令汇总

```bash
cd /workspace-verl/verl

# ---- 0) 只有 CPU 就能做的分析（秒级）----
# 生产 dump 的噪声敏感度曲线：每个 alpha 一行，alpha=0.4 即 0.999 的目标线（§0）
python3 scripts/dsv41_consistency/noise_scale_curve.py
# 逐 token 分解：p 桶 / |dlogp| 桶对 Var(dp) 的贡献、drop-tail 曲线（§1、§5.1）
python3 - <<'EOF'
import torch
d=torch.load('/tmp/p0999/batch_baseline/batch_step0.pt',map_location='cpu',weights_only=False)
rl,ol,m=d['rollout_log_probs'].float(),d['old_log_probs'].float(),d['response_mask'].bool()
r,o=rl[m],ol[m]; pa,pr=o.exp(),r.exp(); dd=o-r
c=pa.pow(2)*dd.pow(2); print("Var(dp) prop:",c.mean().item())
for lo,hi in [(0.3,0.5),(0.5,1.01)]:
    s=(pa>=lo)&(pa<hi); print(lo,hi,"%Var:",c[s].sum()/c.sum()*100,"n",int(s.sum()))
EOF
# 多次 dump / 多个 run 的池化统计（做小效应判读的主要工具，§5）
python3 scripts/dsv41_consistency/pool_dumps.py \
    /tmp/p0999/batch_baseline /tmp/p0999/batch_all /tmp/p0999/batch_am

# ---- 1) padding / RoPE 位置探针（8 卡，~12 min；§2.5 靠它证伪）----
bash scripts/dsv41_consistency/sh/run_padding_probe.sh          # 真实 4 层权重
#   → 日志里的 "PADDING-PROBE len=200 pad=440 ..." 行

# ---- 2) A/B 驱动：一次 2 步 GRPO（8 卡，~15 min）----
# 基线
TAG=baseline bash scripts/dsv41_consistency/sh/run_p0999_ab.sh
# 单开关（可任意组合：attn / router / moe / head / am / all）
TAG=attn ALIGN=attn bash scripts/dsv41_consistency/sh/run_p0999_ab.sh
# 关掉引擎 batch-invariant
TAG=nobi VERL_DSV41_ALIGN_NOBI=1 bash scripts/dsv41_consistency/sh/run_p0999_ab.sh
# 整串（带内存门控与自动重试）
bash scripts/dsv41_consistency/sh/run_p0999_campaign.sh
# 清残留 ray/vllm 进程（pkill -f 会匹配到自己的命令行，所以放在脚本里）
bash scripts/dsv41_consistency/sh/cleanup_ray.sh
```

**本机环境要点（本轮踩到的）**
- 卡上有别的租户（20–37 GB/卡波动）：`ROLLOUT_GPU_MEM_UTIL=0.4`、`SAMPLE_N=2` 才装得下；
  actor 反向里有一次固定的 6.33 GiB 分配（FSDP2 `foreach_reduce` 的 flat grad buffer），
  只差 ~0.8 GB 就会 OOM，**失败点固定在 `update_actor` 的 backward**。
- `pkill -f "<pattern>"` 会杀掉**自己**（命令行里含该 pattern）→ 统一用
  `scripts/dsv41_consistency/sh/cleanup_ray.sh`。
- bash 的 `\` 续行链里**不能插注释**：`#` 会终结逻辑行并**静默丢掉它前面的赋值**
  （本轮 `STEPS=2` 就这么丢过一次，跑成了 10 步）。