# 训推算子差异全景表 —— DeepSeek-V4.1-Flash top32（基于一致性分析）

> ⚠️ **2026-10-13 晚重大口径补注（先读）**：后续实测发现 **trainer 主流是 FP32 精度**（hc_pre 输出 fp32 + torch_npu
> 线性层自动升格 bf16 权重；非内核实现差）。**已解决**：`VERL_DSV41_HC_BF16=1`（训练侧，默认关）把主流拉回 bf16 后，
> q 链差 **×130 坍塌（2.6e-3→2e-5）**、attn_norm 逐位；残差重测为：attn 核 0.00137（ALIGN_ATTN=1 减半）、
> ffn 0.033–0.056、norm 0.033、head（明示 fp32 设计）。结论与完整证据见 `worklog_dsv41_taskB_consistency.md` §6.6/§6.7
> 与 memory `dsv41-dump-fp32-caliber`；下表【核】列的数字以 §6.7 的重测为准。

> 2026-10-13。把 `worklog_dsv41_taskB_consistency.md` §6 的定位结果**按算子**汇总成一张表：
> **训练侧（FSDPTurbo）每个组件用什么实现 ↔ 推理侧（vLLM + vllm-ascend）用什么实现 ↔ 实测差多少 ↔ 是什么性质**。
> 口径：top32-sft116 **4 层切片（层 0–3）**、len200 为主；所有 rel_err 为 **cast-bf16 口径**；
> "增益" = 该子段输出差 ÷ 子段输入差。数字出处：W=worklog §6.x；数字细节见其中表格。

## 0. 怎么读（差异的 4 种性质 —— 本表的核心归类）

| 标记 | 含义 | 可修性 |
|---|---|---|
| 【核】 | **内核实现差**：两侧不同的 GEMM/AR/算子 → 同输入下 ~1 bf16 ULP（≈0.4%）级 | 只能"共用内核"清零 |
| 【并行】 | **并行结构差**：TP 切 K + all-reduce、词表分片等 → 求和域/舍入序不同 | 可用精度补丁收窄（已修一例） |
| 【径】 | **执行路径差**：dtype 落点（权重乘在哪/combine 精度）、decode↔prefill、chunk 结构 | 部分可对齐（ALIGN 门）/部分待归因 |
| 【数】 | **数学增益（不是差异）**：模块对输入差的放大倍数（Jacobian） | 不可修——它决定 ULP 差扩散多快 |
| 离散 | 路由 / 压缩索引选择 | 已实测几乎零偏差 |

**排除项**：harness 的"假差异"（口径陷阱）单列在 §4，不要混进算子差异。

## 1. 主表（按前向顺序）

| # | 组件 | 训练侧实现（算子） | 推理侧实现（算子） | 实测差异 / 增益 | 性质 | 状态 |
|---|---|---|---|---|---|---|
| 1 | embedding | gather | `VocabParallelEmbedding`（vllm_ascend/models/deepseek_v41） | **0** | — | 已一致 |
| 2 | attn RMSNorm | torch fp32 计算 + 正确舍入 | **融合核 `rms_norm_cast`**（compressor 内另有 bf16 RMSNorm, `compressor.py:21`） | **逐位相同 0.00000**（"0.0017"是 dump 口径假象，W §6.2） | — | 已一致 |
| 3 | qkv 投影 + rope | torch matmul（+ 手写 rope） | `cv_wq_a/cv_wkv/cv_wq_b.matmul` + **`inplace_partial_rotary_mul`**（融合 rope；量化分支另有 `npu_quant_matmul`/`npu_rms_norm_dynamic_quant`，本部署 bf16 走非量化） | q 投影 **0.00260（≈1 ULP）** | 【核】 | 共用内核 |
| 4 | 压缩索引选择（compressor/indexer） | torch top-k over 压缩分数 | `_select_sparse_indices` + `npu_quant_lightning_indexer_v2_metadata`（lightning indexer） | 集合 **100% 一致**（3 档 × L2/L3，W §6） | 离散 | 已一致 |
| 5 | 稀疏注意力核 | **eager fallback**（trainer 专用，`ops/cpu/sparse_attention.py`）：`torch.gather` 选 KV → QK einsum（**scores→bf16**）→ `torch.softmax` → PV（**probs→bf16**）；chunk=64 + recompute=1 | **融合核 `torch.ops._C_ascend.npu_sparse_flash_mla`**（`dsa_v41.py:538`；cmp_block_table/cmp_mask_mode=3；核内 scores/softmax 保持 fp32）或 `backend.qsmla` | **1.14×**（0.00260→0.00297）；ALIGN_ATTN（把 trainer 两处 bf16 舍入改 fp32 的 A/B 门）旧栈实测**无增益**（0.99486 vs 0.99464）→ 引擎核同点也在舍入 | 【核】为主 | 共用内核→**已替换（VA_V41_ATTN_EAGER=1）** |
| 6 | **o-proj + TP all-reduce**（attn 收尾） | 8 组 einsum + **单个 K=8192 大 GEMM**（fp32 累加、单次 cast、无 AR） | `dsa_v41.py:345` 调 `dsa_v1.py:1640 AscendDSAImpl._forward_o_proj`：wo_a=`npu_transpose_batchmatmul`(K=4096) → wo_b=RowParallelLinear 局部 GEMM + **bf16 域 HCCL all-reduce**；另经 `init_attention_projections(..., reduce_results)` 处理 attn 尾 AR（model.py TODO 注释） | **1.8×（attn 段唯一放大器）**：0.0030→**0.00534**；HCCL microbench：bf16-AR rel=0.0035 / fp32-AR=0.000000 | 【并行】 | **已修(env)**：`VA_V41_OPROJ_FP32_AR=1` → 0.00473（−11%） |
| 7 | 层间累积 | — | — | MoE 输入差逐层长大（L2 0.012） | — | — |
| 8 | MoE router | torch 线性 + topk/sigmoid（**fp32 bias**，2026-09-22 dtype 差异已修） | `gate_linear` + top-6 选择（AscendC） | **set-match 100%（len64/200）、97.7–98.9%（len700，真翻转 1–2/88）**；logits rel 0.09–0.29% | 【核】+离散 | 已一致 |
| 9 | 专家 GEMM + SwiGLU + 组合 | `alltoall_dispatch`(EP8) → `grouped_matmul`(gate_up) → silu → **路由权重乘在 down-GEMM 之前**（激活 dtype）→ `grouped_matmul`(down) → `alltoall_combine`（**fp32 combine**） | token_dispatcher alltoall → 融合分组 GEMM → **combine 核内乘权重、按激活 dtype（bf16）**（`ops/fused_moe/`，`ascend_moe_forward_shared_sp`；`_should_reduce_routed_before_combine`） | 同输入输出差 **0.0049（ULP）＝实现零缺陷**；**但对输入差放大 ~4×（【数】）**：0.012 → 0.05 | 【径】（权重落点/combine dtype）+【数】 | 落点可对齐（`VERL_DSV41_ALIGN_MOE` 门）；4× 不可修 |
| 10 | shared expert + 残差 | 融合/拆分布局差异（引擎 `.mlp` ≡ `.mlp.experts`，shared 融合） | `maybe_all_reduce_shared_expert` / `maybe_all_reduce_tensor_model_parallel`（MoE 内也有 TP all-reduce） | 并入 #9 的 0.0049 | 【并行】 | — |
| 11 | final norm + head + logprob | norm + head GEMM + trainer 重算 logprob（全量 vocab） | norm 融合核 + `ParallelLMHead`（**TP 词表分片**）+ `LogitsProcessor` + 引擎 logprob（entropy chunk 配置） | **读数段**：模块输出 0.005 → logprob **σ0.0185**（~3.7×）；低 p 区主导（去最差 20% token pearson 仅 +0.003） | 【核】+【并行】 | 部分 |
| 12 | **引擎内部 decode↔prefill**（相对 trainer 全前向口径） | trainer 重算 = 全序列前向（prefill 型） | 生成时 = decode（**KV 分页、逐 token**）vs prefill | 全模型 **0.043–0.19 nats，prompt 依赖**（10-11 实测） | 【径】 | **待归因**（算子级 TODO） |

> **2026-10-13 晚·第5行 kernel 精确规格（读 AscendC 源码得出，`sparse_flash_mla_csa_kernel.h` 等）**：
> kernel 数值链 = QKᵀ（bf16 输入、**fp32 累加、fp32 存**）→ ×scale（fp32，sink 不乘）→ 在线 softmax（**全 fp32**：
> 逐 tile running max/sum，exp 不过 bf16）→ **未归一化 exp 才 Cast→bf16（CAST_ROUND）** → PV（bf16×bf16→fp32 累加）
> → ÷fp32 sum → 输出 Cast **CAST_RINT**（就近偶数）。
> **trainer 链 = scores einsum 输出即 bf16（多舍一次）+ 归一化 probs 再舍 bf16**（PV 输出再舍一次）。
> 即两侧"舍入点位置/对象不同"（kernel 舍**未归一化 exp**/trainer 舍**归一化 prob**；kernel scores 不舍/trainer 舍）——
> 这解释了 ALIGN_ATTN 旧测"无增益"：ALIGN 移掉 trainer 两处舍入后，probs 归一化 fp32 vs 核内 exp-bf16
> 仍然不等价，只是**换了一种不等价**。要真正对齐，需在引擎侧按 trainer 语义替换核（`VA_V41_ATTN_EAGER=1`）。
> 另外：CSA 路径**不做 -inf 掩码**（靠"列范围截断"实现因果：ori 窗口 [max(p-127,0), p]、cmp 取列表前
> N = min(⌊(p+1)/r⌋, 有效前缀数) 条）；cmp 列表契约 = 有效 id 必须**连续升序前缀、-1 只在尾部**。

**终版链（W §6.2）**：
```
embed 0 → norm 0（融合核逐位同）→ qkv GEMM ULP 0.0026 → sparse 核 1.14×（清白）
→ o-proj+AR 1.8×【并行，已修】→ 0.0053 → 层间累积（MoE 输入 0.012@L2）
→ MoE 【数】4× → 0.05 → norm/head 【核】3.7× → logprob σ0.084 → pearson 0.96
```

## 2. 差异源归类（把这堆数字收敛成 4 类）

1. **【核】内核 ULP 差**（qkv GEMM、sparse 核内部、MoE 分组 GEMM、head/读数段）：两侧是不同实现（torch_npu matmul/eager vs AscendC 融合核），相同输入下也差 ~1 bf16 ULP——**逐算子打补丁清不了零，只有共用内核**。
2. **【并行】结构差**：最典型 = o-proj 的 **TP 切 K + bf16 域 HCCL all-reduce**（trainer K=8192 单 GEMM、无 AR）。已用 fp32-AR 补丁收窄；同族还有 head 词表分片、MoE 内 shared-expert AR。
3. **【径】执行路径差**：① 权重乘/combine 的 dtype 落点（trainer down 前乘、fp32 combine；engine combine 核内乘、bf16）——`ALIGN_MOE` 门可对齐；② **decode↔prefill**（RL 指标口径里叠着这条，需归因）；③ chunk/recompute 结构（已证**数值等价**，只是显存手段）。
4. **【数】数学增益**（不是差异、不可修）：MoE ~4×、attn 尾 1.8×、读数段 ~3.7×——它们决定"每一处 ULP 差被扩散成多大"，也是为什么 0.96→0.995 必须把源头 ~6× 压缩。

## 3. 离散选择（单独结论）

- **MoE top-6**：set-match 100%（len64/200）；len700 97.7–98.9%（真翻转 1–2 token/88，其余是 top-k 顺序差）。
- **压缩索引（top-512/滑窗）**：3 档长度 × 2 压缩层 **全部 100%**。
- 结论：真实权重 + 现行栈下，**离散选择不是差异源**（与历史 scaled32 的"82-92% 翻转"完全不同，后者是随机权重时代的现象）。

## 4. 假差异清单（口径纪律 —— 别计入算子差异）

| 假差异 | 真相 | 出处 |
|---|---|---|
| "L0 norm 差 0.0017（第一处差异）" | trainer dump 存 **cast 前 fp32**；cast-bf16 后**逐位相同** | F-75 / W §6.2 |
| router 输入"形状 64 vs 8" | 引擎按 **TP 切 token**（rank0=前 len/8）——layout 差，非数值差 | F-73 / playbook §7 |
| 压缩索引对比"0% match" | 编号基准不同（trainer=S+idx，engine=0 基） | F-74 / playbook §4 |
| top-k "不一致" | 多为**顺序差**非集合差（口径用 set-match） | F-73 |
| chunk 64 vs 256 | 逐 query 独立 → **数值等价** | `sparse_attention.py` 注释 |

## 5. 已修 / 诊断门 / 待修

- **已修**：① router fp32 bias 被转 bf16（2026-09-22，训练侧，实验实锤）；② o-proj TP-AR bf16→fp32（`VA_V41_OPROJ_FP32_AR=1`，引擎侧 env 门，2026-10-13；切片 −7~−16%，两节点同值）。
- **诊断门（默认关）**：`VERL_DSV41_ALIGN_ATTN`（scores/probs fp32，旧栈 A/B 无增益）、`ALIGN_ROUTER/MOE/HEAD`、`ALIGN_NOBI`——旧栈"无增益"结论需在新栈复核（W §3.3c）。
- **待做**：① decode↔prefill 通道的算子级归因（#12，10-11 遗留 TODO ②）；② 共用内核（历史唯一根治，P2）；③ 纯精度收敛扫描（P1，量化"不共用内核"的天花板）。

---
**代码坐标**（现行栈）：训练 `FSDPTurbo/fsdp_turbo/models/deepseek_v41/{adapter,experts}.py`、`ops/cpu/sparse_attention.py`；
推理 `vllm-ascend-v41-private/vllm_ascend/attention/dsa_v41.py`（V41 编排 + 稀疏核调用 :538）、`attention/dsa_v1.py:1640`
（`_forward_o_proj` ＋补丁 :1752）、`ops/fused_moe/*`（融合 MoE）、`models/deepseek_v41/*`（model/compressor/model.py 投影）。
配套：`worklog_dsv41_taskB_consistency.md` §6、`dsv41_consistency_playbook.md`、失败库 §8。
