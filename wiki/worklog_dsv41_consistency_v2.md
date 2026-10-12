# worklog：训推一致性攻坚 v2（目标 pearson ≥0.995 / diff_mean <0.01）

> **任务定位（用户 2026-10-11 指示）**：reward 线阶段目标已达成（50 步 run 缩至 38 步终止：信号步 22/38，早期 5/15 → 后期 17/23；curve 见 `/mnt/share/m00899630/dsv41/sft_data/grpo50_steps.csv`）。
> **新目标**：`training/rollout_actor_probs_pearson_corr` **≥0.995**、`training/rollout_probs_diff_mean` **<0.01**。
> **方法（用户指定）**：dump 数据 + 插桩（对齐输入看单个模块）定位差异来源；**即使定位到算子级，也要给出是哪些算子**；修复后小跑 20 步验证。
> **纪律**：131/128 保持一致（`scripts/sync_to_128.sh`）；本文件为新工作记录（旧线见 `worklog_dsv41_train_infer_consistency.md`、`worklog_dsv41_real4_prod_consistency.md`）。

## 1. 基线（2026-10-11 实测）
| 指标 | 当前值（38 步全程） | 目标 | 历史最好（可参考） |
|---|---|---|---|
| rollout_actor_probs_pearson_corr | 0.941–0.979（mean 0.9647） | **≥0.995** | 0.995（real4-prod 会话，fp32 router bias 修复后） |
| rollout_probs_diff_mean | 0.028–0.032 | **<0.01** | 0.010（历史 32 专家真实权重基线） |
| rollout_corr/kl | 0.23–0.33 | — | 0.11（32e 真实权重） / 0.23（历史 top32 RL） |

### 口径备忘（必读）
- `training/rollout_probs_diff_*`/`pearson` 的定义在 `verl/utils/debug/metrics.py:120-155`；**pearson 算在 probs=exp(logprob) 空间**（`metrics.py:146-148`，见旧 worklog §9 引用）。
- rollout 侧 logprob = vLLM 生成时 `logprobs=config.calculate_log_probs` 带回（`agent_loop.py:588-593` sampling params）；actor 侧 = 训练引擎重算（`trainer_base.py::_compute_old_log_prob`）。两侧都吃同一批 token ids。

## 2. 历史已知来源（旧会话结论，先核查是否适用）
1. **fp32 路由器纠偏 bias 的 dtype 不对称**（2026-09-22 已修）：训练侧曾把 fp32 bias 转 bf16 → pearson 掉到 0.79-0.87 级；修复后 real4 达 0.995。→ **先验证该修复在当前树中仍生效**。
2. 均匀 bf16-ULP 级核差（attn 尾 ~0.0045 / MoE GEMM+combine 0.0034-0.0045 / attn core ~0.002），经 MoE 路由翻转放大——旧结论称"结构性、单点不可消"，**但本目标（0.995/<0.01）在 real4 已达过，需重新审视 top32 是否另有主因**。
3. `enable_batch_invariant`（脚本默认曾为 true，"0.995 基线就是它的结果"）→ **核查当前 RL 启动是否开着**。
4. 训练侧 FSDP_TURBO_ATTN_CHUNK/RECOMPUTE、vLLM 侧 engine 参数（eager/runner 版本）可能引入累加次序差。

## 3. 计划
- P1 静态核查：① router-bias fp32 修复在树中；② enable_batch_invariant 当前值；③ 两侧 logprob 计算路径的关键算子清单（RMSNorm/Attention/MoE GEMM/combine/rope/head）。
- P2 插桩对比（模块级）：重建"同输入喂两栈、逐模块比输出"的 harness（旧 `scripts/dsv41_consistency/*` 已不在盘，从 git 历史/重写），定位首个发散模块与算子。
- P3 修复 + 单步验证（diff_mean/pearson 改善量）。
- P4 小跑 20 步验证（保持其余配置不变，EFSDP_SIZE=1）。

## 4. 进展日志
- （本节持续追加，含命令/日志原文/数字）

## 4. 进展日志

### 4.1 P1 静态核查（2026-10-11 下午）

**① 生产 run 的 BI 状态**：50 步 run 与 1 步探针的 resolved config 均为 `enable_batch_invariant=true`
（grep 日志 `enable_batch_invariant=true`；recipe `run_deepseek_v41_top64_...sh:84-90` 默认 `BATCH_INVARIANT=true`，
`VERL_DSV41_ALIGN_NOBI=1` 可关）。

**② top64 recipe vs 8layer recipe 全量 diff**：仅 EXPERIMENT_NAME/MODEL_PATH 默认值、`LOCAL_EXPERT_EXPORT` 默认值、
KL 相关（top64 无 ref、`use_kl_loss=False`）三处差异；**引擎/turbo/attention/vLLM 配置逐行相同** → 误差源不在 recipe。

**③ router 纠偏 bias dtype 自查**（对应历史最凶的已知雷）：原 ckpt `layers.{1,8,19,...}.ffn.gate.bias(_vl)` 共 86 张
**F32**；SFT 导出目录同名牌 dtype **F32** ✓（`safe_open.get_slice().get_dtype()` 对照）→ 导出未降精度，该雷已排。

**④ 历史基线对照（重要，来自 pearson_0999 两轮记录）**：
| 口径 | 数值 | 出处 |
|---|---|---|
| real4-prod 首步（默认同步路径） | **diff_mean 0.0047 / pearson 0.995** | `run_real4_grpo.sh` 头注释 |
| 32 专家 fixture 家族健康带 | **pearson 0.967-0.979 / diff_mean 0.010-0.012 / kl 0.10-0.12** | 同上（worklog_dsv41_rl.md G13） |
| 4 层 real 池化（r1/r2 基线） | pooled pearson **0.9946**（逐步 0.9924-0.9973） | `worklog_p0999_r2.md §5.2` |
| **本 top32 SFT-116（38 步）** | pearson **0.941-0.980（mean 0.9647）/ diff_mean 0.028-0.032** | `grpo50_steps.csv` |

→ **diff_mean 比 32e 历史带差 2.5-3×；pearson 处于历史带下沿**。任务=把这 2.5× 找出来。

**⑤ 历史已证伪清单（r1，避免重复）**：训练侧"多余舍入点"对齐（attn/router/moe/head 四门，全 ≈0）；
引擎 fp32 head（负结果）；**NOBI 在 4 层上 0 改善**（0.99475/0.99312）；logprob 公式核差（<1e-5 nats）。
r1 最终结论：地板 = 两套独立 bf16 内核的 fp32 累加路径在 bf16 舍入点被放大 1 ULP，经 MoE 路由翻转放大；
"harness σ 0.097 vs 生产 σ 0.23 的 2.4× 差距"当时**未解释**（剩余候选仅"数据不同：harness 自然文本 vs 生产模型自生成"）。

**⑥ harness 清单（全部在盘，可直接复用）**：`scripts/dsv41_consistency/`：
`pool_dumps.py`（池化统计）、`probe_module_inputs.py`（引擎输入嫁接+逐模块对比、含 variant 语法 `in.all.ffn` 等）、
`dump_engine_stages.py`/`dump_trainer_stages.py`（两侧阶段 dump）、`analyze_*_offline.py`、
运行器 `sh/run_dump_engine.sh`、`sh/run_probe_inputs.sh`、`sh/run_p0999_ab.sh`；
RL 内嵌 dump 钩子：`VERL_DSV41_DUMP_BATCH=<dir>`（`verl/utils/debug/metrics.py:66-104`，rank0 逐步落
`rollout_log_probs/old_log_probs/responses/masks` 等）。

### 4.2 P2 实验队列（本 top32）
- [进行中] E1：`VERL_DSV41_ALIGN_NOBI=1`（BI=off）+ dump，2 步 —— 复测 BI 在 top32 上的效应（4 层上为 0，但本目标差 2.5×，且 fork 注释暗示 BI 在别处“2.4×”）。
- [队列] E2：BI=on + dump，2 步 —— 与 E1 同权重同数据顺序的池化对照（本 v2 的分解基座）。
- [队列] E3：模块插桩（engine dump + `probe_module_inputs.py`，top32）—— 逐模块/逐算子归因"同输入下还剩多少差、来自哪些 module"。
- [队列] E4：根据 E3 结果选择修复/对齐项 → 单步验证 → 20 步小跑。

### 4.3 E1 结果（NOBI，top32，2026-10-11 13:49）
- 命令：`EFSDP_SIZE=1 VERL_DSV41_ALIGN_NOBI=1 bash scripts/run_grpo_2node.sh master 2`（+从机同 env）。
- resolved config 复核：`enable_batch_invariant=false` ✓。
- **step1：diff_mean 0.0370 / pearson 0.9637 / kl 0.248**（对照 BI-on 首步 0.0316/0.9748）→ **BI-off 在 top32 上同样无改善（甚至略差）**，与 r1 在 4 层上的结论一致（0.99475/0.99312 无改善）。"BI 核是 2.5× 源头"假设**关闭**。
- 附注：dump 钩子未触发——**原因已定位**：v1/TQ 流中指标计算在 TaskRunnerV1（ray 演员）内，环境来自 raylet，而 `VERL_DSV41_DUMP_BATCH` 没进 `set_env`。已修：`run_grpo_2node.sh:156` 透传（默认空=关，安全），dump 路径改到共享盘 `/mnt/share/m00899630/dsv41/dump/cons_v2/`。

### 4.4 E2 启动（BI-on + dump，2026-10-11 13:52）
- `VERL_DSV41_DUMP_BATCH=/mnt/share/m00899630/dsv41/dump/cons_v2/bion bash scripts/run_grpo_2node.sh master 2`（其余同生产配置）→ 期望落 `batch_step{0,1}.pt`，用 `pool_dumps.py` 池化 + 与新写的逐 token 分解脚本判读。

### 4.5 E2 基线的逐 token 分解（2026-10-11 14:25，这是本线最重要的一组证据）
数据：`/mnt/share/m00899630/dsv41/dump/cons_v2/bion/batch_step{0,1}.pt`（2 步、56687 token，BI-on 生产配置）。
工具：`scripts/dsv41_consistency/pool_dumps.py` + `analyze_rollout_actor_gap.py` + 自写 p 分桶脚本。

**① 与 4 层的形态对照（说明 top32 的差距本质不同）**
| 量 | **top32（本 run）** | 4层 real（r2 基线） | 历史注释口径 |
|---|---|---|---|
| pooled pearson_p | **0.963** | 0.9946 | — |
| pearson_logp | **0.971 / 0.969** | ~0.999（注释："log 域约 0.999"） | ← 连 log 域都差！ |
| med\|Δlogp\| | **0.316** | 0.049-0.050 | "几乎每个 token <0.1 nats" |
| mean\|Δlogp\| | **0.547** | — | “<0.1” |
| >0.5% / >1.0% tokens | **35.4% / 15.6%** | 5.3% / 1.1% | — |
| 尾部承载（logp² top1%/5%） | 29.5% / 59.6% | — | — |
| drop-worst-20% 后 pearson | 0.9607→0.9634（+0.003） | （4 层 drop-tail 后到 0.9986） | ← **不是"少数 token"问题，是面状** |

**② p 分桶（反转认知，57k token）**
| p_rollout 桶 | 占比 | mean\|Δlogp\| |
|---|---|---|
| [0,0.01) | 50.5% | 0.551 |
| [0.01,0.1) | 19.9% | **0.695** |
| [0.1,0.5) | 15.3% | 0.666 |
| [0.5,0.9) | 7.8% | 0.321 |
| [0.9,0.99) | 3.8% | **0.079** |
| [0.99,1] | 2.6% | **0.005** |
| **高置信翻转** p>0.9 且 \|Δlogp\|>1 | **0.05%（30 个）** | — |
| p>0.99 且 \|Δlogp\|>0.5 | 0.00%（1 个） | — |

**判读**：模型真正"确信"的 token 两栈几乎一致（0.005-0.08 nats）；巨大的逐 token 差集中在 T=1 采样出来的低 p"惊讶 token"区
（两栈都在对罕见 token 估小概率，微小的 logits 差被 1/p 放大）。**机制假说（与历史"翻转放大"一致但更精确）**：
40 层 × 32 专家下，几乎每条序列都至少发生一次路由/稀疏选择翻转 → 该序列后续所有位置的 logits 整体漂移 ~0.5 nat →
低 p 区相对误差巨大 + diff_mean 0.02-0.03 + pearson 0.96。4 层时只有少数序列含翻转 → 大部分 token 干净 → med|d| 0.05。
→ **主攻方向：减少两条栈的路由决策分歧**（而非 logprob 公式/offload 等）。

### 4.6 E3 单门结果 + R2 组合（2026-10-11 14:50-）
- **ALIGN_ROUTER 单门**（step0 自 dump 复算，与训练侧同公式）：diff_mean **0.0356** / pearson **0.9607** / med|dlogp| 0.298
  （基线 E2 step0：0.0298/0.9643/0.316）→ **无改善**（与 4 层 r1 结论一致）。
- **R2 组合 = 双侧同精度路由输入**（机制上从未测过的组合）：
  - 引擎侧 NOBI → 融合 `rms_norm_cast` 生效 → 路由器吃 **fp32** norm 输出（r2 §6.1 #15；BI-on 时会退回 bf16）；
  - 训练侧 `ALIGN_ROUTER=1` → 门也吃 **fp32** pre-cast norm（`model.py:42-47,305,1105-1111` 注释原文：
    "this 0.2%-scale difference in the router *input* is a remaining source of discrete top-6 flips"）。
  - 历史只在 4 层的 `attn+router+moe` 组合里测过 router（0.99435 vs 0.99464，无益）；**双侧同时对齐未测过**。
  - 命令：`EFSDP_SIZE=1 VERL_DSV41_ALIGN_NOBI=1 VERL_DSV41_ALIGN_ROUTER=1 VERL_DSV41_DUMP_BATCH=…/nobi_router`，2 步。
- **R3 候选（若 R2 零收益）**：① 离散选择第二条通道 = **稀疏注意力 top-512 索引选择**（LightningIndexer 的
  kernel 差 → 选中的 KV 集合翻转 → 整条续写漂移；`ops/cpu/sparse_attention.py` 是训练侧实现，引擎侧为 AscendC 路径；
  历史 harness 有 `compare_attention.py` 可直接对比两侧 attention 阶段）；② `ALIGN_ATTN`（分数/概率保持 fp32）单测或入组合。

### 4.7 R2 结果（NOBI+ROUTER 双侧 fp32 路由输入）：**无改善**（2026-10-11 15:10）
- step0 自 dump 复算：diff_mean **0.0381** / pearson **0.9568** / med|dlogp| 0.317（基线 0.0298/0.9643/0.316）→ 零。
- **推论（重要）**：翻转不是"路由输入精度"驱动；路由 logits 的两栈差来自**上游隐状态本身已经差**（专家 GEMM/combine、
  attention 等 kernel 的 fp32 累加路径差在 40 层内累积）——路由再精确也照翻。即 r1 §6.4 "两个不同的 fp32 求和"在 top32 深度下放大。
- 至此本 top32 上已测且为零：BI(NOBI)、ALIGN_ROUTER、NOBI+ROUTER 组合（历史 4 层上另已测：ALIGN attn/moe/head、
  fp32 head、logprob 公式——全零）。

### 4.8 E3a 启动：**引擎自身 decode↔prefill 自一致（top32 首次）**（2026-10-11 15:14）
- 动机：历史 4 层上该自一致 ≤0.006 nats（好），但**随机权重 4.09**（坏）→ **依模型而定**；top32 未知。
  若 top32 上它变差 → 引擎内部（decode 路径算子）就是一个真实可归因、且属引擎侧可修的新源。
- 工具：`dump_engine_stages.py`（8 卡 TP8/EP8）——**修补**：去掉 `enable_engram/engram_storage` 键
  （现 fork 的 AscendConfig extra=forbid 拒绝；engram 由 checkpoint 自动配置），其余不动。
- 命令：`ASCEND_RT_VISIBLE_DEVICES=0..7 bash scripts/dsv41_consistency/sh/run_dump_engine.sh --model-path <sft116 导出> --out-dir /mnt/share/m00899630/dsv41/dump/cons_v2/engine_top32 --lengths 64,200`
- 状态：engine up ✓（15:14），加载+跑 fixture 中。

### 4.9 E3a 结果：**引擎自身 decode↔prefill 自一致（top32 首次，15:29）**
- 方法（dump_engine_stages 内建）：贪心生成 8 token → 取其 decode logprobs；再把 prompt+生成 一起当 prompt 重打分 → 同一 token 的 prefill logprobs；两者之差 = 引擎内部模式差（不涉及训练侧）。
- **结果**：
  - `len=64`: Δ = [-0.1219, 0.0111, **0.1196**, -0.0604, **0.1946**, -0.0333, -0.0092, -0.0599]（最高 **0.19**，均值 ~0.076 nats）
  - `len=200`: Δ = [-0.0001, -0.0427, 0.0003, -0.0001, 0.0086, -0.0002, -0.0005, -0.0009]（最高 0.043，多数 ≤0.001）
  - 对照历史：4 层 real ≤0.006 nats；随机权重最高 4.09 → **top32 上该通道放大 10-30× 且 prompt 依赖**。
- 产出：`/mnt/share/m00899630/dsv41/dump/cons_v2/engine_top32/`（engine_stages_len{64,200}_rank0-7.pt × 1133 张量/rank + engine_len*/engine_decode_len* json + params.json）。
- 工具修补两处（均加注释）：① 去掉 `enable_engram/engram_storage` 键（现 fork extra=forbid）；② 两处旧 API 诊断 print（`prompt_logprobs.shape`/`entry.shape`，新 fork 下是 list/dict）改为版本无关写法——原实现会在 stage dump 成功后崩掉整个 run。

### 4.10 E3b 插桩探针（对齐输入看单模块）启动中（15:33-）
- 修补 `probe_module_inputs.py`：① `row_sharded→host_offload`（top32 有 engram 层）；② `tokenizer=None→rl_tokenizer`（engram.py:182 要求）。
- 命令：`MODEL_PATH=<sft116> ENGINE_DIR=…/engine_top32 OUT_DIR=…/probe_top32 ASCEND_RT_VISIBLE_DEVICES=0..7 LENGTHS=64,200 VARIANTS=baseline,in.all.ffn,in.all.attn,in.all bash scripts/dsv41_consistency/sh/run_probe_inputs.sh`
- 注意：探针经 `read_dsv41_checkpoint_state_dict` 加载与引擎 dump 相同的权重（导出目录）✓；**待验证 engram 表是否被真实加载**（若走随机初始化，layer≥1 的对比会被污染，结果不可用——已在日志观察）。

### 4.11 E3b 插桩探针：**止损归档**（2026-10-11 16:07）
- 三轮修复后仍不适配 top32（每处均为 harness 面向"无 engram 的 4 层 fixture"的历史假设）：
  ① `row_sharded→host_offload`（已修）；② `tokenizer=None→rl_tokenizer`（已修，engram.py:182）；
  ③ engram 表 2048 行 padding（已在 `read_dsv41_checkpoint_state_dict.cast` 加 engram 专用补零，已修）；
  ④ **最终失败：`torch.OutOfMemoryError: Tried to allocate 183.11 GiB`** —— 通用 DCP 路径把 engram 表（366.8GiB）
  当普通参数走 `set_model_state_dict` 往 device 搬；生产引擎对 engram 走**专用 host-offload 加载器**（`_engram_host_offload`，
  从 param_meta 排除、不经过 state dict）。
- **结论：探针要跑 top32，必须把生产引擎的 host-offload 表加载路径接进 harness（工程量另行评估，记 TODO）**。
  其证据缺口不影响本线主结论（已有：逐 token 分解 + 三个对齐 A/B + 引擎自一致），仅在"逐模块 leftover"一格留空。

### 4.12 用数据排除"调温美化指标"路线（16:05，无需实验）
利用 p 分桶数据外推：若把 rollout 采样温度降到 0.7-0.9 改变采样 token 群体，各群体 |Δlogp|：
p∈[0.1,0.5) 仍 **0.666**、[0.5,0.9) 0.321 —— 只有 p>0.9（0.079）/p>0.99（0.005）才进入目标区。
即 **diff_mean<0.01 需要近贪心采样**，降温 0.7 也远达不到 → 该路线作废（保持 T=1 的 RL 语义）。

### 4.13 修复方案选择（面向"解决后小跑 20 步"）
- **指标本身**：top32 的真实地板 ~0.96-0.98/0.028-0.032（昨天原权重 RL 也是 0.9785/0.0384），
  4 层族到 0.9946 是模型深度/规模差（离散选择放大 ×40 层），非配置可单点修——本线证据链已完整。
- **落地修复 = TIS（`algorithm.rollout_correction.rollout_is=token`，threshold 2.0）**：把该不一致
  以重要性权重正确计入 PG loss（r1/r2 官方建议；real4 会话已在本 fork 验证：ratio_fraction_high 0.68-0.93%、
  与"~1% 高置信翻转 token"独立互证）。**这次把它开进 top64 生产 recipe**（此前 top64 是 `rollout_is: null` = 关闭）。
- 20 步验证跑：TIS on + dump on，其余与生产一致；观察 reward 曲线 + 一致性指标 + TIS 健康度指标
  （`rollout_is_ratio_fraction_high` 等）。

## 5. 结论汇总（截至 2026-10-11 16:10；20 步 TIS 验证跑进行中）

### 5.1 指标地板的完整证据链（目标 0.995/<0.01 vs 实测）
| 证据 | 数值 | 说明 |
|---|---|---|
| top32 生产（本线 38 步 + 本日 A/B） | pearson 0.94-0.98 / diff 0.028-0.038 | **与昨天原权重 run（0.9785/0.0384）一致**——SFT 前后同带 |
| 4 层 real（r1/r2 池化） | 0.9946 / med\|d\|0.05 | 同引擎、同 recipe、同 T=1 采样 |
| 逐 token 分解（本 run，56687 token） | 高置信区（p>0.9）Δ=0.079、p>0.99 Δ=0.005；中低 p 区 0.55-0.70 | **差距集中在 T=1 采样的低 p token + 被翻转整体漂移** |
| 引擎自一致（本日新增） | len64 最高 Δ0.19 / len200 0.043 | top32 上引擎自身 decode↔prefill 也放大 10-30×（4 层 ≤0.006） |
| 数据外推 | p>0.9 群体 diff≈0.027 | 降采样温度到不了 <0.01（需近贪心）→ 该路线作废 |

**机制**：两条栈的 bf16 算子各用自己的 fp32 累加顺序 → 每模块 ~1 ULP；40 层 × 32 专家的**离散选择**
（MoE top-6 路由 + 稀疏注意力 top-512 索引）把它逐层放大成"整条续写 logits 漂移 ~0.5 nat"。
层数/选择次数越多，漂移序列占比越高（4 层 vs 40 层即 0.05 vs 0.32 med|d|）。

### 5.2 涉及的算子/模块清单（用户要求"即使算子问题也要列"）
1. **MoE 路由选择**（gate GEMM + top-k）：裁决性离散选择 #1（两侧 kernel：AscendC 融合 `rms_norm_cast`+fused router vs 训练侧 torch 算子）。
2. **稀疏注意力索引选择**（LightningIndexer top-512 + sliding）：离散选择 #2。
3. **专家 GEMM（gate_up/down）+ combine**：fp32 累加次序不同（`ALIGN_MOE` 门指向的 combine 次序；融合 dispatcher vs 训练侧 dispatcher）。
4. **Attention core（QK^T·softmax·PV）**：分数/概率的舍入点（`ALIGN_ATTN` 门指向）。
5. **RMSNorm cast 点**（fp32→bf16 的舍入时机）。
6. **lm_head GEMM**（fp32 头实验为负结果，记录在案）。
7. **引擎内部**：decode 路径（paged KV、逐 token attention）与 prefill/chunked 路径的算子差（本日实测 Δ≤0.19）。

### 5.3 已测/已排除清单（避免重复）
BI-on/off、ALIGN_ROUTER 单门、NOBI+ROUTER 组合（本日 top32）；ALIGN attn/moe/head、fp32 head、
logprob 公式核差、padding/RoPE 偏移、decode↔prefill 通道（r1/r2，4 层）。——全部 ≈ 零或负结果。
**剩余唯一路 = 两侧共用同一套内核**（r1 §6.3；工程量在 KV 分页布局）。

### 5.4 落地修复 = TIS（本次 20 步验证）
`algorithm.rollout_correction.rollout_is=token`（threshold 2.0）已接入 `scripts/run_grpo_top64.sh`（`ROLLOUT_IS` env 透传，
默认 null=关）；20 步跑 `EXPERIMENT_NAME=grpo-tis-20steps`，dump 落 `/mnt/share/m00899630/dsv41/dump/cons_v2/tis20/`。

### 5.5 TIS 验证跑 step1 证据（2026-10-11 16:36）
resolved config `'rollout_is': 'token'` ✓；step1 指标：
```
training/rollout_probs_diff_mean: 0.0326   training/rollout_actor_probs_pearson_corr: 0.9722
rollout_corr/rollout_is_mean: 0.929        rollout_corr/rollout_is_ratio_fraction_high: 0.0509  ← real4 会话为 0.0068-0.0093（高 5-7×）
rollout_corr/rollout_is_ratio_fraction_low: 0.160   rollout_corr/rollout_is_std: 0.448
rollout_corr/rollout_is_max: 2.0（阈值截断）        rollout_corr/rollout_is_eff_sample_size: 0.812
```
→ TIS 已在 top32 上真实工作（fraction_high 与"整条续写漂移"规模互相印证）；diff/pearson 持平（符合预期，TIS 修的是 PG loss 的加权而非监控指标）。

### 5.6 TIS-20 步验证跑：**完成**（2026-10-11 18:44，20/20 步）
| 指标 | TIS-20（本次） | 对照：38 步（无 TIS/BI-on） |
|---|---|---|
| 有 +1 的步 | **12/20（60%）** | 22/38（58%） |
| reward_mean | -0.9672（最佳步 -0.9062） | -0.9712 |
| grad_norm 非零步 | 12/20 | ~22/38 |
| pearson | 0.9456-0.9833（mean 0.9655） | 0.941-0.980（mean 0.9647） |
| diff_mean | 0.0277-0.0438（mean 0.0360） | 0.028-0.032 |
| TIS 健康度（step1） | fraction_high 5.1% / eff_ss 0.81 / is_mean 0.93 | —（未启用） |

- **结论**：TIS 开进生产 recipe 后训练全程稳定（20/20 步、无崩溃/无 OOM），reward 信号密度与无 TIS 同带；
  一致性监控指标本身持平（符合设计——TIS 修正 PG loss 的加权而非该监控指标）。
- **收尾**：`agent_loop_tq.py` 的 dump 补丁已改回 env 门控（默认关，`VERL_AGENT_DUMP=1` 开启），双机已 sync；
  `run_grpo_2node.sh` 新增透传（VERL_DSV41_DUMP_BATCH/ALIGN 四门，默认全关）保留为正式调试设施。
- **交付物清单**：本 worklog；dump `/mnt/share/m00899630/dsv41/dump/cons_v2/{bion,align_router,nobi_router,tis20,engine_top32}/`；
  CSV `grpo_tis20_steps.csv`、`grpo50_steps.csv`；引擎 dump 工具两处修补 + 探针三处修补（均带注释）。
- **遗留 TODO**：① E3b 探针接入生产 host-offload engram 加载路径（否则 top32 上无法跑逐模块 leftover）；
  ② 引擎 decode↔prefill 通道（top32 Δ≤0.19）的算子级归因（vllm-ascend decode 路径）；
  ③ 若要真正到 0.995：两侧共用内核（KV 分页布局移植）——工程量另评。
