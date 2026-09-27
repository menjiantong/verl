# DeepSeek-V4.1 真实 4 层权重的**生产一致性**:同步提速 + pearson 0.995 的逐层分解(工作记录,2026-09-23)

> 上一篇:[`worklog_dsv41_gate_bias_fp32_fix.md`](worklog_dsv41_gate_bias_fp32_fix.md)(fp32 纠偏 bias 修复,已完成并有 §5 回归)。
> 本篇记录修复之后**第一次真实权重的生产 GRPO 运行**(`DeepSeek-V4.1-Flash-4layer-real-20260923_104558.log`)暴露的两个问题、
> 它们的定位过程、以及为"提升验证速度 → 继续定位不一致"所做的改动。
>
> 状态:**问题 1(慢)已解决并定版(§5.4,6.3× 提速);问题 2(pearson 0.995→0.999)已完成
> 子段级归因(§4.4–§4.6:残差=o_proj 尾 ~0.0045 + expert GEMM/combine ~0.0037 + core ~0.002,
> 全部为均匀 bf16-ULP 核差,离散/参数级因素为零)**;指标口径已修正为 probs 空间(§3.1);
> fulldet 杠杆证伪(§5.5)、o_proj 形态对齐证伪(§4.7)→ 残余核差是并行结构差;
> **TIS(rollout_correction)已实施并验证可用(§5.8)**;优化路线与遗留见 §7。
> 最后更新:2026-09-23 15:10。

---

## 0. 一句话结论

1. **慢**:step 1230s 里 **1032.7s 是权重同步**,其中 **export 904–1009s** —— 10:45 那次跑没有开
   `VERL_DSV41_LOCAL_EXPERT_EXPORT`,8 个 rank 每个都把全量 104.96 GiB(含 384 专家的 fused 张量从
   CPU-offload 逐块 all-gather)导出了一遍。这正是 `_export_param` docstring 预言过的量级
   ("14 GB→80 s,114 GB→~920 s")。**快速通道今天 09:38 的 `fp32-gate-bias-real4-r2` 运行已在同一份
   真实权重上验证过**(sender 670 tensors/16.37 GiB,export 0.9s,sync 18.5s,pearson 0.988 无断层)
   → 开 `LOCAL_EXPERT_EXPORT=1` 后预期 **~215 s/step(≈5.7×)**。
2. **pearson 0.995 不是新 bug,是"每模块 ~0.5% 均匀核差 × MoE 离散放大"的残余形态**,与修复预期一致。
   两条独立证据链:
   - 生产 token 级 dump(`batch_step0/1.pt`):|Δlogp| 中位数 0.049、49% token >0.05、1.3% >1.0;
     **drop 最差的 5% token 后 pearson 0.9993** —— "宽基底 + 重尾"。
   - 钉输入探针的离线子段分解(新工具 `analyze_moe_floor_offline.py` + `analyze_attn_floor_offline.py`,
     §4.4/§4.6):同一份钉死输入下 **路由器 0 残差**(top-6 选择 200/200 含槽位序全同、权重 |Δ|≤1.2e-7)、
     **attn q 投影 <1e-4**(rope 无罪);地板三块:**MoE expert GEMM/combine 0.0034–0.0045、
     attn 尾(inverse-rope+o_proj 段)~0.0045(§4.7:微基准证伪"调用形态差",归因修正为
     引擎 TP 头切+HCCL all-reduce 的**并行结构差**)、attn core ~0.002**——全部逐 token
     均匀无尾(top-1% 能量占比 ≤2%),即 bf16 1 ULP 级核差;head 也无离散残差。
     harness 另含 attn_sink 值不对称(engine fp32 vs trainer bf16,rel ~0.18%/层),生产同步会抹平。
   - 定量目标:**以 §3.1 的 probs 空间读法为准**(0.995 由 ~1% 高置信翻转 token 主导,drop 1% → 0.9986);
     dtype/参数级/调用形态的路已全部走完或证伪(§3.1/§4.3/§4.7),残余核差是**两套栈并行布局的结构差**,
     不可单点消除 → 现实杠杆是 §5.8 的 TIS(把翻转 token 的后果计入 PG loss)。
3. **TIS 已实施并验证可用**(§5.8,14:38–15:05):dsv41 recipe 首次走通 Decoupled+token-IS 全链路;
   `ratio_fraction_high 0.68%/0.93%` 与 §3.1 的"worst ~1% 高置信 token"**独立互证**;pearson 同带(无误伤)。
   遗留:reward 全 −1 的数据退化让 pg_loss=0,**梯度端效果待有方差的 reward 再验**(§7-F)。

---

## 1. 输入与现场(2026-09-23 12:37 起)

- 分析对象:`logs/DeepSeek-V4.1-Flash-4layer-real-20260923_104558.log`(真实 4 层权重、含 gate.bias 修复,
  `run_real4_grpo.sh` 启动,`total_training_steps=10`)。
- 参照运行:同日 `..._093816.log`(实验名 `GRPO-DSV41-fp32-gate-bias-real4-r2`,同权重、开 fast sync,
  batch/长度是本跑一半)。
- 两份 worklog:本篇上文两篇 + `worklog_dsv41_real_weights_4layer.md`。
- 现场确认:NPU 全部空闲(前一跑只完成 step 1 已终止,TaskRunner defunct),host 内存 2454 GB 可用
  —— 10:45 那跑峰值曾到 1297 GB,全部来自 8 路并发 export 的 105 GiB。

---

## 2. 问题 1:慢在哪里(命令 → 证据 → 根因)

### 2.1 step 时间分解

```bash
grep -o "timing_s/[a-z_]*:[0-9.]*" logs/DeepSeek-V4.1-Flash-4layer-real-20260923_104558.log
```

| 阶段 | 10:45 跑(real,默认 sync) | 09:38 跑(real,fast sync) | scaled32 冒烟(修复记录 §5.5) |
|---|---|---|---|
| **update_weights** | **1032.65s** | **18.49s** | 89.7s |
| gen | 98.62s | 50.45s | 88.8s |
| update_actor | 69.30s | 32.67s | 5.0s |
| old_log_prob / ref / adv | 6.3 / 3.9 / 0.3s | 同型 | — |
| **step 合计** | **1230.1s** | 124.9s | 205.9s |

gen/update_actor 的翻倍不是退化:两跑配置差一倍(10:45:bsz 8×n4、512+128;09:38:bsz 4×n2、256+64),
用配置 diff 核实过(`ppo_mini_batch_size 4→8`、`rollout_n 2→4`、`prompt/response 256/64→512/128`)。

### 2.2 同步的分段计时(`VERL_SYNC_PROFILE=1` 已在 run_real4_grpo.sh 默认开)

10:45 跑(log 3326/3444 行):

```
sender:   4702 tensors, 104.96 GiB, 213 buckets | export 904–1009s, flush(wait for receiver) 12.5–13.3s
receiver: 215 buckets, 4702 tensors | views 0.1s, load 11.0–11.8s, device sync 0.5s
```

09:38 跑(同一份权重、同一 8 卡):

```
sender:   670 tensors, 16.37 GiB, 30 buckets | export 0.9s, flush 11.8s
receiver: 32 buckets, 670 tensors | views 0.0s, load 9.9–10.4s
```

### 2.3 根因

`verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py::FSDPTurboDSV41EngineWithLMHead._export_param`:
默认路径把 fused expert 张量(`[384, 4608, 5120]` 级别)**逐 EP 块 `full_tensor()` all-gather** 回 NPU
再 unfuse;参数在 host(`offload_policy/param_offload=True`)时这个 gather 是 host↔device 往返 +
跨 rank 集合通信,8 个 rank 各做**全量** 8/8 份。docstring 里写着的实测外推
(14 GB→80 s ⇒ 114 GB→~920 s)与本次 904–1009s 吻合——**不是新问题,是已知成本没开开关**。

`VERL_DSV41_LOCAL_EXPERT_EXPORT=1`(`_export_local_experts`):只导出本 rank 分片的 48 个专家
(colocate 下 ZMQ 端点 trainer rank i ↔ engine rank i 1:1 配对,两侧都按 rank 连续切专家维;
前提 `GEN_EP == EP_SIZE`,启动脚本 71–77 行会自动校验并在不满足时降级回默认路径)。
**配错的自检测**:配对错→引擎装错专家→`rollout_probs_diff_*` 从 ~1e-2/~0.99 跳到 O(1)/~0.5,不可能静默通过。

### 2.4 为什么 10:45 没开?

`run_real4_grpo.sh` 18–22 行的注释记录了当时的刻意选择:"默认路径先在真实权重上走通一步再翻开关"。
首步已确认健康(diff_mean 0.0047、pearson 0.995、无断层),条件满足;且 09:38 那次其实已经把快速通道
在真实 384 专家上跑通(16.37 GiB sender + 0.988 pearson)。→ 见 §5 重跑。

### 2.5 附带观察(不影响结论,记录)

- 10:45 跑只完成 step 1 就被终止(log 结尾无 Traceback,step 2 的 dump 已写出);NPU/内存在 12:37 已释放。
- `expert_parallel.py:77` 的 `Cannot create tensor with internal format ...` UserWarning 在每 rank 出现一次,
  来自 EP 元数据刷新时 `.remainder(module.num_local_experts)` 的建张量,无害(base format 回退)。
- reward 全部 −1 → advantage 全 0 → `grad_norm 0/pg_loss 0`:本实验里**权重从始至终不变**,
  每步同步都在传同一份数据。纯一致性实验 `STEPS=2` 足够。

---

## 3. 问题 2a:生产 pearson 0.995 的 token 级分解(离线,零成本)

`run_real4_grpo.sh` 已带 `VERL_DSV41_DUMP_BATCH=/tmp/dsv41_batch_real4`
(`verl/utils/debug/metrics.py:68 maybe_dump_debug_batch`,rank0 每步写
`rollout_log_probs/old_log_probs/responses/response_mask`),10:45 跑留下了 step1/step2 两个 batch。

```python
# 对 batch_step0.pt 的操作(CPU,秒级)
pearson(rl, ol, mask)                    # 0.99756
|Δlogp| 分位数: med 0.049, p90 0.313, p99 1.076, max 2.386
分布:49.3% token >0.05,14.6% >0.2,1.3% >1.0
drop 最差 1/2/5/10% token → pearson 0.99836 / 0.99875 / 0.99934 / 0.99970
按位置分桶(16 tok/桶):[0.137 0.112 0.102 0.112 0.113 0.104 0.135 0.101] —— 无位置效应
step1 batch:pearson 0.99744,diff std 0.244 —— 两步同量级(权重未变,符合预期)
```

- **形态 = 宽基底 + 重尾**:5% 的坏 token 承载了 0.995→0.999 之间的主要差距;与"MoE 路由翻转
  token 吃掉 64–90% 误差能量"的既有结论同一机制。
- ~~本地重算 0.9976 vs 日志 0.9950:聚合口径略有差~~ **已由 §3.1 查明:trainer 指标在 probs 空间
  (metrics.py:146 `exp()`),按同口径重算=0.99505,与日志完全吻合;本节的 logprob 数字只用于看形态。**
- ~~定量到目标:σn ≤ 0.153(压到 63%)~~ **目标函数读法以 §3.1 为准**(probs 空间由 ~1% 高置信
  token 主导,阈值型响应,不是线性压噪声)。
- ⚠️ 作用域提醒:该 fixture 是 4 层未训练 policy(entropy 5.9 / ppl ~366),σs 偏小,pearson 对噪声
  比真实训练模型更敏感;同样的核差在 40 层真实 policy 上预计指标更好看,但**绝对 σ 不保证更小**
  (上一篇 §6.1.3 的层数 caveat 仍然有效)。

---

## 3.1 **口径修正(14:2x,本轮最重要的一个发现)**:指标在 probs 空间,不是 logprob 空间

§3 的离线分解是在 logprob 上做的,得到 pearson 0.9976,与日志 0.9950 有 3e-3 的"口径差"——当时记为
"聚合路径差异,不影响形态判断"。**这个判断错了,现已查明**:`verl/utils/debug/metrics.py:146-148`

```python
actor_probs = torch.exp(actor_old_log_probs)      # ← 先 exp!
rollout_probs = torch.exp(rollout_old_log_probs)
pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, ...)
```

trainer 的 `rollout_actor_probs_pearson_corr` 算在 **probs = e^logprob 空间**。用 dump 张量按同口径重算:

```
probs-pearson(10:45 run step1) = 0.99505   vs 日志 0.995047  ✓ 完全吻合
probs-pearson(fast-sync step1)  = 0.99416   vs 日志 0.994155  ✓
```

** mystery 关闭:没有任何"聚合口径差",dump 就是全部;之前所有判读要在 probs 空间重做**,而重做后
结构变了——**差距几乎完全由 ~1% 的高概率 token 承载**:

| 数量(probs 空间,生产 dump,4096 token) | 值 | 怎么读 |
|---|---|---|
| \|Δp\| 中位数 | **0.00011** | 一半 token 的两侧概率差在 1e-4 量级——低 p token(p≈1e-9)即使 logprob 差 1,绝对 p 差也近零,**对 pearson 无贡献** |
| \|Δp\| p99 / max | 0.079 / 0.32 | 尾部全在"高 p 且两侧分歧"的 token 上 |
| worst-5% token 的平均 p_actor | **0.34**(全体均值 0.079) | **坏 token = 模型很有把握的 token**;它们同时是 logprob 空间"翻转 token"的超集(路由翻转改变了 argmax 分布的尖峰) |
| drop worst **1%** → probs-pearson | **0.9986** | 全 batch 只有 ~40 个 token 在决定 0.995 这个数字 |
| drop worst **5%** → | **0.9997** | 50 个 token 内就够到 0.999 之上 |

**这张表如何改变优化判断**:log 空间下我推过"σn 0.24→0.15 需要三块核差整体 ×0.63";probs 空间下
pearson 对**高置信位置的离散翻转**敏感度远高于对均匀噪声——**翻转数对输入扰动是阈值型(超线性)响应**
(topk 边界竞争,越界概率 ∝ 扰动幅度),所以"把最大的一块核差减半"可能把 1% 的坏 token 数砍掉一半以上,
pearson 呈跳跃式改善。这重新抬高了"单点对齐"的价值——于是去做了 §4.7 的微基准。
(同时保留:两批 fast/default dump 的 probs-pearson 0.9942/0.9951 再次确认同步通道不改变一致性。)

---

## 4. 问题 2b:钉输入地板的子段分解(离线,新工具)

### 4.1 工具

新增 `scripts/dsv41_consistency/analyze_moe_floor_offline.py`(纯 CPU,~1 min,峰值 ~7 GB host):
输入 = 修复后探针 dump(`probe_input_real4_fix/`,2026-09-22 14:27)+ 引擎 stage dump
(`engine_real4/`,2026-09-22 08:24;引擎侧不读 gate.bias 修复的任何东西——它本来就是 fp32,
两份数据可直接对比)。对比点:

| 子段 | 引擎侧键 | 训练侧键 |
|---|---|---|
| 路由决策(仅层 0,hook 只捕获到第一次调用) | `AscendFusedTopKRouter._compute_routing[0]/[1]`(每 DP rank (25,6),8 份拼接) | `stages["model.layers.0.ffn.gate[0]/[1]"]` |
| MoE(含 shared 的融合输出) | `language_model.model.layers.{i}.mlp.experts` / `.mlp` | `ffn.experts + ffn.shared_experts` / `ffn` |
| attn(模块边界) | `...self_attn` | `attn` |
| norm / head | `model.norm` / `logits_processor`(decode 单行) | `model.norm` / `head` 对应行 |

命令与输出存档:`/tmp/moe_floor_real4_fix.txt`(`python3 scripts/dsv41_consistency/analyze_moe_floor_offline.py --length 200 --variant in_all`)。

### 4.2 结果(in_all:所有模块输入钉成引擎逐位值)

```
[router L0] id-set identical: 1.0000 (200/200)  slot 全同  |Δw| max 1.19e-07
L0  ffn-vs-mlp 0.0037   routed+shared-vs-fused 0.0037   ‖routed‖ 202  ‖shared‖ 115  top1%能量 0.02
L1      "     0.0042          "              0.0045         560           65        0.02
L2      "     0.0038          "              0.0037         352           78        0.01
L3      "     0.0034          "              0.0035         268           72        0.02
attn L0..L3: 0.0052 / 0.0048 / 0.0057 / 0.0060(每 token 分布同样均匀,top1% 能量 ≤0.02)
model.norm 0.0062;logits(引擎 decode 行 vs trainer row199)0.0064,argmax 一致
```

### 4.3 结论(这一步把"还剩什么"钉死了)

1. **路由器本身零残差**:同一份 fp32 输入下,AscendFusedTopK 与训练侧 Gate 的 top-6 **选择逐 token 全同
   (连槽位序)且权重差 ≤1e-7** → gate.bias 修复之后,离散决策不存在两侧不一致;
   baseline 里观察到的翻转**全部**是输入差(核差)喂进 `topk` 的结果。
2. **MoE 地板 = expert GEMM/combine 的均匀核噪声**:逐 token 分布无尾(med≈0.004,p99/med≈1.2,
   top-1% token 只占 1–2% 能量),量级 ≈ bf16 1 ULP;L1–L3 里 ‖routed‖≫‖shared‖(560 vs 65 等)
   → 地板由 routed experts 的累加序主导,shared expert 贡献次要。
3. **attn 地板同量级(0.0048–0.0060)、同样均匀、随深度缓增**;它的**内部**分解(qkv/rope/sdpa-core/o_proj)
   训练侧探针没有记录子段 → 下一步给 `probe_module_inputs.py` 加 attn 子段 hook 重跑(§7 方向 A)。
4. **head 也无离散残差**(argmax 一致,纯 logit 噪声)。
5. **len=64 复算同形**(存档 `/tmp/moe_floor_real4_fix_len64.txt`):router 64/64 全同、|Δw|≤1.19e-7;
   MoE 0.0032–0.0048、attn 0.0048–0.0059、model.norm 0.0062、logits 0.0065 argmax 一致——结论与长度无关。
6. 途中排雷:`inputs[key]` 记录的是**钉之前**的自然输入(record hook 在 replace hook 前触发)——
   一度用 `stages["ffn_norm"]` 和引擎 router 输入比出 6.6e-3 的差怀疑映射错误;核对
   `probe_module_inputs.py:332–359` 后确认:gate 与 router 的对比两侧吃的是**同一份钉死输入**,
   1e-7 的一致性成立。(6.6e-3 本身也是有用信息:那是自然态下两侧 norm 输出的典型差,~bf16 1 ULP。)

---

### 4.4 attn 地板的子段分解(带 attn hook 的 probe 重跑 + 新工具,13:08–13:2x)

**E2 验证通过先记**:重跑 `probe_module_inputs.py`(输出新目录 `probe_input_real4_fix_attn/`,
不动 §5.2 引用的旧目录),**12 分钟跑完**(4 变体×2 长度;上次 20 min 是共享 NPU 排队,现在独占);
8 份 dump 全部带 20 个 attn_op 键;**回归:新旧 `in_all` 的 next_logprobs 逐位相同(max diff 0.0)**。

新工具 `analyze_attn_floor_offline.py`(CPU ~1 min):引擎的 attention 是**按头切到 8 个 worker rank**
(每 rank q=(200,8,512),全部 200 token;训练侧本地 64 头),头映射实测为**连续切块**
(contiguous rel_err 0.0000 vs strided 1.0974)。逐层结果(in_all,存档 `/tmp/attn_floor_real4_fix.txt`):

```
L    eps_q  eps_core  q≠0元素  eps_module   core per-token med/p99
0   0.0000    0.0026    896      0.0052      0.0019/0.0054
1   0.0000    0.0024    208      0.0048      0.0020/0.0048
2   0.0000    0.0036   2532      0.0057      0.0029/0.0085
3   0.0000    0.0038     57      0.0060      0.0032/0.0085
```

- **q 投影(wq_a/q_norm/wq_b/rope)基本消灭**:rel_err <1e-4,不一致元素 57–2532 / 6.55M
  (0.001–0.04%,全是 bf16 1 ULP)→ 两侧 GEMM 在这个形状上几乎逐位一致。
- **attention core(SparseFlashMla vs 训练侧 indexed_sparse_attention)≈0.0024–0.0038**。
- **core 之后的 inverse-rope + o_proj 尾巴是 attn 地板大头**:0.0026→0.0052(module),
  正交合成 ≈0.0045 出自这一段。
- **attn_sink 值不对称(仅 harness 态!)**:训练侧 sink 是 bf16 值(`|sink−bf16(sink)|=0`,它是要梯度的
  parameter,被 FSDP 转 bf16),引擎 harness 态持 ckpt-fp32 原值;fp32→bf16 往返 rel_err =
  **1.55e-3 / 1.83e-3 / 1.77e-3 / 1.55e-3**(层 0–3,safetensors 直读)。sink 进 softmax 分母,
  这 ~0.16–0.19% 的**值差**就藏在 eps_core 里 → 生产里开局同步会把它抹平(上篇 §7.2 的降级机制,
  这反而是"生产地板 < harness 地板"的一项)。**判读:attn core 的纯核差 ≲ 0.002**。

### 4.5 合成为"生产相关性地板表"

| 子段 | harness 地板 | 生产态(已同步)估计 | 性质 |
|---|---|---|---|
| MoE router | **0**(§4.3-1) | 0 | 已对齐(fp32 bias 修复) |
| attn q 投影 | <1e-4 | <1e-4 | 基本无差 |
| attn core | 0.0024–0.0038 | **~0.002**(去掉 sink 值差) | 核差(mask/metadata/scale 参数待对齐,§7A′) |
| attn 尾(inverse-rope+o_proj) | **~0.0045** | ~0.0045 | **核差,attn 侧最大项** |
| MoE GEMM/combine | 0.0034–0.0045 | 同 | 核差,全局最大单一项 |
| norm/head | 0.0062 / 0.0064(累积) | 同量级 | 上面各项的传播 |

→ ~~"压 σn 0.24→0.15 = 压三块均匀核差"~~(**此判断被 §3.1/§4.7 修正**:probs 空间下 pearson 由
~1% 高置信翻转 token 主导、微基准又证明 o_proj 形态无罪 → 三块核差是并行布局级的结构差,不可单点消除);
离散/参数级的仗确实打完了(§4.3-1 路由零残差;sink/hc 生产态由同步抹平)。

### 4.6 归因的代码级收口(13:2x,静态读码)

把 §4.4/4.5 的三块核差落到**具体哪个算子不同**(训练侧读 `FSDPTurbo`,引擎侧读
`vllm-ascend-v41-private/vllm_ascend/attention/dsa_v41.py`):

1. **attn 尾 ≈ o_proj,不是 rope**:q 侧两侧都做了 rope 而 eps_q<1e-4 → rope 公式/数值表两侧
   实现基本同值。**o_proj 不同**:`model.py:911–915` 训练侧是"inverse `apply_rotary_emb` +
   block-diag `wo_a` 的 **einsum**(注释:ckpt fp8 已 dequant 成 bf16)";引擎是
   `inplace_partial_rotary_mul(-sin, interleave)` CANN op + `_forward_o_proj`。
   einsum(分组小 GEMM)vs 融合 matmul 的 **累加序**差 = 那个 ~0.0045 尾巴的最可能来源。
2. **MoE GEMM 地板 0.0034–0.0045 不是 GEMM 本身不同**:训练侧 `experts.py:183/200` 用的就是
   **`torch_npu.npu_grouped_matmul`**(`ops/npu/grouped_matmul.py:32`),与 Ascend 融合 MoE 同族;
   差在 **combine/加权求和的顺序与累加 dtype**(引擎 mc2 融合 dispatch→gemm→combine;
   训练侧 grouped matmul 后再按 topk 权重 fp32 求和)。
3. **core ≈0.002**:SparseFlashMla 的元数据/掩码模式(`topk_value_mode=1/ori_mask_mode=4`)两侧
   同一族 op,残差主要是 kv 侧 rope(interleave in-place vs complex mul)+ PA 元数据分块;
   harness 里还叠着 sink 值不对称(生产无,§4.4)。

**含义:0.999 没有"改一行"的参数级路;三个可选攻势(按性价比)**:
- ~~**C(fulldet A/B)**:引擎换 batch-invariant 内核~~ **已证伪关闭**(§5.5:脚本 252/261 行
  默认就开着 `rl_config.enable_batch_invariant=true`,基线 0.995 就是它的结果;增量只是训练侧
  复现标志,不动核差;这次还在 wake_up 处 NPU OOM)。
- **TIS/IS 消化(推荐主路)**:`algorithm.rollout_correction`(`rollout_is: token` +
  `rollout_is_threshold: 2.0` = 截断重要性采样;或 `decoupled_seq_is` 预设)。它不动内核,
  把 0.995 的训推差以重要性权重形式**正确计入 PG loss**——RL 语义上这才是终点;pearson 本身
  不会变,但"差距对训练的影响"归零。(注意默认配置下该差不进 ratio:old_logprobs 是训练侧重算的。)
- ~~**深对齐**:训练侧 o_proj 换引擎的 CANN matmul 调用式、MoE combine 换融合通道、kv-rope 换
  interleave~~ **已被 §4.7 微基准提前证伪**(o_proj 两种调用式 99.98% 逐位一致——"换皮"收益为零;
  真差在 TP 切分+HCCL reduce 结构,不是 op 形态)。要统一只剩"引擎放弃 attention 头切分"一条,
  代价是 rollout 吞吐,不建议。

### 4.7 o_proj/rope 调用形态微基准:**einsum-vs-matmul 被证伪,尾巴归因修正为"并行结构差"**(14:36,NPU×1)

新工具 `oproj_rope_parity_microbench.py`(单 NPU、无分布式;真实 ckpt `wo_a (8192,4096)` bf16 +
真实记录的层 0 core 输出 + 合成对照;两种布局都测)。结果:

```
[T2] o_proj grouped GEMM: einsum("sgd,grd->sgr") vs npu_transpose_batchmatmul(perm=(1,0,2),(0,1,2),(1,0,2))
  recorded 输入: bit-exact 99.98%,rel_err 3.3e-5;两者到 fp32 参考的距离相同(1.659e-3)
  synthetic 输入: bit-exact 99.99%,rel_err 2.7e-5
[T1] inverse rope: op 不可用('_C_ascend' 未注册——bench 里 import vllm_ascend 不足以加载 C++ 扩展;
     见"错误"段)
```

**这张表如何读**:行 = 两种 o_proj 调用形态在**同一输入**上的输出对比;`bit-exact %` 是逐位相同元素占比;
`到 fp32 参考距离相同`意味着 einsum 与 transpose_batchmatmul **不是"谁更糙"的关系,而是同一个 kernel 的两种皮**
(误差都只是 bf16 输出舍入)。

**推论(修正 §4.5/§4.6-1)**:attn 尾的 ~0.0045 **不来自调用形态**(3e-5 可忽略)。真正候选是
**并行结构差**:引擎 attention 按头切到 8 rank(每 rank 8 头),每 rank 只算自己 group 的
partial o_proj,模块输出还要经过 **HCCL all-reduce 求和**——训练侧是 64 头本地一次 einsum,
**没有这次 reduce**。8 份 bf16 partial 的求和顺序/舍入 vs 一次全量 einsum,量级正是 ~1 ULP×√8≈0.4%。
`dsa_v1.py:1559` 注释也印证引擎把 wo_a 重排成 [groups, hidden, rank] 的 A3 布局配合这个切分。
**→ 尾巴属于"两套栈的并行布局"层,不能靠换 op 皮消除;核差三块全部升级为"结构性"**。
结合 §3.1(pearson 由 1% 高置信 token 主导、翻转对扰动呈阈值响应),现实的杠杆排序变成:

1. **TIS(rollout_correction)**:不碰核差,把翻转 token 的后果以 IS 权重计入 PG——正在跑(§5.8)。
2. 引擎侧把 attention 从"TP 头切+all-reduce"换成与训练侧同形的本地全头计算(改引擎,EP8 布局下 = 放弃
   attention 的 TP,rollout 吞吐代价,工程量大)——不建议。
3. 40 层复跑时按"结构差"预期基线,把验收指标定在 TIS 生效上而非 pearson 阈值。

**错误与修复**:T1 失败因 `torch.ops._C_ascend` 需要引擎完整启动流程才注册(bench 只 import 到插件注册层,
`vllm_ascend/__init__` 走的是 platform plugin,不加载 `_C` so)。T2 已足以否定"形态差"假设,T1 未再补;
若将来要测,加载方式为 `import vllm_ascend._ops`(或从 `torch.ops._C_ascend` 改用
`vllm_ascend.utils` 里的 helper)——留给 §7 记录,不在关键路径上。

---

## 5. 重跑:fast sync + 生产 dump(2026-09-23 12:49 启动)

### 5.1 命令

```bash
cd /workspace-verl/verl
LOCAL_EXPERT_EXPORT=1 STEPS=2 \
EXPERIMENT_NAME=GRPO-DSV41-real4-fastsync \
VERL_DSV41_DUMP_BATCH=/tmp/dsv41_batch_real4_fast \
bash scripts/dsv41_consistency/sh/run_real4_grpo.sh
```

(其余环境沿用脚本:真实 4 层权重、bsz 8×n4、512+128、`MASTER_ADDR=127.0.0.1`+`GLOO_SOCKET_IFNAME=enp48s3u1u1`。)

### 5.2 判据

- **等价性**:sender 应为 ~670 tensors/16.37 GiB、export ~1s、`timing_s/update_weights` ~20s;
  指标应与 10:45 同带(`diff_mean ~5e-3 / pearson ~0.995 / kl ~2e-2`),**不允许**掉到 O(1)/~0.5(=配对错)。
- **提速**:step 应从 1230s → ~215s。
- 新 dump(`/tmp/dsv41_batch_real4_fast/batch_step*.pt`)复用 §3 的分解脚本再走一遍,确认 drop-tail 形态与
  fast-sync 前一致(排除"同步通道差异改变噪声结构"的可能)。

### 5.3 记录

- 日志:`logs/DeepSeek-V4.1-Flash-4layer-real-20260923_124957.log`。
- 12:50 起 8 个 WorkerDict 拉起、13:00 模型加载中(56.35B/rank,与 10:45 同路径)。

### 5.4 结果:**快速同步通道在真实权重上定版**(13:08 跑完,exit 0)

```
SYNC-PROFILE sender: 670 tensors, 16.37 GiB, 30 buckets | export 0.1–0.9s, flush 11.2–11.8s   (8/8 rank)
[fsdp_turbo_dsv41] checkpoint buffers loaded: 4 (bit-exact against the checkpoint)             (actor+ref 各一次)
```

| step | 10:45(默认 sync) | 12:49(fast sync) | 变化 |
|---|---|---|---|
| update_weights | 1032.65s | **18.75 / 17.83s** | **55×** |
| gen | 98.62s | 68.72 / 23.36s | step2 引擎已热 |
| update_actor | 69.30s | 77.40 / 62.73s | ≈ |
| **step 合计** | **1230.1s** | **193.9 / 112.7s** | **6.3× / 10.9×** |

指标与默认路径**同带且无断层**(配对正确的自检测通过):

| | 10:45 step1 | 12:49 step1 | 12:49 step2 |
|---|---|---|---|
| diff_mean | 0.00471 | 0.00457 | 0.00447 |
| pearson | 0.99505 | 0.99415 | 0.99554 |
| rollout_corr/kl | 0.02413 | 0.02601 | 0.02877 |

新 dump(`/tmp/dsv41_batch_real4_fast/batch_step{0,1}.pt`)的 token 级分解与旧 dump **逐指标重合**
(pearson 0.99741/0.99750 vs 0.99756/0.99744;|Δ| 中位数 0.051 vs 0.049;diff std 0.242–0.244 vs 0.240–0.244;
drop-5% → 0.99929/0.99937 vs 0.99934)——**同步通道不改变噪声结构**,fast 通道完全可用于一致性验证。
(顺带:两次运行的 pearson 步间波动 ±0.001,量级 ~噪声,后续 A/B 判读时把它当底线。)

### 5.5 fulldet A/B(2026-09-23 13:2x 启动,与 fast-sync 唯一差异 = `full_determinism=True`)

```bash
LOCAL_EXPERT_EXPORT=1 STEPS=2 EXPERIMENT_NAME=GRPO-DSV41-real4-fulldet \
VERL_DSV41_DUMP_BATCH=/tmp/dsv41_batch_real4_fulldet \
bash scripts/dsv41_consistency/sh/run_real4_grpo.sh \
  actor_rollout_ref.rollout.full_determinism=True
```

(`run_real4_grpo.sh` 为此加了一处透传:`"$@"` 追加到 examples 脚本调用——见 §6。)
预期:`main_ppo.py:47-50` 会向全 actor 广播 `VERL_FULL_DETERMINISM=1 + VLLM_BATCH_INVARIANT=1`
+ 固定 `PYTHONHASHSEED`;判据 = 与 §5.4 对照 pearson / diff std(离线 dump 分解)是否改善。

**结果(14:03):失败于引擎 wake_up 的 NPU OOM,且该杠杆判定为死路,不再重试。** 三层信息:

1. **直接死因**:step1 完成、开局同步正常(`SYNC-PROFILE step2 (receive+load) 12.8s`)后,
   下一次 rollout 前的 sleep/wake 循环里 C++ terminate:
   `aclrtMallocPhysical failed ... 207001(OOM)` @ `vllm-ascend-v41-private/csrc/camem_allocator.cpp:67`
   → Executor 崩 → `wake_up cancelled` → 任务退出。fulldet 在 vLLMHttpServer 里额外打开的
   确定性路径(`enable_full_determinism`:HCCL_DETERMINISTIC、`VERL_DISABLE_FLASH_ATTN_CE` 等,
   `workers/engine/utils.py:31-51`)带来常驻 workspace,`gpu_memory_utilization=0.6` 的预算装不下。
2. **为什么这是死路(关键发现)**:启动脚本 `run_deepseek_v41_grpo_fsdp_turbo_npu.sh:252/261`
   **默认就传了** `rl_config.enabled=true + rl_config.enable_batch_invariant=true` ——
   引擎侧 batch-invariant **早就开着**,§5.4 的 0.9942/0.9955 基线就是它的结果。
   `full_determinism` 的增量只剩训练侧可复现性标志(降 CE kernel、NCCL ring 之类),
   它不改变"两套栈的 dense kernel 不同"这件事 → **就算不 OOM,也几乎不会动 pearson**。
   残余核差(einsum-vs-融合matmul、grouped-matmul-vs-mc2-combine)不是开关能对齐的。
3. 处置:杠杆 C 关闭;fulldet 若将来为"复现性"再开,需同时降 `ROLLOUT_GPU_MEM_UTIL`(0.6→0.5)
   再试。**主路转为 D(rollout_correction/TIS)**;A′ 微基准仅保留"钉死归因"的记录价值。
   失败现场的 ray 残骸已清理(`pkill -9 -f "ray::|raylet|gcs_server"`,14:08,NPU/host 已释放)。

### 5.8 TIS A/B(2026-09-23 14:38 启动,结果待回填)

```bash
LOCAL_EXPERT_EXPORT=1 STEPS=2 EXPERIMENT_NAME=GRPO-DSV41-real4-tis \
VERL_DSV41_DUMP_BATCH=/tmp/dsv41_batch_real4_tis \
bash scripts/dsv41_consistency/sh/run_real4_grpo.sh \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0
```

配置树里 `algorithm@algorithm.rollout_correction` 已默认存在(此前 `rollout_is: null` = 关闭);
本次打开 token 级 TIS。实现链路(读码确认,微观依据):`ray_trainer.py:1574/1655-1658`
(fit 中计算 IS 权重并加入 batch)→ `rollout_corr_helper.py:522 compute_rollout_correction_weights`
(权重 + `rollout_is_*` 指标)→ `workers/utils/losses.py:87-111` 把 `rollout_is_weights` 传进
policy loss → `core_algos.py:931-933` 以 ρ̄² 缩放 W(最小化截断 IS 的 MSE 修正)。

**判据**:
- 新指标出现且合理:`rollout_is_mean`(≈1)、`rollout_is_max/min`、
  `rollout_is_ratio_fraction_high`(超阈 token 占比,参照 §3.1 应在 1–5% 量级)、`rollout_is_oob_ratio`;
- 对照:pearson/diff_mean 应与 §5.4 **基本不变**(TIS 不改前向,只改 loss 加权)——它同时是本
  A/B 的"没有误伤前向"检查;
- dsv41 recipe 首次走 `bypass_mode=False`(Decoupled,3 policies)+ vanilla GRPO:跑通即集成验证成功。
**结果(15:05 跑完,exit 0):三条判据全过,TIS 在 dsv41 recipe 上宣布可用。**

| 指标(行=指标;列=含义+读数) | step1 | step2 | 怎么读 |
|---|---|---|---|
| `rollout_is_mean` | 0.9919 | 0.9977 | IS 权重均值 ≈1 → 权重整体中性,没有系统性放大/压扁 |
| `rollout_is_max` | **2.0** | **2.0** | 恰等于 threshold → **截断在发生**(TIS 的"truncated"生效) |
| `rollout_is_min` | 0.110 | **0.033** | 另一侧被压到 1/30 —— 翻转 token 被大幅降权,正是 §3.1 那批高置信分歧 token |
| `rollout_is_ratio_fraction_high` | **0.68%** | **0.93%** | 超阈 token 占比 —— **与 §3.1 独立测得的"worst ~1% 高 p token 决定 pearson"数量级互证**:两条不同的路(dump 离线分解 vs 训练在线统计)指向同一批 token |
| `training/rollout_actor_probs_pearson_corr` | 0.9946 | 0.9929 | 与 §5.4 基线同带(**预期不变**:TIS 不改前向),证明"没误伤" |
| `timing_s/step` | 178.5 | 113.9 | fast-sync 收益保持;TIS 计算本身零成本(在已有张量上逐元素) |
| `actor/pg_clipfrac` / `pg_loss` | 0 / 0 | 0 / 0 | ⚠️ 仍是 reward 全 −1 → advantage 0 的数据退化,**IS 权重对梯度端的实际影响还没被这次 A/B 检验**——权重链路正确性以 `rollout_is_*` 指标 + `core_algos.py:931-933` 读码为证;要检验训练效果需要让 reward 有方差(更长 response/更好数据或真实模型) |

配置生效证据:日志中 resolved config `algorithm.rollout_correction.rollout_is: 'token'`、
`rollout_is_threshold: 2.0`(其余与 §5.5 fast-sync 跑完全一致);集成路径
`ray_trainer.py:1655-1658 → rollout_corr_helper.py:522 → losses.py:87-111 → core_algos.py:931-933`
首次在此 recipe 上走通(Decoupled,3 policies,`bypass_mode=False`)。
日志:`logs/DeepSeek-V4.1-Flash-4layer-real-20260923_143811.log`;dump:
`/tmp/dsv41_batch_real4_tis/batch_step{0,1}.pt`。

**dump 侧交叉验证(15:1x,CPU)**:`ratio = exp(old_log_probs − rollout_log_probs)` 在 mask 上统计:

```
step0: frac(ratio>2.0) = 0.68%   max 6.53  min 0.110   frac(ratio<0.5) = 2.12%
step1: frac(ratio>2.0) = 0.93%   max 3.25  min 0.033   frac(ratio<0.5) = 1.83%
```

与训练进程内指标(`rollout_is_ratio_fraction_high 0.68%/0.93%`、`rollout_is_min 0.110/0.033`)
**逐位吻合** → IS 权重链路端到端数值一致,`rollout_is_*` 不是装饰性指标。
**错误记录(自查)**:第一版验证脚本把 ratio 写成了 `exp(a)/mask`(量纲都不对,得到 95.68% 的荒谬值)——
提醒:mask 用于选取元素(`tensor[mask]`),不参与除法;改对后秒级吻合。另注意 ratio<0.5 的低侧占比(≈2%)
比高侧大 2-3 倍:**翻转主要把训练侧概率压低**(高置信 token 两侧 argmax 分歧 → actor 给低概率),
这与 §3.1 "坏 token 是高 p token" 一致。

### 5.6 重跑中踩到/修掉的两个 harness 问题

**E1(真 bug):`install_attention_op_hooks` 的跨样本陈旧闭包。**
修复回归的 smoke 日志打印 `len=64 … stages=57` 而 `len=200 … stages=37`——差 20 = 4 层×5 个
`attn_op*.{query,key_value,attn_sink,topk_indices,output}`。根因:`dump_trainer_stages.py:339` 的包装器
按样本反复安装,但 `getattr(original, "_dump_wrapped")` 早退守卫让**第二个及以后的样本继续写进第一个
样本的 `stage_sink`**(闭包捕获)→ 后续样本的 dump 永远没有 attn 子段(len64 文件本身是正确的,
其 attn_op 张量在保存之后才被后续样本的 forward 污染?不会——每个样本先存盘再跑下一个,所以
len64 文件干净、len200 文件缺项)。
**修法**:去掉早退守卫,用 `_dump_original` 标记解包原始函数、每次调用**重新绑定**到当前 sink。
(`dump_trainer_stages.py:339–373`,2026-09-23。)

**E2(缺失能力):probe 根本没装 attn 子段 hook。**
`probe_module_inputs.py` 只装 `dt.install_hooks`(模块级),所以 in_all 变体的 0.005 attn 地板
无法下钻。修法:变体循环里加一行 `dt.install_attention_op_hooks(torch, stage_sink, args.max_elements)`
(依赖 E1 的修复才能每变体正确绑定)。**验证方式**:重跑 probe(4 变体×len 64/200,输出
`dump/probe_input_real4_fix_attn/`,13:2x 启动)——len200/in_all 的 dump 里出现 20 个 attn_op 键即成。

引擎侧对应键已勘明(`engine_real4`,层 0 例):`multistream_preprocess@…self_attn[0]`(200,8,512)=query、
`[1]`(200,1280)=压缩 kv、`_attention@…`(200,8,512)=core 输出、`forward@…`(200,5120)=o_proj 后;
与 trainer `attn_op{i}.*` 的语义对齐需要读一次 vllm-ascend `multistream_preprocess`(头数 8 vs 64 的映射,
待回填 §7A)。**注意**:trainer 侧 attn_op 的 q 是 (T,64,512)(64 头),引擎 [0] 是 (T,8,512)(8 头),
不是同一个中间量——对表前先把语义搞清楚,别硬比。

---

## 6. 本篇改动的文件

| 文件 | 改动 | 动机 |
|---|---|---|
| `scripts/dsv41_consistency/analyze_moe_floor_offline.py` | **新增**:钉输入地板的子段离线分解(router/MoE/attn/head),纯 CPU,读现有 dump | §4 的问题不用重新占 NPU 就能答;可复用于 40 层复跑 |
| `scripts/dsv41_consistency/sh/run_real4_grpo.sh` | 注释更新:fast-sync 已在真实 384 专家权重两次验证(09:38/§5.4),real4 迭代默认用 `LOCAL_EXPERT_EXPORT=1` | 防止下一个跑再吃 1032s 同步 |
| `scripts/dsv41_consistency/dump_trainer_stages.py:339` | **修 bug**(§5.6 E1):attn-op hook 每样本重绑定 sink,不再被 `_dump_wrapped` 早退守卫钉死在第一个样本 | 否则除首个样本外所有 dump 都没有 attn 子段 |
| `scripts/dsv41_consistency/probe_module_inputs.py:364+` | **加能力**(§5.6 E2):变体循环装 `install_attention_op_hooks`,attn 内部量随 probe dump 落盘 | 0.005 attn 地板的下钻测量(§4.4,已完成) |
| `scripts/dsv41_consistency/analyze_attn_floor_offline.py` | **新增**:attn 地板子段离线分解(q vs core vs 尾;头映射实测 contig;sink 校验) | §4.4 的产出工具 |
| `scripts/dsv41_consistency/sh/run_real4_grpo.sh` | 追加 `"$@"` 透传 hydra 覆盖项 | §5.5/§5.8 的开关 A/B 不用改脚本 |
| `scripts/dsv41_consistency/oproj_rope_parity_microbench.py` | **新增**:注意力尾两段的调用形态逐位对比(单 NPU) | §4.7 的证伪实验——把"换 op 皮能对齐"的假设提前杀死,避免动生产代码 |

新数据产物:`dump/probe_input_real4_fix_attn/`(带 attn_op 的 8 份 dump + 2 份 report;
`in_all` 与旧目录 logprob **逐位一致**,旧目录未动)、`trainer_real4_fix/`(重录,两长度都 57 stages)、
`/tmp/attn_floor_real4_fix.txt`、`/tmp/dsv41_batch_real4_fast/`。
(未改动任何训练/引擎**生产**代码路径——改动全部在 `scripts/dsv41_consistency/` 的 harness 内。)

---

## 7. 状态板与下一步(13:3x 更新)

- **A attn 子段分解:完成,且尾归因已被微基准改写**(§4.4→§4.7)——地板:core≈0.002、尾≈0.0045,
  但 §4.7 证明尾**不是** einsum-vs-matmul 形态差(99.98% 逐位一致),而是引擎"TP 头切+HCCL all-reduce
  求和"vs 训练侧"本地全头一次 einsum"的**并行结构差**;q 侧 <1e-4,rope 无罪。
- **A′(可选,NPU 5–10 min)rope/o_proj 微基准钉死归因**:同一 NPU 上对同一 bf16 张量分别跑
  训练侧 `apply_rotary_emb(·, inverse=True)`+einsum(wo_a) vs 引擎
  `inplace_partial_rotary_mul(-sin)`+`_forward_o_proj`,逐位比。预期:rope 差 ~0(与 eps_q 一致)、
  einsum-vs-matmul 差 ~1 ULP。它不改变行动(三块核差都已定位),只是把 §4.6-1 从"最可能"变成"实测"。
- **B sink/hc 值不对称:离线完成一半**(§4.4:sink fp32→bf16 rel 1.55–1.83e-3/层,生产被同步抹平)。
  另一半(精确扣除 sink 后的 harness core 值)需要 emulated-sync 引擎 dump(40–70 min),
  优先级降低——只影响 harness 数字的解释,不影响生产。
- **C fulldet A/B:已关闭**(§5.5——引擎 batch-invariant 本来就是脚本默认开,该开关无增量;
  这次还以 OOM 收场。0.995 的残余**没有现成开关**能压)。
- **D TIS(rollout_correction)**:**已实施并验证**(§5.8,token 级 threshold=2.0,三判据全过;
  遗留 F=梯度端效果待有方差的 reward)。
- **F(新)让 reward 有方差,验证 TIS 的训练端效果**:三个便宜选项——① `MAX_RESPONSE_LEN=512`
  让模型有机会不撞长度截断(现在是 clip_ratio 1.0 → 全员 −1);② 换 `TRAIN_FILE` 里更短/更易的样本;
  ③ 用 scaled fixture 造一个"答案在词表里"的合成任务。判据:`actor/pg_clipfrac>0`、
  `rollout_is_mean≠1.0` 且 pg_loss 非零、开/关 TIS 两跑的 advantage-weighted gradient 差异可测。
- **E 40 层复跑**:前置检查见上篇 §7.3;建议带上 `LOCAL_EXPERT_EXPORT=1`(§5.4)+ D 的 correction 一起排期。

(原始候选清单存档如下,已被上面取代:
A=attn 内部分解→已完成;B=同步态对照→离线半完成;C=full_determinism→在跑;D=TIS→升为主路;E=40 层→不变。)

---

## 8. 复现命令汇总

```bash
# §2 慢的定位(任何日志通用)
grep -o "timing_s/[a-z_]*:[0-9.]*" <log>
grep -n "SYNC-PROFILE" <log>

# §3 生产 dump 的 token 级分解(改 dump 目录即可复用)
python3 - <<'EOF'   # 正文里的内联脚本,核心:pearson + drop-tail 曲线
import torch; d=torch.load('/tmp/dsv41_batch_real4/batch_step0.pt',map_location='cpu',weights_only=False)
...
EOF

# §4 地板子段分解(CPU ~1 min,~7 GB host)
python3 scripts/dsv41_consistency/analyze_moe_floor_offline.py --length 200 --variant in_all
python3 scripts/dsv41_consistency/analyze_moe_floor_offline.py --length 64  --variant in_all

# §4.4 attn 子段分解(读 probe_input_real4_fix_attn/,CPU ~1 min)
python3 scripts/dsv41_consistency/analyze_attn_floor_offline.py --length 200

# §5 fast-sync 重跑
LOCAL_EXPERT_EXPORT=1 STEPS=2 bash scripts/dsv41_consistency/sh/run_real4_grpo.sh
```
