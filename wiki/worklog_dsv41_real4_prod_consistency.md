# DeepSeek-V4.1 真实 4 层权重的**生产一致性**:同步提速 + pearson 0.995 的逐层分解(工作记录,2026-09-23)

> 上一篇:[`worklog_dsv41_gate_bias_fp32_fix.md`](worklog_dsv41_gate_bias_fp32_fix.md)(fp32 纠偏 bias 修复,已完成并有 §5 回归)。
> 本篇记录修复之后**第一次真实权重的生产 GRPO 运行**(`DeepSeek-V4.1-Flash-4layer-real-20260923_104558.log`)暴露的两个问题、
> 它们的定位过程、以及为"提升验证速度 → 继续定位不一致"所做的改动。
>
> 状态:**问题 1(慢)已定位并修复中;问题 2(pearson 0.995→0.999)已完成离线分解,残差结构已定形**。
> 最后更新:2026-09-23 13:00(fast-sync 重跑仍在进行,结果回填 §5.4)。

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
   - 钉输入探针的离线子段分解(新工具 `analyze_moe_floor_offline.py`):同一份钉死输入下
     **路由器 0 残差**(top-6 选择 200/200 含槽位序全同、权重 |Δ|≤1.2e-7),地板全部来自
     **expert GEMM/combine(attn 与 MoE 各 ~0.003–0.006)且完全均匀**(top-1% token 只占能量 1–2%),
     head 也无离散残差(logit 行 rel_err 0.0064、argmax 一致)。
   - 定量目标:production 噪声 std 0.24 → **≤0.15** 才够 0.999(信号 std 3.4,`r≈1/√(1+(σn/σs)²)`),
     即核差要整体压 ~1.6×;dtype/参数级的路已经走完了,剩下是内核累加序/精度路径的工程。

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
- 本地重算 0.9976 vs 日志 0.9950:聚合口径略有差(日志端在训练进程内算,dump 端是落盘 fp32 张量重算);
  不影响形态判断,**以 drop-tail 结构为主要证据**。
- 定量到目标:噪声 std σn=0.24、信号 std σs=3.43,`r ≈ 1/√(1+(σn/σs)²)` → 0.999 需要 **σn ≤ 0.153**
  (压到 63%)。
- ⚠️ 作用域提醒:该 fixture 是 4 层未训练 policy(entropy 5.9 / ppl ~366),σs 偏小,pearson 对噪声
  比真实训练模型更敏感;同样的核差在 40 层真实 policy 上预计指标更好看,但**绝对 σ 不保证更小**
  (上一篇 §6.1.3 的层数 caveat 仍然有效)。

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

### 5.5 重跑中踩到/修掉的两个 harness 问题

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
| `scripts/dsv41_consistency/dump_trainer_stages.py:339` | **修 bug**(§5.5 E1):attn-op hook 每样本重绑定 sink,不再被 `_dump_wrapped` 早退守卫钉死在第一个样本 | 否则除首个样本外所有 dump 都没有 attn 子段 |
| `scripts/dsv41_consistency/probe_module_inputs.py:364+` | **加能力**(§5.5 E2):变体循环装 `install_attention_op_hooks`,attn 内部量随 probe dump 落盘 | 0.005 attn 地板的下钻测量(§7A) |

(未改动任何训练/引擎**生产**代码路径——本轮两个问题一个是环境开关,一个是测量分析;改动全部在
`scripts/dsv41_consistency/` 的 harness 内。)

---

## 7. 下一步(按性价比)

- **A. attn 地板的内部分解**(占 NPU ~25 min):给探针加 attn 子段 hook(qkv/rope/core/o_proj;引擎侧
  `multistream_preprocess/_attention/forward` 已有键),把 0.005 的地板落到具体算子 → 才谈得上"改哪个核"。
  同理 MoE 侧想区分 grouped-GEMM 的 tile/split-k 序,可尝试在训练侧把 expert GEMM 换成参考实现
  (fp32 累加路径)做一次 A/B。
- **B. "已同步生产态"对照(上篇 §6.2 遗留,可离线半步)**:修复后两侧仅差 hc_*/attn_sink 这 28 个
  ckpt-fp32 参数在生产的开局同步中被降级成 bf16 值。可先**离线**评估其量级:在钉输入对比里把训练侧
  `hc_pre/hc_post` 相关张量与引擎值(`hc_pre@layer{i}[j]` 已 dump)逐点比 —— 引擎在生产态=bf16 舍入值,
  与 harness 态(ckpt fp32)的差即该项贡献,不需要重跑。若占比可忽略,遗留项直接关闭。
- **C. 现成杠杆 A/B(便宜)**:`actor_rollout_ref.rollout.full_determinism=True`(main_ppo.py:47-50 →
  `VERL_FULL_DETERMINISM=1 + VLLM_BATCH_INVARIANT=1`,vllm-ascend 有 batch-invariant 内核),
  各 2 step 看 σn 是否下降;这是唯一"不改代码就可能压核差"的开关。
- **D. RL 之外的正解**:`algorithm.rollout_correction`(TIS)已存在于配置树
  (`rollout_correction: {'bypass_mode': False...}`,log 522 行)——若 A/C 到不了 0.999,
  一致性差距用重要性修正消化,而不是继续追内核。
- **E. 40 层复跑**(上篇计划方向 3)不变。

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

# §5 fast-sync 重跑
LOCAL_EXPERT_EXPORT=1 STEPS=2 bash scripts/dsv41_consistency/sh/run_real4_grpo.sh
```
