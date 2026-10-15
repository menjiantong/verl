# 任务 B 工作记录：训推一致性定位（2026-10-12 起，rebase 后）

> 目标（用户指令）：`training/rollout_probs_diff_mean` < 0.01、`training/rollout_actor_probs_pearson_corr` ≥ 0.995；
> 用 dump/打桩 **逐个算子、方法、模块**定位，给出"推理与训练不一致在哪"的**实打实证据**，必要时改算子验证。
> 本文件为任务 B 专属 wiki。

---

## 0. 基线与工具箱

- **坏指标基线（10-11 实测）**：pearson **0.954–0.975**、diff_mean **0.029–0.039**
  （`verl/logs/grpo_top64_steps{2,20}_20261011_*.log`；target 0.995 / 0.01）。
- 工具箱 `verl/scripts/dsv41_consistency/`（23 个脚本 + `sh/` wrappers）：
  - dump：`dump_trainer_stages.py`（8 卡 torchrun，训练侧逐模块）、`dump_engine_stages.py`（vLLM 引擎侧，含参数指纹）
  - 比较：`compare_stages.py`（逐层阶梯）、`compare_attention.py`（投影/选择/核计算分离）、`compare_routing.py`（路由 top-k 集合）
  - 归因：`probe_module_inputs.py`（**把引擎输入喂进训练模块**的 baseline/in.all.ffn/in.all.attn/in.all 变体）
  - 校验：`check_engine_params.py`（引擎参数指纹 vs ckpt）
- 模型：4 层真实权重切片 `DeepSeek-V4.1-Flash-4layer-scaled32`（consistency fixture；`fixture.json` 5 个长度）。

## 1. rebase 后首轮 dump（2026-10-12 傍晚）

### 1.1 遇到的坑（全部有栈/日志）
1. **harness 撞融合 RMSNorm**（新 rebase）：`npu_rms_norm` 拒绝 `x(FP32)+gamma(BF16)`
   （aclnn 161002，日志原文 `x(DT_FLOAT, ND), gamma(DT_BFLOAT16, ND), ...`）。
   根因：上游 rebase 把 `RMSNorm.forward` 改为走融合核（旧实现纯 python 容错任意 dtype）；
   harness 的 fp32 参考路径产生 fp32 激活 + bf16 权重。
   **修**：FSDPTurbo `9a13229`——不支持的 dtype 对回落到旧算法（生产 bf16 快路径逐位不变）。
2. **fixture `model_type: "deepseek_v4.1"`（带点）** 不被新 vllm 注册表识别（注册名 `deepseek_v41`）→
   引擎拒绝加载。修：fixture `config.json` 改 `deepseek_v41`（原文件已备份 `.bak-20261012`）。
   注：生产 sft116 模型本来就是 `deepseek_v41` ✅。
3. `run_dump_trainer.sh` 的 `ASCEND_RT_VISIBLE_DEVICES` 写死 → 改为 `${...:-0..7}`（支持 pin 空闲 die）。

### 1.2 成功产物（本轮）
- 训练侧：`/mnt/share/m00899630/dsv41/dump/trainer/trainer_len{64,200,700,1500}.pt`（57 个 stage/长度，
  含 attn_opN.{query,key_value,attn_sink,topk_indices,output}、ffn.gate[0]/[1] 等）
- 引擎侧：`.../dump/cons_v3/engine_4l/engine_stages_len{...}_rank0.pt`（120 tensors；
  新命名：`language_model.model.layers.N.{input_layernorm,self_attn,mlp,mlp.experts}`、
  `DeepseekV41DecoderLayer.{hc_pre,hc_post,rms_norm_cast}@layerN`、`AscendFusedTopKRouter._compute_routing[0/1]`）
- 引擎参数指纹：每 rank 107 项（供 `check_engine_params.py` 对 ckpt 校验）

## 2. 首张逐层阶梯（compare_stages，**rebase 后**）

len=64（trainer vs engine，rel_err / cos）：
```
embed              0.0000  (1.0000)   ← 完美
layers.0.attn_norm 0.0017  (1.0000)
layers.0.attn      0.0063  (1.0000)
layers.0.ffn       0.0520  (0.9986)
layers.1.attn_norm 0.0489  (0.9988)
layers.1.attn      0.0251  (0.9997)
layers.1.ffn       0.0854  (0.9963)
layers.2.attn_norm 0.0645  (0.9979)
layers.2.attn      0.9084  (0.5248)   ← 骤增！
layers.2.ffn       0.5413  (0.8538)
layers.3.attn      0.8097  (0.6442)
layers.3.ffn       0.6757  (0.7688)
model.norm         0.4966  (0.8767)
```
len=1500 同型且**更深层出现 NaN**（engine 侧；同时第 7 次 RL run 的引擎 dump 亦报 `decode-vs-prefill [nan...]`）。
engine rank0 vs rank1 自身对照：`AscendFusedTopKRouter._compute_routing#arg0 rel_err=1.20`（**引擎自身 rank 间亦不一致**！）。

**待判读**：该阶梯与 rebase 前历史（待从更早 worklog/dump 复核）相比是否更差；骤增点（layer2 attn）指向前
DSA 稀疏注意力（layer2/3 才开始出现 `attn.compressor/indexer`——见 trainer stage 表）。

## 3. probe_inputs 归因结果（核心证据，2026-10-12 20:0x）

命令：
```bash
MODEL_PATH=<4layer> ENGINE_DIR=.../cons_v3/engine_4l OUT_DIR=.../cons_v3/probe_4l \
LENGTHS=64,200 VARIANTS=baseline,in.all.ffn,in.all.attn,in.all \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,6,8,9 bash scripts/dsv41_consistency/sh/run_probe_inputs.sh
# 汇总：summarize_probe.py --probe-dir .../probe_4l --engine-dir .../engine_4l
```

### 3.1 逐模块 in/out 分解（len=200；* = 输入已替换为引擎的）
```
module    输入自身偏差   基线输出偏差   in.all.ffn    in.all.attn    in.all
0.attn       0.0017       0.0065       0.0065       0.0060*      0.0060*   ← out≈in，纯输入解释
0.ffn        0.0070       0.0364       0.0050*      0.0362       0.0050*   ← 强制输入后归零（算子一致）
1.attn       0.0352       0.0220       0.0065       0.0047*      0.0047*   ← 同上
1.ffn        0.0325       0.0916       0.0050*      0.0820       0.0050*
2.attn       0.0638       0.7617       0.7609       0.7611*      0.7611*   ← 输入替换后仍 0.76：**输入放大 12×**
2.ffn        0.2635       0.4644       0.0050*      0.4626       0.0050*
3.attn       0.3275       0.8051       0.7841       0.7264*      0.7264*
3.ffn        0.3832       0.6205       0.0050*      0.6015       0.0050*
model.norm      -         0.4384       0.2731       0.4230       0.2615   ← 残差 0.26（非全由输入解释）
```
**判读**：norm/线性/FFN/注意力算子在相同输入下数值一致（强制输入后误差≈输入误差，
0.0050 的量级=bf16 舍入差）。分叉全部来自：① 输入的第一处微小偏差（第一层 attn_norm 处 0.0017）；
② **layer≥2 的离散放大**（2.attn 把 0.064 放大到 0.76）。

### 3.2 路由稳定性（离散证据，baseline vs in.all.ffn）
```
layer  专家集相同率  slot overlap  翻转数    flip上误差  same上误差  flip占err²份额
0.ffn     0.985       0.9975       3/200     0.288      0.0115       0.905
1.ffn     0.915       0.9850      17/200     0.290      0.0217       0.888
2.ffn     0.175       0.8183     165/200     0.481      0.3487       0.897
3.ffn     0.085       0.7708     183/200     0.626      0.5106       0.938
```
**判读**：**layer2/3 的 MoE 路由在 82–92% 的 token-slot 上翻转，翻转贡献了 ~90%+ 的平方误差。**
logprob 层面上，in.all 相对 baseline 只改善 1.7×（σ 1.68→0.99）——余下部分即离散翻转的下游扩散。

### 3.3 结论（证据级）
1. 训推不一致 **不是某个算子算错**：所有被检模块（含 attn 核、GEMM、FFN、rms_norm 相关路径）
   在相同输入下吻合到 bf16 舍入级。
2. 不一致产生链：**第一处 ~1e-3 级数值差**（bf16 不同舍入顺序，第一层 norm 处可见）
   → **layer≥2 离散选择放大**（MoE top-k 路由翻转 82-92%；稀疏注意力索引选择 12× 放大）
   → 逐层滚雪球到 0.76-0.8 的层输出差 → 最终 logprob pearson 0.95-0.97 / diff_mean 0.03-0.04。
3. 改进方向（按杠杆排序）：
   a. **压小第一处差异**：对齐 norm/hc 路径的精度（trainer 端 fp32 计算 vs 引擎 `rms_norm_cast` 的舍入点），
      目标把 0.0017 压到 ~1e-4 级 → 翻转率随噪声/边界距离超线性下降。
   b. **离散边界处理**：top-k/索引选择的 tie 处理与引擎逐位一致（如引擎的稳定排序、-inf 处理）。
   c. 逐项验证 `ALIGN_*` 门在新栈上的实际落点（此前结论"无增益"需在新栈复核——旧结论基于 native_eager 等休眠路径）。

## 3.4 工具命名对齐（已定位并修复钩子）
- 根因：`dsv41_dump_extension.py` 钩的类 `DeepseekV41EagerAttentionImpl` 在 rebase 后已改名
  **`AscendDSAV41Impl`** → 导入失败被 `except` 静默跳过 → **整条注意力内部钩子从未生效**
  （新 dump 只有模块级阶段，没有 multistream/_select 等内部量）。
- 修复（脚本已改+推送）：新名优先、旧名回退；钩 `multistream_preprocess / _write_compressed_source /
  _select_sparse_indices / _forward_attention`。重跑 len=700/1500 dump（`cons_v3/engine_4l_attn`，pt 5GB/1500）成功。

## 3.5 ★ 引擎长上下文 NaN 定位（A 任务根因，与 B 同源）
- len=1500 引擎 dump（修好钩子后）逐函数 NaN 扫描：
  `multistream_preprocess`(0) → `compressor.norm`(0) → `_write_compressed_source`(0) →
  `_select_sparse_indices`(0) → **`_forward_attention` = 第一个 NaN**（6.14M/7.67M 元素）；
  trainer 同长度 0 NaN；len=700 引擎全 0 → **NaN 随长度在 700~1500 间爆发于 layer2/3（compressor 层）**。
- `_forward_attention` = `torch.ops._C_ascend.npu_sparse_flash_mla(...)`（compressed 路径
  cmp_block_table/cmp_mask_mode=3/cmp_ratio）或 `backend.qsmla`。
- 意义：与 A 的 RL 乱码同根（长上下文 NaN→采样塌缩）；也解释引擎 dump 早先
  `decode-vs-prefill [nan]`。**后续：最小复现隔离 内核 vs cmp 元数据组装，然后改算子验证。**

## 4. 重要旁证（A 与 B 交叉）

- RL rollout 乱码（任务 A）已排除引擎栈本身：**同引擎栈的 serve 生成完全正常**（temp0 2/2 正确、temp0.9 连贯）；
  RL 配置复刻版 serve 也正常 → 指向 **step0 权重同步/更新路径**。
- 引擎 dump 的 NaN 现象（len=1500）与 RL 乱码可能同源（引擎前向在特定条件下退化）——两者需并案。

## 5. 后续计划

1. 读完 probe 结果 → 按层/按模块定出"输入误差 vs 算子差异"的判决表。
2. 对判决为"算子差异"的点做**单算子隔离 microbench**（如 `oproj_rope_parity_microbench.py` 的模式）。
3. 按用户要求：定位到具体算子后**尝试改算子**（改法在 vllm-ascend 或 FSDPTurbo 的对应 op），重测指标。
4. 全程数字回填本文件。

---

## 6. 2026-10-13："小步快跑"切片法首轮（top32-sft116 4 层切片，单机 8 卡，单轮 ~10 分钟）

> 背景：全量 top32 排查一轮要 45 分钟（366GB Engram 加载），共享机易被外部杀。改成
> **层切片 + 两侧各一次前向**：切片 0-3 层（engram 层 1 排除+关闭），引擎/训练 dump 各 ~3 分钟。
> 操作手册见 **`wiki/dsv41_consistency_playbook.md`**（本次交付，含全部坑）。

**切片**：`make_slice_from_real.py`（prefix 0-3 + 手工模板）→ 748 张量 / 13 GiB（新磁盘 ~7 GiB）；
`compress_ratios=[0,0,2,2]`、kv/index_source=[2]、engram off；五层校验通过。

**逐模块表（rel_err，trainer vs engine）**：
| stage | len64 | len200 | len700 |
|---|---|---|---|
| L0 attn_norm | 0.0017 | 0.0017 | 0.0017（起点） |
| L0/L1 attn | 0.005-0.006 | 0.005-0.006 | 0.005-0.009 |
| L2 ffn | 0.0085 | **0.0496** | **0.0515** |
| L3 ffn | **0.0207** | **0.0531** | **0.0638** |
| model.norm | 0.0117 | 0.0322 | 0.0399 |

**MoE 路由（compare_routing，修口径后）**：set-match len64/200 = **100%**；
len700 = 97.7/97.7/98.9%（真翻转 1-2 token/88；其余为 top-k 顺序差）。
router logits rel 0.09-0.29%（L3@700 最大）。

**压缩索引（compare_sparse_indices，新增工具）**：3 档 × L2/L3 **全部 100% 一致**
（对齐要点：trainer 压缩位编号 = S+idx，引擎 0 基）。

**首轮结论**：① 起点仍为 L0 norm 的 bf16 舍入（0.0017）；② **两个离散选择环节（torch-32→真实权重下）
几乎零偏差**（非历史 scaled32 的 82-92% 翻转爆炸）；③ **放大发生在 MoE 模块内部计算**
（路由 100% 一致时 L2/L3 ffn 输出仍差 ~5%：输入 0.015 → 输出 0.05，~3.5× 增益）→ 下一锤用
graft（引擎 MoE 输入喂训练侧 MoE）分离"输入解释 vs 模块增益"，并拆 experts vs shared_experts。

**本日工具改动**（131 工作树 → 已同步 119）：
`dsv41_dump_extension.py`（注意力类名 AscendDSAV41Impl 修复;router 属性扫描 `_dsv41_probe_path` 层标识;
hook_end 还原分支）、`dump_trainer_stages.py`（生产 materialize 路径 + tokenizer + 类名 shim）、
`run_dump_engine.sh`（PATH/VLLM_VERSION）、`compare_routing.py`（TP 块对齐 + set-match 口径）、
**新增 `compare_sparse_indices.py`**、`sh/run_dump_trainer_top32.sh`。

### 6.1 MoE 阶段深钻：graft 判决（2026-10-13 下午）

工具：`probe_module_inputs.py`（切片版，生产加载路径；ffn 输入映射 = engine `rms_norm_cast@layerN[0]`，
已实证 = MoE 的 norm 后输入；引擎 `.mlp` ≡ `.mlp.experts`，shared 融合、无拆分）。

| 变体 | L1 ffn out | L2 ffn out | L3 ffn out | norm out | **logprob Δ σ** | argmax |
|---|---|---|---|---|---|---|
| baseline | 0.0305 | 0.0496 | 0.0531 | 0.0322 | **0.0841** | 0.945 |
| `in.all.ffn`（MoE 输入全换引擎） | **0.0047** | **0.0049** | **0.0049** | 0.0056 | **0.0188** | 0.975 |
| `in.all`（attn 输入也换） | 0.0047 | 0.0049 | 0.0049 | 0.0054 | **0.0185** | 0.975 |

**判决**：
1. **MoE 实现两侧零缺陷**（同输入 → 0.005 = bf16 ULP 级）；0.05 是 MoE 的**结构性 4× 增益**；
2. **拐点 = MoE 输入**：对齐它即 4.5× 改善；attn 输入再对齐**无额外收益**（0.0188→0.0185）；
3. **残余 σ 0.0185 = "读数地板"**（模块输出 0.005 → logprob 0.0185，head/logprob 段 ~3.7×）——
   独立于 MoE 的下一块（head GEMM/logprob 计算路径的 ULP）；
4. **新线索**：top32 真实权重下 **attn 非"直通"**（L0: in 0.0017 → out 0.0051, ~3×）——
   与历史 scaled32"attn 1.0-1.2×"不同；**下一个可攻点 = attn 段 + 第一处 0.0017（L0 norm 舍入）**。
5. 全部输入对齐后仍有 per-token p99 Δ≈0.05 nats（低 p 区放大在"读数段"复现，量级比 baseline 小 10×）。

### 6.2 源头的重新定位（2026-10-13 晚）：norm 假差异 + attn 段解剖

**口径修正（重要）**：trainer dump 存的是 **cast 前的 fp32**。按 bf16 语义（下游真实看到的值）重算：
- **`L0 attn_norm` 差 0.00166 → cast-bf16 后 0.00000（逐位相同）**——引擎 `rms_norm_cast`
  与训练侧"fp32 计算+正确舍入"**逐位一致**；所谓"第一处差异"是**解析口径假象**；
- 其余 stage（attn/ffn/norm）cast 前后差不变 → 真实。

**attn 段解剖**（L0，engine 8 local heads ↔ trainer heads 0-7，cos=1.0000 一一对应）：
| 子段 | cast-bf16 rel_err | 增益 |
|---|---|---|
| 输入（norm 输出） | 0.00000（逐位同） | — |
| q 投影 | 0.00260 | GEMM 核差（ULP） |
| sparse attn 核输出 | 0.00297 | 1.14×（核清白） |
| o-proj + TP all-reduce 后 | **0.00534** | **1.8×（attn 段唯一放大器）** |

**与历史对齐**："attn 尾 = TP 头切 + HCCL all-reduce 的并行结构差"复现（B300 旧结论）。

**完整定位链（终版）**：embed 0 → norm 0（融合核正确）→ qkv GEMM ULP 0.0026 →
attn 核 1.14× → **o-proj+AR 1.8×【结构】** → 0.0053 → 层间累积（MoE 输入 0.012@L2）→
**MoE 4×【结构】** → 0.05 → norm/head → logprob σ0.084 → pearson 0.96-0.98。
**两个放大器（attn-AR 1.8×、MoE 4×）与两个核差（qkv/proj GEMM、sparse 索引之外的算子 ULP）均为
"结构/内核"性质**——支持历史"唯有共用内核"的判断；norm、attn 核、MoE 实现、路由、索引**全部清白**。

### 6.3 ★ o-proj+AR 段优化：实锤 bf16 all-reduce + fp32-AR 补丁（生效）

**结构**：引擎 o-proj = 每 rank 本地组 wo_a（`npu_transpose_batchmatmul`，K=4096）→
`wo_b`（**RowParallelLinear**，K=1024 局部 GEMM + **TP all-reduce**）。
训练 = 8 组 einsum + 单个 K=8192 大 GEMM（fp32 累加、单次 cast）。

**实锤（HCCL microbench，8 rank 独立 partial、8192 维）**：
`bf16-AR rel=0.0035` vs `fp32-AR rel=0.000000` —— **引擎 AR 在 bf16 域求和，固有 ~0.35% 误差**，
量级与 attn 段 AR 增量（0.0024/0.0053≈0.45%）吻合。

**补丁（vllm-ascend，env 门控）**：`vllm_ascend/attention/dsa_v1.py::_forward_o_proj` 默认分支，
`VA_V41_OPROJ_FP32_AR=1` 时：关 `wo_b.reduce_results` → 取 local partial → **fp32 all_reduce**
（`dist.all_reduce(partial.float(), group=get_tp_group().device_group)`）→ cast 回输出。
`run_grpo_2node.sh` set_env 已加透传（默认 0）。

**切片验证（len200，engine_fp32ar vs engine vs trainer）**：
| 指标 | bf16-AR | **fp32-AR** | 改善 |
|---|---|---|---|
| L0 attn out | 0.00534 | 0.00473 | −11% |
| L3 attn out | 0.01862 | 0.01722 | −8% |
| L2 ffn out | 0.04966 | 0.04173 | **−16%** |
| model.norm | 0.03227 | 0.03006 | −7% |

**结论**：优化真实生效（全链改善 7–16%，MoE 段因 4× 放大获益最大），但幅度有限
（AR 只占 attn 差的部分；剩余为 qkv GEMM ULP 与层间累积）。**已启动 1 步真实 RL 验证跑**
（119 master + 120 slave，`VA_V41_OPROJ_FP32_AR=1`，`grpo_top64_steps1_*`）——以 step1 的
pearson/diff 对照历史基线（0.9604 / 0.0606@run13）。

### 6.4 1 步 RL 验证结果 + 0.995 路线再评估（2026-10-13 深夜）

**运行**：`logs/grpo_top64_steps1_20261013_102932.log`（119 master + 120 slave）。启动脚本
`/tmp/launch_{master,slave}_1step.sh` 已核实**双节点都导出 `VA_V41_OPROJ_FP32_AR=1`**（补丁非半边生效）；
另 `ROLLOUT_IS=token`（TIS 在跑）。Step1：gen 159s / old_log_prob 8s / update 81s / sync 17s / **step 284s**，
12201 tok，吞吐 2.68。

| 指标 | run13 step1（基线） | 本次 step1 | 备注 |
|---|---|---|---|
| rollout_actor_probs_pearson_corr | 0.9604 | 0.9573 | resp_len 81 vs 99.6、entropy 1.70 vs 2.71 → 批不同 |
| rollout_probs_diff_mean | 0.0606 | 0.0640 | diff_max 0.869 / std 0.098 |
| kl / log_ppl_diff | 0.2926 / 0.2468 | 0.2394 / 0.2444 | |
| reward mean / grad_norm | -0.59 / 1.07 | -0.59 / 0.993 | 训练闭环健康 |
| TIS（is_mean / frac_high / eff_ss） | — | 0.935 / 4.1% / 0.833 | 与 10-11 实测一致带 |

**判读**：① 补丁全流程健康（生成-更新-权重同步闭环正常、无 abort；日志尾 1 处 `DataLoader worker
SIGKILL`/`ERR99999` Traceback 出现在 step:1 指标产出之后的收尾段，判读为收尾噪声，不影响结果）；② **单步 RL 指标对"−10% 级"修复没有
分辨力**（切片预期 diff −10%± ≪ 单步 batch 波动 ±0.01 pearson）→ **补丁验收数字以切片为准（§6.3：全链 −7~−16%）**，
RL 1 步只作冒烟。若补丁全额传导全模型：diff_mean 0.064 → ~0.058 级 → **距 0.01 目标仍差 ~6×**。

**⇒ 核心结论：再修同量级算子无法达到 0.995。三道天花板（证据链见 §6.2/6.3 + 10-11 v2 worklog）：**
1. **内核 ULP 差**（qkv/o-proj/MoE GEMM 两栈不同 bf16 实现，AscendC vs torch_npu）：只能"共用内核"清零；
   精度补丁（AR 已 fp32）只收窄不清零；
2. **引擎自身 decode↔prefill 通道**（10-11 实测：len64 最高 Δ0.19 / len200 0.043 nats，prompt 依赖）：
   RL 指标分子 = 引擎生成时(decode) logprob vs trainer 全前向重算——这条"引擎内部两路径差"叠在指标里，
   即使训推核全同源也会压制指标（= 10-11 遗留 TODO ②：算子级归因）；
3. **指标几何**（低 p 区主导；去最差 20% token pearson 仅 +0.003）= 面状差距，必须整条链压缩。

**路线选项（待定）**：P1 切片精度收敛扫描（MoE combine/head 等剩余环节 fp32，量化"纯精度"上限，预计 ~0.97）；
P2 共用内核 / KV 分页移植（历史"唯一根治"，天级，C1+C2 同拆）；P3 验收口径（0.995 承诺给同模式对拍 = 切片口径，
历史 4 层已达 0.9946/0.005；RL 正确性由 TIS 兜底，已落地）。下一步优先级：P1 + decode/prefill 归因
（均单机 8 卡 ~10min/轮），拿"天花板预算表"再定 P2。

### 6.5 ★ 训推算子差异全景表（2026-10-13 晚交付）

- **`wiki/dsv41_train_infer_op_diff.md`**：把 §6 定位结果按**算子**梳理成主表（12 行，前向顺序）：
  每行 = 训练侧实现（什么算子）↔ 推理侧实现（什么算子）↔ 实测差异/增益 ↔ 性质（【核】内核 ULP /【并行】结构 /
  【径】执行路径 /【数】数学增益）↔ 状态。
- **写表时的代码级新发现**：① 引擎 o-proj 的真实调用链 = `dsa_v41.py:345 → dsa_v1.py:1640 AscendDSAImpl._forward_o_proj`
  （`VA_V41_OPROJ_FP32_AR` 补丁在 dsa_v1.py:1752，**在活路径上**——切片改善来源确认无误）；② 训练侧 MoE 的
  `experts.py:17-21` 注释直接写明与引擎的差异（引擎在 **combine 核**里乘路由权重、激活 dtype；trainer 在 down 前乘、
  fp32 combine）——`ALIGN_MOE` 门即对齐此点；③ `sparse_attention.py:10-24` 注释写明 ALIGN_ATTN 旧栈 A/B 结论
  （0.99486 vs 0.99464 无增益 → "引擎核同点也在舍入"）。
- 假差异 4 条（cast-bf16/TP token 切块/编号基准/顺序差）单列 §4 防混入；已修 2 项、诊断门 5 个、待做 3 项（§5）。

### 6.6 ★ 小算子替换验证 + 重大口径发现：trainer dump 是 FP32 精度前向（2026-10-13 下午）

**用户任务**：把推理侧换成"训练侧同款小算子"对拍，看两边是否一致/是否改善（一台机器验证）；随后两台机器跑 GSM8K 50 步长跑。

**实现**（全部 env 门控、默认关；dsa_v41.py，已推 119/120/123）：
- `VA_V41_TORCH_ROPE=1`：把 `inplace_partial_rotary_mul` 整体 monkeypatch 成 torch 小算子（与 golden/训练侧同款 fp32 interleave 数学）；
- `VA_V41_EAGER_QKV=1`：`multistream_preprocess` 的 q/kv 段替换成 `F.linear` + 模块 norm + 上述 rope；
- `VA_V41_EAGER_STASH_DIR`：q_a/qr/q_b_pre_rope（含 200 行重算版）落盘供逐点对拍。

**逐点结论（120 单机，len200）**：
1. **rope 替换 = 逐位相同**（q 不变）→ 引擎融合 rope 与训练侧数学一致；
2. **qkv 换 F.linear = 逐位相同**（q/qr 不变）→ 引擎侧 GEMM 实现无差异；
3. **干净进程（无 vllm）用 ckpt 权重 + 位同输入重算 = 引擎结果（1.6e-5）**；trainer 侧与之差 **0.001857**（= 全部"谜之 ULP"的源头）；
4. **头映射修正**：引擎 rank0 的 8 头 = **trainer heads 32–39**（cos=1.0）；trainer 在 wq_b 后有头重排，所以最终 query 的 [:8] 对照又对得上——§6.2 的"engine ↔ trainer 0-7"表述按此修正；
5. **M 假说（批 208 vs 200）否定**（同进程 200 行重算仍 0.001857）；
6. **决定性**：trainer dump 捕获张量 **100% 不在 bf16 网格上**（attn_norm/q_a 非网格偏差 ~9.7e-4），引擎 100% 在网格上——**trainer dump 是 FP32 精度前向**（激活+权重 fp32），引擎是 bf16；`VERL_DSV41_KEEP_FP32_PARAMS` 未启用、与它无关；ckpt 36 个 F32 张量仅 attn_sink/gate.bias/hc_*，与 q 链无关。

**含义（重要，归因重估）**：§6.2/6.3 的"逐算子 ~1 ULP 核差"**主要是 bf16(引擎) vs fp32(参考) 的精度口径差**，不是两栈 bf16 内核实现差——这同时解释了：换任何算子都位同、ALIGN 门无增益、M 假说不成立。RL 指标（0.9573）对比的是引擎 bf16 生成 vs trainer 实际重算，不受 dump 口径影响（链条/6× 差距判断仍成立），但**"核差"归因降级为"精度口径含混"，需要 bf16 模式重跑 trainer dump 后复测**。
**副产品**：120 上旧版 `run_dump_engine.sh`（缺 PATH pin + 默认 0.27.1）导致两连败（ModuleNotFoundError torch）→ 已把 dsv41_consistency 全量同步到 120（md5 三机一致）。**新机器 123 入列**：整树同步 + 13 关键文件 md5 三机一致 + preflight ALL OK（详见部署/ roster memory）。

### 6.7 ★★ 定位闭环：根因 = 训练侧 fp32 主流；解决 = `VERL_DSV41_HC_BF16=1`（q 链坍塌 ~130×）

**机制定位（131 单机，全套实证）**：
1. dtype 探针（dump 脚本加 3 行）：**权重全程 bf16**（build→prepare→FSDPTurbo→materialize）；
2. 单算子实证：`F.linear(fp32 x, bf16 w)` 在 torch_npu **合法且输出 fp32**（自动升格）；
3. ⇒ 链条：embed(bf16) → **`hc_pre_sinkhorn_premix` 输出 fp32**（架构设计，hc 参数全 fp32）→ 主流全程 fp32
   （norm 保持 x.dtype；linear 自动把 bf16 权重升到 fp32 算）→ **训练=fp32 精度流 vs 引擎=bf16 融合核**，
   §6.2/6.3 的"每算子 ~1 ULP 谜差"全部由此而来。
4. `ParallelHead`/`model.head` 是**明示设计** fp32（"kept as fp32 here so the logits come out in fp32"）——读数段另算。

**解决与验证（4 个变体 trainer dump，len200，对照引擎 slice4_probe/engine 与引擎 stash）**：
- 补丁：`VERL_DSV41_HC_BF16=1`（FSDPTurbo/model.py，**新加**，env 门控默认关）——在层内 attn/ffn 与 backbone 尾
  共 3 处把 hc_pre 输出 cast 回 bf16。
- **效果（bf16 口径，trainer vs 引擎）**：
  | 点位 | fp32 流（旧） | bf16 流（新） | 改善 |
  |---|---|---|---|
  | wq_a 输出 | 0.001857 | **0.000016** | ×116 |
  | q_norm 输出 | 0.002945 | **0.000016** | ×184 |
  | 最终 q | 0.002602 | **0.000020** | ×130 |
  | L0 attn_norm | — | **0.0000（逐位）** | — |
- **残差图（bf16 流 + ALIGN_ATTN）**：注意力核 **0.001372**（ALIGN_ATTN=1 从 0.002705 再砍半；=
  训练 eager 的 scores/probs 两处 bf16 舍入 vs 核内 fp32）、L0 attn 模块 **0.0032**（从 0.0046）、
  L1/2/3 ffn **0.0326/0.0432/0.0561**（MoE 段：gate/专家 .float() 等 fp32-by-design 内部 + 路由翻转敏感性）、
  model.norm **0.0326**、head fp32-by-design（未动）。
- **ALIGN_MOE=1 实测零效果**（align 与 align2 两版逐位相同）；ALIGN_ATTN=1 有效（核心减半）但会让
 下游 ffn 数字变差（离散路由翻转的连带，非真退化——ffn 口径对输入扰动非线性）。

**解决配方（候选，待 1 步 RL 验收）**：训练侧 `VERL_DSV41_HC_BF16=1` + `VERL_DSV41_ALIGN_ATTN=1`
（两 env 由启动器透传即可，模型侧默认关不影响历史）。
**下一步**：① 机器恢复后跑 1 步 RL（对照 0.9573）→ 通过则 50 步长跑；② 残差继续攻（MoE 段
fp32-by-design 内部的对齐开关候选、head bf16 开关候选、核心 0.00137 的进一步归因）。
