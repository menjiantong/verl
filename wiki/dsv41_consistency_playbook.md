# 训推一致性排查操作指南（小白版 · 2026-10-13 首版）

> 目标读者：第一次接手"训练-推理一致性"排查的同学。
> 场景：给一个 DeepSeek-V4.1（FSDPTurbo 训练栈 + vLLM-Ascend 推理栈），回答
> **"训练与推理两条栈的数值差在哪、有多大、下一步修哪里"**，产出可复现的数字。
> 本文所有内容均来自 2026-10-13 的实测（4 层切片 sft116，单机 8 卡，单轮 ~10 分钟），
> 配套脚本在 `verl/scripts/dsv41_consistency/`（23 个脚本 + `sh/` 启动器）。

---

## 0. 一句话方法论（先读这个）

**自上而下，逐级下钻；每一步都要有"两栈同一输入下的输出差"这个数字。**

```
第 1 级：整链逐模块表（哪一层、哪个模块的输出差最大）
   ↓ 锁定放大点
第 2 级：模块内的"离散选择"与"子算子"（路由 top-k / 稀疏索引 / GEMM 子段）
   ↓ 锁定环节
第 3 级：单点隔离复现（微基准 / graft 输入替换）→ 修复 → 回测第 1 级
```

三个必须建立的前提观念：
1. **"算子本身算错"很少见**——实测所有算子在**相同输入**下一致到 bf16 ULP 级（0.005 量级）。
   差距来自：①某处的第一点舍入差；②被"离散选择"或"高增益模块"放大。
2. **一切对比必须先对齐**：token 子集（TP 切分！）、压缩位编号基准、键名映射——**对齐错会得到 0% 这种假数字**（本指南 §5 有全部陷阱）。
3. **成本控制**：能用"层切片"就不要跑全模型（见 §3.1），单轮迭代压到 ~10 分钟、单机、可随时中断。

---

## 1. 整体排查思路（对应上图的第 1 级）

对**同一段输入**（同一个 prompt token 序列），分别在两条栈上跑一次前向，把**每个里程碑模块的输出**（embed、每层 attn_norm/attn/ffn、norm、logits）都存下来，逐项算：

- `rel_err = ‖d‖ / ‖ref‖`（相对误差，主看这个）
- `max|d|`、`cos`（方向一致性）

得到"逐层阶梯"表：正常形态是 **数值随层数缓慢爬升**；出现**某个模块之后的跳变**（如某层 ffn 从 0.01 → 0.05），那就是放大点，进入第 2 级。

再叠加两个"端到端"指标（用户在训练日志里看到的）：

| 指标 | 定义（`verl/utils/debug/metrics.py`） | 含义 |
|---|---|---|
| `training/rollout_probs_diff_mean` | 引擎 rollout logprob 与训练侧重算 logprob 的 \|Δp\| 均值（probs 空间） | 越小越好（目标 <0.01） |
| `training/rollout_actor_probs_pearson_corr` | 上述两个概率向量的 pearson（**probs 空间**，不是 log 空间） | ≥0.995 是目标；0.96-0.98 是当前常态 |
| `rollout_corr/rollout_is_*` | 同上的重要性采样比值统计（TIS） | 监控 PG 加权健康度 |

**经验 band（top32 真实权重）**：pearson 0.94–0.98 / diff 0.028–0.06 属正常波动；
单看某一步的 0.96 不必惊慌，要看 ①per-token 分解（差集中在哪）②逐模块表（差从哪来）。

---

## 2. 两条栈各自"怎么被dump的"（引擎接入 & 打桩）

### 2.1 引擎侧（vLLM + vllm-ascend）——通过 Worker Extension 接入

引擎跑在 8 个 TP worker 进程里，外部拿不到内部激活。**接入方式是 vLLM 官方的 worker extension 机制**：

1. `scripts/dsv41_consistency/dsv41_dump_extension.py::Dsv41DumpExtension`
   —— 一个"混入"类。启动引擎时传 `LLM(..., worker_extension_cls="dsv41_dump_extension.Dsv41DumpExtension")`
   （`dump_engine_stages.py` 已封装），vLLM 会把它的方法逐 worker 注入；
2. 外部通过 `llm.collective_rpc("dsv41_xxx")` 在**每个 TP worker 内部**执行方法、回传结果；
3. 记录生命期：
   - `llm.collective_rpc("dsv41_hook_begin")` → 挂 hooks；
   - `llm.generate(...)`（跑一次 prefill，最大 1 token）；
   - `llm.collective_rpc("dsv41_hook_end", args=(out_dir, tag))` → 每个 rank 写
     `engine_stages_len<N>_rank<r>.pt`。

**三类 hook**（`dsv41_hook_begin` 里）：

| 方式 | 覆盖对象 | 键名形式 |
|---|---|---|
| forward hook（按模块名） | milestone 模块（embed / layers.N / self_attn / mlp / input_layernorm …正则 `_MILESTONE_RE`）+ 类名含 Router/Experts/Indexer/Compressor/MoE 的模块 | `language_model.model.layers.2.mlp` |
| 方法包装（按类，`_dsv41_wrap_methods`） | 注意力实现 `AscendDSAV41Impl`（multistream_preprocess / _select_sparse_indices / _forward_attention / _write_compressed_source）、解码层 `DeepseekV41DecoderLayer`（hc_pre/hc_post/rms_norm_cast）、两个 router 类（`_compute_routing`） | `类名.方法@层标识`；同时记 `#argN`（浮点输入）与 `[i]`（tuple 返回） |
| 函数包装（`_dsv41_wrap_functions`） | vLLM 默认 fused_topk 路径的模块级函数 | `模块.函数#argN` |

**层标识的三条来源（重要，踩过坑）**：
1. `self.prefix`（注意力 impl 自带）；
2. `self.layer_idx`（解码层）；
3. `_dsv41_probe_path` —— **router 是宿主模块的普通属性（`RoutedExperts.router`）、不在 `named_modules()` 里**，前两条都取不到；hook_begin 会遍历所有模块属性、给 router 对象**盖上宿主路径戳**，wrapper 靠它命名
   （键形如 `AscendFusedTopKRouter._compute_routing@language_model.model.layers.2.mlp.experts.routed_experts`）。

**引擎启动参数**（`dump_engine_stages.py`，与 GRPO 生产同款）：
`tokenizer_mode="deepseek_v41"`、`enforce_eager=True`、TP=EP=8、
`additional_config={mc2_comm_alg: fullmesh_v2, ascend_compilation_config:{...false}, rl_config:{enabled:true}}`。

### 2.2 训练侧（FSDPTurbo）——独立 harness

`scripts/dsv41_consistency/dump_trainer_stages.py`（8 卡 torchrun）：
1. **构建模型**：`build_deepseek_v41_model(...)`（engram 层存在时必须 `engram_storage_backend="host_offload"`、
   有 Engram 层还需传 tokenizer——它要 tokenizer 建 compressed token map）；FSDPTurbo 包装；
2. **加载权重**：见 §3.2（生产 materialize 路径）；
3. **挂 hook 前向**：与引擎侧同构的 milestone + attn-op + gate 钩子（`attn_opN.{query,key_value,attn_sink,topk_indices,output}`、
   `ffn.gate[0]=weights/[1]=ids`、`ffn.experts`、`ffn.shared_experts`、`attn.compressor`、`attn.indexer`）；
4. 对 fixture 的每个长度写 `trainer_len<N>.pt`（含 `stages` / `logprobs` / `topk_ids` / `checkpoint_buffers`）。

**输入必须与引擎一致**：两侧都吃 `fixture.json` 的 `input_ids`（同一段 token！见 §4.1）。

---

## 3. 模型与权重：怎么"load"（两栈各自的路）

### 3.1 层切片——把"大模型排查"变成"小模型排查"（本指南核心技巧）

top32 全量 40 层 + 2 张 196GB 的 Engram 表 → 加载 20-45 分钟，不适合迭代。做法：

```bash
# 1) 造"4 层切片模板"（复制 sft116 config，改 4 层化字段）
#    关键字段：num_hidden_layers=4 / engram_layer_ids=[] / num_nextn_predict_layers=0 /
#    dspark_target_layer_ids=[] / compress_ratios=[0,0,2,2] / kv&index_source_layer_ids=[2] /
#    _dummy_no_engram=True /（别名键 kv_source_layers 等一并给）
#    模板目录还需一个 index.json（名字集合）+ 可读 header 的 shard（symlink 即可）
python3 /tmp/make_top32_slice.py          # 注：本次的胶水脚本，逻辑=上面的手工步骤

# 2) 用官方切片工具产切片（硬链不复制字节，秒-分钟级）
python3 scripts/dsv41_consistency/make_slice_from_real.py \
  --src  .../DeepSeek-V4.1-Flash-top32-sft116 \
  --out  .../DeepSeek-V4.1-Flash-top32-sft116-4layer \
  --layers 0,1,2,3 --config-template <模板>/config.json
```

**切片选层的三条规则**：
1. **必须 0-based prefix**（`pick_names` 不做层号重排——用 0,2,3,4 会得到"名字/模型错位"的坏切片，
   引擎会报 `KeyError: layers.2.self_attn.compressor...`）；
2. **避开 Engram 层**（sft116 的 engram 在层 1/14）：层 1 虽然落在 prefix 里，但它那 196GB 的表
   会被 `is_excluded`（`.engram.` 子串）排除，同时 config 里 `engram_layer_ids=[]` 让模型不建该模块；
3. 保留布局信息：层 2 是"压缩源层"（kv_source），层 2/3 有 compressor/indexer——**要测稀疏选择就必须含它们**。

产物体积（sft116 4层）：748 张量 / 13 GiB 表面积 / ~7 GiB 新磁盘。引擎与训练都能直接加载它。

### 3.2 训练侧加载 top32 的正确姿势（生产 materialize 路径）

**历史坑**：harness 自己实现"rank0 读全量 + DCP broadcast"→ top32 上把 366GB Engram 表当普通参数 → `Tried to allocate 183 GiB` OOM。

**解法（2026-10-13 已修入 harness）**：
- 构建时 `engram_storage_backend="host_offload"`；
- 加载时**复用生产函数**（`verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`）：
  ```python
  engram_tables = host_offload_engram_tables(model)
  for name, _ in engram_tables:
      param_meta.pop(_MODEL_PREFIX + name, None)      # 别让任何 loader 碰这张表
  # ... FSDPTurbo 包装后：
  shim = FSDPTurboDSV41EngineWithLMHead.__new__(FSDPTurboDSV41EngineWithLMHead)  # 空壳实例
  shim.model_config = SimpleNamespace(local_hf_config_path=str(model_path))
  shim._materialize_dsv41_parameters(model, param_meta, buffer_meta,
                                     offload_policy=True, engram_tables=engram_tables)
  ```
- 生产路径内部三件事：DCP/直读加载常规参数 → `hide_engram_tables`（摘除）→
  `_load_host_engram_tables`（**每 rank 只 safe_open 读自己那一段行**，跳过 padding，冻结 requires_grad）。
- 引擎侧不用管：vLLM 的 v4.1 loader 自己走 host-offload。

---

## 4. 对比：怎么把两边的数字对起来（本指南最易翻车之处）

### 4.1 对比前的必做对齐（错了会得到 0% 或形状报错）

| 对齐项 | 规则 | 踩过的坑 |
|---|---|---|
| 输入 | 两侧吃**同一份** fixture `input_ids` | — |
| **TP/token 切分** | 引擎 MoE 的 router 输入是**每 TP rank 一段连续 token**（rank0=前 ⌈len/8⌉ 个）；训练侧是全序列。比较时取 `trainer[: engine_rows]` | 形状 64 vs 8 报错；先做"最近邻匹配"验证 rank0=前 8 行 |
| 注意力 vs router 的切分不同 | 注意力内部量（_select_sparse_indices 等）是**全 token**（切 head，本地 8 heads） | 以为都要切 token |
| **压缩位编号基准** | 训练侧压缩位编号 = **序列长度 S + 序号**（len700 → 700..799）；引擎侧 = **0..99** 直接。换算 `id = value - S`（S=该档长度） | 用错基准（减 200）→ len700 显示"0% 一致"的假数字 |
| 键名映射 | `compare_stages.py` 的 `STAGE_MAP`（trainer `model.layers.N.ffn` ↔ engine `language_model.model.layers.N.mlp` 等） | 引擎侧键前缀/regex 改动会让映射失配 |
| **trainer dump 的精度口径** | trainer dump 存 **cast 前的 fp32**；下游实际吃 bf16。**比较模块输出前先把 trainer 侧 cast 到 bf16**，否则会把"fp32/bf16 表示差"误算成真差异（曾把 norm 的 0 差报成 0.0017） | 只对"bf16 语义的输出"cast；核内的 fp32 内部量（sink/score 等）不 cast |
| top-k 比较口径 | "集合一致"（排序后比）≠"逐位一致"；**顺序差是常态**（tie 排序实现不同），要分开统计 | 只看逐位会误判"87% 匹配" |

### 4.2 三个现成比较器（`scripts/dsv41_consistency/`）

```bash
D=/mnt/share/m00899630/dsv41/dump/slice4
SLICE=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer

# ① 逐模块阶梯（核心表）
python3 scripts/dsv41_consistency/compare_stages.py --len 200 \
  --trainer-dir $D/trainer --engine-dir $D/engine

# ② MoE 路由（top-k 集合一致率 / 真翻转计数 / logits 差）
python3 scripts/dsv41_consistency/compare_routing.py --len 700 \
  --trainer-dir $D/trainer --engine-dir $D/engine --model-path $SLICE

# ③ 注意力子段（投影/选择/核计算分离）
python3 scripts/dsv41_consistency/compare_attention.py --len 200 \
  --trainer-dir $D/trainer --engine-dir $D/engine --heads 8

# ④ 压缩索引一致率（本日新增口径，见附录 I-9 脚本）
python3 /tmp/cmp_idx.py   # set-match + Jaccard，压缩位基准自动按 S 换算
```

### 4.3 指标都什么意思

- `rel_err = ‖e−t‖/‖t‖`：模块输出的相对差。**0.005 ≈ bf16 1 ULP 级**（"算子没毛病"的判据）；
  0.05 表示"被放大了 10 倍"。
- `cos`：方向。>0.999 基本同向。
- **路由 set-match**：两侧 top-6 专家**集合**相同的 token 占比（排序后比较）。
- **order-only**：集合相同只是顺序不同的 token 数（正常现象，不用修）。
- **真翻转（real flip）**：集合不同的 token——**这才是离散差异**，对照 logits 差（翻转 token 通常
  在 top-6 边界上，logits 差跨过了边界 gap）。
- **Jaccard**：两个索引集合的交并比（部分重叠时比"exact"更细）。

---

## 5. 一条完整流水线的命令清单（复制可跑）

```bash
# ===== 0) 造夹具（不需要 NPU）=====
python3 scripts/dsv41_consistency/make_fixture.py   # 产 fixture.json（tokenizer + 4~5 档 input_ids）

# ===== 1) 造切片（CPU，分钟级）=====
python3 /tmp/make_top32_slice.py        # 模板 + make_slice_from_real.py 一条龙

# ===== 2) 引擎侧 dump（单机 8 卡，~3 分钟）=====
#    注意：必须 login shell（bash -lc）启动，否则丢 CANN（libhccl.so）
bash -lc 'cd /workspace-verl/verl &&
  nohup bash scripts/dsv41_consistency/sh/run_dump_engine.sh \
    --model-path $SLICE --out-dir $D/engine --lengths 64,200,700 \
    > /tmp/engine.log 2>&1 &'

# ===== 3) 训练侧 dump（单机 8 卡，~3 分钟）=====
bash -lc 'cd /workspace-verl/verl &&
  MODEL_PATH=$SLICE OUT_DIR=$D/trainer LENGTHS=64,200,700 \
  nohup bash scripts/dsv41_consistency/sh/run_dump_trainer_top32.sh > /tmp/trainer.log 2>&1 &'

# ===== 4) 对比（CPU，秒-分钟）=====
for L in 64 200 700; do
  python3 scripts/dsv41_consistency/compare_stages.py  --len $L --trainer-dir $D/trainer --engine-dir $D/engine
  python3 scripts/dsv41_consistency/compare_routing.py --len $L --trainer-dir $D/trainer --engine-dir $D/engine --model-path $SLICE
done
```

**验收线**：引擎 log 出现 `engine up` + 每档 `stage dumps [...]`；训练 log 出现每档
`wrote trainer_len<N>.pt`；对比脚本打出逐层表。

---

## 6. 参考：2026-10-13 首轮结果（top32-sft116 4 层切片）

| 环节 | len64 | len200 | len700 | 判定 |
|---|---|---|---|---|
| 起点：L0 attn_norm 输出差 | 0.0017 | 0.0017 | 0.0017 | bf16 舍入级（无算子缺陷） |
| MoE 路由 top-6 集合一致率 | 100% | 100% | 97.7–98.9%（真翻转 1–2 token/88） | 基本一致 |
| 压缩索引集合一致率 | 100% | 100% | 100% | 完全一致 |
| **MoE（L2/L3 ffn）输出差** | 0.009–0.021 | **0.050–0.053** | **0.052–0.064** | **最大放大点（~3.5× 增益）** |
| attn 输出差 | 0.005–0.007 | 0.011–0.019 | 0.011–0.023 | 温和 |
| norm（末端） | 0.012 | 0.032 | 0.040 | 累积到位 |

**读法**：top32 真实权重下"离散选择"几乎零偏差——放大发生在 **MoE 模块的"结构性增益"**。

**graft 判决（`probe_module_inputs.py`，同页下午完成）**：
把引擎的输入强制喂给训练模块（`in.all.ffn` / `in.all` 变体）后：

| 变体 | L2/L3 ffn out | logprob Δ σ |
|---|---|---|
| baseline | 0.050/0.053 | 0.0841 |
| `in.all.ffn`（MoE 输入对齐） | **0.0049/0.0049** | **0.0188（4.5×）** |
| `in.all`（全输入对齐） | 0.0049 | 0.0185（拐点已到） |

⇒ ① **MoE 实现零缺陷**（同输入=ULP 级）；0.05 是 MoE 的**结构性 ~4× 增益**；
② **改善端到端的杠杆=压 MoE 输入差**（对齐即 4.5×；attn 输入再对齐无收益）；
③ 残余 σ≈0.0185 是 **"读数地板"**（head/logprob 段 0.005→0.0185）；
④ 下一步目标：**attn 段（真实权重下有 ~3× 增益，非直通）+ 第一处 0.0017（L0 norm 舍入）**。

**o-proj+AR 段已修复验证（同日）**：引擎 `wo_b`(RowParallelLinear) 的 TP all-reduce 在 **bf16 域**
（HCCL microbench：bf16-AR rel=0.0035 vs fp32-AR 0.000000）。补丁 `VA_V41_OPROJ_FP32_AR=1`
（dsa_v1 `_forward_o_proj` 内 fp32 all_reduce）→ 切片全链改善 7–16%（L2 ffn 0.0497→0.0417）——
**"选择正确的 dtype 做通信归约"是单点可修的一类问题**（对照：MoE 4× 是不可修的增益）。

---

## 7. 附录：今天踩过的问题汇总（照单避坑）

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| I-1 | 引擎 `KeyError: layers.2.self_attn.compressor.norm.weight` | 切片用了非前缀层集合（0,2,3,4），`pick_names` 不做层号重排 | 只用 0-based prefix（0,1,2,3；engram 层靠排除+config 关闭） |
| I-2 | 训练 harness `ImportError: cannot import name 'FSDPTurboEngine'` | 类名实际叫 `FSDPTurboDSV41EngineWithLMHead` | 用真实类名 |
| I-3 | `ValueError: Engram requires the model tokenizer` | top32 有 Engram 层，构建时需要 tokenizer | `AutoTokenizer.from_pretrained(rl_tokenizer)` 并传给 build |
| I-4 | 训练加载 `Tried to allocate 183.11 GiB` | 通用 DCP 路径把 196GB×2 的 Engram 表当普通参数搬 | 接生产 materialize 路径（§3.2） |
| I-5 | `compare_routing: no engine router capture` | router 键无层号（4 层互相覆盖）——router 不在 named_modules、无 prefix/layer_idx | hook_begin 里**属性扫描给 router 盖宿主路径戳**；wrapper 三级 fallback 取层标识 |
| I-6 | 注意力内部量 dump 里完全缺失 | 扩展里 import 的类名 `DeepseekV41EagerAttentionImpl` 不存在（两代栈都叫 `AscendDSAV41Impl`）→ except 静默跳过 | 改成真类名；**新增 hook 后先 `grep` dump 键验证生效** |
| I-7 | `compare_routing` 形状 64 vs 8 报错 | 引擎 MoE 的 router 输入按 TP 切了 token（rank0=前 1/8 连续块） | 对齐时取 `trainer[:engine_rows]`；先最近邻验证切块规则 |
| I-8 | 路由"exact match 87%"看着很差 | 口径问题：逐位比较把 top-k **顺序差**算成不匹配 | 分开统计 set-match（排序后）与 order-only；真翻转才重要 |
| I-9 | 索引对比 len700 显示 "0% 一致" | 压缩位编号基准不同：训练侧=序列长 S + 序号，引擎侧=0 起 | 换算 `id = v - S`（S=该档长度） |
| I-10 | 训练 dump 被 SIGKILL(-9)，无栈 | 作业跑在**共享宿主**上，邻居占用（内存/卡）导致外部杀进程；131/126 均发生过 | 排查作业尽量短（本指南的切片法）、观察 npu-smi 占用者 pid 是否属于自己；**被外部杀过的机器别再用** |
| I-11 | `npu-smi` 输出为空 / "0 卡占用"假象 | 非 login shell 缺 CANN 库（libc_sec.so） | 一律 `bash -lc "npu-smi info"`；占用者看 "Process id in container" 是否为自己容器 |
| I-12 | nohup 启动后 `ModuleNotFoundError: torch` / `libhccl.so` | 非 login shell 丢 PATH 与 CANN 环境 | 启动一律 `bash -lc '...'`；runner 脚本内也补 `export PATH=/usr/local/python3.11.15/bin:$PATH` |
| I-13 | `pkill -f <模式>` 把自己的 shell 杀了（exit 144） | pgrep/pkill 的 `-f` 会匹配到**发起命令自身**的命令行 | 用字符类规避自匹配（`watch_steps1[0]`）或按 PID 杀 |
| I-14 | 同一条 ssh 里两个 `cat > f` 只有一个成功 | 第二个 cat 的 stdin 已被第一个吃掉 | 每个文件单独一条 ssh，或 tar 管道 |

---

## 8. 脚本速查（`verl/scripts/dsv41_consistency/`）

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `dump_engine_stages.py` + `sh/run_dump_engine.sh` | 引擎侧 dump（参数指纹/激活/logprobs） | `--model-path --fixture --out-dir --lengths --tp --ep` |
| `dump_trainer_stages.py` + `sh/run_dump_trainer_top32.sh` | 训练侧 dump（8 卡 torchrun） | `--model-path --tokenizer-path --fixture --out-dir --lengths` |
| `dsv41_dump_extension.py` | 引擎 worker extension（hooks/参数指纹） | 由引擎加载，`collective_rpc` 调用 |
| `compare_stages.py` | 逐模块阶梯表 | `--len --trainer-dir --engine-dir` |
| `compare_routing.py` | MoE 路由一致率 | `--len --trainer-dir --engine-dir --model-path` |
| `compare_attention.py` | 注意力子段 | `--len --trainer-dir --engine-dir --heads` |
| `make_slice_from_real.py` | 真实 ckpt → N 层切片 | `--src --out --layers --config-template` |
| `probe_module_inputs.py`（进阶） | "把引擎输入喂训练模块"（in.all 变体） | 需要引擎侧 dump 同 fixture；top32 需 §3.2 的加载路径 |
| `graft_trainer_stages.py`（进阶） | 把引擎某段输出灌进训练侧前向 | 同上 |
| `analyze_rollout_actor_gap.py`（RL 端） | 从生产 RL 的 `VERL_DSV41_DUMP_BATCH` 逐 token 分解 | `--dump-dir` |
