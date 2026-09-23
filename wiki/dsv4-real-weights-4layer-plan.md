# 从真实 DeepSeek-V4.1-Flash 权重切出 4 层，并在其上重跑训推一致性全套 harness

> 计划文档 · 2026-09-22 · 状态：**待执行**（尚未创建切片、未跑任何任务；所有"事实"均已核实到文件/行号）
> 本目录（`/mnt/share/m00899630/plans/dsv41-real-weights-4layer/`）存放本工作流的计划与记录。
> 相关文档：`verl/wiki/worklog_dsv41_module_input_probe.md`（机制闭环与 §5.3 方向 B）、
> `verl/wiki/worklog_dsv41_train_infer_consistency.md`（§9–§11）、`verl/wiki/worklog_dsv41_rl.md`（健康信号阈值）。
> 产物位置约定：切片 `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real/`；
> dump `/mnt/share/m00899630/dsv41/dump/{engine_real4,probe_input_real4}/`。

## Context（为什么做）

- 上一轮已闭环机制（`verl/wiki/worklog_dsv41_module_input_probe.md`）：**训推差距 = 每模块 ~0.5% 核差（≈ bf16 的 1 ULP）× MoE 离散路由翻转放大**；专家数只加强放大器，不改变核差。
- 但那是**随机权重**：路由处在近平局、`--head-gain 4` 把 logprob 噪声放大 4×（KL ~16×）→ 绝对值全偏悲观。方向 B 就是用真实权重复跑，拿生产数字决定 (a) 只换健康指标 还是 (b) 上 TIS/尾部纠正。
- 真实权重 `/mnt/share/DeepSeek-V4.1-Flash-bf16/`（40 层 + 3 MTP、384 专家、1.53 TB、无量化配置）；harness 与 fixture 都是围绕 **4 层**模型建的 → 需要 **4 层切片目录**，两套加载路径都指向它。
- **用户已定**：切**第 0–3 层**；跑**全套 harness**。

## 调研结论（已核实；★ = 本轮复核新发现，直接改变方案）

| # | 事实 | 出处 |
|---|---|---|
| F1 | 真实 ckpt：268 shard / 48,496 张量 / 1.53 TB；无 `quantization_config`；`num_hidden_layers 40`、384 专家、`compress_ratios [0,0,2×18,1×20,0,0,0]`、`kv_source_layer_ids [2,8,14,20]`、`index_source_layer_ids [2,8,14,20,24,28,32,36]`、`candidate_source_layer_id 20`、`engram_layer_ids [1,14]`、`num_nextn_predict_layers 3` | 盘点 agent |
| F2 | **切片名字集合 = 现有 `-4layer-scaled` 的 index 名字集合**（本机实算：4972 个名字在真实 ckpt **0 缺失**；名字构成 layers 4703 + vision 259 + aligner 4 + base 6）；只需 **31 个 shard**，**不需要碰两个 183 GB 的 engram shard**；`total_size = 113,675,759,792` | 本机核算 |
| F3 | 现有 `-4layer-scaled/config.json`（384 专家）**就是切片要用的 config**：与真实 config 的差异恰好是"4 层化"的全部改写，且同时带新旧 key 拼写 → 两套消费方都认。**注意：不能改用真实 config** —— `engram_meta_init=True` 会给层 1/14 建 meta Engram 模块，切片里没有对应张量 → DCP 直接失败 | 盘点 agent + 复核 |
| F4 | 引擎侧硬校验（`vllm-ascend-v41-private/vllm_ascend/models/deepseek_v41/model.py::build_layer_plan 117-195`）：`len(ratios) ≥ n_layers`、每个 source `> num_layers` 或 `ratios[source]==0` 都报错、`kv ⊆ index`、`candidate ∈ kv` → 三个 id 列表必须截断为 `[2]`（= `-scaled` 的值） | 引擎 agent |
| F5 | 引擎必需 `tokenizer.json` + `tokenizer_config.json`（engine init 时构造 tokenizer，即使 harness 只喂 token ids）；**不需要** `generation_config.json`/`quarot.safetensors`（后者只在 engram 开时读）。真实目录与模板的 tokenizer 文件 **md5 相同** | 引擎 agent + 复核 |
| F6 | 架构选择依赖 vision：`vision_n_layers > 0` → 改写 `architectures` 为 `DeepseekV41ForConditionalGeneration`（VL wrapper 建塔，缺张量会 `KeyError`）→ **切片必须保留 vision/aligner 张量**。而**训练侧不建 vision**（`include_vision` 默认 False）→ 那些名字会进 `skipped`，无害 | 复核 |
| ★H1 | **致命：vLLM 的加载是"按文件"而不是"按 index"**：`weight_utils.py:968-972` 迭代**文件里每个 key**，`deepseek_v4/model.py:1331` 是裸的 `params_dict[name]` → 链接进来的真实 shard 里含 **1086 个不属于切片的张量**（898 个 layers.4–39 + 149 个 mtp，其中 1047 个不是 `.bias`）→ 引擎加载 **KeyError**。两个已跑通的 4 层 ckpt 的 shard **零多余张量**（复核逐文件核对过），所以这个坑以前从未暴露。**换成整份拷贝也不能解决**（同样含多余张量） | 复核 |
| ★H2 | **真实权重下新增一个参数级差异**：`attn_sink`、`gate.bias`、`gate.bias_vl`、`hc_*{fn,base,scale}`（每层 9 个 ×4 层 = **36 个**）在真实 ckpt 里是 **fp32**，而训练侧 `prepare_deepseek_v41_model_for_fsdp` 把**所有**浮点参数统一转 bf16（`adapter.py:395-398`，其 docstring 明说"FP32-sensitive forward paths cast their operands locally"→ 用 `hc_fn.float()` 上转的是**已被舍入的 bf16 权重**）；引擎保持 fp32 并 fp32→fp32 拷贝。**随机 fixture 里看不出来**（那边这两类张量本来就存 bf16，两侧取值相同）。`image_start/end/newline` 是反向（真实 bf16 / 模板 fp32）。`check_engine_params.py` **不会**报这个（两侧读到的都是 fp32 原值） | 复核 + 本机实读 (dtype 已验) |
| H3 | `check_engine_params.py:167` 硬编码 `n_experts = 32` → 384 专家会**假 FAIL**（这正是"证明引擎持有 checkpoint 值"的脚本，假 FAIL 代价高） | 复核 |
| H4 | 所有 harness 入口默认指向 32 专家；不过 384 专家**引擎侧 dump 在 2026-09-21 已成功跑过**（`worklog_dsv41_train_infer_consistency.md` §11.1 有命令与结果）→ 不是首次。仍建议先做一次 params-only 冒烟 | 复核 + 本历次记录 |
| H5 | 切片的语义偏差（要写进 `SLICE.md`）：`candidate_source_layer_id 2`（真实是 20）→ **开了层 3 的候选预筛**，真实模型层 0–3 是关的；engram 关闭；且引擎侧 v4.1 专有代码路径挂在**带点**拼写 `deepseek_v4.1`（只有模板有）→ 本 harness 验证的是"模板 config 的引擎路径"，不是真实 ckpt 自带的 config | 复核 |
| H6 | 训练侧缺 key 的失败点是 `load_state_dict(strict=True)` 的 `RuntimeError`（DCP 的 `_broadcast_state_dict` 只是静默 pop），在 harness 自己的 `assert not still_meta` **之前**；三个 trainer 脚本都**丢弃**了 `report["missing"]` | 复核 |
| H7 | `/mnt/share` 是 NFS，vLLM 判定 `is_net_fs` 且 113.7 GiB ≤ 0.9×MemAvailable → **会后台整文件预取 ~120 GiB**（时间成本，非正确性问题） | 复核 |
| — | 已核实前提：fixture token id 0–128821 < `vocab_size 129280` ✓；真实 `tie_word_embeddings=false`（必须保留 `head.weight`）✓；本机 RAM 2.4 TB / NPU chips 2–9 空闲 ✓ | 本机 |

## 方案

**总思路：不改任何加载代码 —— 生成一个 4 层切片目录（config + 定向 index + **键集干净**的 shard + tokenizer），两套消费方都用 `--model-path` 指过去。**

### 1. 新脚本 `verl/scripts/dsv41_consistency/make_slice_from_real.py`

```
--src      /mnt/share/DeepSeek-V4.1-Flash-bf16
--out      /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real
--layers   0,1,2,3
--config-template /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled/config.json
--mode     hybrid|copy      # 默认 hybrid：干净的 shard 直接 hardlink，含多余张量的 shard 重写
```

步骤：
1. 读真实 `index.json` 与模板同目录 index；**选名白名单 = 模板 index 的名字集合**，断言真实侧零缺失，并输出名字构成统计（F2 的实算已给出预期值）。
2. **按 shard 分类（H1 的修复）**：对每个需要的 shard，比较"文件内实际 key 集合"与"index 指向该文件的选中名字集合"：
   - 相等 → **hardlink**（预计 24 个）；
   - 不等 → **重写**：`safe_open` 流式取出选中张量（**保持原 dtype，不改值**）写到新 shard（预计 7 个：`model-00079/00085/00158/00164/00228/00235/00267`；读 ≈27.5 GiB、写 ≈13.7 GiB）。
   重写后**断言不变量**：切片里每个 `*.safetensors` 的 `set(keys) == {index 里指向它的名字}`（这正是两个已跑通 ckpt 都满足、而当前方案会破坏的条件）。
   `--mode copy` = 对**所有** shard 走重写路径（自包含；代价是 113.7 GiB 读写）。
3. 写 `model.safetensors.index.json`：`weight_map` 指向切片内的 shard 名（干净 shard 保持真实文件名；重写的用同名覆盖写入切片目录即可保持一致）、`total_size` 重算、加 `metadata.dsv41_slice`（源目录、层列表、排除项、时间）。
4. 链接/复制 `tokenizer.json`、`tokenizer_config.json`（F5）。
5. 写 `config.json`：**逐字节复制模板**（F3）；仅当 `--layers` 非 `0,1,2,3` 时按规则重算（`num_hidden_layers := len(layers)`；`compress_ratios := [real[l] for l in layers]`；三个 source 列表过滤+映射局部下标；`candidate_source_layer* := max(kv_source ∩ slice)`；engram/MTP 关闭），任何一步不满足 F4 的约束就**报错退出**。
6. 写 `SLICE.md`：来源、层范围、**名字/dtype 差异清单（H2 的 39 项）**、排除项与理由（engram 197 GB、mtp、其它 36 层）、**语义偏差（H5：candidate_source 2 vs 20、engram 关闭、带点拼写的引擎路径）**、**"不要改用真实 config"**的警告（F3）、复现命令。
7. 打印摘要：选名数/字节数、link vs rewrite 的 shard 数、dtype 差异条数、config 差异。

### 2. 校验（五层，任一失败即停）

| 层 | 做法 |
|---|---|
| L1 结构 | 每名命中真实 index；**shard 键集不变量**（H1）成立；`total_size` = 选中张量字节和；重写过的 shard 抽查键集与 dtype |
| L2 config | `--layers 0,1,2,3` 时与模板**零字段差异**；断言 F4 四条约束 |
| L3 可加载 | (a) 训练侧 params-only 冒烟：`dump_trainer_stages.py --lengths 64 --out-tag _real_smoke`（覆盖 DCP strict / `still_meta`）；(b) **引擎侧先 params-only**：`run_dump_engine.sh --skip-params`（覆盖 H1 键集、F5 tokenizer、F6 vision/架构解析），再跑完整 dump |
| L4 数据不变 | 链接的 shard 天然一致；重写的 shard 抽样 `torch.equal`（含 fp32 的 `hc_attn_fn`、bf16 的 `layers.3.ffn.experts.0.w1.weight`） |
| L5 dtype 清单 | 复算真实 vs 模板的 dtype 差异，必须**恰好 39 项**（36 fp32→bf16 + 3 bf16→fp32），与 H2 一致 |

### 3. 用切片跑全套 harness

```bash
SLICE=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real
DUMP=/mnt/share/m00899630/dsv41/dump

# 3.1 引擎侧 dump（先 --skip-params 冒烟，再完整）
MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=enp48s3u1u1 ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
bash scripts/dsv41_consistency/run_dump_engine.sh --model-path $SLICE \
  --out-dir $DUMP/engine_real4 --lengths 64,200 --mem-util 0.85

# 3.2 参数指纹核对（H3：脚本改为自动读 config 的 n_routed_experts）
python3 scripts/dsv41_consistency/check_engine_params.py --params $DUMP/engine_real4/params.json --model-path $SLICE

# 3.3 训练侧输入打桩（4 变体 × 长度 64,200）
MODEL_PATH=$SLICE ENGINE_DIR=$DUMP/engine_real4 OUT_DIR=$DUMP/probe_input_real4 \
MASTER_PORT=59565 bash scripts/dsv41_consistency/sh/run_probe_inputs.sh

# 3.4 汇总（含路由翻转分解）
python3 scripts/dsv41_consistency/summarize_probe.py --lengths 64,200 \
  --probe-dir $DUMP/probe_input_real4 --engine-dir $DUMP/engine_real4

# 3.5（可选）§9 输出嫁接、§10 MoE dtype A/B；长序列档（真实权重下才有意义）：--lengths 64,200,700

# 3.6 ★H2 专项（推荐，见下）
```

**★H2 专项：36 个参数（hc_*/attn_sink/gate.bias）的 dtype 不对称到底贡献多少**
在训练侧加一个 env 门控（如 `VERL_DSV41_KEEP_FP32_PARAMS=1`）：在 `prepare_deepseek_v41_model_for_fsdp` 之后把这 36 个参数**还原成真实 fp32 值**（从 ckpt 读），跑 `baseline` 变体与 §9 的 `all.attn`/`all.ffn` 变体，与"当前 bf16 版本"对照 → 得到"参数级差异"这一新种子的量化贡献。
这一项既是 harness 的一部分，也是一个**可能的生产可修项**（若贡献显著，训练侧应让这些参数保持 fp32；注意 FSDP 对同组 dtype 一致性的限制，需要单独的 param group）。

### 4. 顺手要改的小东西

- `check_engine_params.py:167`：`n_experts` 改为从 `--model-path/config.json` 自动读（H3），保留 `--n-experts` 覆盖。
- 三个 trainer 脚本（`dump_trainer_stages.py:377`、`probe_module_inputs.py:262`、`graft_trainer_stages.py:200`）：把丢弃的 `report["missing"]` 变成断言（H6），失败信息比 DCP 的 RuntimeError 清楚。
- `dump_engine_stages.py:21-28` 的 `PARAM_PATTERNS` 不含 `layers.1.`：模板 config 下已无必要，建议补上（让层 1 也进指纹）。

## 风险与 caveat

1. **切片 ≠ 真实模型的前 4 层**（H5）：`candidate_source=2` 让层 3 开了候选预筛（真实模型 0–3 层是关的）；engram 路径被移除；引擎走的是"带点拼写 config"的路径。→ 两栈之间的**对照是公平的**（同一个 4 层模型），但**绝对数值不代表真实 40 层模型的训推差**，也不能评估 engram 相关问题。σ/KL 外推到整模型需要额外做分层累积估计。
2. **H1 是本方案的关键新步骤**：没有"按 shard 重写多余张量"，引擎侧一定 `KeyError`。这也是为什么"直接链接/整份拷贝"都不行。
3. **H2 会污染 baseline**：真实权重下 baseline 里含"参数精度不对称"这一新种子（随机 fixture 无此问题）→ 与历史数字对比时要显式扣除/标注；专项实验（§3.6）量化它。
4. **时间**：切片创建（hybrid）读 ~27.5 GiB + 写 ~13.7 GiB ≈ 10–20 分钟；引擎侧会整文件预取 ~120 GiB（H7）+ 加载，首轮 dump 预计 40–70 分钟；训练侧 4 变体 × 2 长度 ≈ 15–25 分钟。
5. **rank0 匿名内存 ~110 GB**、NPU 用 chips 2–9、别与大任务并发（本机另有两个交互 session）。
6. 忽略 `config.json_bak`、`optional/quarot.safetensors`、`-scaled/.config.json.swp`。

## 验证（怎么算成功）

- L1–L5 全过（L1 的键集不变量 + L5 的 39 项 dtype 清单是新增关键项）。
- 引擎侧 `check_engine_params` 全 OK（`--n-experts 384` 或自动读）。
- 训练侧 baseline：`in.all` 残余仍是"均匀无尾"小量（形态对照 §2.3/§2.4）；**H2 专项**给出"参数级差异"的贡献占比。
- 路由翻转分解：真实权重下翻转率/误差能量占比与随机权重的对照 → 定量回答"随机权重把放大夸大了多少"，写进工作记录作为方向 B 的判据。

## 交付物

1. `make_slice_from_real.py`（含 H1 的 shard 重写 + 五层校验 + `SLICE.md`）
2. 切片目录 `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real/`
3. `engine_real4` / `probe_input_real4` dump 与汇总表
4. H2 专项结果（可用 `VERL_DSV41_KEEP_FP32_PARAMS` 门控 + 对照表）
5. 工作记录：`worklog_dsv41_module_input_probe.md` 增补「真实权重 4 层复跑」；`worklog_dsv41_rl.md` 的信号阈值若有变化一并回写；H1/H2 两条坑写进 caveat
6. 顺手修：`check_engine_params.py` 的 n_experts；三个 trainer 脚本的 missing 断言
