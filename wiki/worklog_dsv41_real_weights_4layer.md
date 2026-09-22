# DeepSeek-V4.1 训推一致性：**真实权重 4 层复跑**（完整工作记录，2026-09-22）

> 本文是这条工作流的**完整、自包含**记录：从"为什么要用真实权重"到切片构造、两栈 dump、四组 fp32 A/B、
> 离线路由器机制、生产含义、代码/工具变更与复现命令。只读这一篇即可接手。
>
> 上游背景：`worklog_dsv41_module_input_probe.md`（随机权重下的"核差 × 路由翻转放大"机制）、
> `worklog_dsv41_train_infer_consistency.md`（§0–§11 逐阶段对比）、`worklog_dsv41_rl.md`（链路打通，§4.6 = 本篇摘要）。
> 详细计划与"核实到行号"的事实表：`/mnt/share/m00899630/plans/dsv41-real-weights-4layer/{plan.md,worklog.md}`。
> 原始 dump/日志：`/mnt/share/m00899630/dsv41/dump/{engine_real4,probe_input_real4,probe_input_real4_fp32*,trainer_real4}/`、
> `/tmp/{dump_engine_real4,probe_input_real4*,check_params_real4*}.log`。

---

## 0. 摘要（先看这里）

**一句话**：随机权重得出来的"训推差距 = 每模块 ~0.5% 核差 × MoE 路由翻转放大"**只在随机权重下成立**；
真实权重下，差距的主体换成了**一个可修的参数级 dtype 不对称**——

> checkpoint 里 **fp32** 的 MoE 路由器纠偏 bias（`layers.<i>.ffn.gate.bias`，引擎侧 `e_score_correction_bias`）
> 被训练侧的 `prepare_deepseek_v41_model_for_fsdp` 统一转成 **bf16**（`adapter.py:395-398`），而引擎保持 fp32。
> 架构本身声明它是 fp32（`Gate.__init__`，`model.py:936`），**两套真实权重（bf16 与 W8A8 量化版）实测都是 F32**。
> 该 bias **只进专家选择、不进路由权重**（`indices = (scores + bias).topk(...)`），所以它的舍入是**纯离散**的：
> 两侧喂**逐位相同**的输入，top-6 专家集合仍在 **20–62%** 的 token 上不同。

**量化（真实 4 层切片，384 专家，len=200）**：

| 配置 | baseline σ(Δlogp) | 输入全钉死后 σ | 钉死后 MoE 地板 | `model.norm` | argmax 一致率 |
|---|---|---|---|---|---|
| 现状（训练侧 bf16） | 0.2573 | 0.1882（**仅 1.4× 改善**） | 0.0316–0.0449（非均匀） | 0.0735 | 0.819 |
| 只把 `hc_*`+`attn_sink` 还原 fp32 | 0.2467 | 0.1882（无变化） | 同上 | 0.0735 | 0.829 |
| **只把 `gate.bias` 还原 fp32** | **0.0967** | **0.0202** | **0.0034–0.0042** | **0.0062** | 0.970 |
| 全部 32 个 fp32 参数还原 | 0.0668 | 0.0191 | 0.0034–0.0042 | 0.0056 | 0.970 |
| （对照）随机 scaled384 + head-gain 4 | 0.7062 | 0.0310（34.8× 改善） | 0.0054（均匀无尾） | 0.0070 | — |

→ **`gate.bias` 一项解释了 ~9× 的"钉死地板"与 2.7× 的 baseline σ**；还原后真实权重**回到随机权重那套"均匀核噪声"形态**
（地板 0.0034–0.0042 甚至低于随机权重的 0.0054）。机制**离线可复算**（CPU、秒级，
`analyze_gate_bias.py`），对任意 checkpoint 都适用。

---

## 1. 背景与目标

### 1.1 为什么必须换真实权重

随机权重（`DeepSeek-V4.1-Flash-4layer-scaled{32,}`）是 `--init scaled --head-gain 4` 生成的：
矩阵按 `1/sqrt(fan_in)`、norm=1、**bias/sink/HC 全 0**。这带来两个致命偏差：

1. **路由处在近平局**：bias 为 0、权重随机 → 专家得分互相接近，任何微小扰动都容易翻转 → 放大被夸大；
2. **`head-gain 4` 把 logprob 噪声放大 4×（KL 约 16×）** → 所有绝对值偏悲观。

更关键的是：**随机 fixture 里 `gate.bias` 两侧都是 bf16 且取值为 0，因此"参数精度不对称"这一种子在随机权重下根本不存在**。

### 1.2 目标

用真实权重切出 **4 层**（harness/fixture 都是围绕 4 层模型建的），两套加载路径都指向它，复跑全套一致性 harness，
拿生产量级的数字回答：随机权重把放大夸大了多少、要不要上 TIS/尾部纠正。

**用户已定**：切**第 0–3 层**；跑**全套 harness**。

---

## 2. 切片：真实 ckpt → 4 层（`make_slice_from_real.py`，新增工具）

源：`/mnt/share/DeepSeek-V4.1-Flash-bf16`（40 层 + 3 MTP、384 专家、1.53 TB、268 shard、无 `quantization_config`）。
产物：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real/`（+`SLICE.md`，可复现）。

### 2.1 关键不变量：**引擎按"文件"加载，训练侧按"index"加载**

| | 训练侧 FSDPTurbo | 引擎侧 vLLM |
|---|---|---|
| 取数 | **index 驱动**：`for name in weight_map`；不在 `param_meta` 的名字进 `skipped`、**不读字节**（`fsdp_turbo_dsv41_impl.py:140-150`）；shard 懒打开 | **文件驱动**：`for name in f.keys()`（`weight_utils.py:968-972`），唯一过滤 `should_skip_weight` 只丢"非本 rank 专家"（`ep_weight_filter.py:64-86`）；随后裸查表 `params_dict[name]`（`vllm_ascend/models/deepseek_v4/model.py:1331`） |
| 只改 config.json 后 | ✅ 能加载 | ❌ 31 个 shard 里 1086 个多余张量 → **KeyError** |

→ 切片必须满足：**每个 `*.safetensors` 的 key 集合 == index 指给它的名字**（两个已跑通 4 层 ckpt 都满足该不变量）。

### 2.2 做法与结果

- 选名白名单 = 现有 `-scaled` 模板 index 的名字集合（4972 个，真实 ckpt **0 缺失**）；
- 按 shard 分类：键集干净 → **hardlink**（24 个，0 字节）；含多余张量 → **重写**（7 个）：
  `model-00079 (101 extras) / 00085 (11) / 00158 (88) / 00164 (15) / 00228 (175) / 00235 (108) / 00267 (588)`
  → 读 27.5 GiB、写 13.68 GiB；**两个 183 GB 的 engram shard 完全不碰**；
- config **逐字节复制模板**（`-scaled` 的 4 层 config）——**不能改用真实 config**：`engram_meta_init=True` 会给层 1/14 建 meta Engram 模块，切片里没有对应张量 → DCP 直接失败；
- 结果：`total_size = 113,683,600,480 B (105.88 GiB)`，新增磁盘仅 13.68 GiB。

**内置五层校验全部通过**：① 每个 shard 的 key 集合 == index 名字；② config 与模板零字段差异；
③ 抽样张量（`hc_attn_fn`/`embed.weight`/`compressor.wkv`）与源**逐位相同**；
④ **dtype 审计 `{BF16→F32: 36, F32→BF16: 3}`**（36 个 = 每层 `hc_*`×6 + `attn_sink` + `gate.bias` + `gate.bias_vl`）；
⑤ 逻辑大小与 shard 字节和一致。

### 2.3 切片的语义偏差（写在 `SLICE.md` 里，务必带上看）

- `candidate_source_layer_id = 2`（真实是 20）→ 切片**层 3 开了候选预筛**，真实模型层 0–3 是关的；
- engram 关闭（真实层 1 有 197 GB engram 表）；MTP 关闭；
- 引擎走的是模板 config 的 `deepseek_v4.1`（带点）拼写代码路径；
- 训练侧不建 vision（`include_vision=False`）→ 落进 `skipped`；引擎侧建 vision（F6）→ 切片必须保留 vision/aligner 张量。
- **结论：两栈之间的对照是公平的（同一个 4 层模型），但绝对数值不代表 40 层真实模型的训推差。**

---

## 3. 环境坑（本轮新增/复现）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| E1′ | 主机名 `node-29-117` 解析到旧 IP（141.61.29.117，实际 80.5.25.117） | `/etc/hosts` 未更新 | 用 `MASTER_ADDR=127.0.0.1` + `GLOO_SOCKET_IFNAME=enp48s3u1u1`；torchrun 直连 127.0.0.1 的 harness 不受影响，**GRPO 脚本要注意** |
| E2′ | 引擎启动失败 `Free memory on device (49.59/61.28 GiB) ... less than 0.85`；降到 0.72 反而更差（free 42.97/**29.02**） | **其他租户任务瞬时占用**（各卡 3→12→27 GiB 跳动） | `--mem-util 0.5`（模型只需 19.2 GiB）+ 重试包装 `/tmp/run_engine_real4_retry.sh`（失败等 180 s，最多 8 次）→ 第 1 次即成功 |
| **E3′** | fp32 还原第一版直接改 `parameter.data` → `AssertionError: FSDP expects uniform original parameter dtype but got {torch.bfloat16, torch.float32}` | **FSDP2（`_fully_shard`）要求每个 param group 内 original dtype 一致**，且断言在**首次 forward 的 lazy init** 才触发（`_fsdp_param_group.py:238`） | 改成"**用点替换**"：参数保持 bf16，在 owner module 的 forward 前/后 hook 里临时换上 fp32 张量 |
| **E4′** | 用点替换第一版 → `RuntimeError: Expected all tensors to be on the same device, but got other is on cpu` | 加载用 `StateDictOptions(cpu_offload=True)`，`parameter.device` 是 cpu | 显式把 fp32 张量放到计算设备（`dt.init_distributed()` 的 `device`） |

> 另：`ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7,8,9`（chips 0/1 常有他人常驻）；后台跑用 `setsid nohup`。

---

## 4. 引擎侧 dump 与训练侧冒烟

### 4.1 引擎侧（`engine_real4`，vLLM-Ascend，8 rank）

```bash
MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=enp48s3u1u1 ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
bash scripts/dsv41_consistency/run_dump_engine.sh --model-path <slice> \
  --out-dir .../dump/engine_real4 --lengths 64,200 --mem-util 0.5
```

- 两档各 **179 张量/rank**、8 rank、`params.json`（84 张量指纹/rank）+ `engine_len{64,200}.json`（scored/argmax）+ `engine_decode_len*.json`。
- **引擎自身 decode↔prefill 自一致性（真实权重）**：len=64 max **0.006 nats**；len=200 max **0.20 nats**。
  对照随机权重（§11.3-3）：384 专家下同一口径最大到 **+4.09 / +2.65 nats** → **真实权重把这条通道压了 10–20×**（与"随机路由近平局 → 易翻转"一致）。

### 4.2 训练侧加载冒烟（L3a）

```
[dump-trainer] model built in 31.8s (102 parameters)
[dump-trainer] parameters materialized in 255.1s
[dump-trainer] len=64: next_logprobs[1:4]=[-17.9408, -9.5695, -19.1052] logits absmax=21.11 stages=57
```

→ 加载成功（无 meta 残留、DCP strict 通过）；logprob 是真实量级（随机 scaled 因 head-gain 4 会更高）。

### 4.3 参数指纹核对（L3c）

`check_engine_params.py --params engine_real4/params.json --model-path <slice>`

```
checked 672 engine parameters, 672 match the checkpoint
```

→ **引擎持有的确实是切片里的真实权重**（不是加载错位）。
过程修正：首轮 664/672，8 个 `vision.norm.weight` 是**检查器缺 vision 映射**造成的**假 FAIL**
（逐位核对切片值 `std=0.004180 absmax=0.070801` 与引擎指纹完全相同），已在检查器里补上 `vision.`/`aligner.` 分支。

---

## 5. 输入打桩：现状（训练侧 bf16）—— 与随机权重的对照

做法（`probe_module_inputs.py`）：在 `model.layers.<i>.<attn|ffn>` 的 forward **pre-hook** 里把输入换成引擎侧 dump 的
同一位张量（attn ← `language_model.model.layers.<i>.input_layernorm`，ffn ← `DeepseekV41DecoderLayer.rms_norm_cast@layer<i>[0]`），
输出仍与引擎的对应模块输出比。**输入逐位一致后剩下的差异 = 该模块自身的差异**，不含上游传播。

### 5.1 表（真实 4 层，`rel_err = ‖engine − trainer‖/‖trainer‖`，`*` = 输入被钉）

**len=64**

| module | in_nat | baseline | in.all.ffn | in.all.attn | in.all |
|---|---|---|---|---|---|
| 0.attn | 0.0000 | 0.0052 | 0.0052 | **0.0052\*** | **0.0052\*** |
| 0.ffn | 0.0067 | 0.0243 | **0.0239\*** | 0.0243 | **0.0239\*** |
| 1.attn | 0.0354 | 0.0243 | 0.0241 | **0.0048\*** | **0.0048\*** |
| 1.ffn | 0.0301 | 0.0106 | **0.0090\*** | 0.0102 | **0.0090\*** |
| 2.attn | 0.0378 | 0.0353 | 0.0297 | **0.0058\*** | **0.0058\*** |
| 2.ffn | 0.0384 | 0.0396 | **0.0353\*** | 0.0382 | **0.0353\*** |
| 3.attn | 0.0920 | 0.0690 | 0.0654 | **0.0059\*** | **0.0059\*** |
| 3.ffn | 0.0741 | 0.0478 | **0.0377\*** | 0.0446 | **0.0377\*** |
| model.norm | — | 0.0932 | 0.0777 | 0.0855 | 0.0754 |

| variant | Δmean | σ | σ vs base | \|Δ\|mean | p99 | max | argmax |
|---|---|---|---|---|---|---|---|
| baseline | +0.0330 | 0.2417 | 1.0× | 0.1874 | 0.581 | 0.626 | 0.794 |
| in.all.ffn | +0.0067 | 0.1992 | 1.2× | 0.1426 | 0.577 | 0.646 | 0.873 |
| in.all.attn | +0.0028 | 0.2171 | 1.1× | 0.1555 | 0.542 | 0.574 | 0.794 |
| in.all | −0.0004 | 0.1964 | **1.2×** | 0.1299 | 0.593 | 0.675 | 0.841 |

**len=200**：baseline σ 0.2573 / in.all.ffn 0.1978 / in.all.attn 0.2038 / **in.all 0.1882（1.4×）**；
argmax 0.819 / 0.839 / 0.834 / 0.864；钉死后 MoE 地板 0.0316–0.0449，`model.norm` 0.0735。
路由翻转（自然输入 vs 钉死输入）：len64 `3/5/7/21 of 64`，len200 `14/20/20/57 of 200`，翻转 token 承担误差能量 0.13–0.59。

### 5.2 与随机权重的对照（同 harness、同 384 专家）

| | 随机 scaled384（head-gain 4） | 真实 4 层（head-gain 1） |
|---|---|---|
| baseline σ（64 / 200） | 1.0786 / 0.7062 | 0.2417 / 0.2573 |
| **in.all σ** | **0.0310 / 0.0311** | **0.1964 / 0.1882** |
| 输入钉死后改善 | **34.8× / 22.7×** | **1.2× / 1.4×** |
| 钉死后 MoE 地板 | 0.0054（**均匀无尾**） | 0.0090–0.0449（**非均匀**） |
| `model.norm`（in.all） | 0.0070 | 0.0754 / 0.0735 |

**读法**：随机权重下"把输入钉死"几乎消灭全部噪声（34×）→ 噪声 = 输入差被放大；
真实权重下几乎没用（1.2×）→ **噪声主体不在"输入差传播"这条链上，而是模块内部一个新种子**。下一节把它揪出来。

---

## 6. ★ H2 专项：路由器纠偏 bias 的 dtype 不对称

### 6.1 事实（逐条核实）

| # | 事实 | 出处 |
|---|---|---|
| H2-1 | 真实 ckpt 里 `layers.<i>.ffn.gate.bias` = **fp32**、`gate.weight` = bf16、`hc_*`/`attn_sink` = fp32；**W8A8 量化版同样**（`quant_model_weights-*.safetensors` 里 `gate.bias` 也是 F32） | `safe_open` 实读两套 ckpt |
| H2-2 | 架构本身声明 fp32：`Gate.__init__` 里 `self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))`（`model.py:936`） | 源码 |
| H2-3 | 训练侧 `prepare_deepseek_v41_model_for_fsdp` 把**所有**浮点参数 `.to(parameter_dtype)`（bf16，`adapter.py:395-398`）；`read_dsv41_checkpoint_state_dict` 的 `cast()` 按 param dtype 落盘（`fsdp_turbo_dsv41_impl.py:129-135`） | 源码 |
| H2-4 | 引擎侧保持 fp32（dtype 审计 36 项；`check_engine_params` 672/672 全 OK，指纹按 fp32 原值比对） | 本机 |
| H2-5 | bias **只影响选择**、不影响权重：`indices = (scores + bias).topk(...)`；`weights = scores.gather(indices)`（`model.py:955-962`）→ 舍入误差是**纯离散**的 | 源码 |

### 6.2 机制（离线复现，**CPU、秒级**：`scripts/dsv41_consistency/analyze_gate_bias.py`）

1. **配方先验证**（否则后面的翻转率不可信）：引擎 dump 里捕获的 router
   （`AscendFusedTopKRouter._compute_routing`，8 个 decode token，**layer 0**）：
   - 用 `sqrtsoftplus(x @ W.T) + bias_fp32` 重算 top-6 → **48/48 槽位完全一致（100%）**；
     sigmoid 只有 37.5%、softmax 0% → 确认 score_func 是 `sqrtsoftplus`（config 不给 `score_func` 时的默认分支）；
   - 捕获的 logits（`#arg1`）与 `x @ W.T` 相比 **max|diff| = 1.9e-5**（该张量 std=1.13）→ 引擎 = fp32 计算 + fp32 bias。

2. **训练侧拿的确实是舍入值**：用 `probe_len*_baseline.pt` 里训练侧**自己记录的** top-6 反查——

   | bias 取值 | 与训练侧记录一致的 token 比例 |
   |---|---|
   | bf16 舍入值 | **100.0%**（4 层 × 2 长度全中） |
   | ckpt 的 fp32 原值 | 21.5% – 62.0% |

3. **两侧输入逐位相同时的路由翻转率**（引擎侧 MoE 输入 + 各自 bias）：

   | 层 | len=64 | len=200 | 翻转槽位占该 token 归一化路由权重（中位数） |
   |---|---|---|---|
   | 0.ffn | 37.5% | 61.5% | 8.5% / 9.0% |
   | 1.ffn | 20.3% | 24.5% | 8.0% / 8.6% |
   | 2.ffn | 60.9% | 62.5% | 11.9% / 12.2% |
   | 3.ffn | 59.4% | 60.0% | 9.0% / 9.1% |

4. **翻转一次的代价**（抽样 6 个 token，用 ckpt 专家权重重算 routed 输出）：
   单 token 的 routed 输出变化 = 该 token MoE 输出范数的 **0.3% – 22%** → 与"钉死输入后仍有 2–4% 的 MoE 残差"量级一致。

### 6.3 A/B：把 checkpoint 的 fp32 值还回去（`VERL_DSV41_KEEP_FP32_PARAMS`）

训练侧开关（env，默认关）：

| 值 | 含义 |
|---|---|
| `1` | 把 checkpoint 里所有 fp32 参数（本切片 = 32 个）在**用点**还原成 fp32 值 |
| `gate` | 只还原路由器纠偏 bias（`*.ffn.gate.bias`） |
| `continuous` | 只还原 `hc_*` + `attn_sink`（不还原 bias） |

实现要点（`dump_trainer_stages.py::restore_fp32_parameters`）：参数**保持 bf16**（FSDP2 要求每组一种 dtype，见 E3′），
在 owner module 的 forward 前/后 hook 里把 fp32 张量换进 `module.__dict__`，forward 结束再换回；
张量按**计算设备**放置（加载用 `cpu_offload=True`，见 E4′）。**引擎侧不需要重跑**（引擎本来就是 fp32）。

| 配置（len=200） | baseline σ | in.all σ | in.all.ffn 地板（4 层） | `model.norm`(in.all) | argmax |
|---|---|---|---|---|---|
| 现状（bf16） | 0.2573 | 0.1882 | 0.0316 / 0.0115 / 0.0411 / 0.0449 | 0.0735 | 0.819 |
| `continuous`（hc_*+attn_sink） | 0.2467 | 0.1882 | 同上（**毫无变化**） | 0.0735 | 0.829 |
| **`gate`** | 0.0967 | **0.0202** | **0.0037 / 0.0042 / 0.0038 / 0.0034** | 0.0062 | 0.970 |
| `1`（全部 32 个） | **0.0668** | **0.0191** | 同上 | **0.0056** | 0.970 |

len=64 同表：现状 0.2417 / 0.1964；continuous 0.2422 / 0.1979；**gate 0.0492 / 0.0178**；all 0.0621 / 0.0186。
（len=64 上 gate-only 甚至略优于 all-fp32 → `hc_*`/`attn_sink` 的贡献是**二阶且带噪声**；
两档一致的结论只有一条：**`gate.bias` 是主项**。）

**自然输入差同时塌缩**：`3.attn in_nat` 0.0923 → 0.0220（all-fp32, len200），`3.ffn in_nat` 0.0740 → 0.0183。
→ "路由一致"不仅让模块输出一致，也让残差流不再分叉，两个效应是**耦合**的。

### 6.4 生产含义

- **现象在生产权重上同样成立**：W8A8 量化 ckpt 的 `layers.0.ffn.gate.bias` 实测也是 `F32`。
- **偏差方是训练侧**：引擎忠实于 checkpoint；训练侧因为 FSDP 的"每 group 一种 dtype"约束把 fp32 **静默舍入**成 bf16。
- 修法（按代价排序）：
  1. **训练侧让这些参数保持 fp32**（最正）——需 FSDPTurbo 支持**每个 param group 各自的 dtype**
     （FSDP2 的断言是 group 内一致，`Gate`/`hc_*` 可作为独立 group），工程量中等；本 harness 的"用点替换"就是它的**效果预演**。
  2. 引擎侧把纠偏 bias 也按 bf16 参与选择（与训练侧对齐）——最便宜，但会改变推理语义（等于两侧一起降级）。
  3. 都不动：默认配置下 `old_log_probs` 由训练侧重算（`trainer_base.py::_compute_old_log_prob`），
     这 0.07–0.26 nats 只体现为 **off-policy 程度**与**监控指标偏差**（见 `worklog_dsv41_rl.md` §5.2 / §4.6）。
- **判据建议**：修完后 `in.all` 地板应 ≤0.005、且 `gate` 组数字与 `1` 组一致（当前 0.0202 vs 0.0191 → 已接近）。

---

## 7. 结论汇总（含对旧结论的修订）

1. **随机权重下的机制（核差 × 路由翻转放大）仍然成立，但只是"随机世界的机制"**：
   真实权重下输入钉死只改善 1.2–1.4×（随机 22–35×），说明噪声主体换成了参数级种子。
2. **主因是 `gate.bias` 的 fp32→bf16 舍入**：一项参数解释 ~9× 的钉死地板、2.7× 的 baseline σ；
   只还原 `hc_*`/`attn_sink` 无任何效果。
3. **可离线复算、可修**：`analyze_gate_bias.py` 对任意 checkpoint 给出"训练侧若用 fp32 bias 会选哪些专家"；
   训练侧保 fp32（或引擎侧对齐 bf16）即可。
4. **模块级"没有 bug"的结论不变**：权重逐张量一致（672/672 指纹），差异是**精度策略不对称**，不是加载/同步/内核错。
5. **监控指标的含义随之修订**：`rollout_corr/kl`、`pearson` 等在当前配置下混进了这个 10× 的离散种子 →
   在修 dtype 之前**不能**当作"同步是否健康"的判据；健康判据用本 harness 的直接量（模块 rel_err / Δlogp σ / 权重指纹）。

---

## 8. 代码与工具变更清单（本轮）

| 文件 | 变更 | 说明 |
|---|---|---|
| `scripts/dsv41_consistency/make_slice_from_real.py` | **新增** | 真实 ckpt → N 层切片（白名单选名 + shard 键集重写 + 五层校验 + `SLICE.md`）；`--mode hybrid|copy` |
| `scripts/dsv41_consistency/analyze_gate_bias.py` | **新增** | 离线路由器 dtype 分析：验证配方、复算训练侧/引擎侧路由、统计翻转率与槽位权重占比 |
| `scripts/dsv41_consistency/dump_trainer_stages.py` | 改 | ① 新增 `restore_fp32_parameters()`（用点替换，支持 `only=""|gate|continuous`）；② `report["missing"]` 断言（H6） |
| `scripts/dsv41_consistency/probe_module_inputs.py` | 改 | 接入 `VERL_DSV41_KEEP_FP32_PARAMS` + `missing` 断言 |
| `scripts/dsv41_consistency/graft_trainer_stages.py` | 改 | 同上 |
| `scripts/dsv41_consistency/check_engine_params.py` | 改 | ① `n_experts` 自动从 `config.json` 读（H3，旧版硬编码 32 会对 384 专家假 FAIL）；② 补 `vision.`/`aligner.` 映射（消 8 个假 FAIL） |
| `scripts/dsv41_consistency/dump_engine_stages.py` | 改 | `PARAM_PATTERNS` 补 `layers.1.`（**现有 `engine_real4/params.json` 是补之前生成的** → 84 参数/rank、层 1 只有 4 个；层 1 其余参数已由激活级对比 0.005–0.012 间接覆盖，要补齐指纹需重跑一次引擎 dump） |

**新增 dump**：`engine_real4/`、`probe_input_real4/`、`probe_input_real4_fp32/`、`probe_input_real4_fp32_gate/`、
`probe_input_real4_fp32_continuous/`、`trainer_real4/trainer_smoke_len64.pt`。

---

## 9. 复现命令（一条龙）

```bash
cd /workspace-verl/verl
SLICE=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real
DUMP=/mnt/share/m00899630/dsv41/dump

# 0) 切片（86 s；新增磁盘 13.68 GiB）
python3 scripts/dsv41_consistency/make_slice_from_real.py \
  --src /mnt/share/DeepSeek-V4.1-Flash-bf16 --out $SLICE --layers 0,1,2,3 \
  --config-template /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-scaled/config.json

# 1) 引擎侧 dump（含租户波动重试；mem-util 0.5）
setsid nohup bash /tmp/run_engine_real4_retry.sh > /tmp/run_engine_retry_driver.log 2>&1 &

# 2) 参数指纹（应 672/672）
python3 scripts/dsv41_consistency/check_engine_params.py \
  --params $DUMP/engine_real4/params.json --model-path $SLICE

# 3) 训练侧输入打桩：四种配置（引擎侧不用重跑）
#    现状 / 只 gate / 只连续项 / 全还原
run_probe() { MODEL_PATH=$SLICE ENGINE_DIR=$DUMP/engine_real4 OUT_DIR=$2 \
  VERL_DSV41_KEEP_FP32_PARAMS=$1 MASTER_PORT=$3 \
  bash scripts/dsv41_consistency/sh/run_probe_inputs.sh; }
run_probe ""          $DUMP/probe_input_real4                59565
run_probe gate        $DUMP/probe_input_real4_fp32_gate      59571
run_probe continuous  $DUMP/probe_input_real4_fp32_continuous 59572
run_probe 1           $DUMP/probe_input_real4_fp32           59573

# 4) 汇总（含路由翻转分解）
python3 scripts/dsv41_consistency/summarize_probe.py --lengths 64,200 \
  --probe-dir <上面任一目录> --engine-dir $DUMP/engine_real4

# 5) 离线路由器 dtype 分析（CPU，秒级）
python3 scripts/dsv41_consistency/analyze_gate_bias.py --lengths 64,200 \
  --model-path $SLICE --engine-dir $DUMP/engine_real4
```

---

## 10. Caveat 与遗留

### 10.1 Caveat

- **4 层切片 ≠ 40 层真实模型**：`candidate_source=2`（层 3 开了候选预筛）、engram 关闭、MTP 关闭、引擎走"带点拼写 config"路径。
  但 `gate.bias` 机制是**逐层局部**的（每层一个 bias），层数更多只会让翻转更多 → 40 层下的绝对值预计**不小于**本处测得值；具体数字要等整模型复跑。
- **只测 prefill↔prefill**：RL 用 rollout（decode）产出的 logprob，decode↔prefill 是另一条独立通道（384 专家下需 `rl_config.enable_batch_invariant=true`）。
- **只覆盖两种长度 64/200**：更长序列（压缩 KV / index_topk 全开）未测。
- **引擎侧 routed/shared 拆不开**：`language_model.model.layers.<i>.mlp.experts` 看起来像"只含 routed"，实测就是**整块 MoE 输出**
  （Ascend 融合 MoE 把 shared 吃进去了；与训练侧 `ffn` 的 cos=0.99996、rel_err 同 `.mlp`）→ 别拿它做 routed-only 对比。
- **引擎 dump 里的 router 捕获只有 1 层 × 8 个 decode token**：本轮的"引擎侧路由"结论靠"8 token 100% 命中 + logits 吻合 1.9e-5"来锚定，
  其余翻转率是"引擎输入 + 各自 bias"的离线复算（不是引擎那次前向的真实选择）。
- **`hc_*`/`attn_sink` 的贡献是二阶**：`continuous` 组单独看不出变化，但与 `gate` 组合后 baseline σ 从 0.0967 → 0.0668，属于交互项，未单独量化。

### 10.2 遗留

1. **整模型（40 层）复跑**：需要处理 engram（真实 config `engram_meta_init=True` 要 197 GB 表；可像切片一样关掉并写进 caveat）。
2. **修 dtype 不对称**（修法 1：FSDPTurbo per-group dtype）→ 再做一次回归，判据：`in.all` 地板 ≤0.005、
   `gate` 组与全还原组数字一致、`rollout_corr/kl` 回到"随机权重那种与专家数无关"的形态。
3. 长序列档（700/1500）。
4. 层 1 参数指纹补齐（重跑一次引擎 dump，`PARAM_PATTERNS` 已补 `layers.1.`）。
5. 384 专家下 `VERL_DSV41_MOE_BF16=1` 的激活 dtype A/B（旧遗留；优先级已降低——真正的 dtype 问题在**参数**而非**激活**）。
