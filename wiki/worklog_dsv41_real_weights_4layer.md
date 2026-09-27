# DeepSeek-V4.1 训推一致性：**真实权重 4 层复跑**（完整工作记录，2026-09-22）

> 本文是这条工作流的**完整、自包含**记录：从"为什么要用真实权重"到切片构造、两栈 dump、四组 fp32 A/B、
> 离线路由器机制、生产含义、代码/工具变更与复现命令。只读这一篇即可接手。
>
> 上游背景：`worklog_dsv41_module_input_probe.md`（随机权重下的"核差 × 路由翻转放大"机制）、
> `worklog_dsv41_train_infer_consistency.md`（§0–§11 逐阶段对比）、`worklog_dsv41_rl.md`（链路打通，§4.6 = 本篇摘要）。
> 详细计划与"核实到行号"的事实表：`/mnt/share/m00899630/plans/dsv41-real-weights-4layer/{plan.md,worklog.md}`。
> 原始 dump/日志：`/mnt/share/m00899630/dsv41/dump/{engine_real4,probe_input_real4,probe_input_real4_fp32*,trainer_real4}/`、
> `/tmp/{dump_engine_real4,probe_input_real4*,check_params_real4*}.log`。
>
> **§13（2026-09-24 补写）** 是"**bias 是 fp32 这个结论是怎么定下来的**"的取证细节：三个子命题分开证、
> 每条证据的强度与证伪条件、今天还能复现什么。其中今天的复算数字标 [2026-09-24 复算]；
> **注意 real4 系列 dump 已被清理**（§13.6 列了要重跑才能复现的项）。

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
**"bias 是 fp32"这个结论的取证过程（含引擎侧那个会"自己降级"的分支、证据强度与证伪条件）见 §13。**

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

> 本节是结论速览；**每一步"凭什么这么判"的取证细节见 §13**（含引擎侧那个会"自己降级"的 dtype 分支、
> 48/48 的证据强度、以及阳性/阴性对照的清单）。

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
  **两条证据的分工与强度**：真正把"引擎用 fp32 bias"钉住的是 **logits 的 1.9e-5**（配引擎代码里那个
  `text_bias.to(router_logits.dtype)` 分支）；48/48 只够锚定配方（8 token 对 bf16 假设的排除力约 0.05–2.3%）。
  详见 §13.2。
- **`hc_*`/`attn_sink` 的贡献是二阶**：`continuous` 组单独看不出变化，但与 `gate` 组合后 baseline σ 从 0.0967 → 0.0668，属于交互项，未单独量化。

### 10.2 遗留

1. **整模型（40 层）复跑**：需要处理 engram（真实 config `engram_meta_init=True` 要 197 GB 表；可像切片一样关掉并写进 caveat）。
2. **修 dtype 不对称**（修法 1：FSDPTurbo per-group dtype）→ 再做一次回归，判据：`in.all` 地板 ≤0.005、
   `gate` 组与全还原组数字一致、`rollout_corr/kl` 回到"随机权重那种与专家数无关"的形态。
3. 长序列档（700/1500）。
4. 层 1 参数指纹补齐（重跑一次引擎 dump，`PARAM_PATTERNS` 已补 `layers.1.`）。
5. 384 专家下 `VERL_DSV41_MOE_BF16=1` 的激活 dtype A/B（旧遗留；优先级已降低——真正的 dtype 问题在**参数**而非**激活**）。
6. **复现性**：[2026-09-24 查] real4 系列 dump 已被清理（只剩空的 `engine/`）→ 本文里依赖 dump 的数字
   （logits 1.9e-5、48/48、翻转率、identification）要重跑 §9 的引擎 dump + probe 才能复核；仍可秒级复核的项见 §13.6。

---

## 11. 附录 A：这个结论是**怎么被定位的**（推导过程）

整条链每一步都有明确的**证伪条件**；同时记下中途哪一步差点被推翻。

### 11.1 起点：两个"不该出现"的数字

跑完 `summarize_probe.py`（真实权重）后，有两件事和随机权重的经验对不上：

| 观测量 | 随机 scaled384 | 真实 4 层 | 为什么反常 |
|---|---|---|---|
| 输入全钉死后 σ 的改善 | 22–35× | **1.2–1.4×** | 若噪声主体是"输入差 × MoE 放大"，钉死输入就该像随机那样把噪声打掉 |
| 钉死后的 MoE 残差 | 0.0054（均匀无尾） | **0.0090–0.0449（非均匀）** | 同样是"输入逐位一致"，地板却比核差高 2–8× |

→ **推论 1：偏差不在"输入差被传播放大"这条链上。**

### 11.2 排除法收窄到"权重里的某个参数"

输入逐位一致时，两侧模块的输出还能不同的原因只有两类：① 内核算术不同；② 两栈**手里持有的权重值**不同。
随机权重用的是同一套内核、同一套 harness，已经把 ① 钉在 0.0054 → **只能是 ②**。
叠加已知事实（切片 dtype 审计：ckpt 里有 36 个 fp32 张量、训练侧统一转 bf16），候选缩小到那 32–36 个参数。

### 11.3 挑出唯一"能离线判定"的候选

这 32 个里，只有 `gate.bias` 进**离散选择**（`(scores + bias).topk()`，而路由权重取自**无偏** scores）。
离散 = 可以直接问"两侧各自会选哪些专家"，**不需要跑 NPU**。所以先验证它，而不是先跑 A/B。

### 11.4 先验证测量工具，再相信测量结果

要让"翻转率"有意义，必须先证明我的离线路由复现与两栈一致：

- 引擎 dump 里捕获了 router（`AscendFusedTopKRouter._compute_routing`：logits + 选中的 6 个专家，8 个 decode token）；
- 用 `sqrtsoftplus(x @ Wᵀ) + bias_fp32` 重算 → **48/48 槽位全中（100%）**（sigmoid 只有 37.5%、softmax 0% → 顺带确认了 score_func）；
- 捕获的 logits 与我算的 `x @ Wᵀ` 差 **1.9e-5**（该张量 std = 1.13）→ 引擎 = fp32 计算 + fp32 bias。

> **"引擎 = fp32 bias"这一步需要证据而不是读代码**：引擎侧路由器的代码是 dtype 自适应的
> （`text_bias.to(router_logits.dtype)`），logits 若不是 fp32，引擎自己就会把 bias 降级成 bf16。
> 上面这两条的推理细节、各自的证据强度与证伪条件，见 **§13.2**。

### 11.5 决定性一步：**识别**，不是相关

训练侧 dump 里存着**它自己算出来的 top-6**。于是把同一个输入重算两遍，只换 bias 的取值：

| bias 用哪个值 | 与训练侧记录一致的比例 |
|---|---|
| bf16 舍入值 | **100.0%**（4 层 × 2 长度全中） |
| ckpt 的 fp32 原值 | 21.5% – 62.0% |

（逐层 8 行的细表、以及"为什么这是识别而非相关"的读法见 **§13.3**。）

这一步测的不是相关性，而是**训练侧手里实际拿的是哪个值**。内核差、输入差、代码 bug 都造不出这种
"100% vs 21–62%"的二分；同时它把"是引擎错了还是训练侧错了"这个问题**定死在训练侧**。

**自检（如果结论错，会看到什么）**：若两侧都用 fp32 → 两个值都该接近 100%；若都用 bf16 → fp32 值也该 100%。两者都没出现。

### 11.6 中途的自我怀疑（差点推翻自己）

37–62% 的 token 换了专家集合，可钉死输入后的 MoE 残差只有 2–4% —— 量级看起来矛盾。于是拆了两层：

- 翻转槽位占该 token **归一化路由权重的中位数 ~9%**（不是可忽略的尾巴）；
- 用 ckpt 专家权重重算 6 个翻转 token：单 token 输出变化 = 该 token MoE 输出范数的 **0.3%–22%**。

→ 与 2–4% 的全局残差量级一致（全局是范数比、且有抵消），假设存活。

### 11.7 因果闭环：A/B + 阴性对照

把 fp32 值还回去（`VERL_DSV41_KEEP_FP32_PARAMS`）跑三组：

- **只还原 `gate.bias`** 就复现了几乎全部效果（σ 0.188→0.020、地板→0.0034–0.0042）；
- **只还原 `hc_*`/`attn_sink`**（`continuous`）**毫无变化** —— 这是关键的**阴性对照**，排除"随便哪个 fp32 参数都有用"。

### 11.8 最后一环：确认不是 bf16 导出的偶然

生产用的 **W8A8 量化权重**里 `layers.0.ffn.gate.bias` 实测同样是 `F32` → 结论对生产栈成立。

### 11.9 为什么前几轮都没照出来（方法论小结）

| 盲区 | 说明 |
|---|---|
| 随机 fixture 里这个种子**不存在** | `--init scaled` 把 `gate.bias` 与 `hc_*` 全置 0，且两侧都是 bf16 → 无差可测 |
| 参数指纹检查**读错对象** | `check_engine_params` 读的是 **checkpoint 原始值**，不是训练侧实际持有的值 → 672/672 照样全 OK |
| 端到端指标**被离散放大淹没** | kl/pearson 这类指标在随机权重下本就大，量级断层不明显 |
| 单看"内核差"会**找错方向** | 真正的载体是**加载期的一次静默精度降级**，不是某个 kernel 不准 |

**能照出它的组合恰好是本 harness 的两条设计**：① 输入打桩（把"输入差传播"这条链掐断，暴露模块内种子）；
② dump 里保存了两栈**各自的中间决策**（使"谁手里是哪个值"可以离线二分定位）。

---

## 12. 下一步方向（含成本与判据）

### 12.1 总览

| # | 方向 | 价值 | 成本 | 前置 |
|---|---|---|---|---|
| **1（推荐）** | **训练侧让 `gate.bias` 保持 fp32** ✅ **已完成（2026-09-22）** | 把"诊断"变成"修复+回归"；引擎不动、推理语义不变 | 模型侧小改 + 加载侧补 buffer ≈ 半天；回归 ~1 h | 无 |
| 2 | 引擎侧对齐（bias 按 bf16 参与选择） | 两栈自洽，1 行 | 极小 | 无（但改 rollout 分布） |
| 3 | 整模型 40 层复跑 | 生产绝对 σ/kl，决定要不要 TIS | 大（1.53 TB、engram 要处理、小时级） | 建议在 1 之后 |
| 4 | RL 端到端 A/B（同一份数据"修 vs 不修"） | 决定 (a) 换指标 / (b) 上 TIS 的**最终仲裁** | 中（一次 GRPO step） | 需要 1 或 2 |
| 5 | 遗留小项（层 1 指纹、长序列档、激活 dtype A/B） | 完整性 | 小 | 无 |

### 12.2 方向 1 的做法（✅ **已实施并验证，2026-09-22；完整记录见 [worklog_dsv41_gate_bias_fp32_fix.md](worklog_dsv41_gate_bias_fp32_fix.md)**）

> 结论先行：采纳了路线 **(a) fp32 buffer**。修复后**无任何 env 开关**跑同一套 harness，
> 数字与本节 §6.3 的"只还原 `gate.bias`"逐项相同（len200：baseline σ 0.0967 / in.all σ 0.0202 /
> 地板 0.0037-0.0042 / `model.norm` 0.0062 / argmax 0.970），离线 identification 也从"bf16 100%"翻成
> "fp32 100%"。实施中踩到的两个真问题（DCP 广播加载要求模型状态**单一 device**；`assign=True` + `cpu_offload`
> 会把载入的 buffer 变成 host 张量）与三个自伤（诊断代码覆盖 `device` 变量名、指纹读旧引用、`dt.` 跨脚本引用）
> 都记在那篇的 §4。

1. **可行性事实**：FSDPTurbo 的 `fully_shard_parallel.py:44-49` 是 `module_config.update(plan)` ——
   `fsdp_plan.apply_modules` **支持逐模块覆盖**（含 `mp_policy`）；模式是精确分段匹配
   （实测 `model.layers.{*}.ffn.gate` 只命中 Gate，不连带命中 `...gate.bias`）。
2. **卡点**：Gate 组内 `weight` 是 bf16、`bias` 是 fp32 → **同组混 dtype 会踩同一个断言**
   （`FSDP expects uniform original parameter dtype`，`_fsdp_param_group.py:238`）。所以不能只加一条配置了事。
3. **两条可行路线**（择一）：
   - **(a) `gate.bias` 改成 fp32 buffer**（**采纳**）：bias 只进 `topk` 的**选择**，梯度根本流不过去（路由权重取自无偏 scores）
     → 改成 buffer 语义等价、且不进 FSDP 参数组；代价是加载侧要补"buffer 也要从 ckpt 读"（现有 loader 是 `param_meta` 驱动的）。
   - **(b) 把 bias 拆成独立子模块**（`gate.correction_bias` 之类），让它自己成一个 fsdp group 并给 fp32 mp_policy；
     代价是模型结构/SD 名字要动（引擎侧 WeightsMapper 也在等这个名字），已否决。
4. **回归判据**（跑现成 harness，`baseline` 一项即可，不必设 env 开关）：
   `in.all` 地板 ≤0.005、`model.norm` ≤0.006、argmax ≥0.96、且**不再需要** `VERL_DSV41_KEEP_FP32_PARAMS` 才成立。
5. **风险**：改动落在训练栈的模型代码（FSDPTurbo）与加载路径上 → 要跑一次完整性检查：训练侧冒烟（`dump_trainer_stages.py`）
   的 logprob 与修前一致（除了该参数本身带来的差异）、DCP strict 通过、8 卡 GRPO 能起。

### 12.3 方向 2（备选）：引擎侧对齐

- 做法：vllm-ascend 路由处把 `e_score_correction_bias` 按 bf16 参与选择（与训练侧对齐）。
- 代价：**改变 rollout 分布**（不再是 checkpoint 的忠实行为）；只解决"两栈自洽"，不解决"谁更对"。
- 适用：训练侧改不动 / 只想让监控指标干净时。

### 12.4 方向 3/4 的判据

- **方向 3**（40 层）：需要先解决 engram（真实 config `engram_meta_init=True` 要 197 GB 表；可像切片一样关掉并写进 caveat）
  与 fixture 的 4 层约定；产出"生产绝对 σ / kl"，用来判断线上 kl 的落点。
- **方向 4**（RL A/B）：同一份 prompt 数据、同一 seed，跑"修 vs 不修"各一步，比
  `rollout_corr/kl`、`rollout_actor_probs_pearson_corr`、`rollout_probs_diff_mean`、reward 曲线与 grad norm。
  **判据**：修后 kl/pearson 应回到"与专家数无关"的形态（参考随机权重 32 专家的 0.010/0.979）；
  若修后仍有量级断层，才考虑开 `algorithm.rollout_correction` 的 token 级 TIS。

---

## 13. 附录 B：**"bias 是 fp32"是怎么定下来的**（取证细节；2026-09-24 补写）

> §11 记的是"结论怎么被定位"的思路链；本节把其中**指向 dtype 的那一步**展开到可复核的粒度：
> 每个子结论的证据、去哪个文件/哪一行取证、这条证据**能**证明什么**不能**证明什么、以及今天还能不能复现。
>
> 标注约定：**[2026-09-22 记录]** = 当时实测（原始 dump 已清理）；**[2026-09-24 复算]** = 今天重新算过（CPU、秒级）。

### 13.0 先拆成三个问题：混在一起就会得出过强的结论

现象端只有一条线索：**输入逐位钉死后，两侧的 MoE 残差仍有 0.9–4.5%、且非均匀**（随机权重同口径是 0.54%、均匀无尾）。
输入相同还能不同，只可能是"**两侧手里持有的权重值不同**"（内核差已被随机权重钉在 0.54%）。但"bias 是 fp32"其实是三个命题：

| # | 命题 | 主要证据 | 强度 |
|---|---|---|---|
| **Q1** | checkpoint 里它是 **fp32** | safetensors 直读 + 架构/引擎的显式声明 + 切片 dtype 审计 | 静态事实，硬 |
| **Q2** | **引擎运行时是拿 fp32 值做选择** | ⓐ 引擎代码里那个 **dtype 自适应分支**（说明"只看声明不够"）ⓑ 捕获 logits 与 fp32 `x@Wᵀ` 差 1.9e-5 ⓒ 48/48 配方命中 ⓓ 引擎自报 dtype/指纹 | ⓑ 是主证，ⓒⓓ 是辅证 |
| **Q3** | **训练侧手里是 bf16 舍入值** | identification：用训练侧**自己记录的** top-6 反查 | 最强（二分式） |

**Q3 是"两侧不对称"的定音锤；Q1/Q2 决定"偏差方是谁"**——而"偏差方是训练侧"这一点直接决定了修复方向（改训练侧保精度，而不是改引擎对齐 bf16）。

### 13.1 Q1：静态事实——两套 ckpt、两套代码都声明 fp32

**（a）直读 safetensors**（[2026-09-22 记录] 结论；[2026-09-24 复算] 同一张表，数字相同）

```bash
python3 - <<'PY'
import json, torch
from pathlib import Path
from safetensors import safe_open
mp = "/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real"
idx = json.load(open(Path(mp) / "model.safetensors.index.json"))["weight_map"]
for i in range(4):
    n = f"layers.{i}.ffn.gate.bias"
    with safe_open(Path(mp) / idx[n], framework="pt") as h:
        t = h.get_tensor(n)
    d = (t.to(torch.bfloat16).float() - t.float()).abs()
    print(n, t.dtype, f"range [{t.float().min():.4f}, {t.float().max():.4f}]",
          f"|d|max={d.max():.4f} mean={d.mean():.5f} changed={(d > 0).float().mean() * 100:.1f}%")
PY
```

| 张量 | dtype | 值域 | bf16 往返 \|Δ\|max | \|Δ\|mean | 被改动的行 |
|---|---|---|---|---|---|
| `layers.0.ffn.gate.bias` | **float32** | 9.5922 – 9.8844 | 0.0312 | 0.02317 | **100%** |
| `layers.1.ffn.gate.bias` | **float32** | 9.0820 – 9.4042 | 0.0309 | 0.01058 | **100%** |
| `layers.2.ffn.gate.bias` | **float32** | 10.4826 – 10.8067 | 0.0307 | 0.02279 | **100%** |
| `layers.3.ffn.gate.bias` | **float32** | 10.6187 – 10.8808 | 0.0312 | 0.01831 | **100%** |
| `layers.0.ffn.gate.weight`（对照） | bfloat16 | −0.3848 – 0.3926 | 0.0000 | 0.00000 | 0% |
| `layers.0.ffn.gate.bias_vl`（对照） | float32 | 20.5443 – 21.4099 | 0.0621 | 0.02132 | 100% |

**这三个数是整个机制的量纲**，值得单独读一遍：

1. bias 的值域在 **~9–11**；bf16 在该量级的 ULP 是 **0.0625** → 一次舍入最多走 **0.031**；
2. 而这个 bias 是**加在 O(1) 的得分上**（`sqrtsoftplus` 的 `x=0` 处是 √ln2 ≈ 0.83）——**量级完全不对等**；
3. 于是"0.03 的绝对误差落在**被选择量**上"：反过来说，37–62% 的翻转率意味着 top-6 边界处的得分间隔普遍 **< 0.03**
   （这是推论，不是独立测量）。

对照列说明这不是"所有参数都在被舍入"：`gate.weight` 本身就是 bf16（往返零误差），bias 才是那个**声明 fp32 却被统一 cast**的张量。

**（b）代码声明**（[2026-09-24 核对行号]；修复前的行号若不同在括号注明）

| 侧 | 出处 | 内容 |
|---|---|---|
| 训练侧架构 | `FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py:922`（`class Gate`）的 `__init__`（当时 = :936） | `self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))` —— **显式 fp32**（修复后同一位置是 `register_buffer("bias", torch.zeros(..., dtype=torch.float32))`，见 :949） |
| 训练侧 forward | 同上 `:959-969` | `scores = linear(x.float(), self.weight.float()) / self.gate_temp` → `indices = (scores + bias).topk(...)` → `weights = scores.gather(...)`。**两侧的 GEMM 本来就都是 fp32**，bias **只进选择、不进权重** |
| 训练侧加载 | `FSDPTurbo/fsdp_turbo/models/deepseek_v41/adapter.py:384` `prepare_deepseek_v41_model_for_fsdp` → `:403`（当时 = :395-398） | `for parameter in model.parameters(): if parameter.is_floating_point(): parameter.data = parameter.data.to(dtype=parameter_dtype)` → **"架构声明 fp32"在这条路上被统一改成 bf16** |
| 引擎架构 | `/workspace-verl/vllm-ascend-v41-private/vllm_ascend/models/deepseek_v4/model.py:346` | `self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))` |
| 引擎加载 | 同上 `:1183` | `.gate.bias` → `.gate.e_score_correction_bias` 名字映射（`default_weight_loader` 的 `param.data.copy_` **不改 param 自身 dtype**；bf16→fp32 是加宽、fp32→fp32 是原样） |

**（c）切片 dtype 审计** [2026-09-22 记录]：口径是"**模板（`-scaled`）dtype → 真实 ckpt dtype**"，只读 header、不读数据
（`make_slice_from_real.py::audit`，L5 校验）。结果 `{BF16 → F32: 36, F32 → BF16: 3}` ——
那 36 个 = **4 层 ×（`hc_*`×6 + `attn.attn_sink` + `ffn.gate.bias` + `ffn.gate.bias_vl`）**
→ 它是**每层都有**的系统性布局，不是某个张量的偶发（"真实 ckpt 里哪些张量是 fp32"因此可以一次列全）。
（与 §6.3 表里"全部 32 个 fp32 参数"差 4：`bias_vl` 在训练侧因 `include_vision=False` 不存在——`register_buffer("bias_vl", ... if args.vision_enabled else None)`；36 − 4 = 32 ✓）

**（d）生产权重（W8A8）上同样成立** [2026-09-22 记录；2026-09-24 复读]：

- `quant_model_description.json` 里 **129 个 `*.gate.*` 项全部标 `FLOAT`**（量化配方**不量化** gate。
  注意 `FLOAT` 是**配方标签**——"这一项保留浮点"，不等于存储 dtype；存储 dtype 以实读为准，见下一行）；
- 实读 `/mnt/share/l00896581/DeepSeek-V4.1-Flash-w8a8` 的 shard `00009`：
  `layers.0.ffn.gate.bias` = **torch.float32**、值域 **9.5922 – 9.8844**（与 bf16 切片**逐个值同一区间** → 同一份模型）；`layers.0.ffn.gate.weight` = bf16。
- 命令：

```bash
python3 - <<'PY'
from pathlib import Path
from safetensors import safe_open
p = Path("/mnt/share/l00896581/DeepSeek-V4.1-Flash-w8a8/quant_model_weights-00009-of-00272.safetensors")
with safe_open(str(p), framework="pt") as h:
    t = h.get_tensor("layers.0.ffn.gate.bias")
print(t.dtype, tuple(t.shape), t.float().min().item(), t.float().max().item())
# 配套：json.load(open(.../"quant_model_description.json")) 里 ".gate." 项全部是 "FLOAT"
PY
```

→ **生产栈不是"bf16 导出时的偶然"**（§11.8 的那一步，今天可秒级复核）。

> **Q1 的边界**：以上只证明"设计上/文件里是 fp32"。**不能**推出"引擎运行时一定用它算" —— 见下。

### 13.2 Q2：引擎运行时——代码里有一个"会自己降级"的分支，所以**必须**验 logits 的精度

这是整条链里**最容易跳过**、也最值得单独讲的一步。引擎侧融合路由器是 **dtype 自适应**的
（`vllm-ascend-v41-private/vllm_ascend/ops/fused_moe/router/fused_topk_router.py`）：

```python
# :181-189  sqrtsoftplus 融合路径（本 recipe 走这条）
bias_vl = self.bias_vl
if bias_vl is not None and bias_vl.dtype != router_logits.dtype:
    bias_vl = bias_vl.to(router_logits.dtype)
text_bias = self.e_score_correction_bias
if text_bias is not None and text_bias.dtype != router_logits.dtype:
    text_bias = text_bias.to(router_logits.dtype)   # ← 若 logits 是 bf16，这里就把 bias 降级成 bf16 了
topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
    x=router_logits, k=self.top_k, bias=text_bias, ...)

# :208-209  非融合回退路径（更狠：就地改写参数本身，持久降级）
if self.e_score_correction_bias is not None and self.e_score_correction_bias.dtype != router_logits.dtype:
    self.e_score_correction_bias = self.e_score_correction_bias.to(router_logits.dtype)
```

**引擎并没有"硬编码用 fp32 bias"。** 如果 `router_logits` 是 bf16，引擎自己就会把 bias 换成 bf16 ——
那样两侧反而"一致"（都 bf16），也就不该有本次现象。所以"引擎用 fp32 bias"这句话的**真正落点是"引擎的 router_logits 是 fp32"**。

**（a）代码链：两条分支都产 fp32 logits**

- `self.gate.precast_fp32_weight = True`（`models/deepseek_v4/model.py:289`），非 internal-router 路径
  `router_input = hidden_states.float() ...; router_logits = F.linear(router_input, self.gate.weight)`（`:408-409`）；
- internal-router 路径同样：`hidden_states_fp32 = ... hidden_states.float(); router_logits = F.linear(hidden_states_fp32, gate.weight_fp32)`
  （`ops/fused_moe/fused_moe.py:229-234`、`:250-253`）。

**（b）经验证据一（主证）：捕获 logits 的精度** [2026-09-22 记录]

dump 的 `_dsv41_wrap_methods` 不只记返回值，还把方法的**浮点实参**记了下来
（`scripts/dsv41_consistency/dsv41_dump_extension.py`），键名 `AscendFusedTopKRouter._compute_routing@layer<i>#arg0` = hidden_states、`#arg1` = router_logits、`[0]/[1]` = (topk_weights, topk_ids)。

> **`#arg1` 与离线 fp32 `x @ Wᵀ`（x 取 `#arg0`）的差：max = 1.9e-5；该张量 std = 1.13。**

判读（参照量今天补算 [2026-09-24 复算]）：把 std=1.13 的 fp32 张量舍入到 bf16，误差是
**mean ≈ 1.3e-3、max ≈ 1.2e-2**（同尺度随机量实测）。观测到的 1.9e-5 比它**小 600× 以上**
→ `#arg1` 不可能是 bf16；1.9e-5 正是 fp32 GEMM 换一次累加次序（K=5120）该有的差。
**于是 `text_bias.to(router_logits.dtype)` 是恒等变换 → 引擎确实用 ckpt 的 fp32 bias 做 topk。**

- **证伪条件**：若引擎 logits 是 bf16，这里应当看到 ~1e-2 的差，同时（按上面的代码）bias 会被降级 → 本次现象不该存在。
- 这一口径（`router logits: mean|d| / max|d| / rel`）现在由 `scripts/dsv41_consistency/compare_routing.py` 打印（跑时用 `--model-path/--engine-dir/--trainer-dir` 指向 real4 的那套）。

> **(b) 与 (c) 的分工**：(b) 只能证明"**传进 op 的是 fp32 张量**"，op **内部**有没有降精度，
> 要靠 (c) 的 48/48 兜住 —— 两者合起来才是"引擎这次选择 = fp32 复算"。

**（c）经验证据二（辅证）：48/48 配方命中** [2026-09-22 记录]

用 `sqrtsoftplus(x @ Wᵀ) + bias_fp32` 复算引擎那次前向的 top-6 → **48/48 槽位全中（100%）**；
换 sigmoid 只有 37.5%、softmax 0% → 同时把 **score_func 钉死**（切片 config 里 **既没有** `score_func` **也没有** `gate_temp`
→ 走 `ModelArgs` 默认 `"sqrtsoftplus"` / `1.0`，`model.py:77-78`）。
**顺带**：这一层能被"`sqrtsoftplus` + 纠偏 bias"复现出来，本身就说明**它不是什么按 `tid2eid` 查表的 hash 层**
（hash 层没有纠偏 bias，选出来的专家由查表决定，不可能由 bias 复算命中）。

- **强度声明（别读成"证明了 fp32"）**：只有 **8 个 decode token、只捕获到一层**。按该层在 prefill 输入上测得的翻转率
  （0.375–0.615）粗算，"引擎若用 bf16 bias"假设下 8/8 全同的概率约 **0.05%–2.3%** —— 这是方向性排除，不是证明
  （decode 与 prefill 的输入分布不同，翻转率未必可搬）。它真正的价值是 **锚定配方**（score_func/temp）+ 与 (b) 一起定性引擎的算术精度。

**（d）经验证据三（引擎自报）：** [2026-09-22 记录，日志 `/tmp/check_params_real4_v2.log`]

引擎 dump 的参数统计 `dsv41_param_stats` 在**运行中的引擎进程里**逐张量取 dtype 与统计量
（打印于 `dump_engine_stages.py:102`、写进 `params.json`）；672 个参数与 ckpt 的 std/absmax/mean 指纹**全部吻合**：

```
checked 672 engine parameters, 672 match the checkpoint
OK [rank0] language_model.model.layers.0.mlp.gate.e_score_correction_bias
        engine: std=0.035926 absmax=9.884364 shape=[384]
        ckpt  : std=0.035926 absmax=9.884364 shape=[384]  <- layers.0.ffn.gate.bias
```

同一份日志里也能看到"引擎 fp32 的地方就是 ckpt fp32 的地方"这个整体形态（`hc_*`/`attn_sink` 都是 `torch.float32`）：

```
language_model.model.layers.0.hc_attn_base   [24]  torch.float32  std=11.55080 absmax=25.49090
language_model.model.layers.0.self_attn.attn_sink [8] torch.float32 std=0.12605 absmax=0.25219
language_model.model.layers.0.self_attn.wq_a.weight  torch.bfloat16  ...
```

- ⚠️ **这条不能单独用**：指纹比较的 rtol 是 **0.02**（`check_engine_params.py:29`），而 bf16 舍入只动 ~0.2% 的值
  → **指纹分不出 fp32/bf16**。它的作用是排除"**加载错位**"（引擎手里是 ckpt 的值）；
  "引擎手里是 fp32"要靠 dtype 那一列（`params.json`）+ (a)(b)。
  （`params.json` 已随 dump 清理，见 §13.6。）

### 13.3 Q3：训练侧拿的**确实是**舍入值（identification，不是相关）

方法（工具 `verify_gate_bias_fp32.py` 的第 2 项检查）：probe dump 里存着训练侧**自己算出来的** top-6
（`stages["model.layers.<i>.ffn.gate[1]"]`）和它 gate **真正吃到的输入**；用同一输入重算两遍，**只换 bias 的取值**，比命中率。
[2026-09-22 记录，逐层]

| len | layer | fp32 原值命中 | bf16 舍入值命中 |
|---|---|---|---|
| 64 | 0 / 1 / 2 / 3 | 39.1% / 65.6% / 20.3% / 21.9% | **100% / 100% / 100% / 100%** |
| 200 | 0 / 1 / 2 / 3 | 38.0% / 62.0% / 21.5% / 24.0% | **100% / 100% / 100% / 100%** |

为什么这一步是"**识别**"：它问的是"训练侧手里**是哪个值**"。内核差、输入差、代码 bug 都造不出这种
**"100% vs 21–62%"的二分**；而且它把归属**定死在训练侧**（引擎侧对应的观测就是 13.2(c) 的 48/48）。
- **自检**：若两侧都用 fp32 → 两列都该 ~100%；若都用 bf16 → fp32 那列也该 100%。两者都没出现 ✓
- **为什么会这样（机制链）**：`prepare_deepseek_v41_model_for_fsdp` 把所有浮点参数 `.to(bf16)`（`adapter.py:403`），
  而 **FSDP2 硬约束"每个 param group 只能一种 original dtype"**（`_fsdp_param_group.py:238`，见 E3′）
  → 在一个 module 里保留 fp32 的 bias 不是"加一行配置"能解决（修复方案/回归见 `worklog_dsv41_gate_bias_fp32_fix.md`）。

### 13.4 阴性对照与因果闭环（把这些放在一起才叫"定到"）

| 检查 | 结果 | 排除的解释 |
|---|---|---|
| 随机权重（两侧同 bf16、bias 全 0） | 地板 0.0054、**均匀无尾** | 内核差只有 ~0.5%，量级不符 |
| 输入打桩 `in.all` | 残差仍 0.9–4.5% | "输入差被传播放大"这条链 |
| 672/672 参数指纹 | 全 OK | 加载错位 / 加载错栈 |
| `continuous` 组（只还原 `hc_*`+`attn_sink`） | baseline σ 0.2573 → 0.2467、**地板毫无变化** | "随便哪个 fp32 参数都有用"（阴性对照） |
| **只还原 `gate.bias`** | 地板 0.0316–0.0449 → **0.0034–0.0042**、σ 0.1882 → 0.0202 | —— |
| 修复后用**同一工具**再跑 identification | 100% 那一列**从 bf16 翻到 fp32** | 因果闭环（值变了，工具与 dump 结构没变） |

### 13.5 一页速览（这条结论是怎么被一步步钉死的）

1. **现象反推**：钉死输入后仍有非均匀残差 → 参数值不同，不是内核；
2. **候选收窄**：切片审计的 36 个 fp32 张量里，只有 `gate.bias` 进**离散选择**（`(scores+bias).topk`，权重取自无偏 scores）→ 只有它能**离线**判定；
3. **静态事实**：ckpt（bf16 版与 W8A8 版）与两套代码都声明 fp32（13.1）；
4. **引擎侧不能只看声明**：代码里那个 `text_bias.to(router_logits.dtype)` 会让引擎"自己降级"→ 用捕获 logits 的 **1.9e-5**
   把"logits 是 fp32"钉住 → 引擎实际用 fp32 值选择；48/48 复核配方（13.2）；
5. **训练侧 identification**：100% vs 21–62% → 训练侧手里是 bf16 舍入值（13.3）；
6. **阴性对照 + 修复后翻转**：闭环（13.4）。

### 13.6 复现现状：哪些今天还能跑、哪些要重跑

- **CPU、秒级可复现**：13.1(a) 的 bias dtype/值域/bf16 往返误差（切片仍在，命令见上）；
  13.1(d) 的 W8A8 复核（`quant_model_description.json` + shard 00009 实读，命令见上）。
- **需要重跑才能复现**：13.2(b) 的 1.9e-5、13.2(c) 的 48/48、§6.2 的翻转率表、13.3 的 identification
  —— 它们都依赖引擎 dump / probe dump。
  **⚠️ 现状（2026-09-24 查）：`/mnt/share/m00899630/dsv41/dump/` 下的 real4 系列 dump 已被清理**
  （只剩一个空的 `engine/` 目录与一个 NFS 残留句柄；`/tmp/*.log` 的日志还在，本文引用的日志行均可回溯）。
  要复现须重跑 §9 的 `run_dump_engine.sh`（~40–70 min）与 probe（~20 min）；
  13.3 的 identification 只要 probe dump 即可（不必跑引擎）。
