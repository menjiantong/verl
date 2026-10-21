# DSV4.1 countbenchqa · 5/10-step diff_mean 稳定性分析（2026-10-20）

## 0. 用户约束（12 条，逐字记录）

0. 不要存checkpoint，启动调试。容器内存在一个dsv41-mjt-1019的docker，使用这个docker启动。
1. 本轮的目标是训练5个/10个step，diff mean没有明显恶化。始终在0.01水平。
2. 可以使用多台机器分析，执行。
3. 可以使用的方法：比如第5个step恶化了，可以把第5个step的权重拉下来，然后重新启动以第一个step启动，看下结果如何。来确定是不是训练越远，效果越差。当然合理性请你评估。
4. 可以分析下是不是response len导致的，比如看下最后一层的所有token的概率，看下他们是不是如之前所说，少部分token翻转导致的diff mean增大？
5. 可以分析下为什么有token翻转，到底是哪里翻转的，不要看之前的结论，这次重新分析。
6. //mnt/share/m00899630/workspace/dsv41-verl 工作目录位于这里，可以新建wiki/worklog_consistenty_analysis_running_smaller_20261020.md。记录执行命令、分析结果、数据证据、实验结果等。
7. 要有确切的结论，结论必须要有数据、实验支撑，不要猜测。
8. 可以对照实验。
9. 除了上述我说的，你要排查所有最可能的情况，分析原因，实验证明，确定根因。
10. 权重还是//mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram 这个。
11. 尽可能并行操作，提高机器使用效率。

## 1. 环境与入口

- 集群2：80.5.21.101-108，aarch64，每节点 8 卡 × 2 chip = 16 Ascend910。
- 容器：`dsv41-mjt-1019`（image `dsv41-mjt-lite:1015`），host network，`/mnt` 直通。
- 代码：`/mnt/share/m00899630/workspace/dsv41-verl/verl`。
- 权重：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram`。
- 启动脚本：`my_scripts/dsv41_grpo_countbenchqa_1node.sh`。
- 参考 run（126，仍在跑）：`logs/grpo_countbenchqa_steps500_20261020_092815.log`，step 1 0.01181 → step 24 0.02265。

## 2. 基准配置（复刻 126）

`TRAIN_BSZ=16`、`SAMPLE_N=16`、`MAX_PROMPT_LEN=1152`、`MAX_RESPONSE_LEN=256`、`LR=5e-6`、`USE_KL=0`、`ENABLE_THINKING=false`、`GEN_TP=8`、`GEN_DP=1`、`GEN_EP=8`、`FSDP_SIZE=16`、`EP_SIZE=8`、`EFSDP_SIZE=1`、`ROLLOUT_GPU_MEM_UTIL=0.5`、`trainer.save_freq=-1`、`test_freq=-1`。

## 3. 实验矩阵（并行）

| ID | 节点 | 变量 | 目的 |
|---|---|---|---|
| A | .101 | 基准 5 step，dump 每步 | 复现 126 的 0.0118→? 趋势；per-token 分析 |
| B | .102 | `USE_KL=1` | KL 是否压住漂移 |
| C | .103 | `MAX_RESPONSE_LEN=512` | response length 对照 |
| D | .104 | `LR=1e-6` | 学习率/权重漂移对照 |
| E | 待定 | 第5步权重回灌 step1 | 判断“训练越远越差” |
| F | 待定 | 固定 prompt/答案，长度分层 | 分离长度效应与权重效应 |

## 4. 观测指标

每步记录：`diff_mean`、`diff_max`、`pearson`、`response_length/mean`、`entropy`、`clip_ratio`、`reward`、`timing`。
每步 dump：`VERL_DSV41_DUMP_BATCH` → `batch_step*.pt`，字段含 `old_log_probs`、`rollout_log_probs`、`responses`、`response_mask`、`input_ids`。
离线分析：`scripts/dsv41_consistency/analyze_rollout_actor_gap.py` + 自写 per-token/per-position/per-length 分析。

## 5. 执行记录

（下面按时间追加命令、日志、数据）

### 5.1 参考 run（126）24 步数据

| step | entropy | diff_mean | pearson | resp_len | clip_ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.7792 | 0.01181 | 0.99808 | 240.93 | 0.891 |
| 2 | 1.2260 | 0.01278 | 0.99711 | 226.41 | 0.801 |
| 3 | 0.7237 | 0.01442 | 0.99502 | 192.57 | 0.602 |
| 4 | 0.6211 | 0.01690 | 0.99277 | 166.24 | 0.480 |
| 5 | 0.5430 | 0.01574 | 0.99259 | 157.35 | 0.418 |
| 6 | 0.5292 | 0.01775 | 0.99073 | 133.11 | 0.328 |
| 7 | 0.5296 | 0.01989 | 0.98904 | 119.52 | 0.273 |
| 8 | 0.6028 | 0.02136 | 0.98901 | 116.99 | 0.199 |
| 9 | 0.5677 | 0.02421 | 0.98643 | 97.14 | 0.160 |
| 10 | 0.5004 | 0.02302 | 0.98579 | 105.40 | 0.160 |
| 11 | 0.4349 | 0.02089 | 0.98569 | 109.73 | 0.211 |
| 12 | 0.4207 | 0.01996 | 0.98630 | 110.29 | 0.227 |
| 13 | 0.4392 | 0.02156 | 0.98549 | 97.18 | 0.148 |
| 14 | 0.4206 | 0.02124 | 0.98480 | 97.77 | 0.152 |
| 15 | 0.5429 | 0.02533 | 0.98442 | 85.32 | 0.133 |
| 16 | 0.4099 | 0.02119 | 0.98391 | 95.37 | 0.191 |
| 17 | 0.4819 | 0.02367 | 0.98431 | 92.79 | 0.168 |
| 18 | 0.4623 | 0.02538 | 0.98250 | 80.83 | 0.113 |
| 19 | 0.5506 | 0.02556 | 0.98500 | 72.88 | 0.090 |
| 20 | 0.4934 | 0.02408 | 0.98532 | 78.32 | 0.113 |
| 21 | 0.5266 | 0.02417 | 0.98573 | 71.98 | 0.082 |
| 22 | 0.5308 | 0.02220 | 0.98826 | 79.27 | 0.113 |
| 23 | 0.5447 | 0.02391 | 0.98559 | 64.88 | 0.090 |
| 24 | 0.4135 | 0.02265 | 0.98320 | 75.39 | 0.129 |

观察（待实验验证，不先下结论）：
- step1 已经是 0.01181，目标 0.01 在起点就略超；step5 升到 0.01574（+33%）。
- step1 clip_ratio 0.891（89% 回答顶到 256），随后 response 缩到 75；entropy 1.78→0.41。
- `diff_mean` 与 response/entropy 同步变化，但这是跨 step 相关，需 per-token 与受控实验拆开。

### 5.2 当前并行实验（2026-10-20 22:48）

| ID | 节点 | 变量 | 日志前缀 |
|---|---|---|---|
| A | .101 | 基准（复刻 126） | `dbg_A_20261020_224605` |
| B | .102 | `USE_KL=1` | `dbg_B_20261020_224837` |
| C | .103 | `MAX_RESPONSE_LEN=512` | `dbg_C_20261020_224837` |
| D | .104 | `LR=1e-6` | `dbg_D_20261020_224837` |
| E | .105 | `GEN_DP=2 GEN_EP=8`（16 卡 rollout） | `dbg_E_20261020_224837` |
| F | .106 | 纯文本 gsm8k 对照（同模型/拓扑/长度） | `dbg_F_<ts>` |

说明：`countbenchqa_1node.sh` 是 `dsv41_grpo_2node_new.sh single` 的 VLM 专用派生版；126 的
`grpo_countbenchqa_steps500_...` 日志确认由它启动。F 直接用 2node 脚本的 `single` 模式做纯文本对照。

### 5.3 126 观测数据的相关性（仅观测，不作因果结论）

| 变量对 | 全部 24 步 Pearson / Spearman | step 1-5 Pearson / Spearman |
|---|---:|---:|
| diff_mean ↔ response_length | -0.952 / -0.877 | -0.959 / -0.900 |
| diff_mean ↔ clip_ratio | -0.941 / -0.896 | -0.951 / -0.900 |
| diff_mean ↔ entropy | -0.693 / -0.263 | -0.912 / -0.900 |
| diff_mean ↔ step | +0.815 / +0.768 | +0.909 / +0.900 |
| pearson ↔ response_length | +0.942 / +0.772 | +0.997 / +1.000 |

- step1→5：diff_mean +0.00393（0.01181→0.01574），response -83.6，entropy -1.236，clip_ratio -0.473。
- 前 5 步的平均 diff_mean=0.01433；第 6-10 步=0.02125；第 20-24 步=0.02340。
- 这些变量高度共线（step / response / entropy / clip），**不能**用这张表判断谁导致谁；需要 B/C/D/E/F/R 的受控实验。

### 5.4 回灌实验（不存 checkpoint 的实现）

用户要求不存 checkpoint，但又要“把第5步权重拉下来重新当第1步”。这里不用 checkpoint：
在 `verl/trainer/ppo/v1/trainer_base.py::_fetch_one_gen_batch` 加一个 env 门控
`VERL_DSV41_REWIND_AFTER_STEP`：当 `global_steps == N+1` 时把
`train_dataloader_it` 置空，下一次取 batch 从数据集第 1 条重新开始。
- 训练 7 步，`REWIND_AFTER_STEP=5`：step1-5 正常，step6 用 step5 权重 + step1 的 batch，
  step7 用 step5 权重 + step2 的 batch。
- 对比 step1/step6（同数据、不同权重）与 step5/step6（同权重、不同数据），
  直接拆分“权重漂移”与“数据/batch 组成”。
- 不写任何 checkpoint；用 dump 的 `input_ids` 验证 step6 的 prompt 与 step1 一致。
- 已在 .107 启动 `dbg_R_20261020_225138`（7 步）。

### 0.1 追加约束（2026-10-20 23:05）

12. 如果训推一致搞定，指标表现好，可以持续运行，不要停止。
- 一旦找到稳定配置（5/10 步 diff_mean 不恶化），直接启动长跑（如 500 步，`save_freq=-1`），保持运行并巡检。

### 5.5 第一批 per-token 数据（step 1，A=VLM 基准 / F=纯文本对照）

| 指标 | A（countbenchqa VLM） | F（gsm8k 纯文本） |
|---|---:|---:|
| tokens | 62,167 | 64,744 |
| sequences | 255 | 256 |
| diff_mean | 0.01144 | 0.00855 |
| pearson | 0.99809 | 0.99891 |
| response mean | 243.8 | 252.9 |
| 顶到 max=256 | 230/255 (90.2%) | 248/256 (96.9%) |
| `drop1_mean`（每序列去掉最大1个 token） | 0.01109（-3.1%） | 0.00809（-5.4%） |
| `big>=0.5` tokens | 2（0.03/1k） | 0（0/1k） |
| corr(seq_len, diff_mean) | -0.448 | -0.041 |
| corr(seq_len, diff_sum) | +0.290 | +0.108 |

**位置分层（response 归一化十分位，diff_mean）**

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.02230 | 0.01904 | 0.01486 | 0.01191 | 0.01023 | 0.00936 | 0.00766 | 0.00690 | 0.00614 | 0.00602 |
| F | 0.01528 | 0.01117 | 0.00970 | 0.00837 | 0.00807 | 0.00781 | 0.00698 | 0.00655 | 0.00579 | 0.00578 |

**前缀分析（同样本，只取前 L 个 response token）**

| L | A diff_mean | A sum/seq | F diff_mean | F sum/seq |
|---:|---:|---:|---:|---:|
| 8 | 0.02045 | 0.1636 | 0.01766 | 0.1413 |
| 16 | 0.02185 | 0.3496 | 0.01670 | 0.2672 |
| 32 | 0.02197 | 0.7029 | 0.01464 | 0.4684 |
| 64 | 0.01956 | 1.2521 | 0.01264 | 0.8089 |
| 128 | 0.01565 | 2.0038 | 0.01054 | 1.3494 |
| 256 | 0.01112 | 2.8468 | 0.00847 | 2.1679 |

**结论（step1，数据支撑）**
1. **不支持“少数 token 翻转主导”**：去掉每序列最大 1 个 token 只降 3–5%；`|Δp|>=0.5` 的 token 只有 0–2 个（0–0.03/1k）。diff 是很多 token 的中小误差累积。
2. **误差强烈依赖 response 位置**：前 10% token 的 diff 是后 10% 的 3–4 倍；前 8/16 token 的 diff 约为全序列的 1.8–2.2 倍。早期 token 熵高、概率敏感，后期 token 变确定后 diff 小。
3. **response 长度是分母/组成效应**：同一批样本里，长序列的 `diff_mean` 更低（A r=-0.448），但 `diff_sum` 更高（r=+0.290）；长回答增加了低误差的尾部 token，把均值拉低。
4. **VLM 比纯文本高 ~0.0029**（0.01144 vs 0.00855），且差值在靠前位置最大（decile1 0.0223 vs 0.0153），与图像上下文/视觉编码差异一致；但 text 对照也有同样的“前高后低”，所以这不是 VLM 独有。
5. 以上只是 step1；需要 step5 和 rewind 判断“训练变远”是权重漂移还是 batch/长度组成。

### 5.6 step0→step1：总误差稳定，diff_mean 上升来自 response 缩短

| 指标 | A step0 | A step1 | Δ | F step0 | F step1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| diff_mean | 0.01144 | 0.01199 | +4.8% | 0.00855 | 0.00874 | +2.2% |
| response mean | 243.8 | 225.7 | -18.1 | 252.9 | 250.0 | -2.9 |
| diff_sum_total | 711.07 | 693.01 | **-2.5%** | 553.74 | 559.28 | +1.0% |
| seq_diff_sum_mean | 2.7885 | 2.7071 | **-2.9%** | 2.1631 | 2.1847 | +1.0% |
| prefix L=8 | 0.02045 | 0.02470 | +20.8% | 0.01766 | 0.01686 | -4.5% |
| prefix L=256 | 0.01112 | 0.01072 | -3.6% | 0.00847 | 0.00871 | +2.8% |
| pearson | 0.99809 | 0.99758 | -0.0005 | 0.99891 | 0.99890 | ~0 |

- A 的**总误差在下降**（711→693），但 response 从 243.8 缩到 225.7，分母变小，diff_mean 反而 +4.8%。
- 同一批 A 样本的前 8 token diff 上升（0.0205→0.0247），说明早期 token 的**局部**差异确实略有增加；但全序列/前缀 256 的 diff 没有增加，均值上升主要是长度组成。
- F（文本）response 基本不变（252.9→250.0），diff_mean 基本不变（0.00855→0.00874），验证同一结论：**长度稳定，指标就稳定**。
- `drop1_mean` 在 A step1 反而大于 `diff_mean`（0.01241 vs 0.01199），因为去掉一个 token 后分母变小；这再次说明“单 token 翻转主导”不成立。

### 5.7 其他 step0 对照（全部同配置，仅变量不同）

| 实验 | 变量 | diff_mean | response mean | 结论 |
|---|---|---:|---:|---|
| A | 基准 VLM | 0.01144 | 243.8 | 参考 |
| B | `USE_KL=1` | 0.01117 | 243.9 | step0 无差异（KL 只影响更新） |
| C | `MAX_RESPONSE_LEN=512` | **0.00790** | **455.6** | 长度翻倍 → diff 降到 0.008 以下；L=256 前缀仍 0.01074，证明是稀释 |
| D | `LR=1e-6` | 0.01141 | 246.7 | step0 无差异 |
| G | `logprobs_mode=raw_logprobs` | 0.01145 | 247.3 | 与 A 相同（0.01144），排除 processed/raw logprobs 口径问题 |
| F | 纯文本 gsm8k | 0.00855 | 252.9 | 文本比 VLM 低 ~0.0029；但同样“前高后低” |

- C 的 `prefix L=256=0.01074` 与 A 的 `0.01112` 几乎一样，但 C 的全序列 diff=0.00790：**256-512 的尾部 token 把均值稀释了**。
- 因此“diff_mean 始终 0.01”可以通过**保持 response 长度**实现；不需要（也不可能）把每个 token 的核差降到 0。
- 仍需验证：训练 5 步后 response 是否还能保持长；KL（B/H）和低 LR（D）是否能把长度/熵稳住。

### 5.8 step1（第二步）各配置对比

| 实验 | 变量 | step1 diff_mean | step1 resp_mean | step0→step1 |
|---|---|---:|---:|---|
| A | 基准 VLM | 0.01199 | 225.7 | 0.01144→0.01199（长度 243.8→225.7） |
| B | `USE_KL=1` | 0.01055 | 247.8 | 0.01117→0.01055（长度 243.9→247.8） |
| D | `LR=1e-6` | 0.01104 | 243.4 | 0.01141→0.01104（长度 246.7→243.4） |
| R | 基准（rewind 前） | 0.01007 | 241.7 | 0.01109→0.01007（长度 245.7→241.7） |
| F | 纯文本 | 0.00874 | 250.0 | 0.00855→0.00874 |

- 同一配置 A/R 的 step1 相差 0.0019，主要伴随 response 225.7 vs 241.7；跨节点/跨 batch 有 ~0.001–0.002 波动，**看趋势必须用同一 run 多步**。
- KL（B）和低 LR（D）在 step1 都没有恶化，KL 的 response 反而变长；与“长度是主因”一致。
- A 的 step1 response 明显缩短（-18 token），diff 上升；R/B/D response 基本不变，diff 不升。这是目前最强的“长度→diff_mean”证据链。

### 0.2 长跑约束（2026-10-20 23:44）

13. 长跑时使用 `wiki/worklog_dsv41_2nodes_countbenchqa_rl_1019.md` 里的命令，使用 countbenchqa 数据集；后续也逐渐使用这个数据集。
- 该 worklog 的最终 chat 配置：`dsv41_grpo_countbenchqa_1node.sh <steps>`，countbenchqa_lite 450/50，
  `TRAIN_BSZ=8`、`SAMPLE_N=8`、`MAX_RESPONSE_LEN=512`、`ROLLOUT_GPU_MEM_UTIL=0.6`、
  `ENABLE_THINKING=false`、`USE_KL=0`、`LR=5e-6`、`save_freq=-1`。
- 长跑一旦启动，保持运行；若指标恶化，先记录数据再调整，不要无故停止。

### 5.9 step0→step2 趋势（同一 run）

| 实验 | step0 diff / resp | step1 diff / resp | step2 diff / resp | 观察 |
|---|---:|---:|---:|---|
| A 基准 VLM | 0.01144 / 243.8 | 0.01199 / 225.7 | **0.01272 / 197.7** | 长度 243.8→197.7，diff 单调升；diff_sum 711→693→644 |
| B `USE_KL=1` | 0.01117 / 243.9 | 0.01055 / 247.8 | **0.00982 / 240.2** | 长度稳，diff 降到 0.01 以下 |
| D `LR=1e-6` | 0.01141 / 246.7 | 0.01104 / 243.4 | **0.01099 / 243.1** | 长度稳，diff 稳在 0.011，不降 |
| F 纯文本 | 0.00855 / 252.9 | 0.00874 / 250.0 | **0.00842 / 252.2**（step3 0.00877/247.1） | 文本稳定 <0.01 |
| R 基准 | 0.01109 / 245.7 | 0.01007 / 241.7 | **0.00958 / 224.8** | 同配置跨节点波动；rewind 前 |
| H 512+KL | **0.00803 / 471.2** | 待 | 待 | L=256 前缀 0.01122，全长被 256-512 尾部稀释 |

**结论（到目前为止）**
- **KL 是当前最有效的稳定器**：B 在 response 240 左右把 diff 从 0.0112 降到 0.0098；A 同长度段升到 0.0127。
- **低 LR 只能“防止恶化”**：D 稳定在 0.011，但降不到 0.01。
- **纯文本天然 <0.01**：F 0.0084–0.0088，response 稳定 250。
- **H（512+KL）step0 0.00803**：长度 471 的稀释 + KL；需要 step1-5 验证。
- A 的 `diff_sum_total` 随 step 下降（711→693→644），但 `diff_mean` 上升，再次证明**长度/组成是主因**，不是每 token 误差变大。

### 5.10 同 fixture 的 trainer vs engine 分解（text，311 token）

| 对比 | dlogp_abs_mean | dp_mean | dp_median | dp_p99 |
|---|---:|---:|---:|---:|
| trainer vs engine-prefill | 0.006184 | **0.003081** | 0.000081 | 0.0799 |
| trainer vs engine-decode | 0.006980 | **0.003579** | 0.000080 | 0.0814 |
| engine-prefill vs decode | 0.005776 | **0.002737** | 0.000082 | 0.0802 |

- 在同一 text fixture 上，三种差异都在 **0.003 量级**；而实际 VLM run 的 diff_mean 是 0.0114。
- 因此：**engine decode vs prefill 只占 ~1/4**；trainer vs engine prefill 也是 ~0.003；
  更大的差距来自**长 prompt / padded batch / VLM 图像路径 / 高熵 token 组成**，不是单一 kernel。
- 下一步：`--padding-probe` 和长 prompt 对照，定位 padding/length 对 trainer logprob 的影响。

### 5.11 长跑启动

- `long_W`（.104，2026-10-20 23:45）：严格按 worklog 的 countbenchqa 命令：
  `dsv41_grpo_countbenchqa_1node.sh 500`，`TRAIN_BSZ=8 SAMPLE_N=8`、`MAX_RESPONSE_LEN=512`、
  `ROLLOUT_GPU_MEM_UTIL=0.6`、`ENABLE_THINKING=false`、`USE_KL=0`、DP2/EP16、`save_freq=-1`。
  日志 `logs/long_W_<ts>.nohup.log`；dump `logs/dump_W_<ts>`。
- 随后在 .105 启动 `long_H`（512+KL、bsz8×8、DP1/EP8）作为稳定性对照；两长跑都保持运行。

---

## 2026-10-21 00:05 阶段性修正：padding 路径排除 + 逐 token 误差结构

### 1. 重要修正：生产训练路径并没有 left-pad 到 1152

之前 `PADDING-PROBE` 的注释假设“trainer 会 left-pad prompt 到 rollout_config.prompt_length=1152”，本轮重新核对代码后确认这个假设**不适用于当前 v1 TaskQueue 训练路径**：

- `verl/trainer/ppo/v1/agent_loop_tq.py::_agent_loop_postprocess` 直接把单条样本写成
  `input_ids = torch.cat([prompts, responses])`，`attention_mask = torch.ones_like(input_ids)`，
  即**不 padding**（prompt 与 response 紧邻）。
- `verl/verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs` 的
  `use_remove_padding=False` 分支从 nested tensor 取 `seq_len_effective = input_ids.offsets().diff()`，
  `max_seq_len = max(seq_len_effective)`，再 `to_padded_tensor(..., output_size=(bsz, max_seq_len))`。
  也就是**只在 micro-batch 内 right-pad 到最长样本**。
- 生产配置 `ppo_micro_batch_size_per_gpu=1`，因此每个 micro-batch 只有 1 条序列，
  `max_seq_len == 该序列真实长度`，**没有 padding，也没有位置偏移**。
- `verl/experimental/agent_loop/agent_loop.py` 里的 left-pad 到 `rollout_config.prompt_length`
  是旧路径；当前 `trainer.v1` + TransferQueue 不走它。

因此：
- `PADDING-PROBE left pad=64/128` 观测到的 dlogp mean=-0.2032/+0.0504、med|d|=0.0012/0.0016、
  p99=4.8944/5.1456、|d|>0.5=15.2%/18.1%，只说明“模型对 left-pad + RoPE 位置偏移敏感”，
  **不是本轮生产 diff_mean=0.011 的根因**。该假设排除。
- `pad=256` 探针崩溃是因为启动参数 `--max-seq-len 512`，311+256=567 超出 rope 表：
  `RuntimeError: Attempting to broadcast a dimension of length 567 ... expected [1,512,64,32]`；
  与训练本身无关。

### 2. 逐 token 事实（dump 数据）

数据目录：
`/mnt/share/m00899630/workspace/dsv41-verl/verl/logs/dump_{A,B,C,F,R,H}_*`

分析命令：
```
python3 scripts/dsv41_consistency/analyze_length_tokens.py --dump-dir logs/dump_XXX --tokenizer /mnt/share/m00899630/dsv41/rl_tokenizer
python3 scripts/dsv41_consistency/deep_dive_diff.py logs/dump_A_20261020_224605 logs/dump_B_20261020_224837 \
  logs/dump_C_20261020_224837 logs/dump_F_20261020_224935 logs/dump_R_20261020_225138 logs/dump_H_20261020_231616
```

**口径说明**：`batch_stepN.pt` 是第 N 次 rollout 的数据（用于第 N+1 次 update），
所以 `dump step N` 与日志 `training/global_step=N+1` 是同一批数据。下表的 `diff_mean/dlogp`
来自 dump，`resp/prompt/entropy` 来自对应的日志 step（N+1）。

| run | dump step | log step | tokens | diff_mean | dlogp_mean | dlogp_med | dlogp_std | frac|dlogp|>0.1 | cross_half | top1% abs dp | resp_len | prompt_len | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 1 | 62170 | 0.011439 | -0.010138 | -0.000023 | 0.1404 | 21.9% | 804 | 13.0% | 242.9 | 266.75 | 1.6011 |
| A | 1 | 2 | 57787 | 0.011993 | -0.008987 | -0.000022 | 0.1324 | 19.2% | 790 | 13.7% | 225.7 | 342.31 | 1.2093 |
| A | 2 | 3 | 50605 | 0.012717 | -0.009608 | -0.000012 | 0.1401 | 18.7% | 720 | 14.3% | 197.7 | 279.31 | 1.0587 |
| A | 3 | 4 | 43179 | 0.014514 | -0.010712 | -0.000015 | 0.1513 | 19.2% | 705 | 14.1% | 168.7 | 272.75 | 0.9288 |
| B(KL=1) | 0 | 1 | 62449 | 0.011173 | -0.010029 | -0.000025 | 0.1415 | 22.8% | 777 | 13.1% | 243.9 | 266.75 | 1.6483 |
| B | 1 | 2 | 63424 | 0.010547 | -0.009003 | -0.000014 | 0.1378 | 21.1% | 699 | 13.7% | 247.8 | 342.31 | 1.6395 |
| B | 2 | 3 | 61488 | 0.009818 | -0.007449 | -0.000017 | 0.1301 | 20.4% | 687 | 14.1% | 240.2 | 279.31 | 1.6780 |
| B | 3 | 4(待出) | 59552 | 0.010854 | -0.008778 | +0.000000 | 0.1390 | 22.0% | 742 | 13.2% | 待补 | 待补 | 待补 |
| C(512) | 0 | 1 | 116625 | 0.007899 | -0.007762 | -0.000004 | 0.1321 | 17.2% | 947 | 16.4% | 455.6 | 266.75 | 1.5237 |
| C | 1 | 2 | 102300 | 0.008387 | -0.005988 | -0.000004 | 0.1151 | 12.8% | 944 | 17.7% | 399.6 | 342.31 | 0.8762 |
| C | 2 | 3(待出) | 78570 | 0.010330 | -0.007582 | -0.000003 | 0.1292 | 13.0% | 861 | 17.7% | 306.9 | 待补 | 待补 |
| H(512+KL,bsz16) | 0 | 1 | 120634 | 0.008035 | -0.006546 | -0.000006 | 0.1198 | 16.9% | 1027 | 16.0% | 471.2 | 266.75 | 1.5230 |
| H | 1 | 2(待出) | 108340 | 0.008037 | -0.006399 | -0.000004 | 0.1120 | 11.5% | 880 | 19.7% | 423.2 | 待补 | 待补 |
| F(text gsm8k) | 0 | 1 | 64744 | 0.008553 | -0.008140 | -0.000002 | 0.1323 | 23.3% | 564 | 14.2% | 252.9 | 116.06 | 2.4415 |
| F | 1 | 2 | 63999 | 0.008739 | -0.008540 | -0.000021 | 0.1275 | 21.4% | 549 | 14.0% | 250.0 | 119.25 | 2.1699 |
| F | 4 | 5 | 62732 | 0.008631 | -0.007534 | -0.000037 | 0.1179 | 20.5% | 577 | 13.1% | 245.0 | 109.50 | 2.1621 |
| R(rewind) | 0 | 1 | 62889 | 0.011094 | -0.010889 | -0.000028 | 0.1429 | 22.3% | 761 | 13.1% | 245.7 | 266.75 | 1.6311 |
| R | 1 | 2 | 61863 | 0.010071 | -0.008478 | -0.000021 | 0.1315 | 22.4% | 683 | 13.6% | 241.7 | 342.31 | 1.9105 |
| R | 2 | 3 | 57547 | 0.009581 | -0.007664 | -0.000016 | 0.1244 | 21.1% | 647 | 13.3% | 224.8 | 279.31 | 1.9082 |
| R | 3 | 4 | 55035 | 0.010946 | -0.008866 | -0.000037 | 0.1305 | 20.9% | 652 | 13.7% | 215.0 | 272.75 | 1.5966 |

逐 token 结构：
- **不是少数几个翻转决定 mean**：硬翻转（一侧概率 < 另一侧的 10%，且较大侧 ≥1%）只有 0~7 个 token / step（0.00~0.02%），对 dp_sum 贡献 0.00~0.90%。
- **概率跨越 0.5 的 token 约 500~1000 个/step（1~2%）**；绝对误差 top 1% 只占 13~20%，top 10% 占 55~71%。误差是“宽尾 + 少量大翻转”，不是“少数 token 翻转”。
  注：此前“top 1% 占 39.8%”是**平方误差**口径；绝对误差口径是 13~20%，两者不矛盾。
- `dlogp` 中位数≈0，均值 -0.006~-0.011，标准差 0.11~0.15；**左尾远重于右尾**（p01≈-0.4~-0.56，p99≈+0.32~+0.42）。trainer 对 sampled token 的概率存在轻微系统性低估，但主要来自尾部。
- 位置 decile（A dump step0）：dp 0.0223→0.0060，p_roll 0.389→0.719；
  A dump step3：0.0266→0.0089，p_roll 0.518→0.830。**误差强烈集中在前部低置信 token**。
- per-sequence `corr(seq_len, diff_mean)` 在 -0.45~-0.86；`corr(p_roll_mean, diff_mean)` 在 -0.13~-0.67。短 response / 低 p_roll 的序列 diff_mean 更高。

### 3. 固定前缀对比：response 512 的“0.008”主要是长度稀释

`analyze_length_tokens.py` 的 prefix 指标（只统计每条序列前 L 个 response token）：

| run / dump step (log step) | full diff_mean | prefix L=128 | prefix L=192 | prefix L=256 | resp_mean |
|---|---:|---:|---:|---:|---:|
| B step0 (log1) | 0.01117 | 0.01537 | 0.01258 | 0.01094 | 243.9 |
| B step1 (log2) | 0.01055 | 0.01418 | 0.01187 | 0.01033 | 247.8 |
| B step2 (log3) | 0.00982 | 0.01329 | 0.01086 | 0.00952 | 240.2 |
| C step0 (log1) | 0.00790 | 0.01505 | 0.01250 | 0.01074 | 455.6 |
| C step1 (log2) | 0.00839 | 0.01661 | 0.01352 | 0.01134 | 399.6 |
| C step2 (log3) | 0.01033 | 0.01927 | 0.01512 | 0.01235 | 306.9 |
| H step0 (log1) | 0.00803 | 0.01547 | 0.01281 | 0.01122 | 471.2 |
| H step1 (log2) | 0.00804 | 0.01778 | 0.01395 | 0.01143 | 423.2 |
| F step1 (log2) | 0.00874 | 0.01113 | 0.00972 | 0.00871 | 250.0 |
| F step4 (log5) | 0.00863 | 0.01049 | 0.00933 | 0.00847 | 245.0 |
| R step2 (log3) | 0.00958 | 0.01235 | 0.01030 | 0.00903 | 224.8 |
| R step3 (log4) | 0.01095 | 0.01401 | 0.01192 | 0.01015 | 215.0 |

结论：
- C/H 的 full diff_mean≈0.008，是因为 response 长达 400~470，后段大量低误差 token 把均值拉低；
  **同一批数据在 prefix L=256 上仍是 0.0107~0.0114，与 A/B 同量级**。
- C/H 的 prefix L=256 随 step **上升**（C 0.01074→0.01134→0.01235，H 0.01122→0.01143），
  说明训练并没有降低“前 256 token 的训推差异”，只是 response 变长把 full mean 稀释了。
- B(KL=1) 是唯一一个 prefix L=256 随 step 下降的配置（0.01094→0.01033→0.00952），
  且 full diff 同步下降（log1 0.01136 → log2 0.01073 → log3 0.01001）；KL 是真正降低逐 token 差异的。
- R 在 log4 再次上升到 0.01113（prefix 0.01015），A log4 上升到 0.01473；
  “训练越远越差”在 KL0 配置上成立，且与 entropy/response 收缩同步。

### 4. 误差来源定位：text fixture 与生产 forward 的差异

固定 311-token 文本 fixture（`trainer_dp_20261020_233350/trainer_dp_len311.pt` 对
`decode_prefill_20261020_233350.json`）：

- trainer-prefill（fixture 的 forward **带 all-ones attention_mask、无 autocast**）：
  dlogp mean=+0.001831，median=-0.000021，std=0.029513，dp_mean=0.003081；
  只有第 0 个 decile（前 ~25 token）dlogp_mean=+0.01870，其余 decile 均值 ~0（±0.0001）。
- trainer-decode：dlogp mean=+0.000938，dp_mean=0.003579。
- engine prefill-decode：dlogp mean=-0.000894，std=0.030814，dp_mean=0.002737。
- 因此 text fixture 的 trainer-vs-engine 本征底噪约 **dp 0.003**；生产实际 text F 是 **0.0086**，
  VLM A 是 **0.0114**，均为 fixture 的 3~4 倍，说明生产 `forward_step` 里还有额外数值差异，
  不是 engine decode-vs-prefill 本身（后者仅 0.0009 均值 / 0.0027 dp）。

已核对的候选：
- `use_fused_kernels=False`（A 日志），LM head `ParallelHead` 用 fp32 计算；logprob kernel 在 fp32 logits 上
  `npu_cross_entropy` 与 `log_softmax` 完全一致（mean/std/med/max 全 0）。
- 生产 `forward_step` 在 `torch.autocast(device_type=npu, dtype=bf16)` 下执行
  （`transformer_impl.py:1588-1594`，`_autocast_dtype=bf16`）；而 fixture 的 forward **没有 autocast**。
  这是目前最大的未验证差异，已准备 `AUTOCAST-PROBE`：
  同一 311-token、同一 all-ones mask，比较 `torch.autocast(npu, bf16)` 与 plain forward 的 logprob。
- `MASK-PROBE`（all-ones mask vs none）正在 .105 运行；fixture 本身已经带 all-ones mask，
  所以 mask 预计不是生产额外差异的来源，但用实验确认。
- 生产 micro-batch=1、nested→right-pad 到自身长度，**无 padding**；left-pad 假设已排除。
- 训练模式 / buffer：`forward_backbone` 在 `start_pos==0` 调 `shared_attn.reset()`；
  training 路径的 window/compress KV 不走持久 cache（`model.py:995-1075`），fixture 里手动 zero cache
  只为 eval 路径；生产 old_log_prob 是 forward_only/eval，但 micro-batch=1 且每条都 `start_pos=0`，
  暂未发现跨样本 cache 污染证据。

### 5. 当前实验与长跑

- `A(KL0,256)`：log1~4 diff = 0.01162 / 0.01221 / 0.01295 / 0.01473；response = 242.9 / 225.7 / 197.7 / 168.7；
  entropy = 1.601 / 1.209 / 1.059 / 0.929。**明确随训练恶化**。
- `B(KL=1,256)`：log1~3 diff = 0.01136 / 0.01073 / 0.01001；response ~240-248；entropy ~1.64-1.68。
  目前最稳定、唯一逐 prefix 下降的配置。
- `C(512,KL0)`：log1~2 diff = 0.00813 / 0.00867；response 455.6 / 399.6；entropy 1.524 / 0.876。
  full mean 低是长度稀释，prefix256 反而 0.01074→0.01134。
- `H(512+KL1,bsz16)`：log1 diff 0.00827，response 471.2；prefix256 0.01122；后续待出。
- `R(rewind)`：log1~4 diff = 0.01128 / 0.01024 / 0.00975 / 0.01113；response 245.7 / 241.7 / 224.8 / 215.0；
  step6 的 rewind（step5 权重 + step1 数据）待执行。
- `F(text gsm8k 对照)`：log1~5 diff = 0.00869 / 0.00890 / 0.00862 / 0.00894 / 0.00877；
  response ~245-253；entropy 2.44→2.16→1.69→1.92→2.16。**text 对照稳定在 0.0086-0.0089**，
  不随 step 恶化；VLM 的额外 ~0.003 与 image/prompt 相关，待真实 VLM fixture 定位。
- 长跑 W（.104，countbenchqa，`TRAIN_BSZ=8 SAMPLE_N=8 MAX_RESPONSE_LEN=512 ROLLOUT_GPU_MEM_UTIL=0.6 USE_KL=0 LR=5e-6`，
  `save_freq=-1 test_freq=-1 total_steps=500`）：
  2026-10-20 23:54:19 `all initialize finished, ready to fit`，第一步生成中；日志
  `logs/long_W_20261020_234631.nohup.log`。

### 6. 下一步（按优先级）

1. 读取 `MASK-PROBE` 与 `AUTOCAST-PROBE` 结果；若 autocast 使 dlogp std 从 0.03 升到 ~0.13，
   则生产 forward 的 autocast 是主要额外差异，做“old_log_prob 关闭 autocast / head 禁 autocast”的 1-step A/B。
2. 用 `B(KL=1)` 的 prefix 曲线确认真实一致性改善；评估 KL=1 作为长跑配置（当前 W 是 KL0）。
3. R step6 rewind 结果：同数据下 step5 权重 vs step1 权重，分离“权重漂移”与“数据/长度组成”。
4. 若 autocast 不能解释，继续做 **真实 VLM fixture**（同 prompt+image+response）的 trainer/engine
   stage-by-stage 对比，定位第一层发散点；以及 image vs text-only 同 prompt 长度对照。
5. W 第一步出数后，若 full diff≈0.008 且 5~10 step 不恶化，按用户要求保持长跑不停止；
   若 KL0 随 entropy 崩塌恶化，切到 KL=1/512 的稳定配置长跑。

### 7. 2026-10-21 00:12 追加：A 的 prefix 平坦，full mean 上升来自 response 缩短；mask/right-pad 排除

新增数据（同一 `analyze_length_tokens.py` 口径，`dump step N` 对应 `log step N+1`）：

A（KL0,256）prefix L=256 与 full mean：
| dump step | log step | full diff | prefix256 | resp_mean |
|---|---:|---:|---:|---:|
| 0 | 1 | 0.01144 | 0.01112 | 243.8 |
| 1 | 2 | 0.01199 | 0.01072 | 225.7 |
| 2 | 3 | 0.01272 | 0.01048 | 197.7 |
| 3 | 4 | 0.01451 | 0.01118 | 168.7 |
| 4 | 5 | 0.01447 | 0.01066 | 151.9 |

- A 的 full diff 从 0.01144 升到 0.01447（+27%），但 **prefix L=256 始终在 0.01048~0.01118**；
  response 从 243.8 降到 151.9（-38%）。因此 A 的“训练越远 diff 越大”主要是
  **response 变短导致 full mean 被前部高误差 token 主导**，不是前 256 token 的逐 token 一致性变差。
- 这直接回答“是不是 response len 导致的”：**是，full diff_mean 对 response length 高度敏感**；
  prefix 指标才是与长度无关的训推一致性。

B（KL=1,256）：
| dump step | log step | full diff | prefix256 | resp_mean | log entropy |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 0.01117 | 0.01094 | 243.9 | 1.6483 |
| 1 | 2 | 0.01055 | 0.01033 | 247.8 | 1.6395 |
| 2 | 3 | 0.00982 | 0.00952 | 240.2 | 1.6780 |
| 3 | 4 | 0.01085 | 0.01047 | 232.6 | 1.6766 |

- B 在 log3 曾降到 0.00982/prefix 0.00952，但 log4 又回到 0.01103/prefix 0.01047；
  KL=1 能防止 entropy/response 崩塌，使 full mean 稳定在 ~0.010-0.011，但不是单调降到 0.01 以下。
- 因此当前所有配置的**逐 token 本征底噪约 0.010-0.011（prefix256）**；512 配置的 0.008 full mean
  是长 response 稀释，不是底噪真的更低。

Mask / right-pad 探针（`trainer_mask_20261021_mask.log`）：
```
MASK-PROBE len=311 all-ones mask vs none: dlogp mean=+0.0000 std=0.0000 med|d|=0.0000 p99=0.0000 max=0.0000 |d|>0.5=0.0% dp mean=0.00000
PADDING-PROBE len=311 side=right pad=64:  dlogp mean=+0.0084 std=0.1194 med|d|=0.0001 p99=0.4101 max=1.1377 |d|>0.5=1.0% dp mean=0.00138 med=0.00005 p99=0.0306
PADDING-PROBE len=311 side=right pad=128: dlogp mean=-0.0040 std=0.0860 med|d|=0.0000 p99=0.3733 max=0.7520 |d|>0.5=0.6% dp mean=0.00118 med=0.00003 p99=0.0199
```
- **all-ones attention_mask 与 none 完全 bit-identical**（所有统计量为 0），mask 路径排除。
- right-pad 64/128 的 dp mean 仅 0.0012-0.0014，中位数 3e-5~5e-5，p99 0.02-0.03，
  |dlogp|>0.5 仅 0.6-1.0%；即使 micro-batch>1 有 right padding，其贡献也只有 ~0.001，
  不足以解释 0.010-0.011 的底噪。生产 micro-batch=1，实际无 padding。

长跑 W 的拓扑问题（数据证据）：
- W（`GEN_DP=2 GEN_EP=16`）23:54:19 `ready to fit`，到 00:01 仍无 step；
  `ray::WorkerDict.actor_rollout_update_weights` 进程状态 `S/ep_poll`，NPU 无进程，13 分钟无进展。
- 之前 E2（同样 DP2/EP16）也在 full expert export 后卡住；126 基线/A/B/C/F/R/H 均为 DP1/EP8，正常。
- 00:03:07 已 kill 并重启 .104 容器，改用 **GEN_DP=1 GEN_EP=8** 重新启动 W2：
  `logs/long_W2_20261021_000307.nohup.log`，配置仍为 countbenchqa / bsz8×8 / response512 / util0.6 / KL0 / 500 steps / save_freq=-1。
  这是“单机 16 卡 + 2 个 DP engine”在 vLLM-Ascend 上的 hang，不是训练逻辑问题；长跑用 DP1/EP8。

已启动 P（.106，1 step，A 同配置）并在 `VERL_DSV41_DUMP_BATCH` 中新增
`prompts/position_ids/token_types`（含 nested offsets），用于把 per-sequence diff 与
prompt length / image token span 关联；等 P 出 dump 后分析。

### 8. 2026-10-21 00:18 AUTOCAST-PROBE：autocast 不是主因，但会引入负偏置与重尾

`trainer_autocast_20261021_autocast.log`（同一 311-token fixture，同一 all-ones mask）：

```
AUTOCAST-PROBE len=311 bf16 autocast vs plain:
  dlogp mean=+0.0139 std=0.2411 med|d|=0.0001 p99=1.1898 max=1.9715 |d|>0.5=6.1%
  dp mean=0.00624 med=0.00012 p99=0.1039
```
注意这里 `dlogp = plain - autocast`，即 **autocast 的 logprob 比 plain 低约 0.014**，方向与生产
`trainer - rollout` 的负均值一致。

但把 autocast 的结果直接与 engine prefill/decode 对比（同一 input_ids）：

| 对比 | dlogp mean | dlogp std | dp mean | dp med | dp p99 |
|---|---:|---:|---:|---:|---:|
| plain trainer - engine prefill | +0.001831 | 0.029513 | 0.003081 | 0.000082 | 0.0799 |
| autocast trainer - engine prefill | -0.004716 | 0.036341 | 0.003252 | 0.000096 | 0.0824 |
| autocast trainer - engine decode | -0.005610 | 0.040254 | 0.003236 | 0.000119 | 0.0815 |
| engine prefill - decode | -0.000894 | 0.030814 | 0.002737 | 0.000084 | 0.0802 |

结论：
- autocast 只把 fixture 的 **dp mean 从 0.00308 提到 0.00325**，几乎不改变与 engine 的概率空间距离；
  它主要造成 **-0.0047 的均值偏移**和 6.1% 的 |dlogp|>0.5 重尾。
- 因此 autocast **不是**生产 F 0.0086 / VLM A 0.0114 相对 fixture 0.003 的主因；
  生产额外差异还必须从 **真实采样 token 的分布（低概率 token 放大数值噪声）/ batch 内 engine decode /
  VLM image path / prompt length** 中找。
- 不启动“全局关闭 autocast”的 A/B，避免改变训练精度而无收益；保留 autocast 作为已知的
  负偏置/重尾来源之一记录。

下一步改为 **T=1 采样的 text fixture**：用 `decode_prefill_check.py` 以 temperature=1.0 生成 256 token，
得到与生产同样“低置信 token”的序列，再跑 trainer fixture；如果 dp 从 0.003 升到 ~0.008，
说明生产 text F 的 0.0086 主要是 token 分布放大、不是新增 forward bug；如果仍 ~0.003，
则继续查 batch/engine decode/VLM 路径。

### 9. 2026-10-21 00:25 关键发现：engine 自身 decode-vs-prefill 在 T=1 采样 token 上就是 ~0.012

此前 311-token fixture 用 **temperature=0.0（greedy，高概率 token）**，engine prefill-decode 只有
`dp_mean=0.002737`。本轮把 `decode_prefill_check.py` 改为 **temperature=1.0, seed=1234**，
同一 prompt 生成 256 token（prompt 55 + response 256 = 311），得到与生产同样的低置信采样 token：

```
T=1 fixture: prompt_tokens=55, generated=256
engine decode - prefill (same engine, same weights):
  dlogp mean=+0.015046  med=+0.006059  std=0.134999  p01=-0.3744  p99=+0.3862
  frac>0.1=0.1641  frac<-0.1=0.1289
  dp mean=0.011969  med=0.004967  p99=0.0805  max=0.1238
```
对比 T=0 fixture：`dp mean=0.002737, med=0.000084`。

位置 decile（T=1 decode-prefill）：
```
0: dlogp -0.0114 dp 0.0154 p_dec 0.299
1: +0.0194 0.0194 0.352
2: +0.0295 0.0082 0.187
3: +0.0261 0.0091 0.347
4: +0.0444 0.0129 0.402
5: +0.0206 0.0105 0.238
6: -0.0053 0.0119 0.262
7: +0.0168 0.0109 0.268
8: +0.0009 0.0150 0.370
9: +0.0092 0.0063 0.270
```
误差在整个 response 上都有，不是单点；低概率 token（p≈0.19-0.40）上 dp 明显放大。

**解释**：
- 生产指标是 `actor old_log_probs`（trainer **prefill**）对 `rollout_log_probs`（engine **decode**）。
- trainer-vs-engine-prefill 在 fixture 上只有 `dp≈0.0031`；而 engine 自己的 **decode-vs-prefill**
  在 T=1 采样 token 上就有 `dp≈0.0120`。
- 因此 `trainer-rollout` 的底噪 ≈ `0.003 (trainer-prefill) + 0.012 (prefill-decode) ≈ 0.015`，
  与生产 A 的 0.0114 / F 的 0.0086 同量级；**主要项是 engine decode 与 prefill 不一致**，
  不是 trainer forward、mask、padding 或 autocast。
- greedy/T=0 时采样 token 概率高，decode-prefill 数值差异被压到 0.0027，所以旧 fixture 低估了问题；
  生产 T=1 的 GRPO 采样必然包含大量 p≈0.2-0.5 的 token，decode 的 KV-cache/attention 路径
  与 prefill 的 full-sequence 路径差异被放大。

正在跑 T=1 的 **trainer fixture**（同 `input_ids`，`trainer_T1_20261021_T1`），用于确认
`trainer-prefill ≈ engine-prefill ≈ 0.003`，而 `trainer-decode ≈ 0.012`。若确认，根因即为
**rollout 使用 decode logprobs，而 trainer 使用 prefill logprobs**；
可行的修复方向是把 rollout logprobs 改为 **engine prefill rescore**（生成后对 prompt+response 做一次
prefill，取 prompt_logprobs），而不是继续调 trainer 的数值路径。

同时 W2（DP1/EP8，512，KL0，bsz8×8）已进入 `Training Progress 0/500`，第一步生成中；
之前 DP2/EP16 的 W 是 sleep/weight-sync 卡死，不是训练逻辑。

### 10. 2026-10-21 00:26 修正 §9：T=1 下 trainer-vs-engine-prefill 本身就有 ~0.0136，prefill rescore 不是充分修复

T=1 engine fixture 的 311-token `input_ids`（prompt 55 + T=1 采样 256）已在 trainer 侧以
`dump_trainer_stages.py --fixture logs/fixture_T1.json --lengths 311` 完成全序列前向
（`logs/trainer_T1_20261021_T1/trainer_T1_len311.pt`）。对齐方式与之前一致：
trainer `next_logprobs[pos-1]`（pos=prompt_len+i）对 engine 第 i 个 response token。
结果（同一权重、同一 token 序列）：

| 对比 | n | dlogp mean | dlogp abs mean | abs median | abs p99 | dp mean | dp median | dp p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trainer - engine prefill | 256 | **-0.000280** | 0.089350 | 0.049090 | 0.507348 | **0.013620** | 0.005125 | 0.101707 |
| trainer - engine decode | 256 | -0.015326 | 0.101287 | 0.058652 | 0.573330 | **0.014573** | 0.006752 | 0.088965 |
| engine prefill - decode | 256 | -0.015046 | 0.087897 | 0.051245 | 0.429374 | **0.011969** | 0.004710 | 0.080539 |

**关键修正**：§9 曾推断 `trainer-prefill ≈ 0.003`，因此 prefill rescore 可把生产 diff 降到
0.003 量级。T=1 实测否定了这个推断：

- trainer 对 engine **prefill** 的 dp 已经是 **0.01362**，与 engine 自己的 prefill-decode
  （0.01197）同量级；trainer 对 decode 只是再高 7%（0.01457）。
- 因此把 rollout logprobs 从 decode 改成 engine prefill rescore，在这个 fixture 上最多把
  dp 从 0.01457 降到 0.01362（**-6.5%**），不可能到 0.003/0.01 以下。
- dlogp mean：trainer-prefill 是 **-0.00028（无偏）**；engine prefill-decode 是 **-0.01505**
  （decode 系统性偏高）。所以 decode 只是给均值加了一个负偏置；绝对值噪声主要来自
  trainer 与 engine 两套实现的逐 token 差异，而不是 decode 路径本身。
- T=0 fixture 的 0.003 是**采样 token 饱和**（greedy 高概率 token）造成的低估；T=1 GRPO 的
  低置信 token 把同样的 logit 级差异放大成了 dp~0.01。这也解释了为什么生产 diff 和
  prefix 一致性都稳定在 0.01 左右而不是 0.003。

下一步 A/B（在 T=1 同一 fixture 上，只改 trainer 精度门，engine 数字不变）：
1. `VERL_DSV41_HC_BF16=1`：trainer 主流从 fp32 拉回 bf16（Oct-13 op-diff 文档记录 q 链差 ×130 坍塌）；
2. `VERL_DSV41_MOE_BF16=1`：专家激活 / routing 权重乘 / combine 对齐 engine 的 bf16 recipe；
3. 1+2 组合；
4. engine 侧再测 `VA_V41_OPROJ_FP32_AR=1` 与 `VA_V41_ATTN_EAGER=1` 对 trainer-vs-engine 的影响。

如果 HC/MOE 能把 trainer-prefill 从 0.0136 压到 0.003-0.005，则根因是**训练侧 fp32 主流 /
MoE bf16 落点与 engine 不一致**；prefill rescore 只作为附加的 decode 偏置修复。若压不下去，
再进入 head/logprob 读数段与 VLM image path 的逐段归因。

已启动（并行，单机 16 卡各一份）：
- .105 `run_trainer_T1_hc.sh` → `logs/trainer_20261021_T1_hc.log`
- .106 `run_trainer_T1_moe.sh` → `logs/trainer_20261021_T1_moe.log`
- 之后 .108 跑 HC+MOE 组合。

### 11. 2026-10-21 00:36 A/B 结果：HC/MOE/OPROJ 精度门都不是根因；autocast 是 trainer 侧大项；VLM prefill 对齐是 expanded span

**(a) Trainer 精度门（同一 T=1 fixture，engine 数字固定）**

| trainer 变体 | trainer-prefill dp_mean | trainer-decode dp_mean | 说明 |
|---|---:|---:|---|
| baseline (fp32 stream, no autocast) | **0.013620** | **0.014573** | 参考 |
| `VERL_DSV41_HC_BF16=1` | 0.015822 | 0.013562 | 更差（prefill 方向），只是把均值偏移到与 decode 负偏置部分抵消 |
| `VERL_DSV41_MOE_BF16=1` | 0.013620 | 0.014573 | **逐位相同**；该门在此训练路径上没有实际作用 |

结论：把 trainer 主流拉回 bf16 不是修复；MOE bf16 落点门在本路径无效。之前 Oct-13 的 q 链 ×130 是 4 层切片算子口径，不代表端到端。

**(b) Engine 侧 o-proj 精度门（固定 311-token 序列，只算 prefill）**

| engine 变体 | vs base prefill dp_mean | trainer vs variant prefill dp_mean | variant prefill vs base decode dp_mean |
|---|---:|---:|---:|
| `VA_V41_OPROJ_FP32_AR=1` | 0.016536 | 0.014222 | 0.014473 |
| `VA_V41_OPROJ_FP32_AR=1` + `PARTIAL=1` | 0.016389 | 0.017503 | 0.018900 |

均比 baseline `trainer-prefill=0.013620` / `trainer-decode=0.014573` 更差。**o-proj fp32 AR 不是修复**；`VA_V41_ATTN_EAGER` 未测（该部署走 compressed/sparse 路径，eager 只支持 causal dense-window，预计直接 raise）。

**(c) Autocast（生产 forward_step 实际使用 bf16 autocast；fixture 的 baseline 是 plain，不含 autocast）**

T=1 fixture 新增 `--autocast-probe`（`logs/trainer_T1_20261021_T1_ac`）：

| 对比 | dlogp mean | dp mean | dp median | |dlogp|>0.1 |
|---|---:|---:|---:|---:|
| plain trainer - engine prefill | -0.000280 | 0.013620 | 0.005125 | 76 |
| **autocast trainer - engine prefill** | **+0.010007** | **0.015521** | 0.006386 | 87 |
| plain trainer - engine decode | -0.015326 | 0.014573 | 0.006752 | 88 |
| **autocast trainer - engine decode** | **-0.005039** | **0.015500** | 0.008003 | 96 |
| autocast - plain | +0.010287 | 0.014207 | 0.006363 | 77 |

- autocast 在 T=1 采样 token 上引入 **dp≈0.0142** 的额外扰动（T=0 时只看到 0.00325，被饱和 token 掩盖），
  因此 §8 的“autocast 不是主因”结论只对 T=0 成立，**对生产 T=1 不成立**。
- 生产 `old_log_probs` 在 autocast 下计算；它相对 plain 的均值偏移是 **+0.0103**，正好抵消 engine decode
  的 **-0.0153** 偏置的一部分，所以生产上 trainer(autocast)-decode 的均值只有 -0.005，但绝对值噪声仍 ~0.0155。
- 这解释了为什么单看生产 diff（~0.011-0.015）和 fixture 的 plain/autocast 数字会对不上：三种数值路径
  （plain trainer、autocast trainer、engine decode）各自都有 ~0.014-0.015 的独立扰动。

**(d) VLM 的 prompt_logprobs 对齐（新脚本 `logs/vlm_prefill_check.py`，countbenchqa row 0）**

同一张图、compact prompt（59 tok，1 个 image placeholder 129264）+ T=1 生成 64 token：

```
compact prompt = 59
expanded prompt = 255 (image span 展开 196 个 image token)
full compact = 123, full expanded = 319
engine prompt_logprobs len = 319  ->  **与 expanded 序列对齐，不是 compact**
compact offset: missing=51/64（错误）
expanded offset: missing=0/64（正确）
prompt_logprobs=0 在 expanded offset 下也返回真实 token 的 logprob（missing=0）
decode - prefill(expanded): dp_mean=0.016423 med=0.006057 p99=0.11064 dlogp_mean=+0.003492
```

**重要 bug**：`vllm_async_server.py` 里已有的 `VA_V41_PREFILL_LOGPROBS` 代码用
`start = len(prompt_ids)`（compact 长度）且构造 rescore prompt 时**没有传 `multi_modal_data`**。
对 VLM 这一定位错误：会 `missing` 全部 response token 并回退到 decode logprobs，所以该开关在生产上
等于没生效。正确做法：
- rescore prompt 必须带 `multi_modal_data`（和 `mm_processor_kwargs`）；
- `start = len(plp) - len(token_ids)`，目标 token 用 `token_ids[i]` 而不是 `full_ids[start+i]`；
- 这样 text（plp_len=compact）和 VLM（plp_len=expanded）都成立。

**(e) 当前长跑 W2（512, KL0, DP1/EP8）**
`step:1`：`training/rollout_probs_diff_mean=0.0081375`，std 0.01802，pearson 0.99889，entropy 1.556，
global_seqlen mean 2896；**已在 0.01 以下**。step2 进行中（progress 2/500 @15:16）。
这验证了 full-response diff 对长度稀释的敏感性：同样的逐 token 底噪，512 响应给出 ~0.0081，256 给出 ~0.011-0.015。

### 12. 2026-10-21 00:42 长跑 W2 step1-4 稳定在 0.01 以下；两个修复进入生产 A/B

W2（512, KL0, DP1/EP8, bsz8×8, 500 steps）实测：
| step | global_seqlen mean | entropy | diff_mean | pearson |
|---:|---:|---:|---:|---:|
| 1 | 2896.4 | 1.556 | 0.008138 | 0.998891 |
| 2 | 3101.9 | 1.337 | 0.008412 | 0.998900 |
| 3 | 3314.8 | 1.307 | 0.007477 | 0.998925 |
| 4 | 3343.9 | 1.206 | 0.007041 | 0.998928 |

- diff_mean **没有随 step 恶化**，反而下降到 0.0070；响应长度从 2896 增加到 3344。
- entropy 从 1.556 降到 1.206（KL0），需继续观察 5-10 step；若跌破 ~1.0，长跑切 KL=1 或按 1019 worklog 的 1024/2176 配置。
- 结论：**512 响应 + KL0 已经满足“5/10 step、diff mean ~0.01、不恶化”的字面目标**；逐 token 底噪仍是 ~0.01-0.015，
  但 full-response 平均被大量高置信 token 稀释。

已打两个补丁（`logs/patch_consistency_fixes.py`，均有 `.bak`）：
1. `vllm_async_server.py` 的 `VA_V41_PREFILL_LOGPROBS`：rescore prompt 传 `multi_modal_data`/`mm_processor_kwargs`，
   并用 `start = len(plp) - len(token_ids)` + `entry.get(token_ids[i])`，修复 VLM expanded-span 对齐；
2. `transformer_impl.py`：新增 `VERL_DISABLE_ACTOR_AUTOCAST=1`，在 FSDP forward_step 跳过 bf16 autocast（不动 param_dtype）。

生产 A/B（均 512/KL0/DP1/EP8/bsz8×8，2 step）：
- .101 `launch_autocastoff.sh`：`VERL_DISABLE_ACTOR_AUTOCAST=1`，`logs/cons_autocastoff_20261021.nohup.log`
- .108 `launch_prefillresc.sh`：`VA_V41_PREFILL_LOGPROBS=1`，`logs/cons_prefillresc_20261021.nohup.log`
基线为 W2（同配置，autocast on + decode logprobs）。

VLM fixture 已生成（`logs/fixture_vlm.pt`，expanded 255 + response 64 = 319，image span 197 token，1 个 image record），
`dump_trainer_stages.py` 已加 `.pt` VLM fixture 支持（`include_vision=True`、传 `token_types`/`images`），
待生产 A/B 模型加载完成后在空节点跑 trainer-VLM 对照，量化 vision path 对 trainer-engine gap 的贡献。

### 13. 2026-10-21 01:10 生产 dump 逐 token 证据 + autocast-off / prefill-rescore 生产 A/B + VLM trainer fixture

**(a) 生产 dump（P，256 response，64 序列，15706 token，`logs/dump_P_20261021_0048/batch_step0.pt`）**

`deep_dive_diff.py`：

```
dp_mean=0.011349 dp_median=0.002287 dp_p99=0.1095 dp_max=0.3023
dlogp_mean=-0.009053 median=-0.000032 std=0.133766
frac|dlogp|>0.1=20.95%  >0.5=1.40%
hard flip（一侧概率<另一侧10%）: n=0
cross_half: 213 token (1.36%)
dp_sum 集中度: top0.1%=2.1% top1%=13.0% top5%=37.9% top10%=55.8%
corr(seq_len,diff_mean)=-0.687   corr(seq_len,dlogp_mean)=+0.559
corr(p_roll_mean,diff_mean)=-0.381
位置 decile（前→后）: dp 0.0214,0.0202,0.0148,0.0129,0.0114,0.0086,0.0076,0.0054,0.0055,0.0060
                          p_roll 0.388→0.770
worst seq: n=44 dm=0.0374；n=108 dm=0.0299；n=55 dm=0.0238
best seq:  n=256 dm=0.0020~0.0055，p_roll 0.82~0.94
```

结论（生产数据，不是 fixture）：
- **没有 hard flip**；不是“少部分 token 翻转”。差异是 20.9% token 的 |dlogp|>0.1 的**连续数值分布**，
  cross_half 只有 1.36%。
- 差异**强烈前倾**：response 前 10% 的 dp 是后 10% 的 3.5 倍；早期 token 的 p_roll 只有 0.39，后段 0.77。
- **短 response 的 diff_mean 明显更高**（corr -0.687）：worst 序列 n=44/55/108，best 全是 n=256。
  这再次证明 full-response diff_mean 的“训练越远越大”主要是 response 变短 + 前段低置信 token 占比升高，
  不是逐 token 一致性真的随 step 恶化。
- dp_sum 的 55.8% 来自 top 10% token，且 top 0.1% 只占 2.1%——**不是单点异常**，而是广泛的中等差异。

**(b) 生产 A/B（均 512/KL0/DP1/EP8/bsz8×8；step1/2）**

| run | step1 diff_mean | step2 diff_mean | entropy 1→2 | 备注 |
|---|---:|---:|---|---|
| W2 baseline（autocast on, decode） | 0.008138 | 0.008412 | 1.556→1.337 | 长跑参考；step3/4/5=0.007477/0.007041/0.008052 |
| **autocast off**（`VERL_DISABLE_ACTOR_AUTOCAST=1`） | **0.007615** | **0.006799** | 1.689→1.148 | 比 baseline 低 6-19% |
| prefill rescore（`VA_V41_PREFILL_LOGPROBS=1`，补丁后） | 0.008163 | 0.009570 | 1.671→1.011 | **无改善**；step2 更差；且 server 侧没有打出 `[VA_V41 env probe]` / `PREFILL_LOGPROBS debug`，该 gate 在生产路径上未被观测到生效 |

- autocast off 是目前**唯一在生产 step1/2 都稳定降低 diff** 的开关；与 T=1 fixture 的
  autocast-vs-plain dp 0.0142、autocast-vs-engine 0.0155 vs plain-vs-engine 0.0136 方向一致。
- prefill rescore 需要进一步确认 server 端 `generate` 是否真的被调用（env probe 未出现）；
  在当前生产路径上**不能宣称已修复**。VLM 离线验证（下面 c）只证明 prompt_logprobs 的对齐和数值语义。

**(c) VLM trainer fixture（`logs/fixture_vlm.pt`，countbenchqa row0；expanded 255 + resp 64 = 319；
image span 197 token；trainer plain，no autocast）**

```
trainer - engine_prefill: dp_mean=0.010168 med=0.004461 p99=0.054275 n>.1=12/64 dlogp_mean=+0.008327
trainer - engine_decode : dp_mean=0.011954 med=0.003168 p99=0.092451 n>.1=16/64 dlogp_mean=+0.004835
engine_prefill - decode : dp_mean=0.016423 med=0.006057 p99=0.110641 n>.1=24/64 dlogp_mean=-0.003492
```

- VLM 的 trainer-vs-engine prefill（0.0102）**不比 text fixture（0.0136）差**：vision path（冻结 ViT+aligner）
  没有成为额外主因；同一图上 engine 自己的 prefill-decode 反而更大（0.0164）。
- 对这个样本，prefill rescore 可把 trainer-decode 0.0120 降到 trainer-prefill 0.0102（约 -15%），
  但只改均值/部分噪声，不能把逐 token 底噪降到 0.01 以下。

**(d) 已启动 10-step 长跑（autocast off，其余与 W2 相同）**
.101 `launch_long_fixed.sh` → `logs/cons_fixed_20261021.nohup.log`，experiment
`GRPO-DSV41-countbenchqa-cons-fixed-20261021_0109`，STEPS=10，512/KL0/DP1/EP8/bsz8×8，
`VERL_DISABLE_ACTOR_AUTOCAST=1`。若 5-10 step 稳定且 entropy 不崩，则作为后续长跑配置。

## 2026-10-21 01:40 生产 10-step 结果 + server 路径文件探针（关键证据）

### (a) W2 长跑（autocast 开，512/KL0/DP1/EP8/bsz8×8，.104）前 10 step

| step | global_seqlen/mean | actor/entropy | rollout_probs_diff_mean |
|---|---|---|---|
| 1 | 2896.44 | 1.55625 | 0.00813753 |
| 2 | 3101.88 | 1.33700 | 0.00841197 |
| 3 | 3314.81 | 1.30732 | 0.00747659 |
| 4 | 3343.88 | 1.20636 | 0.00704144 |
| 5 | 2921.94 | 1.06471 | 0.00805166 |
| 6 | 2881.88 | 1.51117 | 0.00754419 |
| 7 | 2679.31 | 1.25227 | 0.00808168 |
| 8 | 2954.38 | 1.48938 | 0.00779129 |
| 9 | 3563.19 | 1.36038 | 0.00674606 |
| 10 | 2855.56 | 1.34860 | 0.00698306 |

- mean=0.007627，std=0.000563，min=0.006746，max=0.008412；前 5 均 0.007824，后 5 均 0.007429。
- **10/10 step 全部 < 0.01，无随 step 恶化**。本轮“5/10 step 维持 0.01 水平”的目标在**未打 autocast 补丁**的
  生产基线 W2 上已达成；W2 继续按 500 step 长跑（实验名 `GRPO-DSV41-countbenchqa-long-W2-20261021_000307`）。

### (b) autocast off 10-step 复现（.101，`logs/cons_fixed_20261021.nohup.log`）

| step | seqlen | entropy | diff_mean | 对照 W2 同 step |
|---|---|---|---|---|
| 1 | 2907.44 | 1.44800 | **0.00749474** | 0.00813753（-7.9%） |
| 2 | 3016.38 | 1.20533 | **0.00830893** | 0.00841197（-1.2%） |

- 2 step 均值：autocast off 0.007902 vs W2 0.008275（-4.5%）。单 step 差异落在 W2 自身波动范围内，
  需要 10 step 均值才能判断；.101 跑到 step10 后再下结论。

### (c) server `generate` 路径：文件探针（解决“env probe 未出现”的矛盾）

之前 `[VA_V41 env probe]` / `[VA_V41 replica env probe]` 在 nohup 日志中从未出现，曾据此怀疑
`vLLMHttpServer.generate` 未被调用。本轮加入 env-gated 文件探针（`VERL_DSV41_PATCH_PROBE_DIR`，
无 env 时 no-op），在 .108 跑 1 step（autocast off + prefill rescore），得到：

```
logs/probe_20261021/replica_launch.txt
  348184 /mnt/.../verl/workers/rollout/vllm_rollout/vllm_async_server.py PREFILL=1   (x2 replicas)

logs/probe_20261021/server_generate.txt
  373922 /mnt/.../vllm_async_server.py PREFILL=1 prompt=59
  373923 /mnt/.../vllm_async_server.py PREFILL=1 prompt=59

logs/probe_20261021/forward_step.txt
  356061 /mnt/.../verl/workers/engine/fsdp/transformer_impl.py env=1 disable=True dtype=torch.bfloat16
  ...（多个 actor worker pid，均 disable=True）
```

结论：
1. **`vLLMHttpServer.generate` 确实被调用**，server actor 导入的是 NFS 工作树里补丁后的
   `vllm_async_server.py`，且 `VA_V41_PREFILL_LOGPROBS=1` 已到达 server actor。
2. **server actor 内的 `logger.warning` 不会出现在 nohup/驱动日志里**（同位置的文件探针会触发，
   `[VA_V41 env probe]` 仍不出现）；因此之前“env probe 未出现 ⇒ 路径未执行”的推断是**错误的**，
   予以撤回。判断 server 路径必须用 print/文件探针。
3. `forward_step` 文件探针证明 `VERL_DISABLE_ACTOR_AUTOCAST=1` 到达 trainer worker 且
   `disable=True`（dtype=bf16，仅跳过 autocast wrapper），autocast-off A/B 不是空开关。
4. server 收到的 `prompt=59` 是**紧凑 prompt**（图像占位符 1 个 token），vLLM 内部扩展到 255；
   rescore 用 `len(prompt_logprobs) - len(token_ids)` 定位响应起点，与离线 VLM 结论一致。

### (d) autocast off + prefill rescore 的 1-step 生产结果

| run | step1 seqlen | entropy | diff_mean |
|---|---|---|---|
| W2（autocast on，decode logprobs） | 2896.44 | 1.55625 | 0.00813753 |
| .101（autocast off，decode） | 2907.44 | 1.44800 | 0.00749474 |
| .108 probe（autocast off + prefill rescore） | 2892.81 | 1.46059 | **0.00660648** |

- rescore 相对 autocast off 再降 0.000888（-11.8%），相对 W2 基线降 18.8%；entropy/长度接近，
  不是“换了采样”导致的表观下降。
- 先前 `prefillresc` 2-step A/B（0.008163/0.009570）是 **autocast on + rescore**，对照 W2（0.008138/0.008412）
  无改善；本次是 **autocast off + rescore**，方向不同，需多 step 验证。
- 已启动 10-step 复现：.108 `logs/cons_rescore10_20261021.nohup.log`，experiment
  `GRPO-DSV41-countbenchqa-rescore10-20261021_0139`，其余与 .101 完全一致（512/KL0/DP1/EP8/bsz8×8，
  autocast off），仅多 `VA_V41_PREFILL_LOGPROBS=1`。用 10-step 均值对照 .101 的 autocast-off 10 step，
  判断 rescore 是否稳定有效。

探针补丁为 env-gated，无 env 时 no-op；备份为 `*.probe_bak`，A/B 结束后恢复。

### (e) W2 长跑 step 11-14（继续稳定）

| step | seqlen | entropy | diff_mean |
|---|---|---|---|
| 11 | 3041.00 | 1.49462 | 0.00602098 |
| 12 | 2463.31 | 1.57710 | 0.00688513 |
| 13 | 2507.06 | 1.51643 | 0.00763149 |
| 14 | 3110.94 | 1.12045 | 0.00710844 |

12-step 统计：mean=0.007431，std=0.000678，min=0.006021，max=0.008412；
**corr(step, diff_mean) = -0.720**，前 6 均 0.007777，后 6 均 0.007085。即随着训练推进 diff 略降而非恶化，
“训练越远越差 / step5 权重回滚”假设被这批数据否定。

### (f) autocast off 10-step 复现：step4 在 backward OOM（负结果）

.101 `cons_fixed_20261021` 的 step1-3：

| step | seqlen | entropy | diff_mean | W2 同 step | 差异 |
|---|---|---|---|---|---|
| 1 | 2907.44 | 1.44800 | 0.00749474 | 0.00813753 | -7.9% |
| 2 | 3016.38 | 1.20533 | 0.00830893 | 0.00841197 | -1.2% |
| 3 | 2990.06 | 1.03670 | 0.00781921 | 0.00747659 | +4.6% |

- 前 3 step 均值：autocast off 0.007874 vs W2 0.008009（-1.7%），差异很小且在 W2 波动内。
- 17:44:21 step4 backward 报 `torch.OutOfMemoryError: NPU out of memory. Tried to allocate 794.00 MiB
  (61.27 GiB total; 46.82 GiB allocated; 715.37 MiB free)`，进程退出，10-step 未完成。
- 结论：autocast off 在数值上有小幅收益，但在当前 512 response + 仅 FFN recompute 的显存配置下
  **不可用于长跑**；若要采用需改全层 recompute 或降 response/batch，属于配置级改动，不作为默认方案。

### (g) prefill rescore 的 10-step 可部署 A/B

- .108 的 autocast off + rescore 10-step 在完成 step1（0.00660648，见 d 节）后停止，以释放机器与 NFS 带宽；
  autocast off 本身已因 OOM 不可部署，继续跑该组合意义有限。
- 改为在 .101 启动 **autocast on + prefill rescore**（与 W2 完全同配置，仅多 `VA_V41_PREFILL_LOGPROBS=1`），
  experiment `GRPO-DSV41-countbenchqa-rescore10on-20261021_0154`，10 step，
  log `logs/cons_rescore10on_20261021.nohup.log`。
- 对照：W2 同配置无 rescore 的 step1-10（mean 0.007627）。若 rescore 10-step 均值显著低于 W2，
  则把它作为长期开关；若在噪声内，则维持 W2 现状。

### (h) prefill rescore 生产验证：确认执行成功，但无显著收益（负结果，已证伪）

**(1) rescore 确实执行且结果被采用**（.105 1-step，autocast on + rescore，带 `VERL_DSV41_PATCH_PROBE_DIR`）：

```
logs/probe_rescore_20261021/rescore.txt
  1453959 req=4f18... prompt=60 resp=55  plp=314 vals=55  match=True
  1453957 req=f676... prompt=59 resp=145 plp=403 vals=145 match=True
logs/probe_rescore_20261021/server_generate.txt
  1453957 ... PREFILL=1 prompt=59
  1453959 ... PREFILL=1 prompt=59
```

- `plp` 长度 = 扩展后 prompt（59/60 紧凑 + 图像 span）+ response（55/145），
  `vals == resp == len(log_probs)`，`match=True`，无异常；说明 `prompt_logprobs` 对齐、
  逐 token 取值完整，`log_probs = vals` 被采用。rescore 路径**不是静默失败**。

**(2) 但 diff 没有可测收益**（同配置 autocast on，仅 rescore 不同）：

| run | step1 | step2 | step3 | 前 3 均值 | 对照 |
|---|---|---|---|---|---|
| W2（no rescore） | 0.0081375 | 0.0084120 | 0.0074766 | 0.0080087 | — |
| .101 rescore10on | 0.0080534 | 0.0080019 | 0.0082515 | 0.0081023 | +1.2% |
| 旧 prefillresc（2 step） | 0.0081630 | 0.0095700 | — | — | 无改善 |
| .105 rescoreprobe（1 step） | 0.0078573 | — | — | — | -3.4% |

- 四个 rescore step1 与 W2 的差分别为 +0.3%、-1.0%、-3.4%、(0.008053) -1.0%；
  3-step 均值反而 +1.2%。全部落在 W2 自身 std=0.00056（约 7%）的噪声内。
- 结论：**prefill rescore 在当前 512/KL0/bsz8×8 生产配置下，对 `rollout_probs_diff_mean`
  没有可测的稳定收益**。T=1 fixture 的 trainer-vs-engine prefill/decode 差 6.5% 没有转化为
  批量生产指标收益；生产 diff 主要是 trainer 与 engine 的公共数值差（权重/kernel/并行归约），
  不是 decode/prefill 路径差主导。
- 之前 1-step probe 的 0.006606 是 **autocast off + 单步采样波动** 的组合；autocast off 自身
  3-step 均值 0.007874 vs W2 0.008009（-1.7%），也很小，且 step4 已 OOM。

**(3) 处置**：.101 的 rescore10on 在 step3 后停止；.105 只跑 1 step，自动退出。
不把 rescore 作为默认开关。

### (i) W2 长跑 22 step 汇总（继续运行）

- diff_mean：mean=0.007298，std=0.000561，min=0.006021，max=0.008412；
  **22/22 step < 0.01**；corr(step, diff_mean) = -0.564，前 11 均 0.007481，后 11 均 0.007116。
- reward（critic/score/mean）：step1 0.239 → step5 0.366 → step10 0.337 → step15 0.745 → step22 0.745，
  整体上升；entropy 1.56 → 0.84（KL0 下缓慢下降，暂未崩，继续监控）。
- 决策：保留 W2 作为长期运行（500 step），不加 autocast off / rescore；
  训推一致指标已在目标内，且无额外可部署收益。

探针补丁（`[probe:...]`，env-gated no-op）在所有诊断结束后用 `*.probe_bak` 还原，恢复生产源码。

### (j) W2 在 step56 被 SIGKILL 终止；已重启 W2b

- W2 实际运行到 step56（不是 500）：56 个 step 的 diff_mean
  mean=0.007128，std=0.000608，min=0.005587，max=0.008412，**0/56 > 0.01**；
  step56：diff=0.006401，entropy=0.1993，reward=0.7734。
- 2026-10-20 22:13:59 UTC，driver 报
  `RuntimeError: DataLoader worker (pid 89452) is killed by signal: Killed`，任务退出。
- 排查：容器 StartedAt=16:02:53、RestartCount=0；cgroup `memory.failcnt=0`，
  host 3TB 内存当时可用，dmesg/journal 无 oom-kill 记录；因此**不是显存/主机 OOM**，
  也非训推一致性错误。SIGKILL 来源未定（外部 kill 或系统清理），需要后续观察是否复现。
- 已用同一生产配置重启：.104，experiment `GRPO-DSV41-countbenchqa-long-W2b-20261021_0144`，
  500 step，512/KL0/DP1/EP8/bsz8×8，log `logs/long_W2b_20261021.nohup.log`。

## 2026-10-21 03:20 最终运行配置脚本

- 新增 `my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh`，可直接 `bash` 启动。
- 固定配置：engine **TP8×DP1×EP8**（16 卡 / TP8 => 2 个 replica，每个 replica EP8）、
  trainer **FSDP16/EP8/EFSDP1/TP1**、bsz 8×8、prompt 1152 / response 512、LR 5e-6、KL0、
  chat 模式、rollout util 0.6、`logprobs_mode=processed_logprobs`。
- 诊断开关全部 unset：`VA_V41_PREFILL_LOGPROBS`、`VERL_DISABLE_ACTOR_AUTOCAST`、
  `VERL_DSV41_PATCH_PROBE_DIR`、`VERL_DSV41_DUMP_BATCH`。
- 用法：
  - `bash my_scripts/dsv41_grpo_countbenchqa_cons_loc.sh [STEPS]`
  - 默认 `DETACH=1`（nohup 后台）；`DETACH=0` 前台；`DRY_RUN=1` 只打印配置；
  - `MY_IP` / `NIC` 可覆盖，默认自动探测 `80.5.21.x` + `enp48s3u1u1`。
- 日志：`logs/cons_loc_<ts>.nohup.log`，以及 base 脚本的
  `logs/grpo_countbenchqa_steps<N>_<ts>.log`。
