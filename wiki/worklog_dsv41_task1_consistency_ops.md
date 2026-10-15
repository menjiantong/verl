# worklog · 任务1：训推一致性——逐个排查小算子 + 推理融合算子替换为小算子（2026-10-13 启动）

> 用户任务书（原文摘要）：继续训推一致性定位，挨个排查小算子，将推理的融合算子替换为小算子，然后简单验证；
> 验证 ok → 启动 verl+gsm8k 验证跑 1 步，看 pearson 是否 ≥0.99（最低 0.985）。
> **验收标准：第 1 步 pearson > 0.99（最低 0.985）且 diff_mean < 0.01；达标后启动 50 步长跑。**
> 未达标则继续分析（自动循环）。所有过程（命令/代码修改/错误日志/思路）记入本文件。

## 机器与流程
- 验证对：**131（本机, master）+ 119（slave）**（proven 组合）；切片 dump 复用 131。
- 循环：改（env 门控开关）→ 切片 dump 对拍（len200，~6min）→ 131+119 跑 1 步 RL → 对照 pearson/diff_mean。

## 已知基线（启动时状态）
- 根因已定位并修复其一：训练侧主流 fp32（hc_pre→fp32 出生 + 混合 dtype 自动升格）vs 引擎 bf16
  → 开关 `VERL_DSV41_HC_BF16=1`（q 链 ×130 坍塌）+ `VERL_DSV41_ALIGN_ATTN=1`（attn 核减半）。
- 1 步 RL（131+119, 16:0x）：pearson 0.9630 / diff_mean 0.0571（**未达标**）；50 步 run 前 3 步 0.9599/0.9564/0.9722。
- 切片 logprob 级（本轮新口径，shift 对齐，见下）：mean|d|=0.047（未含 MOE_BF16）/ **0.041（含 MOE_BF16）**。

## 小算子排查清单（引擎侧融合算子 → 小算子/对齐）
| # | 引擎算子 | 替换/验证 | 状态 |
|---|---|---|---|
| 1 | `inplace_partial_rotary_mul`（rope） | → torch 小算子（VA_V41_TORCH_ROPE） | ✅ **逐位相同**（排除） |
| 2 | qkv 投影（CVLinear/matmul） | → F.linear（VA_V41_EAGER_QKV） | ✅ **逐位相同**（排除） |
| 3 | o-proj wo_a/wo_b + TP-AR | AR→fp32（VA_V41_OPROJ_FP32_AR=1）；GEMM 已验证=F.linear 等价 | ✅ 已修/排 |
| 4 | **注意力核 `npu_sparse_flash_mla`** | → eager torch（trainer 同款 indexed_sparse_attention 数学） | ⏳ **进行中（下一个）** |
| 5 | MoE 融合（grouped/combine/gate） | trainer 侧 `VERL_DSV41_MOE_BF16=1`（engine recipe） | ✅ 已加（σ −11%），继续攻内部 fp32 设计点 |
| 6 | head/logits | 引擎 logits 实测 **fp32**（与 trainer 设计 fp32 一致） | ✅ 已对齐（排除） |
| 7 | norms | rms_norm_cast vs torch | ✅ 逐位相同（§6.2） |

## 本轮关键方法/发现（2026-10-13 傍晚）
- **logprob 对齐口径（新增，必读）**：引擎 dump 的 `language_model.logits_processor` = [1,V]+[199,V] 两段 logits，
  **行序比 trainer 预测行整体偏移 1**（engine 行 i ↔ trainer 预测行 i-1）；正确对齐后 argmax 一致率 92.5%（之前错位时报 0%）。
- 对齐比较：`eng = log_softmax(cat(lp))[1:].gather(ids[1:])` vs trainer `next_logprobs[:199]`。
- 当前结果：align2（HC+ATTN）σ=0.1155 / mean|d|=0.0472；**align3（+MOE_BF16）σ=0.1032 / 0.0410（−11%）**；
  |d|>0.5 的位置 3-4/199（离散翻转少数点主导 σ）。
- 结论：**下一步 = 攻注意力核（引擎融合核 → eager 小算子）**，然后攻 MoE 段残余 fp32 设计点。

## 下一步实验预案（1 步 RL 结果出来后执行；131+119 释放即点火）
- **靶点 #5：MoE 段【径】差（trainer 权重乘在 down-GEMM 前 vs 引擎 combine 核内乘）**。
  修法 = trainer 侧 `VERL_DSV41_ALIGN_MOE=1`（experts.py:211/227，把权重乘移到 combine，**对齐引擎顺序**）。
  - 切片验证（引擎 dump 复用 slice4_attneager_al，不用重跑引擎）：
    `cd /workspace-verl/verl && VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1 VERL_DSV41_ALIGN_MOE=1 bash scripts/dsv41_consistency/run_dump_trainer.sh --len 200 --out-dir /mnt/share/m00899630/dsv41/dump/slice4_moeal/trainer`
    → 与 align3 trainer 对比 MoE/ffn 段残差（expect ffn 0.033-0.056 → 大幅下降）。
  - 通过后 1 步 RL（同前，追加 `VERL_DSV41_ALIGN_MOE=1`）。
- **备选靶点**：ALIGN_ROUTER=1（fp32 归一化喂 gate）、ALIGN_HEAD/ALIGN_NOBI（读数段门）——同一批次切片 A/B。
- **验收**：每个门单独 + 组合跑切片，取"1 步 RL pearson"最优组合冲 ≥0.985；达标 → 50 步长跑（机器待定：
  128+129 被任务2 占用；120/123 视存活）。

## 记录区（追加式）
- [待填] 每个开关的对照表与最终验收跑。

### [2026-10-14 02:2x] ★ 单机轮（131-only）· sft116-4layer 真实权重 —— 立项
- 用户指令："先使用131单机 DeepSeek-V4.1-Flash-top32-sft116-4layer 加这个权重，单机解决训推一致性。"
- 动机：砍掉双机 RL 循环（F-80 同步问题 + 引擎 NaN 疑云 + 慢）；**单机 = 引擎源码即本地树**，
  dump 对拍循环 ~10min/轮、随时可断。
- 权重：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer`（真实权重切片，
  `make_slice_from_real.py --mode hybrid` 2026-10-13 08:18；SLICE.md 注 39 处 dtype 差异 vs 模板）。
  **注意：上轮全部细粒度数字（0.00088 / 0.0335 等）是在 scaled32 合成权重上；本轮起在真实权重上重建基线。**
- 基线配方 = 上轮已验证的全部修复拉满：
  - 引擎：`VA_V41_ATTN_EAGER=1`（+镜像 `VERL_DSV41_ALIGN_ATTN=1`）+ `VA_V41_OPROJ_FP32_AR=1` + `VA_V41_OPROJ_FP32_PARTIAL=1`
  - 训练：`VERL_DSV41_HC_BF16=1` + `VERL_DSV41_ALIGN_ATTN=1` + `VERL_DSV41_MOE_BF16=1`
- NPU 状态（02:15）：chips 0-7 各被"僵尸登记"占 ~26GB（AICore 0%，**实测可分配**：8 卡 matmul OK、
  free 36-38GB）；chips 8-11 邻居活跃（不碰）。⇒ mem-util 0.9 必拒启，用 **0.5**。
- 引擎 dump 已启动（02:19，pid 11001，`/tmp/dump_engine_sft116real.log`；
  out=`/mnt/share/m00899630/dsv41/dump/sft116real/engine`，lengths 64,200，带 params）。
- 计划：引擎 dump → 训练 dump → 对拍（compare_stages/routing/logprobs）→ 残差画像 → 下一刀。
- **[02:2x 引擎 dump 完成]** 产物全：params.json(107 指纹/rank，attn_sink=fp32 ✓)、
  stages len64/200×8rank（182 张量/rank）、engine_len{64,200}.json、engine_decode_len{64,200}.json。
  - **★ NaN 最小复现候选（重大）**：len64 生成 → `generated=[0]*8`（0=bos，疑似 logits NaN→argmax 取 0）
    + decode/prefill logprob **8/8 全 NaN**；**len200 同流程正常**（decoded 值 [-2.77,-2.54,...]）。
    len64 fixture = len200 前 64 token 前缀；**prefill stage dump 182 张量零 NaN、scored 有限**（-20.2/-8.5/-19.1）
    ⇒ 坏点在 **decode/生成侧且长度相关**，第一嫌疑 = **总长 < sliding_window(128) 的 decode 路径**
    （也解释 RL 里 rollout_ppl NaN / trainer_ppl 正常：短 prompt 中招）。
  - len200 decode↔prefill deltas：[-0.002, 0.001, -0.053, -0.051, **-0.102**, 0.065, -0.028, 0.009]（mean|d|≈0.039）。
- **[02:3x 训练 dump 启动]** devices 8-15、同配方（HC_BF16+ALIGN_ATTN+MOE_BF16）、含 len64,200。
- **★ 设备选择变更**：devices 0-7 于 02:20 被**外部邻居容器**吞 ~22GB/卡
  （holder：僵尸 `ray::WorkerDict` 23GB + 外部 `python` 22GB（容器内 pid 35259，/proc 不可见）+ 陈旧 VLLM::Worker_TP）
  → vLLM 启动检查拒启（free 16GB < 0.5×61.27GB）→ **改用 devices 8-15**（free 31-36GB）+ HCCL 端口错开
  （60060-60110/61100-61150）。0-7 已实测 matmul 也 OK（free ~14GB）但 mem-util 需 ≤0.22，弃用。
- **[02:45 ★ 真实权重基线画像完成（全配方）]** 对拍三件套结果：
  - **逐模块阶梯（rel_err，len200 / len64）**：
    | 模块 | len200 | len64 |
    |---|---|---|
    | L0 attn / attn_norm | **0.0007 / 0.0000** | 0.0005 / 0.0000 |
    | L0 ffn | 0.0044 | 0.0044 |
    | L1 attn / attn_norm | 0.0045 / 0.0035 | 0.0045 / 0.0035 |
    | **L1 ffn** | **0.0318** | 0.0132 |
    | L2 ffn | 0.0496 | 0.0304 |
    | L3 ffn | 0.0493 | 0.0394 |
    | 末端 norm | 0.0303 | 0.0256 |
    ⇒ 出生点确认：**L0 ffn 的 MoE ULP 地板 0.0044**（输入已干净 0.0000）；此后每层 ffn ×1.7-3.8 结构性增益
    （len200 的 L1 ×9 是 1 个近-tie 翻转 token 叠加）；与 scaled32 结论同构（"MoE 同输入=ULP、增益不可修"）。
  - **路由（set-match）**：len64：L0/L1/L3 100%，L2 87.5%（1 真翻转/8）；len200：L0 100%、L1 96%（1/25）、
    L2 100%（1 order-only）、L3 100%；router logits rel 0.001-0.006。离散选择基本干净（翻转=近-tie 尾）。
  - **跨栈 logprob（next 约定）**：len200：delta mean **+0.0013**（无偏）std 0.0723，p50 0.0168 / p90 0.0597 /
    p99 0.3061 / max 0.6623；argmax 一致 94.95%。len64：p50 0.0149 / p90 0.0518 / max 0.345；
    argmax 一致 93.55%。（对照 scaled32 全配方：0.0335 均值 → 真实权重分布同量级、p50 已到 0.017。）
  - ⇒ **真实权重没有引入新的主残差**（39 处 dtype 差异未成为主导；AR+PARTIAL+EAGER+HC_BF16+ALIGN 全配方已生效）。
    余下主残差 = **MoE ULP 地板（0.0044/层出生）+ 每层 1.7-3.8× 复利** + 少量近-tie 翻转。
- **[03:0x ★★★ NaN 边界钉死（131 sweep，同一引擎寿命 10 档）]** 对 len200 fixture 前缀逐长度生成（8 token，
  贪婪+logprobs，全配方 env）：
  | prompt 长度 | 32 | **64** | **100** | **120** | **127** | 128 | 129 | 140 | 160 | 200 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | decode NaN | **0/8** | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
  | gen tokens | 真实 | `[0,0,0,0]`（NaN→argmax→bos） | 同左 | 同左 | 同左 | 真实 | 真实 | 真实 | 真实 | 真实 |
  - 判定：**NaN 区 = prompt ∈ [~33..63?, 127]，上边界 = 128（sliding_window）**；127+8=135 全 NaN ⇒
    不是"总长<128"；NaN 出现在**第一个 decode 步**后污染 KV（后续步全 NaN、token 全 0）传播。
  - 32 干净是反常点 ⇒ 下边界需细分（40/48/56/60/63）；也需排除"内容耦合"（换随机 prompt 对照）。
  - **与 RL NaN 的关系（待 A/B 判决）**：RL 的 GSM8K prompt 大量落在 [64,127] → 若 NaN 与补丁无关
    （原始引擎也有）则 RL NaN 直接解释；若为 eager 补丁引入则与 F-80 时间线需再对账。
  - 旧 dump 无 len64 decode（全部 `--lengths 200`），无免费历史 A/B ⇒ 需新跑。
- **[03:1x ★ 迁移 126（用户指示"126可以试一下"）]** 131 邻居波动（free 16-30GB 抖）→ 126 侦察：
  **16 chips 全空**（AICore=0%、占用仅 ~3GB/卡、free≈62GB）；工作树在（FSDPTurbo/vllm/vllm-ascend-v41-private/verl，
  但为 09-24 旧版）→ **tar-over-ssh 全树同步**（rsync 两边都没有）→ **关键四文件 md5 与 131 逐字节一致**
  （dsa_v41 a4bd5574 / dsa_v1 17ff0ec1 / FSDPTurbo model.py c90a6ad3 / vllm_async_server 74bc94c4）。
- **[03:2x NaN 归因计划（126）]** Run A（全配方）：细分长度 32/40/48/56/60/64/80/96/112/120/127/128 +
  随机内容 64/100/127/128 对照；Run B（全关=原始引擎）同列表 → 判决 eager 责任 + 内容耦合。
  已点火 Run A（`/tmp/nan_sweep_A.log`，SWEEP_TAG=A，126 本地）。
- **[03:3x ★★ Run A 结果（全配方，126）]** fixture 前缀逐长度 + 随机内容对照：
  | prompt | 32 | 40 | 48 | 56 | 60 | 64 | 80 | 96 | 112 | 120 | 127 | 128 | rand64 | rand100 | rand127 | **rand128** |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | NaN | 0/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 | 8/8 | 8/8 | 8/8 | **8/8** |
  - **下边界=40（32 干净 → 40 起 NaN）；上边界≈127（fixture 128 干净）**；**内容无关**（随机 token 同区间同 NaN）。
  - ⚠ 反常点：**random_len=128 竟 NaN** 而 fixture_len=128 干净（同长度、不同内容/历史）⇒ 128 附近的"干净"不是纯长度条件，
    疑似还有状态/历史依赖（块复用？前序请求残留？）。
  - NaN 出现即污染（token 全 0、8 步全 NaN 传播）。
  - **代码精读（eager 窗口/掩码段）**：invalid 列用 `masked_fill(-inf)`（NaN 会被替换，安全）；softmax 拼 sink 列
    ⇒ 行内不会全 -inf。⇒ eager 内部产生 NaN 的入口只可能是 **q 输入 / 合法 kv 值 / sink 三者之一为 NaN**（或上游模块）。
  - 待 Run B（全关）判决：基座引擎 bug（则 RL NaN 全解释）or 补丁引入（则 bisect EAGER/AR/PARTIAL）。
- **[03:4x Run B 点火（126）]** 四开关全关（原始引擎），长度表追加末尾再测 32（状态恢复检查）+随机对照。
  `SWEEP_LENGTHS=32,40,48,56,60,64,80,96,112,120,127,128,32`。日志 `/tmp/nan_sweep_B.log`。
- **[03:5x ★★ Run B 判决：全干净（0/8 NaN × 全长表 + 随机对照）]** ⇒ **NaN 不是基座引擎 bug，
  是四个引擎开关（EAGER/ALIGN/AR/PARTIAL）之一引入**。⇒ 双机并行 bisect（用户给 126+127）：
  - 126 Run C：`VERL_DSV41_ALIGN_ATTN=1 VA_V41_ATTN_EAGER=1 VA_V41_OPROJ_FP32_AR=0 VA_V41_OPROJ_FP32_PARTIAL=0`
  - 127 Run D：`VA_V41_OPROJ_FP32_AR=1 VA_V41_OPROJ_FP32_PARTIAL=1`（EAGER/ALIGN 关）
  - 两机树为 **131 整树 tar 同步 + 四文件 md5 复核一致**（127 是 01:xx 后首次接入，同步命令见文末）。
- **[04:0x 本日轮全部启动命令留档（用户要求）]**
  ```bash
  # ===== 引擎 dump（131，devices 8-15，全配方）=====
  bash -lc 'cd /workspace-verl/verl && \
    ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    HCCL_HOST_SOCKET_PORT_RANGE=60060-60110 HCCL_NPU_SOCKET_PORT_RANGE=61100-61150 \
    VERL_DSV41_ALIGN_ATTN=1 VA_V41_ATTN_EAGER=1 VA_V41_OPROJ_FP32_AR=1 VA_V41_OPROJ_FP32_PARTIAL=1 \
    nohup bash scripts/dsv41_consistency/sh/run_dump_engine.sh \
      --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer \
      --out-dir /mnt/share/m00899630/dsv41/dump/sft116real/engine \
      --lengths 64,200 --mem-util 0.45 > /tmp/dump_engine_sft116real.log 2>&1 &'
  # ===== 训练 dump（131，同配方）=====
  bash -lc 'cd /workspace-verl/verl && \
    ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    HCCL_HOST_SOCKET_PORT_RANGE=60060-60110 HCCL_NPU_SOCKET_PORT_RANGE=61100-61150 \
    MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer \
    OUT_DIR=/mnt/share/m00899630/dsv41/dump/sft116real/trainer LENGTHS=64,200 \
    VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1 \
    nohup bash scripts/dsv41_consistency/sh/run_dump_trainer_top32.sh > /tmp/dump_trainer_sft116real.log 2>&1 &'
  # ===== NaN sweep（126/127 通用；wrapper=run_dump_engine.sh 把最后 exec 行换成 /tmp/nan_sweep.py）=====
  ssh -p 50001 root@80.5.25.<126|127> "bash -lc 'cd /workspace-verl/verl && \
    SWEEP_LENGTHS=<表> SWEEP_TAG=<A|B|C|D> <开关们> \
    setsid nohup bash /tmp/run_nan_sweep.sh > /tmp/nan_sweep_<tag>.log 2>&1 &'"
  # ===== 对拍（CPU）=====
  python3 scripts/dsv41_consistency/compare_stages.py  --len 200 --trainer-dir $D/trainer --engine-dir $D/engine
  python3 scripts/dsv41_consistency/compare_logprobs.py --len 200 --trainer-dir $D/trainer --engine-dir $D/engine
  python3 scripts/dsv41_consistency/compare_routing.py --len 200 --trainer-dir $D/trainer --engine-dir $D/engine \
    --model-path /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-4layer
  ```
  （wrapper 生成：`sed 's#exec python3 scripts/dsv41_consistency/dump_engine_stages.py "\$@"#exec python3 /tmp/nan_sweep.py#' scripts/dsv41_consistency/sh/run_dump_engine.sh > /tmp/run_nan_sweep.sh`）
  - 树同步（rsync 两机均无 → tar-over-ssh）：`cd /workspace-verl && for t in FSDPTurbo verl vllm vllm-ascend-v41-private; do tar -cf - --exclude=__pycache__ --exclude=.git --exclude="*.pyc" --exclude=logs $t | ssh -p 50001 root@80.5.25.<ip> "tar -xf - -C /workspace-verl"; done`
- **[04:1x ★★★ bisect 判决]** Run C（126，只开 `VA_V41_ATTN_EAGER=1 + VERL_DSV41_ALIGN_ATTN=1`）：
  **NaN 全谱复现**（64/96/127 NaN；128/200 干净；random 64/100/127/128 全 NaN——含 128 反常点复现）。
  Run D（127，只开 `VA_V41_OPROJ_FP32_AR=1 + VA_V41_OPROJ_FP32_PARTIAL=1`）：**全干净**。
  ⇒ **NaN 由 eager 注意力补丁（`sparse_flash_mla_eager`）引入**（AR/PARTIAL 无罪）。
- **[04:2x NaN 现场捕获（stash）两路并进]**：
  - Run E（127，TP=8）：`VA_V41_ATTN_STASH_DIR=/tmp/attn_stash_E VA_V41_ATTN_EAGER_DEBUG=1`，
    lengths=64,128。stash 抓每层**前 2 次调用** = len64 的 prefill（call0）+ 第一次 decode（call1 = 嫌疑点），
    存 q/kv_sel/valid/sink/out → 定位 NaN 在 哪个张量（q / 合法 kv / sink / 输出）。
  - Run E2（126，**TP=1 单卡**）：8 rank 同写一个 stash 文件会互相覆盖 → 单卡版无竞争、归属明确；
    同时验证 NaN 是否 TP 无关。
  - `_select_sparse_indices` 读码：ids 来自 `attn.indexer.select(...)`（indexer 自身 key cache 的 top-k），
    eager 里 `ids≥0 → valid`、`safe//cmp_bs` 映射到长缓存 —— **嫌疑 = ids 指向"未写入槽位"（未初始化显存）
    且被当 valid 参与 softmax**；长度相关性待 stash 数据定。
- **[04:3x ★ stash 判决：eager 注意力输出全干净]** E2/E（TP=1 与 TP=8）每层前 2 次调用
  （len64 的 prefill + 第一次 decode）的 q / kv_sel（valid 与 invalid 分开算）/ sink / out **全 0 NaN**；
  valid 列数正常（decode L2: 65 window + 32 cmp = 97 ✓）。⇒ **NaN 不在注意力算子内**，在**注意力之后**的链路
  （且 dump 主 prefill 阶段的 182 张量此前也验过 0 NaN ⇒ 更可能是 **decode 步的后半链**）。
- **[04:4x ★★ F 组鉴别（单引擎 8 请求，eager+align）]**：
  | 请求 | 结果 |
  |---|---|
  | raw len64（dump 式 prompt_logprobs=5, max_tokens=1） | **尾部全有限**（last=-17.97） |
  | raw len65 = len64+[0] | **尾部 3 位全 NaN（含 62/63 位）** |
  | raw len67 = len64+[0]*3 | 尾部全 NaN |
  | raw len128 | 全有限（-15.83） |
  | gen len64 temp=0 logprobs=1 | tok=[0,0,0,0]，nan 8/8 |
  | gen len64 **temp=1** logprobs=1 | **tok=[0,0,0,0]，nan 8/8（温度无关！）** |
  | gen len64 **logprobs=0** | tok=[0,0,0,0]——**采样本身就在选 token 0**（非 logprob 记录问题） |
  | gen len128 temp=0 | 正常 |
  - 关键：(a) 同一 64-token prompt，**raw 路径干净、gen 路径坏**（请求形态不同 → 执行路径不同）；
    (b) 温度无关且 lp=0 也选 token 0 ⇒ **logits 在该步的主峰被破坏**（NaN 或退化为 0 峰）；
    (c) **+1 个 token（len64→len65 raw）行为就翻转** ⇒ 与"最后 token 的增量状态"强相关；
    (d) len128 的 gen 正常 ⇒ 上边界仍在 ~127。
  - ⇒ 下一步 = **Run G：hooks 挂在生成请求上跑**（extension 每 key 抓 2 次调用 = prefill + 首次 decode），
    逐模块扫 NaN，钉第一个坏张量。已点火（126，`/tmp/nan_probe_g.log`，out=`…/dump/nan_probe_g`）。
- **[05:0x ★★★ Run G 反转：hooks 挂载中 → 生成完全正常！]** 127 上（E 同机同 env 同 prompt，E=NaN）Run G
  （extension hook_begin 挂着跑生成）→ tok=[120631, 110263, ...] **真实 token、无 NaN**（219 张量 dump 均正常）。
  ⇒ **NaN 对执行时序敏感（竞态）**：extension 的逐模块 `.cpu()` 捕获 = 全程 device 同步 → 掩盖竞态。
  也解释了 dump 流程（hook_end 后生成）与 sweep（无 hooks）都 NaN、而带 hooks 的生成不 NaN。
- **[05:1x 候选机制修复尝试：补 `wait_for_device_metadata(ATTENTION)`]** —— 融合路径在
  `npu_sparse_flash_mla` 前有 `wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(smla_metadata))`
  （元数据流水线：compressor/indexer 任务异步构建，跨流），eager 路径**漏了**该等待（读 paged cache 可早于
  异步写入落盘）。已在 `_forward_attention_eager` 补镜像等待（dsa_v41.py，md5 a4bd5574→**d71066b7**），
  三机同步（131/126/127 对账一致）。
  - **结果：H/H2 双机重跑（127 mem0.85 / 126 mem0.35）→ NaN 谱系完全不变**（64/100/127 NaN，128/200 干净，
    random128 仍 NaN）⇒ **该等待不是（唯一）根因**；补丁保留（与融合路径对齐，属正确性修补）。
- **[05:2x Probe I：逐位置 NaN 地图]** raw 探针（prompt_logprobs）× 17 档长度 [36..200 含 44/65/72/96/126] +
  混合 batch（64+64 / 64+65 / 64+128 / 65+128）→ 输出每请求的 **NaN 行区间**：
  判定形状（尾 K 行？从某行起？全 NaN？）+ batch 依赖性。已点火 127（`/tmp/nan_probe_i.log`）。
  - 已知线索拼图：dump-len64 prefill 全干净；raw65（=64+[0]）尾 3 行（62..64）NaN；dump rescore（64+8=72）尾 8 行 NaN；
    len32 干净（其 8 个 decode 步 p=32..39 也全干净！）⇒ **边界跟"prompt 总长"绑定，不是 decode 位置**。
- **[05:3x ★★★ 图景重写（查 E 完整日志后）]"hooks 掩盖竞态"理论作废]** Run E（127）请求序 [f64, f128, r64, r100, r127, r128]：
  **前 3 个全干净**（f64#1 tok=120631...、f128#2、**r64#3 干净！**），**从 r100#4 起才是 NaN**（含 r128#6）。
  Run G 干净只是因为**它只有 1 个请求**（=E#1）。⇒ **变量 = 请求历史（跨请求状态污染）**：
  - E：3 个干净后翻转；A（131）：f32#1 干净、f40#2 起 NaN、**f128#12 又干净**（f128 偶发免疫）；
  - 同一 prompt（r64）在 E#3 干净、A#13 NaN ⇒ **逐请求判定由历史状态决定**，长度/内容只是二阶因素。
  - 修正后的候选机制：**跨请求共享缓冲残留**（indexer key-cache 的 valid 计数/槽位、`shared.topk_indices`、
    c2 ring 状态）→ 后续请求的 ids 可能指向**非本请求的陈旧/未写槽位**；融合核用 `seqused_cmp_kv/cmp_seq_lens`
    天然屏蔽越界 id，而 **eager 只信任 `ids≥0`** → 读陈旧显存 → NaN（形状与"历史敏感、内容无关"吻合）。
- **[05:4x Probe K：污染请求现场捕获]** extension `hook_begin(max_calls=64)`（默认只 2 → 提上限），
  序列 `[f64, f128, r64, r100, f64]` 复现翻转；逐调用（每层每请求 1 prefill+8 decode）扫 NaN，
  定位**被污染请求（r100）** 的第一个坏张量。**双机并行**（127 原版 / 126 mem0.35 复本）。
- **[06:0x ★★★ K 结果（双机逐位一致）]** 序列复现：f64#1✓ f128#2✓ r64#3✓ → **r100#4 NaN** → **f64#5 NaN**（污染持续）。
  2903 张量捕获；rank0 扫描（call=该 key 的第 i 次调用，每请求 9 次 = prefill+8 decode）：
  - **首个 NaN = call24 = r100 的 prefill，`_forward_attention@layer2` 输出（20000/409600 部分 NaN）**；
    同 call 它的**输入全干净**（input_layernorm/q/select/多流预处理的捕获均无 NaN）⇒ **NaN 诞生在 layer2 注意力内部**。
  - layer2 输出部分 NaN →（o-proj 全维混合）→ self_attn/mlp 输出全 NaN → layer3 全链 NaN（级联）。
  - **NaN 结构极规整**：r100 prefill 的注意力输出 = **100 行 × 每行恰好 200 个 NaN**（行间同 dim 集；
    行内 NaN 位如 flat [10,22,46,52,54,94,...4004,4008,4016,4028,4074]）；后续 decode 每行 NaN 数 **176→168→160→152**
    （每步 −8，8=head 数！）⇒ 形态 = **"被所有行共同 attend 的 kv 条目的特定通道为 NaN"**
    （kv 值 NaN→全行同 dim NaN；−8/步 = 每 head 少 1 个 NaN dim 的规律待 NANDBG 揭）。
  - **NANDBG 补丁**（`VA_V41_ATTN_EAGER_NANDBG=1`，dsa_v41.py）：在 gather 后检测 kv_sel NaN，打印 NaN 列的来源
    （`win(abs=位置)` 或 `cmp(id=…)`，含 valid 标记、q/sink NaN 数）→ 直接指认坏槽位是"窗口位置还是压缩 id、id 值多少"。
    md5 db44f8ee，已同步 127；K 序列带 NANDBG 重跑中。
- **[06:2x ★★★★ NaN 根因 + 修复（收官）]** NANDBG 判决（K 序列重跑）：
  `layers.2 @ p=101: row0 ncols=16 valid=0 cols=[102..110] ['win(abs=102)'..]` ——
  **NaN 列全部是窗口的"未来列"（abs > p，valid=0）** = **本请求尚未写入的 paged-cache 槽**（内存里是 NaN 垃圾）。
  首毒在 r100 prefill 的输出（部分 NaN）⇒ 同一机制：
  - **机制**：eager 把整窗 gather（含未写入/复用槽，垃圾含 NaN 位型）→ softmax mask 只把**概率**压成 0 →
    **PV einsum 里 0 × NaN = NaN** 把输出污染（行间同 dim 的规律 NaN，200/行；−8/步 的规律随窗口推进）；
    下一层输入污染 → 全链级联。融合核按"列范围截断"根本不含未写列 ⇒ 全 off 干净；首请求块的垃圾恰好是零页
    ⇒"首请求干净"伪规律；块分配垃圾内容动态 ⇒ 长度/内容伪边界。
  - **修复**（dsa_v41.py，md5 a4bd5574→d71066b7→db44f8ee→**dd08e172**）：
    gather 组装后 `if not vc.all(): kv_sel = kv_sel.masked_fill(~vc.unsqueeze(-1), 0)`——
    **被掩码条目的值清零**（NaN 无法再过乘法）。同步 126/127 对账一致。
  - 验证（双机）：127 K 序列 [f64,f128,r64,r100,f64]（期望全干净 + NANDBG 静默）；
    126 原位谱系 [32,40,64,100,127,128,200]（期望全干净）。
- **[06:3x ★ 验证 PASSED（双机）]** 127 K 序列：5/5 请求全干净（r100#4 恢复真 token、f64#5 与首请求逐位一致、
  NANDBG 零输出）；126 原位谱系 + 随机对照：**全 nan=0/8**（旧"杀手" 64/100/127 全真 token）。
  ⇒ **NaN 类彻底关闭**（机制收敛于 0×NaN；修复=被掩码条目值清零）。后续：该修复即任务4 重跑的引擎前提。

### [2026-10-13 傍晚] 修改 4：注意力核 eager 小算子（`VA_V41_ATTN_EAGER=1`）
**语义钉定（读码结论，均有出处）**：
- trainer 数学（`FSDPTurbo/.../ops/cpu/sparse_attention.py`，trainer 实跑的就是这个 fallback）：
  每 query 行 gather 选中 KV（窗口在列 0..127 = 绝对位置 [t-127,t]，压缩 id 列在 ≥128 且编号 = S+idx）
  → QK einsum **bf16 舍入** → fp32×scale → mask(-inf) → 拼 sink 单次 softmax（fp32）→ probs **bf16 舍入** → PV。
- 引擎因果部署路径：`ori_sparse_indices=None` + `ori_mask_mode=4` + `ori_win_left=window-1`（稠密窗口掩码，
  见 `dsa_v41.py:966-986`；非因果才走 `build_dspark_swa_indices`）；压缩段 `cmp_mask_mode=3`、`ratio=2`
  （`cmp_seq_lens=floor(S/2)`、residual=S%2，`dsa_v41.py:936-943`）；压缩 id 0 基（与 trainer 集合 100% 一致，
  见 `compare_sparse_indices.py`）。kernel 的地址约定 = **逻辑位置 p，block_table[p//block_size]**（出处：
  `dsa_v1.py build_dspark_swa_indices` 的 `use_logical_indices` 文档）；缓存寻址 slot=block*block_size+offset
  （`scatter_cache_sk`，plane 是 `[blocks, block_size, 1, D]` PA_BBND，store 走 2-D (block, offset) 索引）。
- kernel C++ 源码在位（`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/.../sparse_flash_mla/`），
  已派子代理提炼精确 dtype/舍入说明书（swa/csa 两路 + merge 语义）。

**实现**：`vllm_ascend/attention/dsa_v41.py`
- 新增 `sparse_flash_mla_eager()`（模块级，~110 行）+ `AscendDSAV41Impl._forward_attention_eager()`
  + `_forward_attention` 门控 `VA_V41_ATTN_EAGER=1`（默认关，非因果路径会显式报错而非静默算错）。
- 语义：因果窗口列序与 trainer 一致（有效在前、-1 在后）；ori 先 cmp 后（trainer 同序）；两次 bf16 舍入同 trainer；
  token 分块（`VA_V41_ATTN_EAGER_CHUNK`，默认 256）控显存。
- stash：`VA_V41_ATTN_STASH_DIR` 存前 2 次调用/层的 (q, kv_sel, valid, sink, scale, out)，供离线复算。

**验证进展**：
- **[F-79 坑]** engine 的 paged plane 是 24-51GiB 非连续视图，torch_npu 对它的花式索引整表拷贝（两次 dump OOM，独立复现）
  → 修复 = 逐块视图 gather（`plane[b]` + 小块单轴索引）；已在 131 单卡验证正确+内存有界。
- 18:08 首跑（131 引擎 dump，`run_dump_engine.sh` + eager）→ **crash：einsum 4-D**（plane 布局误判为
  flat [slots,1,D]；实为 [blocks, block_size, 1, D]）→ 修（reshape(-1, D) 兼容两种）→ 18:22 重启 dump。
- 下一步：dump 完 → ①引擎 eager attn 输出 vs trainer（align3）attn 输出（期望残差从 0.00137 大幅下降）；
  ② vs 融合核引擎 dump（slice4_ar2）= 被替换掉的差异量；③ 达标 → 131+119 跑 1 步 RL 验收。

### [验证结果 18:45-19:00] 注意力核 eager 对拍（len200 切片，8 rank 拼 64 头，contig 映射）
头映射修正：**contig**（rank r = heads [8r:8r+8]），引擎 q vs trainer q **rel=0.0000**（HC_BF16 修复后逐位级一致）。

| L | eps_q(输入差) | eager_vs_trainer | fused_vs_trainer | 结论 |
|---|---|---|---|---|
| 0 | 0.00002 | **0.00010** | 0.00133 | **eager = trainer 数学（13× 改善）** |
| 1 | 0.00274 | 0.00304 | 0.00321 | 输入差主导（上游 MoE/norm 残差） |
| 2 | 0.00822 | 0.00798 | 0.00774 | 同上 |
| 3 | 0.01310 | 0.01185 | 0.01173 | 同上 |

- **关键**：L0 输入几乎相同（2e-5）时，融合核 vs trainer = 0.00133，**eager = 0.00010（注意力数学差清零到噪声底）**。
  这是"融合核 → 小算子"替换有效的决定性证据；L1-3 的残差来自上游输入差（不是注意力本身），换核不再有可见增益。
- 实验细节：align3 trainer 用 `VERL_DSV41_ALIGN_ATTN=1`（分数/概率 fp32）→ 引擎侧必须同一开关（已镜像；
  第一版 dump 没带该开关，eager 反而 0.0026 > 融合核 0.0013——修正语义后达标）。
- **1 步 RL 验收跑已启动**（19:00，131+119，全配方 HC_BF16+ALIGN_ATTN+MOE_BF16+AR=1+ATTN_EAGER=1）。

### [1 步 RL 结果 19:45] pearson 0.9581 / diff_mean 0.0602 —— **未达标（<0.985），且落在噪声带内**
- 跑：131+119 全配方 + eager（gen 535s，较基线 353s 慢 52% = eager 生效的侧证；**下次务必带
  `VA_V41_ATTN_EAGER_DEBUG=1` 留直证**）。step1 后 driver 退出（total_training_steps=1）时
  DataLoader worker 被外部 SIGKILL（`ERR99999`，环境老问题，不影响 step1 数据）。
- **★ 噪声带发现（策略级）**：任务2 原始代码 9 步 pearson = 0.951/0.9655/0.9672/0.9684/0.9645/0.9640/0.9750/0.9672（首步 0.9619）
  → **run-to-run σ≈0.007、极差 0.024**。单算子改善 ~0.001-0.005 量级完全被淹没。
  ⇒ ①"0.9630 → 0.9581"不构成回退证据；②后续 A/B 必须**同批/多步均值**口径，单跑 1 步不可判。
- **新主攻方向（★ decode↔prefill 通道，今晚量定）**：用引擎自带 `engine_decode_len200.json`（decode 生成 8 token）
  对 `prefill_logprobs` 逐 token 求差：
  | build | 每 token Δ（8 个） | mean\|d\| |
  |---|---|---|
  | 今晚 eager+ALIGN | [0.058, 0.056, -0.000, 0.004, 0.001, -0.054, 0.010, -0.039] | 0.0277 |
  | ar2 融合核 | [-0.004, 0.059, -0.002, -0.052, 0.065, -0.057, -0.042, 0.001] | 0.0352 |
  | slice4 融合核 | [0.000, -0.002, -0.001, 0.000, 0.061, 0.012, 0.063, 0.007] | 0.0183 |
  → **通道真实存在：多数 token ~0.00x、少数 token ±0.04-0.065**（随机位置，跨 build 都出现）；
  量级 ≈ RL diff_mean 0.06 的 **40-60%**。⇒ **diff_mean<0.01 的验收在通道未消前不可能达标**（与算子对齐无关）。
- **修法线索（已定位代码）**：rollout 的 `log_probs` 取自 `final_res.outputs[0].logprobs`（
  `vllm_async_server.py:730`，**decode 步捕获**）；同文件 `extract_prompt_logprobs(...prompt_logprobs...)` 是
  **prefill 模式**logprob 设施（`utils.py:574`，含 `num_prompt_logprobs=0` 只取采样 token 的分支）——已存在但
  当前未作为 rollout 记录源。⇒ 下一步实验 = 让引擎以 prefill 模式记录 rollout logprobs（对 [prompt+response]
  整段做一次 prefill 取各位置采样 token 的 logprob），把通道从指标里消掉。
- **次主攻（出生点）**：L0 attn 输出差 0.0001（干净）→ L1 输入 q 差 0.0027 ⇒ 发散出生在 L0 的
  attn-out→o-proj→ffn/MoE→norm 段；子段对拍需分片感知（引擎 EP=8 路由 [25,6] vs trainer [200,6]），
  走既有 graft/probe 工具（待排期）。
- **★ 跨栈 logprob 现状（今晚新测，当前全配方 eager+ALIGN+HC+MOE_BF16+AR）**：
  trainer `next_logprobs`（l0birth dump 顶层，199 位）vs 引擎 prefill `scored`（attneager_al json）：
  **mean|d|=0.0362 / std=0.0950 / max=0.742；|d|>0.5 有 2 位、>0.1 有 8 位、>0.02 有 87 位**；
  199 位 token id 完全一致；位置 pearson=0.99959。（对照 align3 无 eager：0.0410/0.1032 → eager 改善 ~12%。）
  ⇒ **验收 diff_mean<0.01 的分解**：跨栈 ~0.036 + decode 通道 ~0.028 —— 两项都要攻：
  ①主攻仍是 MoE 段出生点（把 0.036 的长尾压下去；87 个 >0.02 的位是主体，翻转只占 2 位）；
  ②decode 通道（引擎侧 prefill 记录 rollout logprobs，线索见上）。
- **★ L0 出生点钉定（当晚实测，l0birth trainer + attneager_al engine，精确键+rank 拼接）**：
  | 锚点 | 数字 |
  |---|---|
  | 注意力核输出（eager） | 0.0001 |
  | **ffn 输入（engine 路由输入 vs trainer ffn_norm）** | **rel=0.00282（28× 放大！）** |
  | 路由 | ids 集合一致 **100%**、顺序 96%；weights max\|Δ\|=0.011（mean 1e-4） |
  | MoE 输出（routed+shared vs engine 融合核） | rel=0.00468，per-token med 0.0046/p90 0.0053（**均匀 ULP 型，1.7× 增益**） |
  | ffn 块输出 | 0.00476 |
  ⇒ **出生点 = 注意力尾段（_project_output 的 o-proj+AR+hyper-connection/残差路径）的核级 ULP 地板**：
  attn 核已 0.0001，但到 ffn 输入已 0.0028（≈ 旧测 o-proj 段地板 0.0030，HCCL bf16-AR 0.0035 已在 AR=1 时消掉，
  剩余疑为 o-proj GEMM/hc 混合的另一种实现差）；MoE 本身干净（1.7×，非放大器）。
  **下一步候选**：① 复核 `oproj_rope_parity_microbench.py` 已有结论（GEMM=F.linear 等价 vs 残余）；② 引擎侧
  o-proj / hc 混合的小算子替换（同今晚注意力核思路）——把 28× 放大器按 trainer 数学重写。
- **★ 28× 放大器的真凶锁定（当晚 microbench + 代码链）**：
  - wo_a 形式差**排除**：`oproj_rope_parity_microbench.py` 实跑（131 单卡）——trainer einsum vs 引擎
    `npu_transpose_batchmatmul`：**99.99% 位同、rel=2.7e-5**（两者对 fp32 参考同样 1.66e-3）。
  - 反 rope 此前已验证（VA_V41_TORCH_ROPE 逐位）；AR 已 fp32（VA_V41_OPROJ_FP32_AR=1）。
  - ⇒ **剩余 ~0.0047 的机制 = 引擎 wo_b 的 8 个 per-rank partial 是 bf16 值**（local GEMM 输出即 bf16 舍入），
    fp32-AR 只修了"归约域"、没修"partial 本身已被舍"；trainer = **单个 K=8192 fp32 累加、只舍一次输出**。
    8 处 bf16 partial 舍入 vs 1 处 ≈ 0.4%（1 ULP）→ 与实测 0.0047 吻合。
  - **补丁计划**：`VA_V41_OPROJ_FP32_PARTIAL=1`（扩展现有 fp32-AR 分支）——local wo_b 走 fp32 输入 GEMM
    （`F.linear(x.float(), w.float())`，bf16 升 fp32 无损）→ fp32 partial → 现有 fp32 AR → 单次 cast。
    预期把 L0 attn 尾 0.0047 → ~1e-4 级（即把 attn 段真正清零）。
- **★ 补丁实测（slice4_oprojfp32 dump，当晚）**：
  | 指标 | AR-only | **+FP32 partial** |
  |---|---|---|
  | **L0 ffn 输入 vs trainer** | 0.00282 | **0.00088（3.2×）✅ 机制假设证实** |
  | logprob vs trainer（mean\|d\|） | 0.0362 | 0.0335（−8%）std 0.095→0.072 |
  | logprob 尾部 | >0.5:2 / >0.1:8 / >0.02:87 | 1 / 12 / 94（翻转位是噪声重投） |
  | engine self_attn OLD vs NEW | — | L0 0.0025 → L3 0.019（逐层改写轨迹） |
  ⇒ **机制赢、传播输**：o-proj 尾修好后，L0 的 binding floor 变成 **MoE 自身的 ULP 地板（0.0047，与输入无关）**；
  层间每层 ~2-3× 复利 → 4 层到 logprob 仍有 0.033。
  **结论：逐算子修法余量见底**——剩余 = 每层 MoE/combine"同 bf16 链、不同实现/顺序"的 ULP 地板（≈0.005/层）
  + 近-tie 翻转尾部。再往下的路只剩**共用内核**（引擎侧 MoE eager 大工程，或 trainer 改用引擎核）。
  这是**范围决策点**，需用户拍板（见汇报）。
- **★ (C) 方案实现（当晚）**：`VA_V41_PREFILL_LOGPROBS=1`（verl/workers/rollout/vllm_rollout/vllm_async_server.py，
  generate() 里 log_probs 捕获后）——对 [prompt+response] 以单 prompt 重跑一次取 `prompt_logprobs`
  （prefill 模式、与 trainer 重算同模式），替换 decode 时序 logprob；失败自动回退 decode 值。
  消费链已核实（llm_server.py:304 → rollout 输出）。**验证跑已启动**（131+119，全配方+FP32_PARTIAL+PREFILL_LOGPROBS）。
  - 启动（20:21，两机同 env）：`VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1
    VA_V41_OPROJ_FP32_AR=1 VA_V41_OPROJ_FP32_PARTIAL=1 VA_V41_ATTN_EAGER=1 VA_V41_PREFILL_LOGPROBS=1
    bash scripts/dsv41_deploy/launch_grpo_{master 1|slave 80.5.25.131}`
  - master 日志 `/tmp/grpo_2node_master_20261013_202118.log`；slave ready ✓。判读：diff_mean 应从 0.060 向 ~0.035-0.04 掉
    （通道 ~0.028 + o-proj 尾改善），pearson 期望上移（噪声 ±0.01 内判读用多值）。
- **★★ [F-80] 20:21 跑结果作废 + 重大流程坑**：该跑出 step1 但 **metrics 全 NaN**；排查中做 md5 对账发现
  **119（引擎所在节点）的四个相关文件全是旧版**（无 eager/无 PARTIAL/无 PREFILL/HC_BF16 也缺）——
  **19:0x 的"0.9581"与 20:21 的"NaN"都不是新补丁的结果**（引擎从未加载过今晚的补丁）。
  - 已 scp 四文件到 119 并 md5 复核一致（`dsa_v41.py`/`dsa_v1.py`/`model.py`(FSDPTurbo)/`vllm_async_server.py`）。
  - 失败库已立项 **F-80**；教训同 [[dsv41-npu-stack-sync-so-deps]] 家族：**验证前必须对引擎所在节点对账**。
- **[21:13 重跑]** 真·全配方（四文件已同步）：131+119，同 env（HC_BF16+ALIGN_ATTN+MOE_BF16+
  FP32_AR+FP32_PARTIAL+ATTN_EAGER+PREFILL_LOGPROBS），日志 `/tmp/grpo_2node_master_20261013_211338.log`。
  PREFILL 补丁带一次性 debug warning（plp 长度/取样值/NaN 计数）→ 顺带判别 NaN 机制。
- **[21:13 结果 22:0x] 仍 NaN（pearson/diff_mean=nan，rewards -0.81 正常，gen 1283s）+ PREFILL 诊断行全日志流缺席**
  → 补丁块疑似未执行。逐层排查：
  - 代码落点 ✓（`generate` 唯一一份，插在 logprobs 分支内）；消费链 ✓（llm_server→agent→rollout_log_probs）。
  - **发现层：vLLM server actor 用 `runtime_env={"env_vars": 白名单}` 创建**（1428 行，仅 platform noset + rollout_env_vars）——
    `VA_V41_*` 是否随 raylet env 继承取决于 Ray 合并语义（本 fork 文档称 worker 从 raylet env spawn，应继承）。
  - **双保险已加**（22:0x）：① 白名单显式并入 `VA_V41_*/VERL_DSV41_*`（os.environ 透传）；② `generate` 顶部加
    **无条件一次性 env 探针**（打印 PREFILL/EAGER/PARTIAL/AR 值与 logprobs 请求标志）。
- **[22:03 探针跑]** 131+119 已启动（同 env）；首请求即出 `[VA_V41 env probe]` 行 → 用它对账"env 是否到引擎"，
  并观察 NaN 是否随补丁真启用而消失/复现。日志 `/tmp/grpo_2node_master_20261013_220352.log`。
- **[22:5x 重大实锤]** Ray runtime_env **替换/覆盖语义**：server actor 的 env = 白名单+继承合并；**passthrough 开启后
  runtime_env_agent 日志实见 `VA_V41_ATTN_EAGER/PREFILL/OPROJ_*` 全量注入（23:04）**——
  **此前所有双机 RL 的引擎侧 env 开关（含 AR 补丁）从未生效**（历史"一致性优化"在 RL 里全部是空跑；仅单机 dump 生效）。
  探针行仍未现（两处 probe 均未打出）→ 待活捕排查（列下轮第一项）。
- **[23:0x 22:52 跑结果]** gen 正常跑完但 **update 阶段 OOM**（rank1/HcclAllgather；疑 FP32_PARTIAL 引擎侧 fp32 权重复本
  抬高峰值）→ **缓解：去掉 FP32_PARTIAL 重跑**（23:35 起，其余全开：HC+ALIGN_ATTN+MOE_BF16+AR+EAGER+PREFILL）。
- **[00:3x 23:35 跑结果]** 无 OOM ✓（证实 PARTIAL 是 OOM 源：update 132s 正常）但 **NaN 依旧**：
  定位到 **rollout 侧**（`rollout_ppl`/`rollout_log_ppl`=nan，**trainer 侧 `training_ppl`=30.9 正常**）；
  **PREFILL 调试行 0 次、env 探针 0 次**（在整轮 gen 之后）——补丁块从未执行（代码在模块里、请求走 `server.generate.remote` ✓）。
  ⇒ 请求路径与"我 patch 的 `vLLMHttpServer.generate`"错位（机制待活捕）；NaN 源在引擎侧的真实 logprob 路径。
- **[00:24 判决跑]** 去掉 PREFILL（保留 EAGER+AR+训练侧全开）→ 判据：①若 NaN 消失 = PREFILL 补丁相关；
  ②无论如何，**gen 期用 py-spy 活取 119 引擎进程栈**（找真实 serving 路径）与探针行对照。
- **[00:5x 流程发现]** 之前反复"探针缺失"的一大原因：**每次重启前我在 119 上 `rm -rf /tmp/ray`，
  把上一跑 worker 的日志（探针应出现的地方）也清掉了**——从没搜过正确位置。
  静态链已核尽：`vllm → _load_vllm → vllm_async_server.vLLMReplica → launch_servers（探针就在其中）→
  ray.remote(vLLMHttpServer) → 客户端 `server.generate.remote`（我的 generate）`。**结论：下轮 gen 期就地读取
  119 当前 session 的 worker-*.out**（不再提前清理），配合 py-spy 活栈，一次性收口。
- **[01:1x 用户叫停，全部实验终止]** 收尾状态：
  - 判决跑（131+119，无 PREFILL/PARTIAL）**00:59 被外部杀**（无 step、无 traceback——同"机器杀"家族），无结论；
  - A/B（128+129 干净树）首启撞 F-77（忘带 RAY_PORT=6390，外部 GCS 再占 6379）→ 重跑后按用户指令中止；
  - 今日 3 个 NaN 跑（20:21/22:52/23:35）+ 1 个 OOM 跑（22:52）+ 2 个被杀跑，**NaN 根因未收口**（rollout 侧
    logprobs 全 NaN、trainer 侧正常；补丁执行与探针可见性两个疑点都指向"实际 serving 路径/日志链路≠静态链"）。
  - **下一步（新会话开局项）**：① 干净配对下用最简变量法定 NaN（先关 EAGER，后关 AR，再关训练侧三开关）；
    ② 活捕 server 进程（gen 期 py-spy + 119 当前 session worker 日志，勿提前清理）；③ 13:00 前 131 的卡被
    16×`ray::WorkerDict`（~25GB/卡、非本命名空间可见、疑邻居）占用——按规则等待。


