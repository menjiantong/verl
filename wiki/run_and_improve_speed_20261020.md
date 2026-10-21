# DSV4.1 Flash top32 noengram · GRPO 性能优化与 MFU/Profiling 报告

> 工作目录：`/mnt/share/m00899630/workspace/dsv41-verl-copy`
> 集群：80.5.21.110–116（Kunpeng 920 + Ascend 910，8 NPU × 2 chip = 16 chip/节点，64 GB HBM/chip）
> 容器：`dsv41-mjt-1019`（privileged / host IPC / host network，`/mnt` 挂载）
> 日期：2026-10-20
> 本文所有结论均给出：命令 + 日志/采样证据 + 实验对照。无证据的推断会明确标注“待验证”。

---

## 0. 目标

1. 用 `dsv41_grpo_2node_new.sh` 的单机模式在 80.5.21 段持续运行（不存 checkpoint）。
2. 分析并优化 step 耗时，核心是 `update_actor`。
3. 计算 MFU、采集推理/训练 profiling，输出可复算的公式、数值来源与 profiling 分析。
4. 用多机并行做 A/B，所有优化必须有对照实验与数据证据。
5. 优化后再次采集 profiling/MFU，与基线对比。
6. 单独跑 FSDPTurbo 基准，量化“纯训练”性能。

---

## 1. 环境与启动改造（相对原脚本）

原脚本 `verl/my_scripts/dsv41_grpo_2node_new.sh` 面向 80.5.25.x，直接在 80.5.21 段运行会踩两个坑：

| 问题 | 原行为 | 改造 |
|---|---|---|
| `MY_IP` 只 grep `^80.5.25.` | 在 21 段得到空值，HCCL/GCS 绑错 | 改为 grep `^80.5.21.` |
| `single` 模式 `pkill -9 -f "VLLM::|ray::|vllm serve"` | 会杀掉同节点其他 agent 的进程 | 去掉全局 pkill；只用 `RAY_TMPDIR` 隔离的 `ray stop` |
| `RAY_PORT=6390`、HCCL 60000/61000 | 与现存其他 agent 会话（GCS .102:6390、.109:6766）冲突 | `RAY_PORT=6479`、`RAY_TMPDIR=/tmp/ray_dsv41_perf_copy`、HCCL 62000-62050 / 63000-63050 |
| `MIRROR_ROOT` 指向 `dsv41-verl` | 与其他 agent 工作目录冲突 | 指向 `dsv41-verl-copy` |
| 默认权重是 4layer SFT | 不是本轮目标 | 默认 `DeepSeek-V4.1-Flash-top32-bf16-noengram` |

生成脚本：`verl/my_scripts/dsv41_grpo_1node_21.sh`（由原脚本 patch，保留 master/slave/join 逻辑）。
另加 `${TRAIN_EXTRA_ARGS:-}` 透传，用于 profiling 与 A/B 的 Hydra 覆盖。

启动命令（节点 110 基线）：

```bash
docker exec -d dsv41-mjt-1019 bash -lc '
cd /mnt/share/m00899630/workspace/dsv41-verl-copy/verl/my_scripts
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16-noengram
export EXPERIMENT_NAME=GRPO-DSV41-noengram-perf-baseline-20261020_230207
nohup bash dsv41_grpo_1node_21.sh single 100000 \
  > ../logs/perf_baseline_20261020_230207.outer.log 2>&1 < /dev/null &'
```

---

## 2. 运行矩阵（最终）

| 节点 | 运行 | 配置差异 | steps | 日志 | 结果 |
|---|---|---|---|---:|---|
| 110 | 基线（GSM8K） | 原脚本：KL=1, bsz 8×8, resp 1024, micro=1, recompute FFN, offload 全开, util 0.5 | 5 | `verl/logs/perf_baseline_20261020_230207.outer.log` | step1–5 全量指标；update 270.9→198.2→207.0→216.6s |
| 111 | 基线 profiling | 同上 + `global_profiler.steps=[2]` rank0 level1 analysis=True | 3 | `verl/logs/perf_profile_20261020_230432.outer.log` | DB 1.66GB；actor_update 210.0s |
| 111 | **countbenchqa 长跑** | 工作日志配方：countbenchqa_lite 450/50，VLM，bsz 8×8，resp 512，KL=0，util 0.5，expandable_segments | 500 | `verl/logs/countbenchqa_optlong.outer.log` | 首轮 VLM logprob bug 已修复，重启中 |
| 112 | 关 empty-cache A/B | `VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=0` | 3 | `verl/logs/perf_opt_noemptycache.outer.log` | 稳态 2.740 vs 基线 2.769（≈0） |
| 113 | FSDPTurbo 独立基准 | 无 rollout/offload，batch=1×1024 + FFN recompute | 6 | `logs/fsdp_bench4_20261020_234104.log` | 3.5–3.7s/step，峰值 23.46GB，MFU 7.9% |
| 113 | E5 全无 offload | param/optimizer/offload_policy 全 False | 3 | `verl/logs/perf_opt_full_nooffload.outer.log` | update_actor OOM |
| 113 | E6 参数常驻 | param=False, optimizer=True, policy=False | 3 | `verl/logs/perf_opt_paramresident_optcpu.outer.log` | update_actor OOM |
| 113 | **组合优化** | expandable_segments + AdamWSwap | 3 | `verl/logs/perf_opt_combo_expand_adamwswap.outer.log` | 2.862/2.550/2.385 s/1k（step1–3） |
| 113 | 组合 profiling | 同上 + analysis=True, steps=[2] | 2 | `verl/logs/perf_opt_combo_profile_db.outer.log` | 进行中 |
| 114 | AdamWSwap A/B | AdamWSwap，optimizer_offload=False | 3 | `verl/logs/perf_ab_adamwswap2.outer.log` | 3.367/2.903/2.529 s/1k |
| 114 | 组合 profiling（无 DB） | 组合 + analysis=False | 3 | `verl/logs/perf_opt_combo_profile.outer.log` | step2 2.579 s/1k；仅原始 trace |
| 116 | util0.4+micro2 | util=0.4 + micro=2 | 3 | `verl/logs/perf_opt_util04_micro2.outer.log` | OOM（49.71GB allocated） |
| 105 | 推理统计/独立 vLLM | 1-step RL（vLLM stats）+ TP8/EP8 serve + profiler | 1 | `verl/logs/perf_infer_stats.outer.log`、`logs/vllm_serve_infer_105.log` | gen 390.4s/69,998 tok；decode 159.6 tok/s |

NPU 采样：节点 110 `logs/perf_baseline_monitor_20261020_230207.log`（每 5 s，npu-smi 全 16 chip）。

---

## 3. 硬件标定（实测，节点 114）

命令：`docker exec dsv41-mjt-1019 python logs/bench_hw.py`

| 项目 | 实测 | 说明 |
|---|---:|---|
| 设备名 | `Ascend910_9382` | npu-smi 显示 `Ascend910`，HBM 65536 MB/chip |
| BF16 matmul 4096³ | **318.8 TFLOPS** | `2n³/t`，warmup 5，iter 20 |
| BF16 matmul 8192³ | **300.1 TFLOPS** | 同上 |
| BF16 matmul 16384³ | **312.0 TFLOPS** | 同上 |
| H2D 1 GB pageable | **55.45 GB/s** | `npu.copy_(cpu)`，同步后计时 |
| D2H 1 GB pageable | **39.94 GB/s** | 同上 |
| H2D/D2H pinned | 54.48 / 40.65 GB/s | 与 pageable 基本相同 |

结论：
- 单芯片 BF16 实测可达 ~300–319 TFLOPS，verl `_DEVICE_FLOPS["910B"]=354 TFLOPS` 作为理论峰值合理（实测为峰值的 85–90%）。
- 主机↔设备拷贝带宽约 40–55 GB/s/chip；全节点 16 chip 若同时拷贝会受主机内存带宽限制，但单步 offload 的数据量（~0.4 TB）理论上仅需秒级，**不足以解释 update_actor 中 ~98 s 的 AICore=0**。因此需要 profiling 定位是 CPU optimizer、逐参数小拷贝还是同步等待。

---

## 4. MFU 公式与数值来源

### 4.1 训练 MFU

```
MFU = achieved_FLOPS / (n_devices × peak_FLOPS_per_device)
achieved_FLOPS = FLOPs_per_step / step_time
FLOPs_per_step = 6 × N_activated × T + attention_core_FLOPs
```

- `6 × N × T`：前向 2N、反向 4N（matmul 主导），T 为该阶段有效 token 数。
- `N_activated`：**每 token 实际参与计算的参数**（MoE 只算 top-k 专家，不算全部专家）。
- `peak_FLOPS_per_device`：verl 表值 354 TFLOPS（BF16）；实测 matmul 300–319 TFLOPS，报告中同时给出两者。

本模型（从 checkpoint 实际 shape 统计）：

| 项 | 数值 | 来源 |
|---|---:|---|
| 40 层注意力线性参数 | 126.61 M/层 | `q_lora=1280, head_dim=512, 64 heads, o_groups=8, o_lora=1024`，与 `layers.N.attn.*` shape 逐项核对 |
| MoE 激活参数/层 | 247.70 M/层 | `(top6 + shared1) × 3 × 5120 × 2304` |
| gate/层 | 0.164 M/层 | `32 × 5120` |
| 每层激活参数 | 374.48 M | 上述三项之和 |
| 40 层 + embed/head | **16.30 B** | `40×374.48M + 2×129280×5120` |
| 全量可训练参数（含全部专家） | **53.2 B** | 264.85B(总 ckpt) − 196.61B(Engram 表) − 14.23B(MTP) − 0.49B(vision/aligner) |
| Engram 表 | 196.61 B | `layers.1/14.engram.embed` 2×98.3B，`engram_layer_ids=[]` 不参与训练 |
| MTP/nextn | 14.23 B | adapter `n_mtp_layers=0` 显式关闭 |
| vision+aligner | 0.49 B | `include_vision=False` |

注意：若直接用全量 53.2B 计算会高估训练 FLOPs 约 3.3×；MFU 报告必须用 **16.30B 激活参数**。

### 4.2 推理 MFU

```
prefill: FLOPs = 2 × N_activated × T_prefill + attention_core
decode:  FLOPs = 2 × N_activated × T_decode + attention_core
MFU_infer = FLOPs / (n_devices × peak)
```

decode 阶段每 token 的 attention 需要读 KV cache，属带宽受限，MFU 天然远低于 prefill；报告中会分别给出 prefill/decode 的 token/s 与 MFU。

### 4.3 本集群实测 MFU（节点 110 基线 step3 / 节点 113 组合优化 step3）

计算脚本：`worklog/compute_mfu.py`（公式与 `verl/utils/flops_counter._estimate_deepseek_v41_flops` 一致）。

- `N_activated = 16.304 B`：`[(126.61M + 0.164M + 247.70M) × 40 + 2×129280×5120] = 16.304B`，逐项来自 checkpoint `config.json`。
- 注意力 core：`3 × Σ L_i·min(L_i, 640) × 2D × NH × L`，`index_topk=512 + sliding_window=128 = 640`。
- 序列长度用 batch 均值近似（T/64）。

| 阶段 | tokens | 时间 s | PFLOP | achieved TF/s | MFU（÷354×16） |
|---|---:|---:|---:|---:|---:|
| 基线 update_actor | 71,585 | 198.19 | 7.363 | 37.15 | **0.656%** |
| 基线 old_log_prob（前向） | 71,585 | 26.06 | 2.695 | 103.40 | 1.826% |
| 基线 ref（前向） | 71,585 | 15.77 | 2.695 | 170.87 | 3.017% |
| 基线 gen（前向） | 71,585 | 279.49 | 2.695 | 9.64 | **0.170%** |
| **基线整 step（加权）** | 71,585 | 543.92 | 15.447 | 28.40 | **0.501%** |
| 优化 update_actor | 71,421 | 170.38 | 7.346 | 43.12 | **0.761%**（+16% 相对） |
| 优化 old_log_prob | 71,421 | 26.40 | 2.688 | 101.83 | 1.798% |
| 优化 gen | 71,421 | 280.40 | 2.688 | 9.59 | 0.169% |
| **优化整 step（加权）** | 71,421 | 521.56 | 15.411 | 29.55 | **0.522%** |

推理（节点 105 独立 vLLM TP8 单 replica，8 chip，65,536 输出 token / 390.42s）：

| 项 | 数值 |
|---|---:|
| 输出 token/s（8 chip） | 65,536 / 390.4 = **167.8** |
| achieved TFLOPS（2ND） | 6.75 |
| MFU（÷354×8） | **0.238%** |
| 对照：RL 内 gen（16 chip，2 replica） | 0.169–0.170% |

说明：
1. 这里的 MFU 是**端到端阶段 MFU**，分母包含 offload/通信/CPU 优化器/等待，所以 update_actor 只有 0.66%，而纯 forward 的 ref 有 3.0%。
2. 用实测 matmul 峰值 300–319 TFLOPS 代替 354，则 update_actor 实际算力利用率为 0.73–0.77%。
3. 纯训练对照（FSDPTurbo 独立基准，无 offload/rollout）：16,384 token / 3.6s → **7.9% MFU**。RL 的 update_actor 比纯训练低 12×，差距来自 offload 同步、变长 logprob 与通信等待，而非模型计算。
4. gen 的 2ND 公式把每个 token 当作一次完整前向；decode 实际是带宽受限（读 KV cache），0.17% 应理解为“等效 MFU”，不是算力瓶颈指标。

---

## 5. 基线证据（80.5.25.118，2026-10-20 09:35 UTC）

来源：`worklog/dsv41_grpo_noengram_1step_202610.md`（同一权重/脚本/单机 16 卡）。

| 阶段 | 耗时 (s) | 占 step |
|---|---:|---:|
| gen | 299.33 | 44.1% |
| update_actor | 291.05 | 42.9% |
| old_log_prob | 27.75 | 4.1% |
| update_weights | 23.00 | 3.4% |
| ref | 15.03 | 2.2% |
| adv | 0.30 | 0.04% |
| step 总 | 677.88 | 100% |

update_actor 内部已知事实：
- `opt-state-debug done created=894/918 elapsed=58s`：首步一次性创建 894 个参数的 AdamW 状态，**dtype=bf16、device=cpu**（日志 `param0 ... dtype=torch.bfloat16 device=cpu`）。
- 09:54:53–09:56:31 约 **98 s AICore=0**（monitor line 613–757），与 CPU optimizer step 时间吻合。
- 单芯片 HBM 峰值 65,517/65,536 MB（99.97%），update_actor 期间总 HBM 峰值 1,031,806 MB。

---

## 6. 待验证假设（按优先级）

| # | 假设 | 预期证据 | 对照实验 |
|---|---|---|---|
| H1 | update_actor 的 98 s AICore=0 是 CPU AdamW（bf16 状态，894 个 DTensor 参数） | profiling CPU trace 出现 `aten::adamw`/`_foreach` 长尾；NPU trace 该区间无 kernel | 114：`ppo_mini_batch_size=64` 使 optimizer step 从 8 次/step 降到 1 次 |
| H2 | 每 micro-batch 一次全参数 all-gather/H2D，micro=1 时 64 次/step，是主要通信/搬运成本 | profiling 显示 `all_gather`/`copy_` 次数与 micro 数成正比；standalone FSDP 基准同 token 更快 | 112：micro=2；116：micro=8 |
| H3 | 首步 58 s 是 CPU 状态物化（bf16 零初始化），只影响 step1 | 后续 step 不再出现 `opt-state-debug done` | 观察 step2 日志 |
| H4 | update_actor 峰值 65.5 GB 主要由参数/梯度/优化器状态常驻 + all-gather 缓冲构成，activation 占比小 | `torch_memory` 快照分类统计 | 111 profiling + memory snapshot |
| H5 | vLLM rollout 与训练共卡导致可用 HBM 受限，是 offload 无法关闭的根因 | 对比 vLLM sleep 前后 HBM；util=0.3 对照 | 后续实验 |

---

## 7. Profiling 方案

### 7.1 训练侧（节点 111）

使用 verl 内置 profiler（`torch_npu.profiler`，mstx 标记阶段）：

```bash
TRAIN_EXTRA_ARGS="global_profiler.tool=npu global_profiler.steps=[2] \
  global_profiler.profile_continuous_steps=False \
  global_profiler.save_path=.../profiles/run_111_actor_baseline \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.ranks=[0] \
  actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory] \
  actor_rollout_ref.actor.profiler.tool_config.npu.level=level1 \
  actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True \
  actor_rollout_ref.ref.profiler.enable=True ... "
```

- `steps=[2]`：避开 step1 的 58 s 一次性状态物化，采集稳态。
- `level1`：AICore + HCCL；`contents=[npu,cpu,memory]`：CPU 栈 + NPU kernel + 内存快照。
- 产物：`profiles/run_111_actor_baseline/`（rank0 的 `*_ascend_pt`、`kernel_details.csv`、`memory snapshot`）。

### 7.2 推理侧

- 方案 A：rollout profiler（vLLM engine，`VLLM_TORCH_PROFILER_DIR`）在 GRPO step 内采集。
- 方案 B：独立 vLLM serve + 固定 prompt 集合，分别测 prefill/decode，采集 kernel trace。用于干净的推理 MFU/profiling 报告。

### 7.3 FSDPTurbo 独立基准（节点 113）

`FSDPTurbo/examples/deepseek_v41/config_perf_1node16.yaml`：
- 同一权重、FSDP16/EP8/EFSDP1、bf16、batch=4×1024、无 rollout、无 offload、AdamW on NPU。
- 用途：给出“纯训练 fwd+bwd+optimizer”时间，作为 update_actor 的理论下界/对照。

---

## 8. 实验记录

### 8.1 E0：基线（节点 110，80.5.21 段，2026-10-20 15:01 启动）

命令/配置：见第 1 节；`MODEL_PATH=...noengram`，`STEPS=100000`，KL=1，bsz 8×8，resp 1024，micro=1，mini=8，FFN recompute + attn recompute，param/grad/opt offload=True，vLLM util=0.5。

**step 1（15:13:58 首次 `running: 8`，15:26 左右完成）**

| 阶段 | 耗时 (s) | 备注 |
|---|---:|---|
| gen | **394.18** | 比 118 的 299.33 慢 95s；step1 含引擎首次生成开销 |
| old_log_prob | 28.69 | |
| ref | 15.64 | |
| adv | 0.45 | |
| update_actor | **270.95** | 含首步 `opt-state` 预建 58s |
| update_weights | 22.67 | receiver load ~9s |
| **step** | **752.91** | `perf/throughput=5.64` |
| global_seqlen/mean | 4244.75 | total tokens 67,916 |
| response_length/mean | 951.56 | |
| actor max_memory_allocated | 33.09 GB | reserved 48.57 GB |
| cpu_memory_used | 1350 GB | 主机 3.0 TB |
| perf/mfu/actor | 0.0 | 未打 MFU patch 的原始行为 |

数据来源（Ray driver 日志，tee 的外层日志不含 step 汇总）：
`/tmp/ray_dsv41_perf_copy/session_2026-10-20_15-01-53_076952_274891/logs/worker-a95a012621da8ba500e3be182dc25a28606f44df29723f2f5983c945-01000000-311309.out`

**update_actor 内部栈证据（py-spy，15:20–15:25，节点 110）**

采样命令：

```bash
for pid in $(pgrep -f "actor_rollout_ref_update_actor"); do
  py-spy dump --pid "$pid"
done
```

统计（`logs/pyspy_update2_20261020_232322.log`）：

| 栈帧 | 出现次数 | 含义 |
|---|---:|---|
| `_engine_run_backward` | 459 | fwd/bwd 主循环 |
| `all_gather_inputs` / `foreach_all_gather` | 131 / 29 | FSDP 参数 all-gather（含 recompute 触发的重复 gather） |
| `optimizer_step` | 370 | optimizer 阶段总样本 |
| `_dsv41_prepare_optimizer_step` | 137 | 其中 `torch.npu.empty_cache()` + `gc.collect()` |
| `_single_tensor_adam` | 83 | **CPU 单张量 AdamW**（非 foreach/fused） |
| `compute_grad_norm` | 13 | CPU/DTensor 梯度范数 |

结论（有栈证据）：
1. update_actor 的 AICore=0 区间由 **CPU 优化器路径**主导：`_single_tensor_adam` + `compute_grad_norm` + `empty_cache/gc`；不是 H2D 带宽（实测 55/40 GB/s，理论秒级）。
2. `VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1` 触发的 `empty_cache()+gc.collect()` 在 optimizer 栈中占 137 个样本，是可单独关闭的候选。
3. FSDP all-gather 在 backward 中反复出现，且 `recompute_fn`（activation checkpoint）会再次触发 forward all-gather；这与 micro=1 时每 step 64 次全参数 gather 的假设一致（H2）。

**step 2（稳态，15:35–15:39）**

| 阶段 | 耗时 (s) | 相对 step1 |
|---|---:|---|
| gen | **307.37** | -86.8s（step1 首次生成 warmup） |
| old_log_prob | 26.06 | |
| ref | 15.77 | |
| adv | 0.47 | |
| update_actor | **223.22** | -47.7s（step1 含 58s 状态预建） |
| update_weights | 23.44 | |
| **step** | **596.35** | `perf/throughput=7.417` |
| global_seqlen/mean | 4423.0 | total tokens 70,768 |
| response_length/mean | 983.25 | |
| actor max_memory_allocated | 33.71 GB | reserved 48.57 GB |
| perf/mfu/actor | 0.0 | 未打 patch |

**MFU（step 2，N_activated=16.30B，peak=354 TFLOPS×16）**

| 阶段 | 时间 s | FLOPs | achieved TFLOPS | MFU |
|---|---:|---|---:|---:|
| update_actor | 223.22 | 6×16.30B×70,768 = 6.921e15 | 31.0 | **0.547%** |
| old_log_prob | 26.06 | 2×16.30B×70,768 = 2.307e15 | 88.5 | 1.56% |
| ref | 15.77 | 2.307e15 | 146.3 | 2.58% |
| gen（prefill+decode） | 307.37 | 2×16.30B×69,968 = 2.281e15 | 7.42 | **0.131%** |
| 整 step（有用 FLOPs 合计） | 596.35 | 13.816e15 | 23.2 | 0.409% |

**update_actor 阶段分解（py-spy 时间线，step1，`parse_pyspy.py`）**

| 区间（UTC） | 阶段 | 时长 |
|---|---|---:|
| 15:22:50–15:23:38 | backward + all-gather（采样到的尾部） | ~48s |
| 15:23:40–15:23:41 | grad norm/clip | ~2s |
| 15:23:43–15:24:10 | `empty_cache()+gc.collect()` | **~27s** |
| 15:24:12–15:24:34 | `_single_tensor_adam`（CPU AdamW） | **~22s** |
| 15:19:58–15:22:50（未全程采样） | forward/backward 主体 | 由总时长反推 ~220s |

稳态 update_actor 223s ≈ **fwd/bwd ~172s + optimizer ~51s**（去掉 step1 的 48s 状态预建）。
优化优先级由此确定：
1. `VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=0`：直接去掉 ~27s；
2. AdamWSwap：CPU Adam ~22s → NPU fused（预计省 15–20s，且释放常驻优化器状态显存）；
3. 释放显存后尝试 micro=2 / `reshard_after_forward=False` / 关 FFN recompute，压缩 fwd/bwd 的 ~172s。

### 8.2 E1：micro-batch A/B（节点 112/116）

| 实验 | 配置 | 结果 | 证据 |
|---|---|---|---|
| E1a | `ppo_micro_batch_size_per_gpu=2` | **OOM** | `torch.OutOfMemoryError: NPU out of memory. Tried to allocate 92.00 MiB (NPU 9; 61.27 GiB total; 49.89 GiB already allocated; 112.18 MiB free; 50.80 GiB reserved)`，发生在 `all_gather_inputs` |
| E1b | `ppo_micro_batch_size_per_gpu=8` | **配置断言失败** | `AssertionError: data size must be divisible by force_group_size * micro_batch_size_per_gpu`（64 不能被 16×8 整除） |

结论：
- 当前 util=0.5 下，训练进程峰值已到 49.9/61.3 GB，**micro=2 即 OOM**；micro-batch 杠杆在现有显存布局下不可用。
- micro=8 在数据切分层就被拒绝；micro 只能取 1/2/4。
- 若要用 micro≥2，必须先释放 vLLM 睡眠期间仍占用的 ~10 GB/chip，或降低 util。

### 8.3 显存天花板根因：vLLM 在 Ascend 上只能 sleep level=1

代码证据（`verl/workers/rollout/vllm_rollout/vllm_async_server.py`）：

```python
def _resolve_sleep_level(self) -> int:
    ...
    if mtp_rollout_enabled or self.lora_as_adapter or is_torch_npu_available(...):
        return 1          # <-- Ascend 固定 level 1
    return 2
```

vllm-ascend `worker.sleep(level=1)` 会 `allocator.sleep(offload_tags=("weights",))`：weights 备份到 CPU，其余（KV）释放；但 **persistent tag 的分配不释放**（DSA Hadamard、元数据等）。实测 update_actor/OOM 时：
- 训练进程 `max_memory_reserved=50.80 GiB`，NPU 总容量 61.27 GiB；
- 差额 ~10 GiB 由 vLLM persistent/其他进程持有，训练侧只剩 112 MiB 可用。

这是“为什么不能关 offload / 不能上 micro-batch / 不能把 optimizer 放 NPU”的共同根因之一。待验证：把 `gpu_memory_utilization` 从 0.5 降到 0.4/0.35 能否把这 10 GiB 降下来（E4）；以及强制 level=2 是否可行（E5，有精度/恢复风险）。

### 8.4 E2：AdamWSwap（节点 114）

动机：`FSDPTurbo/fsdp_turbo/memory/swap_optimizer/swap_optimizer.py` 的 `AdamWSwap` 是 DTensor 感知的 fused NPU AdamW，按可用显存分批把 optimizer state 换入/换出，并支持 `param_prefetch`；可替代 `_single_tensor_adam` 的 CPU 路径。

兼容性修复（本工作目录内，已记录）：
1. `torch.accelerator.Stream` / `stream` / `get_device_properties` / `current_device_index` 在 torch 2.10 不存在 → 改为 `torch.npu.*`（16 处）。
2. 首轮试验因此报 `AttributeError: module 'torch.accelerator' has no attribute 'Stream'`。

启用命令（追加到 `TRAIN_EXTRA_ARGS`）：

```
actor_rollout_ref.actor.optim.optimizer_impl=fsdp_turbo.memory.swap_optimizer
actor_rollout_ref.actor.optim.optimizer=AdamWSwap
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

状态：15:32 通过 Hydra 校验并创建 worker group；结果待补。

### 8.5 E3：FSDPTurbo 独立基准（节点 113，已完成）

目的：无 rollout、无 offload、AdamW 在 NPU 上，给出纯训练 fwd+bwd+optimizer 的对照。

- 最终配置：`FSDPTurbo/examples/deepseek_v41/config_perf_1node16.yaml`（FSDP16/EP8/EFSDP1，**batch=1/rank，max_seq=1024，FFN recompute**，max_steps=6，无 checkpoint；**未设置任何 offload**）。
- 数据：`data/perf_text_256.jsonl`（GSM8K prompt 填充到 ~1000 token，256 条）。
- 踩坑：`datasets` 版本不接受直接文件路径 → 给 `fsdp_turbo/training/dataloader.py` 增加本地文件分支（仅基准脚本）；batch=4 无 recompute 在 `loss.backward()` OOM，改为 batch=1 + FFN recompute。
- 结果（`logs/fsdp_bench4_20261020_234104.log`）：

| step | Step Time | Loss | Peak Memory |
|---|---:|---:|---:|
| 1 | 23.25s（warmup/编译） | 11.95 | 23.46 GB |
| 3 | 3.72s | 11.49 | 23.46 GB |
| 4 | 3.54s | 9.61 | 23.46 GB |
| 5 | 3.71s | 6.83 | 23.46 GB |
| 6 | 3.46s | 5.21 | 23.46 GB |

稳态 **3.5–3.7s/step**：16 条 × 1024 token = 16,384 token 的 fwd+bwd+AdamW。
MFU = 6×16.30e9×16384 / 3.6s / (16×354e12) = **7.9%**（含 FFN recompute 的额外前向，硬件算力利用率更高）。

**对照（同 token 量级、同 FSDP/EP 拓扑）**：RL 的 update_actor 每个 micro-batch 是 16 条（1 条/rank）fwd+bwd，耗时 **47s**；独立基准包含 optimizer 只需 **3.6s** → **RL 慢约 13×**。差距不在模型计算，而在 verl 的 offload/同步/变长 logprob 处理。

### 8.6 E4：HCCL 微基准（节点 105，已完成）

目的：判定 profiling 中 allGather 71ms/次、reduceScatter 172ms/次（0.07–0.16 GB/s）是否为 HCCL 硬件能力问题。
方法：`data/hccl_bench.py`，16 进程单机 HCCL，warmup 5 + 20 次取平均，消息大小按 profiling 实际值（1/11.5/23/36/83/142 MB/rank）。

| 操作（每 rank 输入） | 基线 | AIV 展开 | 结论 |
|---|---:|---:|---|
| allGather 11.5MB | **0.946 ms**（聚合 194 GB/s） | 1.092 ms | AIV 无收益 |
| reduceScatter 11.5MB | **1.076 ms**（192 GB/s） | 1.294 ms | AIV 无收益 |
| allToAll 11.5MB | 1.462 ms（132 GB/s） | 1.537 ms | 无收益 |
| allReduce 11.5MB | 1.993 ms（97 GB/s） | 1.951 ms | 持平 |
| allGather 142MB | 10.07 ms（225 GB/s） | 10.00 ms | 持平 |
| reduceScatter 142MB | 12.38 ms | 12.34 ms | 持平 |
| allReduce 1MB | 0.550 ms | **0.291 ms** | 仅小消息 1.9× |

- `HCCL_OP_EXPANSION_MODE=AIV` 只对 ≤1MB 消息有效；本负载 11.5–142MB **无收益**，`AIV_ONLY` 直接初始化失败。
- 机内 HCCL 原始能力 100–225 GB/s；RL 中观测到的 0.066–0.164 GB/s 是**等待/rank 偏斜/排队**，不是硬件带宽。
- 结论：**HCCL 算法/AIV 调参不是本负载的杠杆**（负结果，排除）。

### 8.7 offload 语义与 E5：全无 offload（节点 113）

代码证据 `verl/verl/workers/engine/fsdp/fsdp_turbo_impl.py:55-61`：

```python
offload_policy = None
if self.engine_config.offload_policy or self.engine_config.forward_only:
    self._is_offload_param = False        # 覆盖 param_offload
    self._is_offload_optimizer = False    # 覆盖 optimizer_offload
    offload_policy = True                 # 走 FSDP2 CPUOffloadPolicy
```

即 **`offload_policy=True` 时 `param_offload/optimizer_offload` 开关被忽略**。因此 E-A（只关 param_offload）无效，已改为 E5：
`PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=False OFFLOAD_POLICY=False REF_PARAM_OFFLOAD=False`，其余与基线一致，3 步。
依据：E3 已证明全无 offload 在 batch=1 + FFN recompute 下仅 23.46GB/rank，显存足够。

### 8.8 profiling 深挖（节点 111，step 2，已完成）

**MSTX 阶段**：`actor_compute_log_prob` 23.23s → `ref_compute_log_prob` 10.83s → `actor_update` 209.99s（rank0）。

**actor_update 内部 PyTorch API 时间线**（相对 actor_update 起点）：

| 区间 | 内容 | 时长 |
|---|---|---:|
| 0.25–37.52s | micro_batch0（含首层/首步开销） | 37.28s |
| 37.52–85.91s | micro_batch1 | 48.38s |
| 85.91–133.89s | micro_batch2 | 47.98s |
| 133.89–180.34s | micro_batch3 | 46.45s |
| 180.34–187.92s | grad norm/clip + 清理 | 7.58s |
| 187.92–209.41s | `Optimizer.step#AdamW.step`（CPU） | **21.49s** |

**host 侧阻塞（CANN_API，actor_update 窗口累计）**：

| API | 次数 | 累计 | 说明 |
|---|---:|---:|---|
| `aclrtSynchronizeStream` | 4,368 | **72.03s** | 流同步（offload 拷贝等待） |
| `aclrtMemcpyAsync` | 8,646 | **37.52s** | H2D/D2H 异步拷贝 |
| `aclrtMallocPhysical` | 10,335 | 14.61s | 动态显存分配 |
| `aclrtMemcpy` | 5,093 | 13.55s | 同步拷贝 |
| `aclrtUnmapMem/FreePhysical/MapMem` | ~31,900 | 9.78s | 分配器 map/unmap |

单个最大拷贝：`aten::to`/`aten::copy_`/`Dequeue@acl_memcpy_host_to_device` 均 **3.355s**，位于 `FSDP::pre_forward (model.layers.38.ffn.experts)` —— expert 参数从 CPU 换入的 pageable 拷贝（`VERL_DSV41_OFFLOAD_PIN_MEMORY=0`）。`aten::to` 全窗口累计 **128.82s/81,440 次**，`aten::copy_` 125.63s。

**设备 kernel 只占 ~46s（22%）**：ScatterAdd 24.88s（2,916 次）、Gather 6.31s（6,160）、GroupedMatmul 1.16s、BatchMatMul 1.04s、ReduceSum 0.91s。**矩阵乘仅 ~2.2s**，其余是变长/MoE 数据搬运。

**HCCL（actor_update 窗口，op 时间可多流重叠）**：

| 操作/组 | 次数 | 累计 | 数据量 | 平均每次 |
|---|---:|---:|---:|---:|
| reduceScatter group0（FSDP16） | 172 | 29.65s | 1.97GB | **172ms** |
| allGather group0（FSDP16） | 345 | 24.46s | 4.00GB | **71ms** |
| allToAllV group57（EP8） | 960 | 12.16s | 35.04GB | 12.7ms |
| allGather group57（同步语义，0 字节） | 320 | 11.32s | ~0 | 35ms |
| allReduce group33 | 160 | 6.71s | 22.65GB | 42ms |

显存：actor_update 窗口 `max_allocated=36.20GB`，`max_reserved=52.09GB`（HBM 61.27GB）。

**与微基准对照**：同一 11.5MB allGather 原始 0.95ms，RL 中 71ms（75×）；reduceScatter 1.08ms vs 172ms（159×）。→ 等待/偏斜/offload 同步是主因。

### 8.9 基线 step3（稳态）

节点 110 持续运行到 step3：`update_actor=198.19s`（step2 223.22s，step1 270.95s），`step=543.92s`，`gen=279.49s`，`max_alloc=33.71GB`，`max_reserved=48.57GB`。说明 step2 存在预热效应，**A/B 比较统一取 step3**。

### 8.10 E2：AdamWSwap（节点 114，3 步完成）

| 指标 | 基线 step3（110） | AdamWSwap step3（114） | 变化 |
|---|---:|---:|---:|
| update_actor | 198.2s / 71,585 tok | **180.1s / 71,210 tok** | **-8.7%**（按 token 归一 2.769→2.529 s/1k） |
| step | 543.9s | 543.2s | 持平（gen 波动 279→297s） |
| max_alloc / reserved | 33.7 / 48.6 GB | 34.0 / 48.6 GB | 持平 |
| pearson | 0.99943 | 0.99952 | 一致 |

step1/2 归一值：3.367 / 2.903 vs 基线 3.99 / 3.154（-16% / -8%）。AdamWSwap 的收益来自 `Optimizer.step` 21.5s → 约 12s，量级符合。

### 8.11 E1c：util=0.4 + micro=2（节点 116，失败）

`update_actor` 内 `_fsdp_param_group.wait_for_unshard → alloc_storage` OOM：
`49.71 GiB already allocated; 95.29 MiB free; 50.56 GiB reserved`（HBM 61.26GiB）。
结论：把 rollout util 从 0.5 降到 0.4 **没有**释放训练侧可用显存（vLLM sleep level=1 已释放 KV，persistent 部分不随 util 线性下降）；micro=2 仍需要额外 ~13GB。

### 8.12 E5：全无 offload（节点 113，失败）

`PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=False OFFLOAD_POLICY=False`，3 步。模型加载、rollout、compute_log_prob 均通过；在 `update_actor` 的 expert dispatch `HcclAllreduce` 处 OOM（Ray `OutOfMemoryError`，无显式字节数）。
解释：无 offload 时常驻 = 参数 6.65GB + 梯度 6.65GB + AdamW fp32 状态 ~26.6GB（或 bf16 ~13.3GB）+ 激活 + vLLM persistent ~10GB，逼近 61.3GB。**该方向被显存否决**。

### 8.13 E6：参数/梯度常驻 + 优化器状态 offload（节点 113，失败）

`PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=True OFFLOAD_POLICY=False REF_PARAM_OFFLOAD=False`。
动机：profiling 中 `aten::to/copy_` 128.8s、`aclrtSynchronizeStream` 72s、`aclrtMemcpyAsync` 37.5s 来自参数/梯度的 CPUOffloadPolicy 往返；若去掉可大幅提速。
结果：同样在 `update_actor` 的 expert dispatch `HcclAllreduce` OOM（7 处）。说明 **EP dispatcher 的临时 all-reduce 缓冲需要 CPU offload 释放出的显存余量**，参数/梯度常驻在 61.3GB HBM 下不可行。

### 8.14 E7：expandable_segments + AdamWSwap 组合（节点 113，3 步完成）

配置：基线 + `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` + AdamWSwap（`optimizer_impl=fsdp_turbo.memory.swap_optimizer`、`optimizer=AdamWSwap`、`optimizer_offload=False`），其余不变。

| step | tokens | update_actor | s/1k tok | baseline 同口径 |
|---|---:|---:|---:|---:|
| 1 | 68,630 | 196.45s | 2.862 | 3.99 |
| 2 | 71,716 | 182.90s | 2.550 | 3.154 |
| 3 | 71,421 | **170.38s** | **2.385** | 2.769 |

step2–3 均值 2.468 s/1k vs 基线 step2–5 均值 2.958 → **-16.6%**；step3 单点 -13.9%。`max_reserved` 51.1GB（expandable 的正常水位），pearson 0.9995，数值一致。
这是当前最优可行配置；代价是 reserved 显存 +2.6GB，仍在 61.3GB 内。

### 8.15 长跑命令变更（用户 2026-10-21 指令）

长跑改用 `dsv41-verl/verl/wiki/worklog_dsv41_2nodes_countbenchqa_rl_1019.md` 的命令与数据集（countbenchqa_lite，图文 VQA），不再用 GSM8K。
- 适配脚本：`dsv41-verl-copy/verl/my_scripts/dsv41_grpo_countbenchqa_1node_21.sh`（由原脚本生成，仅改 `MIRROR_ROOT`、80.5.21 IP 探测、Ray/HCCL 端口、去掉全局 pkill；训练参数逐字保留）。
- 配置：单机 16 卡，FSDP16/EP8，GEN TP8×DP2；bsz 8×n8；LR 5e-6；USE_KL=0；prompt 1152 / response 512；util 0.5；vision 冻结；agent loop `deepseek_v41_vl_single_turn`；thinking=false；`save_freq=-1`。
- 启动（节点 111）：`bash dsv41_grpo_countbenchqa_1node_21.sh 500`，日志 `verl/logs/countbenchqa_optlong.outer.log`。
- 本工作副本的 countbenchqa 奖励/agent-loop/example 脚本与 dsv41-verl 完全一致（diff 为空）。
- **2026-10-21 修复后配置**（见 8.17）：`_launch_cbq_long.sh` 额外设
  `VA_V41_PREFILL_LOGPROBS=0` 与 `TRAIN_EXTRA_ARGS="actor_rollout_ref.rollout.load_format=auto"`；
  后者用于让 vLLM 初始化时加载真实 vision tower（dummy 同步不会恢复它）。

### 8.16 优化后 profiling（节点 114 → 113）

节点 114 先用 `analysis=False` 跑 3 步，`global_profiler.steps=[2]`、rank0、`contents=[npu,cpu,memory]`，输出 `profiles/run_114_opt_combo`。结果：step2 `update_actor=182.04s / 70,585 tok = 2.579 s/1k`，与 113 组合（2.550）一致。
`analysis=False` 只产生 2GB 原始 FRAMEWORK 文件、无 `ASCEND_PROFILER_OUTPUT/*.db`；为做 kernel 级对比，在 113 重新启动 `analysis=True` 的 2 步 profiling（`profiles/run_113_opt_combo_db`，step2，AdamWSwap + expandable_segments）。DB 对比（`profiles/compare_prof_out.txt`，均取 rank0 actor_update 窗口）：

| 指标 | 基线（111 原始配置） | 组合优化（113） | 变化 |
|---|---:|---:|---:|
| actor_update 窗口 | 209.99s | 186.31s | **-11.3%** |
| 计算 kernel 总计 | 46.17s | 46.16s | 0%（模型/批量相同） |
| ScatterAdd（EP dispatch 重排） | 24.88s | 24.73s | -0.6% |
| Gather | 6.31s | 6.25s | -0.9% |
| aclrtSynchronizeStream | 72.03s | 69.64s | -3.3% |
| aclrtMemcpyAsync | 37.52s | 26.81s | **-28.5%** |
| aclrtMemcpy | 13.55s | 22.37s | +65% |
| aclrtMallocPhysical | 14.61s | 18.59s | +27% |
| 10 项 CANN host API 合计 | 157.1s | 157.0s | 0% |
| max allocated / reserved | 36.20 / 52.09 GB | 36.17 / 54.80 GB | 持平 / +2.7GB |

结论：**组合优化减少的 23.7s 不来自计算 kernel**（46.17→46.16s 完全不变），也不来自 CANN host API 总量（157.1→157.0s）；收益来自 optimizer 路径——AdamWSwap 把优化器状态换到 NPU、避免 CPU AdamW 的 host 计算（该部分不计入 CANN API），与 8.10 的 -8.7% 单步结果相互印证。expandable_segments 主要降低分配器行为波动（memcpyAsync -28.5%，但 memcpy +65%），不是 wall time 的主因。


### 8.17 countbenchqa 长跑失败与根因（vision tower 未加载）— 已修复

**症状**（节点 111，17:17 / 18:15 两轮）：
- `agent_dump_countbenchqa.*` 的响应是**随机 CJK 乱码**（不是格式问题）；
- reward 0.028–0.042、response 恒为 512、`actor/entropy` 8.04–8.23（接近均匀分布
  log(129280)=11.77）、pearson 0.69–0.84；
- 对照工作日志 118 成功运行 step1：reward 0.211、response 461、entropy 1.53、pearson 0.9989。

**排查与排除**：
1. **文本路径正常**：在 113 独立起 vLLM（8010，真实权重，无 dummy），用与 agent loop 相同的
   chat 模板构造纯文本问题，输出 `The final answer is 2.`（连贯且正确）。
2. **prefill rescore 不是根因**：给 114 的 `vllm_async_server.py` 加文件探针
   （`/tmp/plp_trace_<pid>.log`），64/64 个请求 rescore 全部成功：
   `plp_len=756–770, vals=512, token_ids=512, 0 FAIL`；但该步 pearson 仍只有 **0.687**。
   111 开启 `VA_V41_PREFILL_LOGPROBS=1` 后 step1/2 pearson 0.808/0.840，也远离 0.999。
   说明 decode-vs-prefill 与 VLM 对齐补丁都不是主因（补丁本身正确，保留在副本中）。
3. **同步计数可疑**：sender `1664 tensors / 26.19 GiB / 52–54 buckets`，而 vLLM 以
   `--load_format dummy` 启动；config 的 `fsdp_plan.ignored_modules=['model.vision','model.aligner']`
   又把冻结的 vision 排除在 FSDP 计划外。同步只保证文本参数，vision 是否被写入未经验证。

**决定性 A/B**（节点 114，单步 countbenchqa，其余参数逐字相同，仅改 load_format）：

| 配置 | reward | response len | entropy | pearson | gen | step |
|---|---:|---:|---:|---:|---:|---:|
| `load_format=dummy` + 权重同步 | 0.028–0.042 | 512.0 | 8.10 | 0.808 | 260s | 500s |
| **`load_format=auto`** | **0.1969** | **487.3** | **1.245** | **0.99886** | 253s | 500s |
| 工作日志 118（参照） | 0.211 | 461 | 1.53 | 0.9989 | 213s | 1266s |

**根因**：`actor_rollout_ref.rollout.load_format=dummy` 让 vLLM 以随机初始化启动，而每步的
权重同步**没有恢复 vision tower（32 层 ViT + aligner）**；训练侧 vision 是真实（冻结）权重，
两侧图像 embedding 不一致 → 有图 rollout 乱码、reward≈0、pearson 低。文本路径不经过 vision，
所以纯文本输出仍连贯——这解释了“文本正常、带图必乱”的不对称现象。

**修复（已应用到长跑）**：
- `_launch_cbq_long.sh` 增加 `TRAIN_EXTRA_ARGS="actor_rollout_ref.rollout.load_format=auto"`：
  vLLM 初始化时直接加载完整 HF 权重（含 vision），后续每步仍由 trainer 同步文本权重；
- 同时设 `VA_V41_PREFILL_LOGPROBS=0`：vision 正确后 decode logprobs 已达 0.9989，无需额外
  prefill rescore（省一次全量 prefill）；
- 代价：引擎初始化多读一次 ~100GB checkpoint（一次性 +4–6 min），step 时间不变（~500s）；
- 已按此在 111 重启长跑（容器时钟 19:40）。**step1 实测**：
  reward **0.1547**、response **469.3**（clip 0.844）、entropy **1.596**、
  pearson **0.99894**、gen 160.0s、update_actor 168.3s、**step 402.9s**、max alloc 26.14GB。
  与工作日志 118 的 step1（0.211 / 461 / 0.844 / 0.9989 / 1.53）同量级，修复确认。
- **step2 继续验证**：reward **0.2531**、response 440.6、pearson **0.99669**、step **335.5s**，
  与工作日志 step2（reward 0.2531、pearson 0.9986）一致；无 OOM/异常。长跑按 500 步继续。

### 8.18 推理 profiling 与吞吐（节点 105，独立 vLLM TP8/EP8）

- 服务：`vllm serve` 新 launcher，TP8/EP8（8 chip，单 replica），`--profiler-config {"profiler":"torch"}`，util 0.85，enforce_eager。
- 负载：8 prompts 并发 × n=8 × max_tokens=1024（64 序列），与 RL gen 的 batch 规模一致。
- **实测 decode 吞吐：159.6 token/s（8 chip，Running 57 reqs，Waiting 0）**；65,536 output token 用时约 428s（17:34:33–17:41:41），与日志 153–160 tok/s 一致。
- 推理 MFU（2×16.304B×token / time ÷ 354TF×8）= **0.238%**（8 chip）；RL 内 gen（16 chip，2 replica）0.169–0.170%。decode 为带宽受限，该值应视为等效 MFU。
- profiler trace：`VLLM_TORCH_PROFILER_DIR` 下生成每 rank 34GB 的 FRAMEWORK 原始数据（共 264GB），但 `.pt.trace.json.gz` 只含 `python_function`（CPU），无 NPU kernel；未生成 `ASCEND_PROFILER_OUTPUT/*.db`。已保留 75MB CPU trace，删除 264GB 原始文件（共享盘曾 100%，释放后 1.6TB 可用）。
- 结论：现有 vLLM/ascend 组合下，推理 trace 需要额外 msprof 导出才能做 kernel 级分析；本轮以实测吞吐 + 时间法 MFU 作为推理 profiling 结论。

---

## 9. 结论（每条均有实验编号/日志/DB 支撑）

### 9.1 update_actor 是核心瓶颈，瓶颈在 host/offload/通信而非算力

- 基线（8.1 E0，节点 110）：`update_actor` 198.2–216.6s，整步 543.9–554.9s；
  profiling 窗口 209.99s（8.8，111 DB）。
- 窗口内**计算 kernel 合计仅 46.17s（22%）**，其中 EP dispatch 重排 `ScatterAdd` 24.88s、
  `Gather` 6.31s；其余 ~164s 是 host/offload/同步等待：
  `aclrtSynchronizeStream` 72.03s、`aclrtMemcpyAsync` 37.52s、`aclrtMemcpy` 13.55s、
  `aclrtMallocPhysical` 14.61s、unmap/free 8.55s；HCCL 单 op 71–172ms，而 16 卡 HCCL
  微基准（8.6）同样数据量仅 ~1ms → 实际通信被 host 调度/同步放大。
- 结论：继续压 `update_actor` 的方向是**减少 CPU↔NPU 搬运与同步次数**（offload/优化器路径），
  而不是换 matmul kernel。

### 9.2 已验证有效的优化（A/B，同一脚本/数据/批量）

| 优化 | 证据 | 效果 |
|---|---|---:|
| `expandable_segments:True` | 8.1/8.9，steps 2–3 vs 基线 | steps 2–3 均值 **-10.5%**（2.958→2.647 s/1k） |
| `AdamWSwap`（优化器状态换到 NPU） | 8.10，节点 114 step3 | **-8.7%**（update 170.4s vs 198.2s） |
| 两者组合 | 8.14，节点 113 steps 2–3 | **-16.6%**（2.958→2.468 s/1k）；update 186.3s vs 210.0s |
| 优化后 profiling | 8.16，113 DB vs 111 DB | update 窗口 **-11.3%**；计算 kernel 46.17→46.16s 不变；10 项 CANN host API 157.1→157.0s 不变 |

- 组合的收益来自 **AdamWSwap 消除 CPU AdamW 的 host 计算**（不计入 CANN API），
  不是分配器或 kernel；expandable_segments 主要降低分配波动（memcpyAsync -28.5%，memcpy +65%）。
- MFU（4.1/4.3，`flops_counter._estimate_deepseek_v41_flops`，6·N_act·T + 稀疏 attention，
  N_act=16.304B，峰值 354 TFLOPS/chip）：actor **0.656%→0.761%**，整步 **0.501%→0.522%**。
  绝对值低是 offload/通信占比高所致，不代表 matmul 效率（实测 matmul 300–319 TFLOPS，
  约为峰值的 85–90%）。

### 9.3 已排除/失败的优化（均有 OOM 或对照数据）

- `micro_batch=2`（8.11）：49.71GiB allocated / 95MB free OOM，不可行。
- 全无 offload（8.12 E5）与“参数/梯度常驻 + 仅优化器 offload”（8.13 E6）：均在
  `expert_parallel/dispatcher.py` 的 `HcclAllreduce` OOM —— EP dispatcher 需要 CPU offload 腾出的 HBM 余量。
- `param_offload=False` 单关（8.7）：`fsdp_turbo_impl.py:55-61` 中 `offload_policy=True` 会覆盖
  `param_offload/optimizer_offload`，该实验语义无效。
- HCCL `AIV` 展开模式（8.6）：11.5MB 消息无收益。
- 结论：在 64GB HBM/卡、EP8 的约束下，**CPU offload 必须保留**；可优化的是 offload 的实现方式
  （AdamWSwap）而不是关闭它。

### 9.4 FSDP 独立基准与推理

- FSDPTurbo 独立（8.5，113，batch=1，max_seq 1024，FFN recompute）：**3.5–3.7 s/step**，
  峰值 23.46GB，MFU **7.9%**；按相同 token 折算比 RL 整步快 ~13×。
  → RL 的差距主要来自 colocate 的 offload/同步/权重同步，而不是模型前反向本身。
- 推理（8.18，105，独立 vLLM TP8/EP8）：decode **159.6 tok/s（8 chip）**，
  推理 MFU **0.238%**；RL 内 gen（16 chip，2 replica）0.169–0.170%。decode 带宽受限，
  该 MFU 应视为等效值；vLLM 的 NPU kernel trace 需额外 msprof 导出（8.18 已说明）。

### 9.5 countbenchqa 长跑的关键正确性根因（本轮最重要发现）

- `rollout.load_format=dummy` 启动的 vLLM 引擎，其 **vision tower（32 层 ViT+aligner）没有被
  权重同步恢复**，图像 embedding 为随机值；训练侧 vision 是真实冻结权重 → 带图 rollout 乱码、
  reward≈0、entropy≈8.1、pearson 0.69–0.84。
- 决定性 A/B（8.17）：仅把 `load_format` 改为 `auto`，单步结果
  reward **0.1969**、response 487、entropy **1.245**、pearson **0.99886**，与工作日志 118 的
  成功运行（0.211/461/1.53/0.9989）一致。
- 已修复长跑并在 111 验证 step1/step2：step1 reward **0.1547**、response 469、pearson **0.99894**、
  step 402.9s；step2 reward **0.2531**、pearson **0.99669**、step 335.5s（8.17）。
- 该根因同时解释了此前“pearson 只有 0.8、prefill rescore 无效”的现象：rescore 补丁本身正确
  （114 探针 64/64 成功、vals=512），但无法弥补两侧 vision 权重不一致；vision 正确后
  **无需** `VA_V41_PREFILL_LOGPROBS`，decode logprobs 已达 0.9989。

### 9.6 当前长跑状态与后续建议

- 111 上 countbenchqa-lite 长跑（500 步、`save_freq=-1`）已于容器时钟 19:40 重启，配置：
  原工作日志命令 + `load_format=auto` + `VA_V41_PREFILL_LOGPROBS=0`；
  step1 指标见 9.5，step 402.9s（约 9 步/小时）。
- 建议后续：保持该配置连续跑；若需进一步提速，优先把 AdamWSwap 合入长跑（单步再 -8.7%，
  与 9.2 的实验一致），但会改变工作日志的原始配方，需用户确认后再切换。

