# DSV4.1 pinned memory 现状、CPUOffloadPolicy 影响与 pin_memory 估算

- 日期：2026-10-22（UTC）
- 工作目录：`/mnt/share/m00899630/workspace/dsv41-verl`
- 关联 worklog：`verl/wiki/worklog_dsv41_cons_speed_1node_20261022.md`
- 目的：确认当前长跑配置是否开启 pinned memory；评估 CPUOffloadPolicy 对 HBM/性能的影响；估算开启 `pin_memory=True` 后的 pinned host 内存需求。

---

## 0. 结论摘要

1. 当前**没有训练在跑**：`cons_speed` 日志停在 2026-10-21 23:04 UTC（step 56，DataLoader worker killed）。当前 `Mlocked=24 kB`，pinned 占用为 0。
2. 上次长跑配置中 pinned memory 是**部分开启**：
   - **开启**：AdamWSwap 的 host 优化器状态，`swap_optimizer.py:86` 硬编码 `pin_memory=True`。
   - **关闭**：FSDPTurbo 的 `fsdp_plan.pin_memory=false`（启动脚本显式覆盖；源码默认 True）。
   - DataLoader、delta snapshot（NPU 上 `is_cuda_available=False`）、activation swap、engram host-offload、NIXL 均未 pin。
3. CPUOffloadPolicy 本身**是开启的**（`fsdp_turbo_impl.py:34`：`fsdp_plan.cpu_offload = engine_config.offload_policy`）；关闭的只是它的 pin_memory。
4. 显存（HBM）由 CPUOffloadPolicy 决定，`pin_memory` 不改变 HBM：
   - 开启：`After FSDP allocated 0.96 GB / reserved 15.79 GB`；update 峰值 `42.74 GB allocated / 51.19 GB reserved`（卡 61.27 GB）。
   - 关闭：每 rank 参数 ~6.66 GB + 梯度 ~6.66 GB 常驻，E5/E6 实测在 update_actor expert dispatch `HcclAllreduce` OOM。
5. 开启 `fsdp_plan.pin_memory=true` 的增量估算：**+213 GB pinned/node**（每 rank 6.66 GB 参数 + 6.66 GB 梯度），pinned 总量从 ~213 GB 升到 **~426 GB**（396.8 GiB）。HBM 不变。

---

## 1. 执行命令（可复现）

```bash
# pinned 相关代码/配置
grep -rn "pin_memory" verl/verl FSDPTurbo/fsdp_turbo --include=*.py
grep -n "PIN_OVERRIDES\|pin_memory" verl/my_scripts/dsv41_grpo_countbenchqa_cons_speed.sh
grep -n "After FSDP\|cpu_memory_used_gb" verl/logs/cons_speed_20261021_164948.nohup.log

# checkpoint 组成
python3 - <<EOF
import json, struct, glob
...  # 逐 safetensors header 统计 shape/dtype
EOF

# 机器可用性
scripts/check_npu.sh 80.5.25.120 80.5.25.124 80.5.25.128
```

代码证据位置：
- `FSDPTurbo/fsdp_turbo/memory/swap_optimizer/swap_optimizer.py:86`：`torch.empty_like(..., pin_memory=True, device="cpu")`
- `FSDPTurbo/fsdp_turbo/distributed/fully_shard_parallel/fully_shard_parallel.py:81-85`：`CPUOffloadPolicy(pin_memory=fsdp_plan.pin_memory)`
- `FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py:129`：`pin_memory: bool = True`（源码默认）
- `verl/verl/workers/engine/fsdp/fsdp_turbo_impl.py:34,55-60`：`cpu_offload = offload_policy`；`offload_policy=True` 覆盖 `param_offload/optimizer_offload`
- `verl/verl/trainer/ppo/v1/trainer_base.py:721`：`StatefulDataLoader` 未传 `pin_memory`（默认 False）
- `verl/verl/workers/engine/base.py:205`：delta snapshot `pin=is_cuda_available and ...`
- `verl/verl/utils/device.py:61`：`is_cuda_available = torch.cuda.is_available()`

---

## 2. 运行日志实测数据

| 指标 | 值 | 来源 |
|---|---:|---|
| `After FSDP` allocated/reserved | 0.96 / 15.79 GB（device 16.22/61.27） | nohup 日志第 1245 行 |
| step 35 max allocated | 42.74 GB | 第 12810 行 |
| step 35 max reserved | 51.19 GB | 第 12810 行 |
| step 35 `cpu_memory_used_gb`（psutil 全系统 used） | 1386.26 GiB | 第 12810 行 |
| step 56 `cpu_memory_used_gb` | 1388.07 GiB | 日志末尾 |
| 日志停止时间 | 2026-10-21 23:04 UTC | `ls -l` |
| 停止原因（日志表象） | `DataLoader worker (pid 1250457) is killed by signal: Killed` | 日志末尾 |

---

## 3. checkpoint 组成拆分（529.78 GB）

| 部分 | 大小 | 是否进 actor FSDP |
|---|---:|---|
| engram 表（layers.1/14；`engram_layer_ids: []`） | 393.86 GB | ❌ 不加载 |
| MTP（`mtp.*`，apply_modules 不含） | 28.46 GB | ❌ 不在 actor |
| vision + aligner（frozen，ignored_modules） | 0.97 GB | ❌ 留在卡上（= After FSDP 0.96 GB） |
| text actor 参数 | **106.50 GB** | ✅ |

交叉验证：106.50/16 = **6.66 GB/rank**，与 `run_and_improve_speed_20261020.md` E5 的“参数 6.65GB + 梯度 6.65GB”一致；AdamWSwap debug 日志 1184 param/rank。

---

## 4. CPUOffloadPolicy 的 HBM 与性能

### 4.1 HBM

| | 开（当前） | 关 |
|---|---|---|
| 初始化后 allocated | 0.96 GB | 参数分片常驻 ~6.66 GB/rank |
| update 峰值 | 42.74 allocated / 51.19 reserved | 参数 6.66 + 梯度 6.66 + 优化器/激活，E5/E6 实测 OOM |

E5（全无 offload）、E6（仅 optimizer offload）均在 `expert_parallel/dispatcher.py` 的 `HcclAllreduce` OOM；结论：CPUOffloadPolicy 是 64 GB 卡 / FSDP16 / EP8 下能训练的前提。

### 4.2 性能（`run_and_improve_speed_20261020.md` §8.8，pageable 模式，与当前 `pin_memory=false` 同语义）

| 项目 | 数值 |
|---|---:|
| actor_update 总时长 | 209.99 s |
| 设备 kernel 合计 | ~46 s（22%） |
| `aten::to` | 128.82 s / 81,440 次 |
| `aten::copy_` | 125.63 s |
| `aclrtMemcpyAsync` | 37.52 s |
| `aclrtSynchronizeStream` | 72.03 s |
| 最大单次拷贝 | 3.355 s（layer38 experts 参数 pageable H2D） |

---

## 5. pin_memory=True 的详细估算

### 5.1 每 rank

| 项目 | 大小 | 变化 |
|---|---:|---|
| sharded 参数（pageable→pinned） | 6.66 GB | 新增 pinned |
| sharded 梯度（pageable→pinned） | 6.66 GB | 新增 pinned |
| **新增小计** | **13.31 GB** | |
| AdamWSwap `exp_avg`+`exp_avg_sq`（bf16，2×6.66） | 13.31 GB | 已 pinned |
| **开启后每 rank pinned** | **26.62 GB** | |

### 5.2 节点合计

| 项目 | 数值 |
|---|---:|
| 新增 pinned | 213.0 GB（198.4 GiB） |
| 现有 pinned（AdamWSwap） | 213.0 GB（198.4 GiB） |
| 开启后 pinned 总量 | **426.0 GB（396.8 GiB）** |
| MemTotal | 2.635 TB（2.397 TiB） |
| 当前 run 系统 used | 1386 GiB（1.49 TB） |
| 开启后预计 used | ~1584 GiB（1.70 TB），余量 ~0.9 TB |

### 5.3 风险

`FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py:407-413` 记录：~366 GiB/node 的 engram pinned 表会导致后续大块 pinned 分配失败（`aclrtMallocHostWithCfg`，E10/E11/E13）。本方案总量 ~397 GiB/node，接近/超过该失败阈值；`transformer_impl.py:476-483` 也有同款记录。容量够，但分配器失败风险高。

### 5.4 建议

- 不建议直接开长跑；若测试，3-step A/B（只改 `PIN_OVERRIDES`），监控 `Mlocked`、`cpu_memory_used_gb`、`update_actor`、HBM 峰值。
- 如果只想验证 pinned 收益，可考虑把 AdamWSwap `cpu_state` 改 pageable（`swap_optimizer.py:86`），把 pinned 预算让给每步都搬的参数/梯度；需单独 A/B。

---

## 6. 机器可用性快照（2026-10-22 14:35 UTC）

| 主机 | SSH 50002 | NPU | 结论 |
|---|---|---|---|
| 80.5.25.120 | OK | 16 chip 全 0%，HBM 4.6%（~3 GB/chip 驱动），procs=0 | **空闲可用** |
| 80.5.25.124 | OK | 16 chip 全 0%，HBM 4.6%，procs=0 | **空闲可用** |
| 80.5.25.128 | OK | 32 procs，27.3–27.6 GB/chip（41.9%），load 34；`W2_longrun_20261022_132541` 日志持续写入，14:26 到 step 5 | **占用中，不可用** |
| 其余 12 台 | FAIL | — | docker 暂停，不可用 |

证据：`verl/logs/W2_longrun_20261022_132541.outer.log` mtime 14:36:14 +0000；`monitor_119_longrun.status` 14:26:47 step=5 ALIVE；`.128` `uptime` load average 34.46。

---

## 7. 待办

- [ ] 若要验证 pin_memory：3-step A/B + 监控项（见 5.4）。
- [ ] 长跑（.128）继续时补记显存/内存随时间变化，与本文 42.74/51.19 GB、1386 GiB 对照（准则 3.2）。

---

## 8. 机器状态复查（2026-10-22 14:44–14:51 UTC）

SSH 50002 可达：.119 / .120 / .123 / .124 / .128（.118、.126、.127 等 10 台仍 FAIL）。

| 主机 | NPU | 进程 | 判断 |
|---|---|---|---|
| .119 | 16 chip，HBM 73.1%（45–65 GB/chip），AICore 0–2% | 32 | W2 500-step 长跑（13:26 启动，14:41 step 7 ALIVE）；另有 11:48 的 perf profile 进程残留 |
| .120 | 16 chip，HBM 4.6%（~3 GB/chip 驱动），0% | 0 | **空闲可用** |
| .123 | 16 chip，HBM 4.7%，0% | 16 | 14:41 启动 W2-dummy5 冒烟（5 step，load_format=dummy），14:44 仍在初始化 |
| .124 | 16 chip，HBM **96.3%**（63 GB/chip），0% | 32 | 被任务占用（PID 3142520/3142524，~33 GB/chip），显存近满 |
| .128 | 16 chip，HBM 37.4%，AICore 最高 96% | 32 | 正在计算（PID 1937427/1937428，~28 GB/chip） |

变化：.119/.123 恢复（docker 已起）；.124 由空闲变为 96.3% HBM 占用；.128 由 0% 变为 96% AICore；唯一空闲仍是 .120。

证据：`check_npu.sh` 输出；`.119` ps 中 `experiment_name=GRPO-DSV41-countbenchqa-W2-longrun-20261022_132541`、`trainer.total_training_steps=500`；`.123` ps 中 `W2-dummy5-123-20261022_144123`、`total_training_steps=5`；`npu-smi info -t proc-mem -i 0` 的 PID/内存。

---

## 9. W3 pin_memory + profiler 采集（.120，2026-10-22 14:54–15:26 UTC）

### 9.1 脚本与修复

用户原脚本：`verl/my_scripts/pref_dsv41/_launch_W3_longrun_1022_pin_memory.sh`。发现 3 个阻塞问题：

1. `MY_IP=80.5.25.119`，但本次在 .120 执行；base 用 `MY_IP` 设置 `HCCL_IF_IP` 和 `ray start --node-ip-address`，会绑错 IP。
2. 两次 `export TRAIN_EXTRA_ARGS`：第二次（W3 extras）覆盖第一次的 `PROF_ARGS`，profiler 配置会全部丢失。
3. `EXPERIMENT_NAME` 第二次 export 覆盖 pin/profile 命名。

未改原脚本，新建修复版：
`verl/my_scripts/pref_dsv41/_launch_W3_longrun_1022_pin_memory_v1.sh`
（只改以上 3 处，pin/profiler/3 steps/profile step 2/save path 保持原意；`bash -n` 通过）。

启动命令：

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
nohup bash my_scripts/pref_dsv41/_launch_W3_longrun_1022_pin_memory_v1.sh \
  > /tmp/W3_pin_v1.nohup.out 2>&1 &
# PID 2542283 (launcher) / 2542562 (main_ppo driver)
```

关键配置：`STEPS=3`、`PROFILE_STEP=2`、NPU profiler actor+ref rank0、level1、contents `[npu,cpu,memory]`、`analysis=True`；`VERL_DSV41_OFFLOAD_PIN_MEMORY=1`；无 AdamWSwap、无 expandable_segments；`gpu_mem_util=0.6`。

pin 生效证据：日志中无 `fsdp_plan.pin_memory=false` 注入；example 脚本仅在 `VERL_DSV41_OFFLOAD_PIN_MEMORY=0` 时注入该 false，本次 env=1，因此 FSDPTurbo `FSDPPlanConfig.pin_memory=True` 默认生效。

### 9.2 10 分钟巡检

| 时间 (UTC) | 状态 |
|---|---|
| 14:54 | 启动 |
| 14:57 | 数据加载、worker group 初始化 |
| 15:07 | 首次 rollout 完成，vLLM `on_sample_end` sleep；host used 1041 GB |
| 15:14 | profiler 目录出现 |
| 15:17 | profile 2.2 GB；host used 1415 GB；无报错 |
| 15:26 | **3/3 steps 完成**，`Training Progress: 100%`，`step:3`，`Final validation metrics: None` |

运行总时长 22:18（约 446 s/step；step2 因 profiler 603 s）。结束时日志尾部出现 `DataLoader worker ... killed by signal: Killed`（atexit 清理），与前一次 step 56 停机的尾部同款；发生在 100% 完成与 final validation 之后，**不是外部 kill**。全程 driver PID 存活，未重启。

### 9.3 指标对比（pin vs pageable）

| step | 方案 | step(s) | gen(s) | update_actor(s) | max_alloc(GB) | max_reserved(GB) | cpu_mem(GiB) | diff_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | PIN .120 | 383.6 | 186.7 | 140.3 | 26.06 | 48.45 | 1484.4 | 0.00797 |
| 2 | PIN .120（profiled） | 603.2 | 144.1 | 152.0 | 30.52 | 48.45 | 1488.3 | 0.00822 |
| 3 | PIN .120 | 351.3 | 142.0 | 173.4 | 30.52 | 48.45 | 1490.6 | 0.00842 |
| 1 | W2 .119 pageable | 630.7 | 352.2 | 207.4 | 26.06 | 48.37 | 1041.7 | 0.00782 |
| 2 | W2 .119 pageable | 453.6 | 229.3 | 176.7 | 30.52 | 48.37 | 1040.1 | 0.01062 |
| 3 | W2 .119 pageable | 418.5 | 171.3 | 198.1 | 30.52 | 48.37 | 1042.7 | 0.00994 |
| 5–8 均值 | W2 .119 pageable | 346.9 | 153.3 | 149.4 | 42.44 | 48.49 | 1047.6 | ~0.0101 |

观察：

- **HBM 无变化**：两方案 max_allocated 26.06/30.52 GB、reserved 48.37–48.49 GB 基本一致；pin 不改变显存。
- **update_actor 未见明显收益**：PIN 140.3/152.0/173.4 s；W2 稳态 5–8 均值 149.4 s。3-step 样本 + profiler 开销，暂不能下结论。
- **CPU 内存差 +435 GiB**：PIN ~1484–1491 GiB vs W2 ~1041–1056 GiB。按 §5 估算只应 +213 GB（198 GiB），实测约为估算的 **2.2×**。可能来源：pinned params+grads 213 GB 之外，还有 FSDP 的 pinned 暂存/优化器状态 pin/分配器池开销；需用本次 profile 的 memory 数据进一步定位。不要把 435 GiB 当作最终结论。
- 一致性：PIN 三步 diff_mean 0.00797/0.00822/0.00842，与 W2 同量级，无明显恶化。

### 9.4 Profiler 产物

```
/mnt/share/m00899630/workspace/dsv41-verl-copy/profiles/run_enable_pin_memory_profile_20261022_145433
  2.6 GB
  train/node-29-120_2568210_20261022151023092_ascend_pt/
    ASCEND_PROFILER_OUTPUT/analyse.done            (15:23:44)
    ASCEND_PROFILER_OUTPUT/analysis.db             (2.1 MB)
    ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db (1.2 GB)
    profiler_info_0.json / profiler_metadata.json
    logs/profiler_*_ProfilingParser.log / *_CANNAnalyzeParser.log
```

Profiler finish：rank 0/1/14 的 `[Profiler][finish_hook] ... profiling finished` 在 15:15:39 前后；`analyse.done` 15:23:44。采集完整。

### 9.5 结论与下一步

1. 本次采集**正常完成**（3/3 step），不是被杀；按用户要求未重启。
2. .120 已恢复空闲：`procs=0`，HBM 4.6%（~3 GB/chip）。
3. 下一步分析 `analysis.db` / `ascend_pytorch_profiler_0.db`：H2D/D2H、`aten::to/copy_`、`aclrtSynchronizeStream` 与 §8.8 的 pageable profile 对比，确认 pin 是否降低了 offload 拷贝/同步时间；同时解释 CPU 内存 +435 GiB 的来源。

### 9.6 前三步训推一致性 / reward 指标

PIN 为本次 .120 pinned 采集；W2 为 .119 pageable 长跑同前三步（对照）。

| step | 方案 | diff_mean | diff_max | pearson | kl | log_ppl_abs_diff | ppl_ratio | chi2_seq | reward_mean | reward_max | entropy | resp_len | clip_ratio | grad_norm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PIN | 0.007967 | 0.4175 | 0.998783 | 0.006633 | 0.008481 | 1.007153 | 0.528 | 0.21094 | 0.9 | 1.2745 | 435.5 | 0.750 | 1.110 |
| 2 | PIN | 0.008216 | 0.4604 | 0.998063 | 0.004517 | 0.006466 | 1.005415 | 51.607 | 0.09844 | 0.9 | 0.7280 | 467.0 | 0.844 | 1.791 |
| 3 | PIN | 0.008419 | 0.8591 | 0.997188 | 0.006419 | 0.009086 | 1.008546 | 20.978 | 0.46406 | 0.9 | 0.6191 | 437.9 | 0.781 | 3.990 |
| 1 | W2 | 0.007824 | 0.6232 | 0.998856 | 0.006025 | 0.007529 | 1.006783 | -0.262 | 0.16875 | 0.9 | 1.4228 | 433.5 | 0.750 | 5.228 |
| 2 | W2 | 0.010621 | 0.5054 | 0.997587 | 0.008906 | 0.013412 | 1.012522 | 7.962 | 0.33750 | 0.9 | 1.0670 | 389.7 | 0.594 | 3.605 |
| 3 | W2 | 0.009938 | 0.5251 | 0.997756 | 0.009941 | 0.014540 | 1.012281 | 6.254 | 0.46406 | 0.9 | 1.3623 | 338.9 | 0.484 | 3.934 |

均值（step1-3）：

| 指标 | PIN | W2 |
|---|---:|---:|
| diff_mean | **0.008201** | 0.009461 |
| pearson | 0.998011 | 0.998066 |
| kl | 0.005856 | 0.008291 |
| log_ppl_abs_diff | 0.008011 | 0.011827 |
| ppl_ratio | 1.007038 | 1.010529 |
| reward_mean | 0.257812 | 0.323437 |
| entropy | 0.8739 | 1.2840 |
| resp_len | 446.8 | 387.4 |
| clip_ratio | 0.792 | 0.609 |

结论（仅 3 步，样本 64/步，不下长程结论）：

- **一致性**：PIN 3/3 步 diff_mean < 0.01（0.00797/0.00822/0.00842），pearson 0.9972–0.9988；W2 step2 为 0.010621（略超 0.01）。PIN 在 step2/3 的 diff_mean、kl、log_ppl_abs_diff 均低于 W2，pinned 未引入一致性恶化。
- **reward**：两步 step3 完全一致（0.46406）；step1 PIN 0.211 vs W2 0.169，step2 PIN 0.098 vs W2 0.338。3 步均值 PIN 0.258 < W2 0.323，但 64 样本/步、countbenchqa 奖励离散，3 步不足以判断优劣。
- **响应长度**：PIN 446.8 vs W2 387.4，clip_ratio 0.792 vs 0.609；PIN 回答更长，可能与 reward 波动相关，后续需更多步确认。

---

## 10. pin_memory 单机 500-step 长跑准备（2026-10-22 16:00 UTC）

脚本：`verl/my_scripts/pref_dsv41/_launch_pin_longrun_500_noprof_1023_v1.sh`（bash -n 通过，尚未执行）。

与 .119 W2 长跑的训练配置差异：**仅 `fsdp_plan.pin_memory`**（.119 显式 `false`；本脚本 `VERL_DSV41_OFFLOAD_PIN_MEMORY=1`，不注入 false，FSDPTurbo 默认 True）。另：无 profiler；experiment/log 名称按节点区分。数据、拓扑、bsz、prompt/response、KL0、util0.6、AdamW/offload、`load_format=auto`、agent/thinking、sync_refill、500 steps 均一致。

启动命令（每个节点一条，`NODE` 自动取 hostname）：

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
nohup bash my_scripts/pref_dsv41/_launch_pin_longrun_500_noprof_1023_v1.sh \
  > /tmp/pin_longrun_$(hostname).nohup.out 2>&1 &
```

### 16:00 可用性阻塞

| 主机 | NPU | 结论 |
|---|---|---|
| .120 | 32 procs，HBM 75.9%（~50 GB/chip），AICore 2–10% | 已有完整训练任务，不能起 16-chip |
| .124 | 16 procs，HBM 16.7%（~8–11 GB/chip），AICore 0% | 已有任务（初始化/加载），不能起 |
| .123 | 16 procs，HBM 14.3%（~6–9 GB/chip），AICore ~0 | 已有任务，不能起 |
| .128 | NPU0 两 chip 100% AICore、56 GB；其余 14 chip 空闲 | 16-chip 单机任务会被 NPU0 卡住 |

这些 NPU 进程的 host PID 在 `npu-smi proc-mem` 可见，但 `/proc/<pid>` 在 SSH 容器内不存在（不同 PID namespace），无法确认是否用户自己的任务。未启动任何新长跑，也未停止任何现有任务。

---

## 11. .131 pin_memory 单机 500-step 长跑启动与巡检（2026-10-22 16:05 UTC 起）

- 节点：80.5.25.131（node-29-131），16 chip 全空闲（procs=0，HBM 4.6%），业务 IP/NIC 自动探测成功。
- 脚本：`verl/my_scripts/pref_dsv41/_launch_pin_longrun_500_noprof_1023_v1.sh`（STEPS=500，pin=ON，无 profiler）。
- experiment：`GRPO-DSV41-countbenchqa-pin-longrun-node-29-131-20261022_160550`
- 日志：`logs/pin_longrun_node-29-131_20261022_160550.outer.log` / `.inner.log`
- 启动命令：`nohup bash my_scripts/pref_dsv41/_launch_pin_longrun_500_noprof_1023_v1.sh > /tmp/pin_longrun_$(hostname).nohup.out 2>&1 &`

启动阶段（16:05–16:24）：数据过滤 450/450（16:08）→ vLLM 权重分片 267/267（16:15）→ 首次 rollout 结束（16:19:56）→ step1 日志（16:24）。无 Traceback/OOM/Killed。

### step 1 指标

| 指标 | 值 |
|---|---:|
| training/rollout_probs_diff_mean | 0.008273 |
| training/rollout_actor_probs_pearson_corr | 0.998761 |
| critic/rewards/mean | 0.239062 |
| actor/grad_norm | 1.616755 |
| response_length/mean | 445.22 |
| actor/kl_loss | N/A（KL0，use_kl_loss=False，日志无该键） |
| timing_s/step | 411.36 |
| timing_s/gen | 216.18 |
| timing_s/update_actor | 137.93 |

巡检策略：启动阶段 2–3 分钟一次；step 1 落日志后改为 10 分钟一次，按 step 记录上表 6 项指标与 timing。

### .131 step 1–4 指标

| step | critic/rewards/mean | actor/grad_norm | actor/kl_loss | response_length/mean | rollout_probs_diff_mean | pearson | timing_s/step | gen | update_actor |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.239062 | 1.616755 | N/A(KL0) | 445.22 | 0.008273 | 0.998761 | 411.36 | 216.18 | 137.93 |
| 2 | 0.196875 | 1.398986 | N/A(KL0) | 427.14 | 0.008741 | 0.998638 | 330.67 | 167.03 | 129.68 |
| 3 | 0.239062 | 1.734730 | N/A(KL0) | 466.95 | 0.007972 | 0.998836 | 357.82 | 154.99 | 165.86 |
| 4 | 0.421875 | 6.229812 | N/A(KL0) | 401.77 | 0.007658 | 0.998659 | 365.19 | 152.96 | 170.80 |

无 Traceback/OOM/Killed；host used 1410–1417 GB。巡检：10 分钟一次，若进程被杀只记录、不重启。

### .131 step 5–7 指标（10 分钟巡检，16:58 UTC）

| step | critic/rewards/mean | actor/grad_norm | actor/kl_loss | response_length/mean | rollout_probs_diff_mean | pearson | timing_s/step |
|---|---:|---:|---|---:|---:|---:|---:|
| 5 | 0.548437 | 3.185642 | N/A(KL0) | 273.56 | 0.010188 | 0.998150 | 307.32 |
| 6 | 0.534375 | 4.485131 | N/A(KL0) | 334.89 | 0.009059 | 0.998439 | 311.42 |
| 7 | 0.689063 | 1.669599 | N/A(KL0) | 190.00 | **0.013006** | 0.996589 | 273.88 |

- step7 diff_mean 首次 >0.01（0.013006），pearson 降到 0.996589；同时 response_length 从 445→427→467→402→274→335→**190** 明显缩短，reward 升到 0.689。三者可能相关，需继续观察是否为单步波动。
- 无 Traceback/OOM/Killed；driver 存活；host used 1420 GB。

| 8 | 0.660938 | 1.777885 | N/A(KL0) | 228.36 | 0.011633 | 0.997781 | 306.31 |

- step8 diff_mean 连续第二步 >0.01（0.011633，较 step7 0.013006 回落），pearson 回升到 0.997781；response_length 190→228。继续观察是否回到 0.01 以下。
- 无 Traceback/OOM/Killed；driver 存活；host used 1417 GB。

### .131 step 9–11 指标（17:18 UTC 巡检）

| step | critic/rewards/mean | actor/grad_norm | actor/kl_loss | response_length/mean | rollout_probs_diff_mean | pearson | timing_s/step |
|---|---:|---:|---|---:|---:|---:|---:|
| 9 | 0.787500 | 3.445502 | N/A(KL0) | 231.80 | 0.011266 | 0.997871 | 383.28 |
| 10 | 0.618750 | 3.502626 | N/A(KL0) | 237.56 | 0.010173 | 0.997647 | 296.29 |
| 11 | 0.703125 | 1.093741 | N/A(KL0) | 232.70 | 0.010335 | 0.997725 | 337.89 |

同 step 对照 .119 pageable（W2）：

| step | .131 pin diff_mean / pearson | .119 pageable diff_mean / pearson |
|---|---|---|
| 9 | 0.011266 / 0.997871 | 0.008693 / 0.998341 |
| 10 | 0.010173 / 0.997647 | 0.010030 / 0.998001 |
| 11 | 0.010335 / 0.997725 | 0.010649 / 0.997165 |

- 同步数下两者一致性同量级（step9 .131 略高，step10/11 相当）；.119 在 step31-33 已恶化到 diff_mean 0.0125–0.0163、pearson 0.980–0.985、resp_len 105–149，说明长程漂移是共同趋势，当前 .131 尚未到该步数。
- 无 Traceback/OOM/Killed；host used 1418 GB。

| 12 | 0.562500 | 1.259871 | N/A(KL0) | 261.83 | 0.009416 | 0.997907 | 299.41 |

- step12 diff_mean 回到 0.01 以下（0.009416），pearson 0.997907；response_length 232.7→261.8。
- 无 Traceback/OOM/Killed；driver 存活；host used 1419 GB。

### .131 step 13–18 指标（18:38 UTC 巡检）

| step | critic/rewards/mean | actor/grad_norm | actor/kl_loss | response_length/mean | rollout_probs_diff_mean | pearson | timing_s/step |
|---|---:|---:|---|---:|---:|---:|---:|
| 13 | 0.745312 | 0.942702 | N/A(KL0) | 301.30 | 0.009162 | 0.997904 | 298.93 |
| 14 | 0.632812 | 1.757494 | N/A(KL0) | 278.08 | 0.010120 | 0.998266 | 368.79 |
| 15 | 0.660937 | 3.449417 | N/A(KL0) | 272.94 | 0.008119 | 0.998231 | 392.53 |
| 16 | 0.618750 | 4.119825 | N/A(KL0) | 307.25 | 0.010423 | 0.998035 | 389.46 |
| 17 | 0.675000 | 2.438303 | N/A(KL0) | 251.84 | 0.010653 | 0.997578 | 334.22 |
| 18 | 0.717187 | 9.579253 | N/A(KL0) | 242.56 | 0.011987 | 0.997512 | 291.14 |

### 运行终止（17:57 UTC）

- 17:57:30 `VllmWorker-2`、`VllmWorker-4` **died unexpectedly (exit code: None)**，executor 开始 shutdown。
- 17:57:39 `EngineCore encountered a fatal error` → `RuntimeError: cancelled` → `EngineDeadError`；executor 对 8 个 worker 先 SIGTERM 后 SIGKILL。
- Raylet 同时 `terminated unexpectedly`，节点因心跳丢失被标记 dead；日志最后 18:01:49（step 18 已记完）。
- **非内存 OOM**：dmesg 无 OOM/killed process；cgroup limit 无限；cgroup max_usage 992.8 GB < 2454 GB；failcnt=0。首个事件是 vLLM worker 被杀/异常退出（exit code None），随后 Ray/vLLM 连锁退出。
- .131 已恢复空闲：NPU procs=0，HBM 4.6%。按用户要求**未重启**，已停止 10 分钟巡检 cron。
