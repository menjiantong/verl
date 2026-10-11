# DeepSeek-V4.1-Flash top32 训练加载：显存 / 主机内存 / 并行策略 归档

> 主题：top32（40 层 / 32 专家 / engram 层 1、14）在双机 16 卡 NPU（node-29-131 + node-29-128，2×16 chip）上的
> **训练/加载全过程中遇到的所有 显存(HBM)/主机内存/并行策略 问题 + 解决方式 + 加载时序逐阶段峰值**。
> 数据来源：worklog（`worklog_dsv41_sft.md`、`worklog_dsv41_direct_shard_load.md`、`worklog_dsv41_top64_grpo.md`、
> `worklog_dsv41_rl.md`、`worklog_dsv41_weight_sync_export.md`）+ 本机日志（`/tmp/sft2_master*.log`、
> `/tmp/grpo50_master.log`（写档时仍在跑）、`/workspace-verl/verl/logs/*.log`）+ 源码行号（只读引用）。
> 写作时间：2026-10-11（RL 50 步运行至 step 9 时）。
> **本文件是归档说明，不修改任何运行中的代码/脚本/配置。**

---

## 0. 一句话结论

- 单机 16 卡跑不动 top32 的原因从来不是"模型太大"，而是**并行切分口径**：专家三件套（params+grads+优化器态）在
  EP8/efsdp=1 下单卡常驻约 **44GB**（占 ~55GB 峰的 80%）。**`expert_fully_shard_parallel_size 1→2` 一处改动**
  把每卡权重直读从 11.03→**5.75 GiB**、step1 峰值从 55.15→**30.42GB**，step2+ 稳态 **41.7GB/61.3GiB**，训练从
  "三次逐位相同地死在 step2" 变成 116 步 23 分钟一次跑完（`/tmp/sft2_master5.log`）。
- 主机内存的**结构性占大头**是 engram 表（每 rank 45.8 GiB；SFT 每节点 2 份 ≈733 GiB；RL 再叠引擎一份 ≈366 GiB）
  与 AdamW fp32 态（全局 ≈426GB，推算）。SFT 双机稳态 host ≈**1.24 TB**；RL 16 卡稳态 host ≈**1.66 TB**。
- 加载路径的关键修复是"**每个 rank 直读自己的分片**"（`VERL_DSV41_DIRECT_SHARD_LOAD=1`）：跨机 gloo 逐张量广播实测
  只有 ~30MB/s（小时级），直读后物化 **4~44 秒**、每卡只读 5.75/11.03/11.51 GiB。

---

## 1. 事实底座（权重构成 / 平台口径 / 并行公式）

### 1.1 权重构成（2026-10-11 现场复算，脚本见 §1.4）

来源：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16/model.safetensors.index.json` +
逐分片 safetensors header（shape×dtype 求和）：

| 类别 | 大小 | 张量数 | 说明 |
|---|---|---|---|
| dense（非 experts 全部：文本 dense + mtp dense + vision/aligner） | 14.07 GiB | 1127 | 含 mtp dense 0.99、vision 0.90 |
| experts（全部 "experts" 命名：主模型 routed + shared + mtp experts） | 112.52 GiB | 5121 | 含 mtp experts 25.51 GiB / 1161 张（训练侧不建） |
| **engram（2 张表，layers 1/14）** | **366.81 GiB** | 8 | 每表 ~196.6GB；行数 384,006,168 / 384,016,682，dim 256，bf16 |
| 合计 | **493.40 GiB** | **6256** | 与任务给定值逐项一致（dense 14.07/1127、experts 112.52/5121、engram 366.81/8） |

训练模型实际构建/加载的量（去掉 mtp 与 vision）：
- **主模型可训练参数 ≈ 99.19 GiB**（dense/main 12.17 GiB/758 张 + experts/main+shared 87.01 GiB/3960 张，bf16）
  → 约 53.3B 参数；
- 引擎自报 `DeepseekV41ForCausalLMAdapter contains 250.14B parameters`（含 engram 全表 2×98.3B=196.6B）
  ——`/tmp/sft2_master5.log:115`；
- 加载条目：`materializing 924 parameters + 40 checkpoint buffers`，直读 `964 tensors`
  ——`/tmp/sft2_master5.log:127,128`（RL 侧同口径：`/tmp/grpo50_master.log:1257,1258`）。

### 1.2 平台口径（全部后续数字的背景）

| 项 | 值 | 来源 |
|---|---|---|
| HBM/卡 | 物理 65536MB；**torch 可用 61.27 GiB**（≈-2.4GB 驱动/HCCL）；OOM 时报 "61.27 GiB total capacity" | `/tmp/sft2_master4.log:219`；worklog_direct_shard_load §六.3 末 |
| 主机内存/节点 | total 2454 GB / available ~2418-2261 GB | worklog_top64 §5.4；`/tmp/grpo50_master.log` 采样 |
| Ray 杀线 | 默认 `0.95×2454=2331GB`；脚本已提 `RAY_memory_usage_threshold=0.995`（≈2442GB） | worklog_top64 §5.5b/§5.8；`scripts/run_sft_dsv41_2node.sh:71` |
| 节点时钟 | .128 比 .131 慢约 4 分钟（判读跨机时间戳时换算） | worklog_direct_shard_load §四 |
| 主机内存指标 | `perf/cpu_memory_used_gb = psutil.virtual_memory().used/1GiB`（节点口径，每 step 上报） | `verl/verl/workers/engine_workers.py:221` |

### 1.3 并行公式（本档统一口径）

```
参数/rank        = dense_main / FSDP   +   experts_main+shared / (EP × EFSDP)      [+ buffers]
梯度/rank        = 同参数（bf16，反向期间存在）
优化器态/rank    = 8B × 53.3B / FSDP_world   （fp32 exp_avg + exp_avg_sq，等分，推算）
engram/rank      = 366.81 GiB / EP = 45.8 GiB（EFSDP 不分 engram；每 EP 副本各持全表）
engram/节点      = 16 ranks × 45.8 GiB = 733 GiB（= 该节点上 2 个 EP 组副本）
激活/rank        ≈ 20MB × 每卡 token 数   （经验拟合，见 §3.E5；序列越长线性越大）
```

实测对表（权重直读量，公式 vs 日志）：

| 配置 | 公式（推算） | 实测（"slice-read directly (X GiB this rank)"） | 出处 |
|---|---|---|---|
| 双机 FSDP32 / EP8 / **efsdp=1** | 12.17/32 + 87.01/8 = **11.26 GiB** | **11.03 GiB** | `/tmp/sft2_master{1,2,3,4}.log`（v1-v4） |
| 双机 FSDP32 / EP8 / **efsdp=2** | 12.17/32 + 87.01/16 = **5.82 GiB** | **5.75 GiB** | `/tmp/sft2_master5.log:128`（v5 成功跑） |
| 双机 FSDP16 / EP8 / efsdp=1（RL 16 卡） | 12.17/16 + 87.01/8 = **11.64 GiB** | **11.51 GiB** | `/tmp/grpo50_master.log:1258` |
| 单机 8 卡冒烟（FSDP8/EP8） | 12.17/8 + 87.01/8 = 12.4 GiB | **12.46 GiB** | worklog_direct_shard_load §三.6 |

> 两者差 ~2%（公式多算的部分：部分张量 Replicate/未切成精确字节；量级吻合）。
> **注意 efsdp 只切"专家维（dim0）"**，dense 走 FSDP、engram 走 EP 行切——三者互不干扰。

### 1.4 权重复算脚本（本次用，只读）

```python
import json, os, struct
p="/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16/model.safetensors.index.json"
wm=json.load(open(p))["weight_map"]; GiB=1024**3; cache={}
def sz(name):
    f=os.path.join(os.path.dirname(p), wm[name])
    if f not in cache:
        with open(f,'rb') as fh:
            n=struct.unpack('<Q', fh.read(8))[0]; cache[f]=json.loads(fh.read(n))
    h=cache[f][name]; nb=1
    for s in h["shape"]: nb*=s
    return nb*{"BF16":2,"F16":2,"F32":4,"I8":1,"U8":1,"I64":8}.get(h["dtype"],2)
cat=lambda n: "engram" if ".engram." in n else ("experts" if "experts" in n else "dense")
```

---

## 2. 【A】预估 vs 实际 对照表

> 统一背景：53.3B 可训练参数、每卡 HBM 61.27 GiB、每节点主机 2454 GB。
> "死亡点"=决定该配置不可用的那次 OOM/崩溃；公式行标"推算"，日志行标实测。

| # | 配置（重启后实际跑的） | 预估公式 | 实测峰值（device / host） | 死亡点 | 修复 | 修复后占用 |
|---|---|---|---|---|---|---|
| 1 | **单机16 卡 FSDP16 / EP8 / efsdp=1 / param_offload=True（proven）** | 参数 11.6 + 梯度（瞬时 11.6）+ 优化器 8B×53.3B/16=**26.6GB** + 激活 → 反推 ~57GB 量级 | smoke：max_alloc **28.62** / resv 51.39（61.3GiB 卡）；host `cpu_memory_used_gb` **2068GB**（worklog_sft §5） | 长跑撞 **host 泄漏 +5GB/进程/步**（§3.E3，v4 死在 +358GB 跳变）；步速 144s/步 → 1 epoch 18h39m | 换双机（见 #3/#4） | — |
| 2 | **单机16 卡 no-offload**（v7/v9/v11 试验） | 参数 11.6 + 梯度 11.6 + 优化器 26.6 = 49.9 + 激活 ~7.5 ≈ **57.4**（推算，与实测死点吻合） | **57.4GB alloc**（v7）/ 57.76（v9）/ 57.53（v11） | 均死在"差 182MB 的分配"（同 #3 签名）；无 host 泄漏、步速 42s/步 | 无（差 1-2GB） | host 基线 1305GB（省 760GB vs offload 配置） |
| 3 | **双机32 卡 FSDP32 / EP8 / efsdp=1 / no-offload** | 脚本头注预估"params 3.25+grads 3.25+AdamW ~13+激活 ≈41GB"（**低估**：3.25 按 experts/32 算，实际 EP8 只切 8 份 → experts/8=10.88GiB，真实 params+grads ≈23.3GiB） | step1 峰 **55.146934509277344**（v2/v3/v4 逐位一致）；host cpu 1307GB（v2） | **step2 backward** `Tried to allocate 182.00 MiB (61.27 total; 58.63 allocated; 216.46 MiB free)`（v2/v3/v4 三次逐位同签名） | **efsdp 1→2**（唯一改动） | 见 #4 |
| 4 | **双机32 卡 FSDP32 / EP8 / efsdp=2 / no-offload（SFT v5，proven）** | 参数 5.82 + 梯度 5.82 + 优化器 8B×53.3B/32≈**13.3GB**（经验 12.9）+ 激活 ~10.4 ≈ 41-44（推算；worklog 三数据点模型预测 44.5） | step1 **30.42** → step2 40.01 → 稳态 **41.73** alloc / 56.43 resv（61.27 卡）；host **1233.8→1239.3GB** 稳定无泄漏 | 无（116/116 步跑完） | — | 12s/步量级（115 步 23m13s）；ckpt 55.33GB/rank、全量 1.65TB |
| 5 | **RL 双机16 卡 FSDP16 / EP8 / efsdp=1 / param+optimizer offload（当前 50 步，proven）** | 直读 11.64；update 需把参数+梯度+优化器拉上卡：11.5+11.5+~13+激活 ≈ 38-40（推算） | 加载后 `After FSDP allocated 0.21`（参数在主机）；update 峰 **38.41** alloc / 54.92 resv；host **1656.9→1661.5GB**；vLLM 睡眠期释放 17.4GB（史证） | 无（写档时 step 9/50 正常运行）；历史：efsdp=2 时 sync 权重损坏（§3.E8） | RL 保持 efsdp=1 + LOCAL_EXPERT_EXPORT=1 | step 373-463s；sync 16.4-16.9s |
| 6 | **top64 单机（校准行，2026-09-30）** | engram 2×196.6GB + 专家 offload 362GB + AdamW ~395GB + 引擎 engram 393GB ≈ **2.7TB > 2.4TB 物理**（保留 ref 时） | 训练侧加载完 1.34TB → 引擎初始化峰 **2.35TB** | `ray.exceptions.OutOfMemoryError: 12 worker(s) were killed`，节点 2422.66/2454.43GB | 去 ref/KL + 换 top32 + 提杀线到 0.99 + 启动前清残留 | top32 同期 host 874GB（训练侧加载完）→ 1526GB（ready to fit）→ 1710GB |

---

## 3. 【B】关键事件卡（错误原文 + 数据 + 归因 + 修复）

### E1 双机 efsdp=1：step2 backward OOM（三次逐位复现）→ efsdp=2 根治

**错误原文**（`/tmp/sft2_master4.log:160,219`，v2/v3 同签名）：
```
E1011 03:01:53.286527 741835 NPUCachingAllocator.cpp:1332] [rank:6] NPU out of memory.
Tried to allocate 182.00 MiB (NPU 6; 61.27 GiB total capacity; 58.63 GiB already allocated;
58.63 GiB current active; 216.46 MiB free; 58.80 GiB reserved in total by PyTorch).
torch.OutOfMemoryError: ...（traceback: transformer_impl.py:767 loss.backward()
  → FSDPTurbo/fsdp_turbo/ops/npu/grouped_matmul.py:60 (MoE 反向)）
```
**数据**：step1 峰 55.146934509277344（`/tmp/sft2_master2.log:151`、`master3:151`、`master4`）；死亡点逐位一致。
**归因**：EP8/efsdp=1 → 每卡常驻 4/32 专家的 params+grads+优化器态 ≈44GB（占峰的 80%）；micro/offload/FSDP32 都动不了它。
**修复**：`engine.turbo_config.distributed.expert_fully_shard_parallel_size=1→2`（mesh 自动 edp=2/efsdp=2/ep=8，拓扑不变）。
**验证**：每卡直读 11.03→5.75GiB；step1 55.15→**30.42**；step2 40.01 **通过**；融合专家 `_assemble_fused_expert_window` rank0 独立装配 bit-exact。

### E2 optimizer_offload 的否定实验：峰值逐位不变、步速 4× 劣化、host +500GB

**数据**（v3 = v2 同配置仅开 optimizer_offload，`/tmp/sft2_master3.log:151` vs `master2.log:151`）：

| 指标 | v2（no-offload） | v3（optimizer_offload=True） |
|---|---|---|
| step1 max_memory_allocated_gb | 55.146934509277344 | **55.146934509277344（逐位相同）** |
| step1 cpu_memory_used_gb | 1307.45 | **1810.98**（+503GB） |
| 步速 | 47.10 → 84.07 s/it | **188.68 → 225.48 s/it（4×）** |
| 死亡点 | 58.63+182MiB | 同 |

**归因**：优化器态不是 step2 +3.5GB 的来源；opt-offload 在本引擎不触达主导项，只增加每次 to(device)/to(cpu) 的搬运与主机占用。
**处置**：两处默认回 False（`scripts/run_sft_dsv41_2node.sh:26-27`）。

### E3 单机 offload 配置的主机泄漏：+~5GB/进程/步 ×16 = +80GB/步

**证据链**（worklog_sft §阶段③主机泄漏确诊；采样工具 `/tmp/host_mem_sampler.sh`，15s 一拍的 CSV **`/tmp/host_mem_trace.csv` 现已不在盘上——见 §7 存疑**）：
- v4 死因定量：工作进程 RSS **+~5GB/进程/步 ×16 进程 = +80GB/步线性泄漏**；"358GB 跳变" = 4.5 步累积；
- 否定项：v6 关 `optimizer_offload` 后泄漏照旧（**+108GB/2min**）→ 与优化器 offload 无关；
- 嫌疑：param/offload_policy 的 FSDPTurbo 逐步机制，**泄漏量 ≈ 每 rank 参数分片 6.5GB/步**；
- 对照：全 no-offload（v7）**无泄漏**，host 基线 1305GB；
- engram 787GB 是 CANN 注册主机内存（`aclrtMallocHost`），**ps RSS 不可见**——判读内存必须用 `free/MemAvailable`，不能只看进程 RSS。
**处置**：SFT/RL 双机生产配置弃用"单机+offload 长跑"，改双机 no-offload（SFT）/ 双机 offload（RL 16 卡验证无此泄漏，host 平稳 1656→1662GB/9 步）。

### E4 top64 单机：Ray 因主机内存打满杀 12 个 worker

**错误原文**（worklog_top64 §5.5b）：
```
ray.exceptions.OutOfMemoryError: 12 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 80.5.25.131) was 2422.66GB / 2454.43GB
```
**内存账（实测）**：训练侧加载完 1.34TB（engram 393GB pinned + 专家 offload 181GB + AdamW ~395GB）→ 引擎初始化峰 **2.35TB**（再加引擎 engram 393GB = 16 worker × 22.89GiB）。Ray 默认杀线 0.95×2454=2331GB 先到。
**隐藏推手**：上一轮 serve 残留进程持 **~700GB 设备注册主机内存（ps 不可见）**，`free` 只剩 1.4TB。
**处置**：启动前自动清扫 `VLLM::`/`ray::` 残留 + 杀线 0.995 + **换 top32**（493.4GiB，训练侧同期 host 从 1.34TB 降到 874GB）。

### E5 RL（top32 单机，2026-10-08/09）：从 per-op 尖峰到"与 token 无关的 ~60GB 平台顶"

分三层，逐层被测量钉死：
1. **单次 gather 尖峰**：`Tried to allocate 19.50 GiB`（seq2048）——`indexed_sparse_attention_torch` 的 4-D int64 索引物化（b·s·topk·hd·8B，随 s 线性）。
   修复：**query 维分块** `FSDP_TURBO_ATTN_CHUNK=256`（`ops/cpu/sparse_attention.py`）→ 单次 ~2.4GiB/块，数值严格等价。
2. **每层 saved-for-backward 累积**：chunk 后仍 56.56GiB 活跃 → `_AttentionRecomputeFn`（`FSDP_TURBO_ATTN_RECOMPUTE=1`）把每层 attention 中间量 ~5GiB → O(1)。
3. **平台顶（真正的墙）**：`512/B8（2613 tok/rank，峰 60833MB）与 1024/B2（1165 tok/rank，峰 60856MB）峰值几乎相同` → **update 峰值不随 response 长度缩放**；逐层探针（`FSDP_TURBO_MEM_DEBUG=1`）显示 FWD 单调累积 **1.32→46.54GB（≈1.13GB/层，无中途释放）**，BWD 期间 alloc 稳在 47.6-48.7GB 不释放。
   结论：峰值 ≈ Σ(逐层 saved-for-bwd ~45GB) + base(10-15GB)。
   **ffn 范围 recompute**（`memory.recompute=true` + `recompute_plan=["model.layers.{*}.ffn"]`）**-4.7GiB**、grad_norm 逐位一致 → 恰是 run12（死于 55.03+300MiB）与 run13（alloc 37.98/resv 54.90 **存活**）的生死差。
   **整层 recompute 崩**：与层内 sparse-attention 自带 checkpoint 嵌套冲突（`CheckpointError: different number of tensors ...`）→ 放弃。**swap_activation 也判死**：backward 取回的 device 张量无释放 → 逐层累积回 ~60GB（vendor 实现缺陷）。
4. `hc_pre/hc_post` 的 `[b,s,4,4,d]`（16×hidden/层）物化是另一个大件，已用 matmul 等价改写消灭（hc_post 需转置 comb）。

### E6 clip_grads 的 13GB/rank 瞬态（梯度裁剪 fp32 副本全存 list）

**位置**：`FSDPTurbo/fsdp_turbo/training/clip_grads.py::compute_grad_norm` 原实现把每个梯度的 fp32 副本存 list 直到函数尾（52e9 params/16 rank ≈ **13GB/rank 瞬态**）。
**修复**：逐参数流式累加（现实现 `clip_grads.py:179-184`：`g_f = local_g.float(); term = g_f.norm()**2` 立即释放，按 shard-group 累加标量；数值逐位等价）。
**旁证**：该文件顶部还留着 CPU-offload 修复 `_reduction_device/_all_reduce_on_group_device`（`:82-114`，把 CPU 标量先搬到组设备再 all_reduce），修的是 `No backend type associated with device type cpu`（top64 §5.6）。

### E7 efsdp=2 的 RL 权重同步损坏（内存改动的反噬）

- **症状**：reward 恒 -1、response 全部顶格 1024、dump 出来是**多语种乱码词沙拉**（不是循环思考）、`pearson 0.61-0.70`（健康值 0.978）。
- **判据差异**：SYNC-PROFILE 同模型同导出仅 efsdp 不同：
  ```
  efsdp=1（正常）: sender 1364 tensors, 25.87 GiB, 51 buckets | export 0.4s, flush 10.2s
  efsdp=2（损坏）: sender 1364 tensors, 20.60 GiB, 43 buckets | export 0.5s, flush 7.9s
  ```
- **代码路径**：`fsdp_turbo_dsv41_impl.py::_export_local_experts`（`:865`）要求"专家 dim0 **恰好一个** mesh 维切分"（`len(shard_mesh_dims)==1`，`:875-879`）；efsdp=2 时 efsdp 与 ep 两个维都切 dim0 → 回退 `super()._export_param` 全 gather + `first_expert_id=0` 全量重切，与 LOCAL 模式的接收端契约（rank i 收发同段专家）错位 → vLLM 专家权重损坏。
- **修复**：RL 侧 `EFSDP_SIZE=1`（历史 proven；当前 50 步 run 的 resolved config 实测 `expert_fully_shard_parallel_size: 1`）。**SFT 侧 efsdp=2 与 sync 无关，维持不变**（两种模式用同一脚本不同 env）。
- **遗留**：脚本 `scripts/run_grpo_2node.sh:237` 默认已是 `EFSDP_SIZE:-2` → **RL 复跑必须显式 `EFSDP_SIZE=1`**，否则默认撞回损坏路径（见 §6 黑榜）。

### E8 跨机 gloo 全量广播 ~30MB/s（直载的动机）

- 双机首启三次死亡全部落在"rank0 读全量 state dict + 逐张量 gloo 广播"窗口（8/18/10 分钟，`Raylet is terminated` / `gloo Connection closed by peer`）→ 高度指向 .124 当时被别租户占用（NPU 8-15 被占 ~49GB/卡）。
- 实测吞吐 **~30MB/s**（80.5.25.x 管理网，enp48s3u1u1）；493GB 权重足以"小时级"。
- 修复 = **直读切片加载**（§4 详述），跨机搬运归零；副作用：物化时间从单机 399s / 广播 27-90s+ 降到 **4.3-13s**。

### E9 SFT 单机 param_offload=False 首跑 OOM（编外案）

`E1010 19:14:46 NPU out of memory`（采样时卡 64885/65536MB；rank0 首个 forward：参数常驻 + per-layer 全参 all-gather + 编译工作区超限）；micro8 即使带 offload 也 OOM（`max_batch_size` engram 缓存墙与此并行修复）。
**教训**：单机上"关 param_offload"这条省时路线比 offload 更需要显存余量——它把每步的搬运省了，但把参数常驻进了 61.3GiB 的卡。

---

## 4. 【C】加载时序逐阶段峰值

### 4.0 加载路径总览（2026-10-09 起，`VERL_DSV41_DIRECT_SHARD_LOAD=1` 默认开）

```
[进程/通信初始化] → [模型构建（专家 meta）] → [Before FSDP]
  → [① 每 rank 直读自己的分片（safetensors get_slice）→ 装进 DTensor/模块]
  → [② buffer bit-exact 校验 + meta 替换（cpu_offload 时落 CPU）]
  → [③ engram host 表：每 rank 切片读自己的行区间 → copy_ 进 pinned 主机表 → 冻结]
  → [④ After FSDP 基线（参数常驻或已 offload）]
  → [⑤ 第一步 forward/backward/优化器（优化器态在此首次分配）]
  → [⑥ 稳态每步]
```

关键代码位置（`/workspace-verl/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`）：
`_plan_direct_shard_load:226`（用 `compute_local_shape_and_global_offset` 算窗口，与 DTensor 自身同源；fail-closed 回退广播）
`_read_direct_shards:324` / `_load_direct_shards:393`（真实参数原地 `copy_`、仅 meta 替换、`cpu_offload` 时 `.cpu()` 镜像 DCP 行为）
`_verify_direct_shards:457`（rank0 抽样 + 融合专家独立装配 + 40 buffer bit-exact）
`_materialize_dsv41_parameters:1017` / `_load_dsv41_parameters_directly:1120`（两条路径与 fail-closed）
`_load_host_engram_tables:1174`（行区间覆盖性校验 + 冻结 + `_grad_keepalive` 改写）
`_export_local_experts:865`（RL sync 快路径）／`_export_param:811`

### 4.1 SFT v5 双机成功跑（`/tmp/sft2_master5.log`，FSDP32/EP8/efsdp=2，**全 no-offload**）

时间轴（有日志时间戳的用时间戳；进程打印无时间戳的按顺序+时长标注）：

| # | 阶段 | 时间/耗时 | 设备/主机峰值 | 证据（行号） |
|---|---|---|---|---|
| 0 | Ray 集群 + 32 worker 加入 | 03:08:10 ray start → 03:08:16 ready → "waiting for the slave" → 03:08:56 提交 job → worker 03:09:32 起 | — | master5:1-26（[sft2] 行）、03:08:10-16 时间戳 |
| 1 | 进程/通信初始化 | 03:12:07-08 `[Gloo] Rank 0 connected to 31 peer ranks`（32 rank 全连接） | device used 15.76/61.27（构建即占用） | master5:76,91 |
| 2 | 模型构建（专家 meta） | `Before FSDP: allocated 15.34 / reserved 15.36 / device 15.76` | dense 真建 + 专家 meta | master5:116 |
| 3 | **① 权重切片直读** | `checkpoint read: 964 tensors slice-read directly (**5.75 GiB this rank**; no cross-rank broadcast)`；03:14:40 出现首条 NPUCachingAllocator 告警（读/装进行中） | 读入 CPU 切片 → 装设备；无跨机搬运 | master5:128,80 |
| 4 | ② 校验 + 物化完成 | `materialized in **43.6s** (direct shard load)`；`buffers loaded: 40 (bit-exact)`；融合专家 rank0 独立装配 matched；`5 sampled shard(s) bit-exact` | — | master5:131-134 |
| 5 | ③ buffers 上卡 + **engram host 表** | `moved 139 buffers to npu`；`host-offload Engram tables loaded: 2 table(s), **45.8 GiB per rank in 3.1s (frozen)**` | 主机 +45.8GiB/rank（pinned）；**不占 HBM** | master5:135（moved 139）, 137 |
| 6 | ④ `After FSDP` 基线 | `allocated **5.77** / reserved 15.42 / device 15.82/61.27` | **= 参数常驻（5.75GiB 直读量）**，reserved 里 ~10GB 是通信/引擎缓冲 | master5:138 |
| 7 | ⑤ 第一步（含编译） | 03:20:35 rank16 `is_fx_tracing`（首 trace）→ **03:23:07 rank0 首步**；step1 26.40s/it | step1 `max_memory_allocated **30.42** / reserved 44.72`；**host 1233.80GB** | master5:141-143,152 |
| 8 | ⑤ 第二步（优化器态在此入账） | 16.76s/it | step2 `alloc **40.01** / resv 53.17`；host 1234.52 | master5:153 |
| 9 | ⑥ 稳态 | step3+ 11.4-13.2s/it；115 步累计 23:13（≈12.1s/步） | alloc 40.0 → 40.55（step7）→ **41.73**（step24 起恒定）/ resv 55.06-56.43；host 1234.8→**1239.3GB 单调微增（无泄漏特征）** | master5:154-267 |
| 10 | ckpt 存盘 | 03:46:19 → 03:48:10（~2 分钟），每 rank **55.33GB**（engram 45.8GiB + trainable ~7.9GiB），全量 **1.65TB**（双机各写本地盘） | — | master5 尾部 `[Rank N] Saved model ...` 行；worklog_sft §阶段③完成 |

**阶段分解（本配置下）**：
```
每卡 HBM（FSDP32/efsdp=2, micro1）：
  Before FSDP 15.34 ──(直读+FSDP 切分+engram 上主机)──► After FSDP 5.77（=参数）
  step1 峰 30.42 = 参数 5.8 + 梯度 5.8 + 激活/编译 ~12 + 通信缓冲
  step2+ 峰 41.73 = 上式 + 优化器态（step1→step2 实测 **+9.59GB**；推算 8B×53.3B/32≈13.3GB，见 §7 存疑）
每卡 host：engram 45.8GiB + 参数/梯度/优化器常驻 0（no-offload 全在卡上）→ 节点 16 rank 合计 ~733GiB + 杂项 = 实测 1.24TB
```

### 4.2 RL 50 步（`/tmp/grpo50_master.log`，2×8 卡 = 16 NPU，FSDP16/EP8/efsdp=1，**param+optimizer offload**）

| # | 阶段 | 时间/耗时 | 设备/主机 | 证据（行号） |
|---|---|---|---|---|
| 0 | 集群/初始化 | 08:38:31 precheck → 08:38:47 ray ready → slave 加入（16 NPU）→ 08:39 提交 | — | grpo50:1-36 |
| 1 | 数据/worker 组 | 08:43:09 `train dataset size: 512, val 64`；08:43:11 worker group | — | grpo50:1066-1071 |
| 2 | 模型构建 `Before FSDP` | `allocated 15.54 / reserved 16.02 / device 16.42/61.27` | dense 真建 + 专家 meta | grpo50:1250 |
| 3 | **① 权重直读** | `checkpoint read: 964 tensors slice-read directly (**11.51 GiB this rank**; no cross-rank broadcast)`；`materialized in **7.1s**` | 读自 `/mnt/share/.../DeepSeek-V4.1-Flash-top32-sft116`（SFT 导出目录） | grpo50:1258,1263 |
| 4 | ② buffers 上卡 | `moved 139 buffers to npu`；`buffers loaded: 40 (bit-exact)` | — | grpo50:1264 前后 |
| 5 | ③ engram host 表 | `host-offload Engram tables loaded: 2 table(s), **45.8 GiB per rank in 24.2s (frozen)**` | 主机 +45.8GiB/rank | grpo50:1269 |
| 6 | ④ `After FSDP` 基线 | `allocated **0.21** / reserved 16.10 / device 16.50/61.27` | **参数已按 offload_policy 落到主机**（0.21GB=纯 buffer/杂项）——与 SFT 的 5.77 形成对照 | grpo50:1271 |
| 7 | 引擎构建（vLLM 侧） | 08:54:12 `indexer_topk` 警告（引擎 build 中）；引擎 engram 22.89 GiB/rank 注册主机 | — | grpo50:1274；`/tmp/serve_sft116.log:10416` |
| 8 | ⑤ gen#1 → update#1 → sync#1 | gen 起点 ~08:56:20；**09:01:51** `on_sample_end`（gen 完）→ sleep（2 replica，~8s）→ update → sync；step1 全程 **463.46s** | step1 `alloc **36.68** / resv 54.92 / cpu **1656.9GB**`；gen 295.95s / update 124.56s / sync 16.40s | grpo50:6986-6991；7067 |
| 9 | ⑥ 稳态（step2-9） | step 373-420s（gen 266-296s、update 75-125s、sync 16.4-16.9s、old_log_prob ~9s） | alloc **38.41** 恒定 / resv **54.92** 恒定；host **1659.5→1661.5GB**（无泄漏） | grpo50:7150 起；§5 表 |
| 10 | 权重同步分段（每步） | sender `1364 tensors, 25.87 GiB, 51 buckets | export 0.4-1.1s, flush 9.8-10.4s`；receiver `53 buckets, 1364 tensors | views 0.0s, load 8.5-8.9s`；**step2(receive+load) 合计 11.3-13.1s** | — | grpo50:7032-7046 |

**RL 每步实测量（写档时 step 1-9，`/tmp/grpo50_master.log` 解析）**：

| step | reward mean | grad_norm | resp_len (clip) | alloc / resv (GB) | host (GB) | step_s | update_s | sync_s |
|---|---|---|---|---|---|---|---|---|
| 1 | -1.0 | 0.0 | 496.1 (0.30) | 36.68 / 54.92 | 1656.9 | 463.5 | 124.6 | 16.4 |
| 2 | **-0.96875** | **0.341** | 420.3 (0.22) | 38.41 / 54.92 | 1659.5 | 387.1 | 91.5 | 16.9 |
| 3 | -0.96875 | 0.240 | 342.1 (0.16) | 38.41 / 54.92 | 1660.2 | 388.7 | 97.7 | 16.9 |
| 4 | -0.96875 | 0.195 | 466.7 (0.25) | 38.41 / 54.92 | 1660.7 | 420.6 | 112.9 | 16.4 |
| 5-7 | -1.0 | 0.0 | 281-493 | 38.41 / 54.92 | 1660.9-1661.5 | 373-409 | 75-108 | 16.5-16.9 |

（reward -0.96875 = 64 样本中 1 条答对；grad_norm>0 的步与答对步一致 —— efsdp=1 修复后的预期形态。）

### 4.3 offload 语义专节（"param_offload / optimizer_offload 到底搬什么"）

**代码语义**（`/workspace-verl/verl/verl/workers/engine/base.py:314-335`）：

```python
def _context_switch(self, device):
    if self.disable_auto_offload: return
    if device != "cpu" and not (param_offload or optimizer_offload): return
    if   mode == "eval":  engine.to(device, model=param_offload, optimizer=False, grad=False)
    elif mode == "train": engine.to(device, model=param_offload, optimizer=optimizer_offload,
                                    grad=param_offload)
def __enter__: self._context_switch(get_device_name())   # 训练/eval 入卡
def __exit__ : self._context_switch("cpu")               # 出卡回 CPU
```

即：**param_offload=每步把 模型+梯度 在 host↔device 之间整体往返；optimizer_offload=每步把优化器态往返**。
`engine.offload_policy=${PARAM_OFFLOAD}`（`scripts/run_sft_dsv41_2node.sh:` 引擎参数段）——三者联动。

**实测三档对照**：

| 配置 | `After FSDP` 设备 allocated | 每步代价 | 备注 |
|---|---|---|---|
| param_offload=True（RL 现役） | **0.21GB**（参数在主机） | update_actor 75-125s（含 H2D + 计算）；单机全 offload SFT 144s/步 | 显存最省；`allocated` 就是参数是否在卡的直接证据 |
| param_offload=False, optimizer_offload=True（v3 反面教材） | — | **188s/步（4× 劣化）**，设备峰逐位不变、host +503GB | 混搭是病态组合（E2） |
| 全 no-offload（SFT v5 现役） | **5.77GB**（= 参数 5.75GiB 直读量） | 12s/步 | 需要"每卡放得下参数+梯度+优化器"（efsdp=2 后成立） |

**直读与 offload 的交汇点**：`_load_direct_shards(..., cpu_offload=bool(offload_policy))`（`:393`）——meta 替换在 cpu_offload 时显式 `.cpu()`（`:438`），镜像 DCP `StateDictOptions(cpu_offload=True)` 的终态（"npu mesh + CPU local"）；否则（no-offload）直接落卡（冒烟实证：v1 现代码 param local=npu → 首 forward 被 CPU-offload 校验拒绝；修复后 forward OK，worklog_direct_shard_load §三.5）。

---

## 5. 主机内存账（三口径）

| 口径 | 构成（每节点） | 公式/推算 | 实测 |
|---|---|---|---|
| **SFT 双机 v5**（FSDP32/efsdp2/no-offload） | ① engram：16 rank × 45.8GiB = **733 GiB**（2 个 EP 组副本）② AdamW fp32：8B×53.3B/32×16 ≈ **214GB** ③ 参数+梯度常驻 0（在卡上）④ 数据/杂项 | ≈1.1TB | **1233.8 → 1239.3GB**（`/tmp/sft2_master5.log:152,267`），单调+5.5GB/24min（≈噪声级，无泄漏） |
| **RL 16 卡**（FSDP16/efsdp1/offload） | ① trainer engram：8 rank × 45.8 = **366 GiB** ② engine engram：16 worker × 22.89 = **366 GiB** ③ 参数+梯度+优化器 offload 常驻（全局 ≈106+106+426GB，按分片摊到节点）④ vLLM 权重/KV 池页 | ≈1.6TB | **1656.9 → 1661.5GB**（§4.2），9 步平稳 |
| **top64 单机**（反面） | engram 393×2（trainer+engine）+ 专家 offload 362 + AdamW ~395 + … | 保留 ref ≈2.7TB **> 2.4TB 物理** | 训练侧完 1.34TB → 引擎 init 峰 **2.35TB** → Ray 杀 12 worker（E4） |

**每节点还有两类"隐形内存"，判读体系务必包含**：
1. **CANN 注册主机内存**（`aclrtMallocHost`，engram 表走 UVA）——**ps 的 RSS 看不见**，只有 `free/MemAvailable` 和 Ray 的监控能看见；
2. 僵尸/残留进程（`[VLLM::Worker_TP] <defunct>`、孤儿 ray worker 每个 20-30GB）——启动前必须 `pgrep` + `npu-smi info -t proc-mem` 双向核对（曾累积到 ~1TB，导致下次启动静默死）。

---

## 6. 【D】结论速查 + 复跑建议

### 6.1 配置 → 峰值 → 适用场景（一页表）

| 配置 | 每卡直读/常驻权重 | 设备峰值 | 主机峰值 | 适用场景 |
|---|---|---|---|---|
| 单机16 / FSDP16 / efsdp1 / 全 offload | 参数在主机（HBM≈28.6 峰） | **28.6**（smoke） | **2068GB + 泄漏 80GB/步** | ❌ 长跑（泄漏）；短冒烟/自测可用 |
| 单机16 / 全 no-offload | 11.6GiB 常驻 | **57.4 死**（差 182MB） | 1305GB | ❌ 差 1-2GB |
| 双机32 / FSDP32 / **efsdp=1** | 11.03GiB | **55.15 死 step2** | 1307（opt-off 时 1811） | ❌ 历史死亡配置 |
| **双机32 / FSDP32 / efsdp=2** | **5.75GiB** | **30.4 → 41.7 稳态**/61.3 | **~1.24TB** | ✅ **SFT 生产**（12s/步） |
| **双机16 / FSDP16 / RL / efsdp=1 + offload** | 11.51GiB（加载一次性） | update **38.4**/61.3 | **~1.66TB** | ✅ **RL 生产**（400s/步） |
| 推理 vLLM DP2×TP8（单机16 卡） | 每 worker engram 22.89GiB → 注册主机 | KV 池 33.5-37GB + 权重 | 每节点 engram 366GiB | ✅ SFT 评测/rollout |
| top64 单机 | — | — | 2.35TB（超 2.4TB 物理） | ❌ 结构不可行（必须双机或去 ref） |

### 6.2 开关红黑榜（复跑别踩）

**红（别动）**：
- `VERL_DSV41_DIRECT_SHARD_LOAD=1`（直读切片；被拒时自动回退广播，fail-closed 有校验）；RL 同侧 `VERL_DSV41_LOCAL_EXPERT_EXPORT=1`（sync 25.87GiB/16s vs 全量 104.96GiB/900s+）。
- `FSDP_TURBO_ATTN_RECOMPUTE=1` + `FSDP_TURBO_ATTN_CHUNK=64`（脚本默认；代码默认 256）——关掉它 update 必 OOM（E11/E5 层 1）。
- **SFT**：`EFSDP_SIZE=2` + `MICRO_BSZ=1` + `PARAM_OFFLOAD=False` + `OPTIMIZER_OFFLOAD=False` + `recompute_plan=["model.layers.{*}.ffn"]`（范围化 ffn）。
- **RL**：`EFSDP_SIZE=1`（历史 proven）+ `param_offload=True` + `optimizer_offload=True` + `LOCAL_EXPERT_EXPORT=1`。
- `VERL_DSV41_MAX_BATCH_SIZE=MICRO_BSZ`（engram hash cache 的 `max_batch_size` 墙）；`RAY_memory_usage_threshold=0.995`。

**黑（毒药）**：
- `FWD_PREFETCH=0 / BWD_PREFETCH=0` → 病态慢（>15min/步）。
- 整层 `recompute_plan=["model.layers.{*}"]` → 嵌套 checkpoint 崩（E5.3）；`swap_activation` → backward 不释放、照样 OOM（E5.3）。
- `optimizer_offload=True` 搭配 `param_offload=False` → 188s/步且不省显存（E2）。
- SFT `MICRO_BSZ=2`（双机 no-offload）→ step2 OOM（E1；micro 是峰值与 token 无关的固定项）。
- 单机 `PARAM_OFFLOAD=False` → 首 forward OOM（E9）。
- **RL 不给 `EFSDP_SIZE=1`（用脚本默认 2）→ 权重 sync 走未验证回退、vLLM 权重损坏（E7）**。

**启动检查（每跑必做）**：`npu-smi info -t proc-mem` 核持有者 PID 是否自己（别租户占用会伪装成"卡空闲"）；`free -g` available ≥1.5-2TB；清 `VLLM::`/`ray::` 残留（含 defunct）；双机脚本先 master 后 slave。

---

## 7. 存疑 / 缺数据清单（如实标注）

1. **`/tmp/host_mem_trace.csv` 已不在盘上**（采样脚本 `/tmp/host_mem_sampler.sh` 在）——"+~5GB/进程/步"、"358GB 跳变=4.5步"、"+108GB/2min" 三个泄漏数字**仅存于 worklog_sft §阶段③**，无原始 CSV 可核。
2. **单机 SFT v1 的运行日志（`verl/logs/sft_dsv41_steps467_20261010_181953.log`）已不在盘**——"144.07s/步、epoch 外推 18h39m"同样只有 worklog 出处；且 worklog 自注释"144s 来自首步含编译的 tqdm 外推，稳态从未测过"。
3. **worklog 数字矛盾（SFT 时长）**：worklog_sft 说"全程 21 分钟训练"，而 `/tmp/sft2_master5.log` tqdm 为 115/116 步 = **23:13**（+存盘总 26:14）。以日志为准；21 分钟可能是中途读数。
4. **优化器态入账的差异**：step1→step2 实测 **+9.59GB**（30.42→40.01），而按"8B×53.3B/32"推算 **≈13.3GB**（worklog 经验值 12.9GB）。差 ~3.5GB 原因未定位（可能与首步分配时机/fp32 master 归属有关）——**存疑**。
5. **RL 的优化器态分片口径**：RL 16 卡 update 峰 38.41GB 反推"优化器+激活+通信"余量 ~15GB，与"8B×53.3B/FSDP16=26.6GB"不符；结合 v3 实验（开 optimizer_offload 设备峰逐位不变），说明该引擎 `optimizer_offload` 的实际落位与直觉不同——**机制未完全定位（存疑，不影响现行配置可用性）**。
6. **SFT v5 加载阶段无逐条时间戳**：worker 的 `[fsdp_turbo_dsv41]` 打印是裸 `print()` 无时间戳，只有 `materialized in 43.6s / engram in 3.1s` 的内计时；"03:14:40 读数进行中"由分配器告警时间戳推断。阶段间 wall-clock 只能给量级。
7. **SFT v5 的 43.6s 物化 vs v2-v4 的 4.3-6.8s**：同为直读，慢 6-10×，归因"NFS 冷缓存（前一轮 OOM 后 page cache 被冲掉）"是**推测**。
8. **"engram 每节点 366.81GiB"（任务描述）与实测口径不同**：实测**每 rank 45.8GiB（=366.81/8，EP 行切）**，每节点 16 rank 持 **733GiB（2 份 EP 组副本）**；训练+引擎并存时节点上到 ~1.1-1.2TB。已按实测口径写入本档。
9. **RL 50 步运行仍在进行**：写档时 step 9/50（09:55），step 10+ 的稳态峰值/长时 host 走势未纳入；当前脚本默认 `EFSDP_SIZE=2` 而本 run 实际 `EFSDP_SIZE=1`（resolved config 实证），**launch 命令原文未在日志中回显**——按 11.51GiB 直读 + 25.87GiB sync 判定为 1。
10. **v7/v9/v11 单机 no-offload 系列日志未在 /tmp 留存**（`sft2_master*` 只到 v5）——57.4/57.76/57.53 三个峰值出自 worklog_sft §夜间收束。

---

## 附录：本档引用一览

| 引用 | 内容 |
|---|---|
| `/tmp/sft2_master5.log:115,116,127,128,131-134,136-138,141-143,152,153,154-267` | SFT v5 全程（成功跑） |
| `/tmp/sft2_master4.log:160,219`（同 v2/v3） | efsdp=1 OOM 三次同签名 |
| `/tmp/sft2_master2.log:151` / `master3.log:151` | opt-offload 否定实验（峰值逐位相同/188s/步/cpu +503GB） |
| `/tmp/grpo50_master.log:1250,1257-1264,1269,1271,1274,6986-6991,7032-7046,7067,7150...` | RL 50 步全程（写档时仍在跑） |
| `/tmp/serve_sft116.log:10416` | vLLM 引擎 engram 22.89GiB/rank 注册主机 |
| `verl/verl/workers/engine/base.py:314-335` | offload 逐步语义（to(device)/to("cpu")） |
| `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py:226,324,393,438,457,811,865,1017,1120,1174` | 直读/engram/导出实现 |
| `FSDPTurbo/fsdp_turbo/training/clip_grads.py:82-114,179-184` | 流式 grad-norm + CPU-offload 归约修复 |
| `FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py:176-197,240-256,481-517` | 并行拓扑字段 / MemoryConfig / EP 校验 |
| `scripts/run_sft_dsv41_2node.sh:8-13,22-38,68-79,100-133` | SFT 默认配置与设计注释 |
| `scripts/run_grpo_2node.sh:17-28,147-165,237` | RL 默认拓扑与 EFSDP 默认值 |
| `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16/model.safetensors.index.json` | 权重复算来源 |
| worklog_sft / direct_shard_load / top64_grpo / rl / weight_sync_export | 历史过程与无日志留存的数字 |
