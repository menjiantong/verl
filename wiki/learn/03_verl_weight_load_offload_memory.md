# 学习笔记 03：verl 训练引擎的**权重加载 / offload / 显存账**（DeepSeek-V4.1 · FSDPTurbo 实证）

> 读码基线：`/workspace-verl/verl`（2026-10-12 树）+ `/workspace-verl/FSDPTurbo`。
> 数据基线：SFT-116 双机 run（`/tmp/sft2_master5.log`）、RL 生产 run（`/workspace-verl/verl/logs/grpo_top64_steps*`）、
> 归档 `wiki/dsv41_memory_and_parallel_archive.md`（本日产出，逐阶段数字全表）。
> 本文回答：**① 权重怎么装进去（三路径+代码行号）② offload 三个开关的真实语义（含实测代价）③ 每阶段显存/内存账怎么算（公式+实测）**。

---

## 0. 一句话总览

训练引擎（`fsdp_turbo_dsv41`）= **"meta 构建 → FSDPTurbo 包壳 → 三选一装权重（直载 / 广播 / resume）→ 尾巴（buffers→device、专家元数据刷新、engram host 表加载）"**；
offload 只有**两套机制**：① `offload_policy=True` → torch FSDP2 的 `CPUOffloadPolicy`（**参数放 pinned CPU，按模块前向时搬运**，优化器态随参数落 CPU）；
② `param_offload/optimizer_offload=True`（且 offload_policy=False）→ **每步/每阶段对整套参数或优化器态做 to(device)/to("cpu")**（代价高，实测 188s/步）。
engram 表是第三类特殊"offload"：**永远在主机（CANN 注册内存），不占 device**。

---

## 1. 训练引擎初始化链路（自顶向下）

```
main_ppo / sft_trainer_ray（driver，131 机）
 └─ ray 远端 actor::TrainingWorker.__init__                    workers/engine_workers.py:84
     ├─ initialize_global_process_group_ray(...)               # HCCL/Gloo 建组
     ├─ EngineRegistry.new(model_type, backend=engine_config.strategy, ...)   :136
     │   └─ FSDPTurboDSV41EngineWithLMHead.__init__（→ 父类 FSDPEngineWithLMHead）
     │       ├─ _init_device_mesh() → _init_parallel_state()   fsdp_turbo_impl.py:26
     │       │   ├─ fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, engine_config.turbo_config)  :33
     │       │   ├─ **fsdp_turbo_config.distributed.fsdp_plan.cpu_offload = engine_config.offload_policy**  :34 ★offload 总闸
     │       │   └─ init_parallel_state(cfg)（建 dp/fsdp/tp/ep/edp 各 mesh）  parallel_state.py
     │       ├─ _build_module() → FSDPTurboDSV41._build_module  fsdp_turbo_dsv41_impl.py:898
     │       │   └─ _build_dsv41_module()                       :926
     │       │       ├─ build_deepseek_v41_model(..., engram_meta_init=True, experts_meta_init=True)
     │       │       │     # 全部权重先建在 meta；engram 表 host_offload（权重行也在 meta）
     │       │       ├─ initialize_deepseek_v41_model(...)      # 记录 engram 表 init seed（表本体仍 meta）
     │       │       ├─ prepare_deepseek_v41_model_for_fsdp(...) # 浮点参数统一 bf16；非 meta 的 tensor 搬 device
     │       │       └─ [FSDP_TURBO_MEM_DEBUG 门控 hook（逐层显存探针）]   :951
     │       └─ _build_fsdp_module(module)                      :1006
     │           ├─ param_meta/buffer_meta 采集（**engram 表从 param_meta 弹出** → 不走 DCP）  :1015-1017
     │           ├─ module = FSDPTurbo(fsdp_turbo_config, module).model   :1019   # 并行切分（DTensor 化）
     │           ├─ offload_policy=True 时：_is_offload_param=False; _is_offload_optimizer=False;
     │           │   _uses_fsdp2_cpu_offload_policy=True                :1023-1027
     │           └─ _materialize_dsv41_parameters(...)          :1031  ← ★ 权重入口（§2）
     ├─ _build_optimizer()（torch.optim.AdamW；**状态惰性创建于第一次 optimizer.step**）  transformer_impl.py:496
     └─ （resume_mode=auto 时）checkpoint_manager.load_checkpoint（§2.3 路径 C）
```

要点：
- **meta 构建是显存策略的地基**：参数/专家/engram 全在 meta，装权重时才知道每卡真实分片大小；
- FSDPTurbo 包壳发生在**装载之前**（先切好拓扑，再按本地分片读文件——这是"直载"的前提）；
- optimizer 状态不在初始化时分配，**第一次 step 才出现**（SFT 实测 step1→step2 设备峰 +9.59GB，见 §4 存疑）。

---

## 2. 权重加载的三条路径（代码 + 实测）

入口统一在 `_materialize_dsv41_parameters`（fsdp_turbo_dsv41_impl.py:1031），按 env 三选一：

### 2.1 路径 A：**直载分片**（`VERL_DSV41_DIRECT_SHARD_LOAD=1`，生产默认）
```
_plan_direct_shard_load(module, ckpt_dir, param_meta, buffer_meta)   :226
  └─ 逐参数用 torch 官方 compute_local_shape_and_global_offset(dtensor)  :243
     算出 (local_shape, offset)；融合专家按 _assemble_fused_expert_window 展开  :190
_read_direct_shards(plan, ckpt_dir)                                  :324
  └─ 每个 rank 用 safe_open.get_slice(name)[window] 只读自己那段字节（不跨机广播）
_load_direct_shards(module, plan, values, engram_tables, cpu_offload=bool(offload_policy))  :393
  └─ assign=True 直接替换 meta 参数（offload_policy 时落 CPU）  :437
_verify_direct_shards(...)  :457   # rank0 独立重装几个分片比对（fail-loud）
```
实测（本 top32 模型，双机）：
| 配置 | 每卡直读字节 | 物化耗时 | 日志原文 |
|---|---|---|---|
| FSDP32/ep8/**efsdp=2** | **5.75 GiB** | 4.3-4.8s | `checkpoint read: 964 tensors slice-read directly (5.75 GiB this rank)` |
| FSDP32/ep8/efsdp=1 | 11.03 GiB | 6.0-43.6s | 同上（11.03 GiB） |
| FSDP16/ep8/efsdp=1（RL，装载时） | 11.51 GiB | 6-22s | `grpo` 各 run 日志 |

### 2.2 路径 B：**rank0 读 + DCP 广播**（直载被拒时的回退）
```
rank0: read_dsv41_checkpoint_state_dict(ckpt_dir, param_meta, buffer_meta)   # 整份 state dict（含融合专家）
       set_model_state_dict(modules, full_state, StateDictOptions(full_state_dict=True,
                              broadcast_from_rank0=True, cpu_offload=bool(offload_policy)))
       # engram 表 hide/restore（不参与 strict 加载）  :1092-1098
```
- **跨机 gloo 逐张量广播实测 ~30MB/s** → 双机装载会拖成小时级；这是直载被引入的原因（:1034-1040 注释）。
- 单机（loopback）下尚可；双机生产一律走 A。

### 2.3 路径 C：**resume（存过的 FSDP 分片）**
```
trainer.resume_mode=auto → ckpt_handler.load_checkpoint() → FSDPCheckpointManager.load_checkpoint
  └─ 每 rank torch.load(model_world_size_<W>_rank_<r>.pt)（SHARDED state dict，含 engram 表行）
       model.load_state_dict(...)     # verl/utils/checkpoint/fsdp_checkpoint_manager.py:176-230
```
- 保存同构：`save_checkpoint`（同文件 :272-330，`get_fsdp_state_ctx` 对 FSDP2 = nullcontext → 直接 `state_dict()` 取 DTensor）；
- **约束：load 必须 world size 与切分几何完全一致**（保存时 W=32/efsdp=2 的 run 不能被 16 卡/efsdp=1 直接 resume）；
- 本模型 ckpt 体积：每卡 ~55GiB（engram 45.8GiB + 可训 ~7.9GiB）×32 = 1.65TB（engram 每节点各存一份）。

### 2.4 装完的"尾巴"（`_finish_dsv41_materialization` :1116）
1. `move_buffers_to_device`（139 个 buffer 如 rope 表、engram_hash primes/offsets → device）；
2. `refresh_dsv41_expert_metadata`（EP 派发簿记，纯属性重建）:780；
3. **`_load_host_engram_tables`** :1188：每 rank 只 `safe_open` 切片读自己那段行（2 张表共 **45.8 GiB/rank**，双机 run 实测 9.7-95.7s），随后 `requires_grad=False` 冻结；
4. 残留 meta 检查（fail-loud）。

---

## 3. offload 的真实语义（三个开关 + engram）

### 3.1 `offload_policy`（FSDP2 CPUOffloadPolicy——生产 RL 用的就是它）
- 生效链：`engine_config.offload_policy` → `fsdp_turbo_config.distributed.fsdp_plan.cpu_offload`（fsdp_turbo_impl.py:34）
  → `get_offload_policy()` 返回 `CPUOffloadPolicy(pin_memory=True)`（fully_shard_parallel.py:81）
  → torch FSDP2 在**每个模块前向时把该模块参数从 pinned CPU 搬到 device，用完放回**；
- 副作用：引擎把 `_is_offload_param/_is_offload_optimizer` 强制 False（:1023-1027），**手动 offload 机制整体旁路**；
- 优化器态：参数本体在 CPU → 第一次 step 时 AdamW 状态在 **CPU 侧创建**（主机内存大头之一）；
- 代价：逐模块搬运是固定开销（SFT v1 实测 **144.07s/步**，且主机在早期版本有 +5GB/进程/步 泄漏，与 offload 机制相关）；
- 实测显存 vs 无 offload：**设备峰值只差装载基线**，forward 期间逐层驻留 ≈ 当前+预取层（≈数 GB），稳态峰 28.6GB（单机 SFT v1）vs no-offload 的 55-57GB——**省的是"参数常驻"，不是激活**。

### 3.2 `param_offload` / `optimizer_offload`（手动搬运路径；仅在 offload_policy=False 时有效）
- 语义在 `BaseEngineCtx._context_switch`（base.py:314）：
  ```
  train 模式 enter: engine.to(device, model=param_offload, optimizer=optimizer_offload, grad=param_offload)
  exit:             engine.to("cpu", 同上)
  ```
- `engine.to`（transformer_impl.py:844-875）：device 方向 → `load_fsdp_model_to_gpu / load_fsdp_optimizer`；
  cpu 方向 → `offload_fsdp_model_to_cpu / offload_fsdp_optimizer`；
- **`offload_fsdp_optimizer`/`load_fsdp_optimizer` 是逐参数的 `state[key].to("cpu"/device)`**（fsdp_utils.py:241-260）——
  本 top32 上实测该路径：**设备峰值逐位不变（状态反正要在 step 内上卡）、步速 4× 慢（188s vs 47s）、主机 +513GB**
  → 这就是为什么"想靠 optimizer_offload 省设备显存"在本引擎上是死路；
- SFT 生产配置最终 = **两个都关**（脚本 `PARAM_OFFLOAD=False / OPTIMIZER_OFFLOAD=False`，`engine.offload_policy` 跟随 PARAM_OFFLOAD）。

### 3.3 engram 主机表（第三类 offload，永远在主机）
- `host_offload` 是唯一实现（row_sharded 已被上游删除）：表放 **CANN 注册主机内存**（`ps` 不可见、`free` 可见），device 上只有每层 1 头的计算切片；
- 双机 top32：**45.8 GiB/rank**（2 表），每节点 733GiB（EP 组两副本）；RL 再叠一个 vLLM 侧副本（同样 host）；
- 冻结（requires_grad=False）：不参梯度、不进 weight sync（`_export_param` 对 engram 直接 return）。

---

## 4. 显存账（公式 + 实测，全配置对照）

### 4.1 公式（bf16 训练，AdamW fp32 态）
```
每卡参数(bf16)  = dense_bytes/FSDP + experts_bytes/(EP × EFSDP)
每卡梯度(bf16)  = 同参数量
优化器态(在device时) ≈ 8B × 参数量/2B   （fp32 m+v；实测口径见存疑）
激活            ≈ 单微批 tokens × 每token系数（recompute 只覆盖 ffn；实测对 micro 弱敏感）
engram(device)  = 0（host）
→ 设备稳态峰 ≈ 参数 + 梯度 + 优化器态 + 激活 + 通信/工作区
```
本 top32 实数（checkpoint header 复算）：dense 12.17GiB + experts(main/shared) 87.0GiB（bf16, 可训 99.19GiB）；engram 366.81GiB(host)。

### 4.2 实测对照表（详见归档文档 §2）
| 配置 | 参数/梯度每卡 | 设备峰值 | 主机 | 备注 |
|---|---|---|---|---|
| 单机 FSDP16/efsdp=1，全 offload（v1 proven） | 6.5+6.5GiB | 28.6GB | 2.07TB | 144s/步；主机泄漏（旧版） |
| 单机 FSDP16/efsdp=1，无 offload | 6.5+6.5GiB | **57.4GB → OOM(+182MiB)** | 1.31TB | 专家态+激活撞顶 |
| 双机 FSDP32/efsdp=1，无 offload | 11.0GiB 直读 | **55.15GB → step2 OOM** | — | 专家三件套≈44GB/卡 |
| **双机 FSDP32/efsdp=2，无 offload（SFT 成功）** | **5.75GiB 直读** | **step1 30.4 / 稳态 41.7GB** | 1.23TB | 12s/步 |
| 双机 RL FSDP16/EP8/efsdp=1（offload_policy=True） | 11.51GiB 直读 | **36.7-38.4GB** | 1.66TB | 参数在 host（policy），状态也在 host |
| 双机 RL + 我试的 efsdp=2 | 5.75GiB | 峰位同带 | — | **但 sync 传坏权重（见 §5 坑 5）** |

### 4.3 装载时序（每阶段在干什么 + 数字）
以 SFT v5 双机成功跑（`/tmp/sft2_master5.log`）为准：
```
① 建组/HCCL                    （~1min）
② meta 构建 + FSDPTurbo 包壳    （秒级；"After FSDP"基线 = alloc 5.77GB / reserved 15.4GB）
③ 直载权重切片                 每卡 5.75GiB（efsdp=2），4.3s
④ buffers→device（139 个）+ 专家元数据       秒级
⑤ engram host 表 45.8GiB/rank 切片读（9.7-95.7s，页缓存冷热差 10×）
⑥ step1：优化器态首次创建（设备/主机取决于 offload 配置）→ step1 峰值 30.4GB
⑦ 稳态 41.7GB/卡（含激活 + clip 相位 fp32 瞬态，已流式化 13GB→0）
⑧ 存盘（save_freq）：SHARDED state dict 逐 rank 写本地盘，1.65TB / ~3 分钟
```

### 4.4 主机内存账（一页）
| 组成 | 量 |
|---|---|
| engram host 表（引擎侧） | 733GiB/节点（top32, EP 两副本） |
| RL 再叠 vLLM engram 副本 | +733GiB/节点（共享模式下一副本） |
| 可训参数副本（CPUOffloadPolicy 时） | ~99GiB/节点 + 优化器态（CPU 侧创建） |
| 其余（HCCL 缓冲、加载瞬态、页缓存） | 数百 GB |
| 实测稳态 | SFT 1.23TB / RL 1.66TB（节点，2454GB 总） |

---

## 5. 常见坑（本模型上实测踩过，均有日志）

1. **param_offload=False + 专家全量在卡 → 必然 OOM**：efsdp=1 时专家三件套 44GB/卡；修 = `efsdp=2`（或 offload）。
2. **`optimizer_offload=True` 在该引擎里不能省显存**（峰值逐位不变）且拖慢 4×——别用它做显存杠杆。
3. **`offload_policy=True` 与 `param_offload=True` 同时给是常态**：后者被旁路，真正生效的是前者（FSDP2 policy）。
4. **resume 的几何必须逐位一致**（world size + fsdp/efsdp/ep 切分），否则 load_state_dict 崩或错位。
5. **`EFSDP=2` 只在 SFT（无 sync）安全**：RL 的 weight sync 在 efsdp=2 下走未验证回退路径 → 传出坏权重（乱码，已实证；RL 保持 EFSDP=1）。
6. **direct shard load 是双机唯一实用路径**（广播 ~30MB/s）。
7. **engram 行数有 2048 对齐 padding**（384008192 vs 384006168）：任何自写加载器要补零（`cast()` 已加 engram 分支）。
8. **存疑（归档 §7）**：优化器态 step1→step2 实测 +9.59GB vs fp32 m+v 推算 13.3GB（差 ~3.5GB 未定位）；
   如需精确账目，用 `VERL_DSV41_DUMP_BATCH`/逐层探针复测。

---

## 6. 与推理引擎（vLLM 侧）的对照（一段话）

vLLM 侧**文件驱动**加载（`for name in f.keys()`，`should_skip_weight` 仅按 EP 过滤"非本 rank 专家"，
`vllm_ascend` 的 `weight_utils.py`/`ep_weight_filter.py`），没有 meta/DTensor 机制；
engram 走 `VLLM_PLE_CPU_OFFLOAD=1`（注册主机内存，serve 日志：`Engram table offloaded to registered host memory`）。
训练↔推理之间的权重通道 = `get_per_tensor_param`（学习笔记 01），offload 语义互不相干。
