# 权重同步的 export 段:调用链、契约与快慢两条路径的机制(代码级记录,2026-09-24)

> 相关:[`worklog_dsv41_real4_prod_consistency.md`](worklog_dsv41_real4_prod_consistency.md)(§2 慢因定位、
> §5.4 fast-sync 定版)、[`worklog_dsv41_rl.md`](worklog_dsv41_rl.md)(G13 首次把 export 认定为瓶颈)。
>
> 本篇**不是一次实验记录,而是一份代码级说明书**:把"权重同步为什么慢、`LOCAL_EXPERT_EXPORT=1`
> 为什么能快 55×"从 `update_weights` 入口一路读到 vLLM 的 `load_weights`,逐段钉上行号,
> 并用模型 config 把两条路径的张量数/字节数**算一遍去对实测**。
>
> 状态:两条路径的机制已完全钉死(§3/§4);数字自校验与实测**逐项吻合**(§6);
> 权重目录命名核对**已闭环**(§8.1:曾发现 `-real` 目录不在 → 用户已改名回规范名,现核对通过)。
> 最后更新:2026-09-24 03:2x。

---

## 0. 一句话结论

1. **传输代码本身没有 bug,也没有被改过**。快速路径
   (`fsdp_turbo_dsv41_impl.py:351 _export_local_experts`)是 **2026-09-19 提交 `5cff0d13`** 就写好的
   opt-in 分支;2026-09-23 做的只是**打开开关 + 在真实 384 专家权重上两次验证 + 定版**
   (`git log -S "VERL_DSV41_LOCAL_EXPERT_EXPORT"` → `5cff0d13 2026-09-19 06:38:39`)。
2. 慢的根因是**默认路径让 8 个 rank 各自物化并推送全量 384 专家**(104.96 GiB/rank),
   而 colocate + 1:1 配对 + 两侧按 rank 连续切分的布局下,**每 rank 只有 1/8 是对面需要的**。
3. 快速路径把 all-gather **整个省掉**,只做一次本地 H2D:每 rank 105 → 16.37 GiB、
   张量 4702 → 670、`update_weights` 1032.7s → 18.7s(55×,超过 6.4× 的部分来自少掉的 8 次 HCCL
   all-gather、CPU-offload 的重复 H2D、host 峰值从 1.3 TB 降下来后不再换页)。

---

## 1. 全景调用链(关键是"惰性 generator")

```
ActorRolloutRefWorker.update_weights()                     engine_workers.py:786
  └─ per_tensor_param = actor.engine.get_per_tensor_param(layered_summon=..., base_sync_done=True)
       └─ FSDPEngine.get_per_tensor_param()                fsdp/transformer_impl.py:989
            params = self.module.state_dict()              # :1028  FSDP2 → value 是 DTensor
            params = convert_weight_keys(params, ...)      # :1030  训练侧名 → HF 名
            per_tensor_param = (entry for name, param in params.items()
                                  for entry in self._export_param(name, param))   # :1040-1042
            per_tensor_param = unfuse_moe_params(per_tensor_param, ...)           # :1043
       └─ FSDPTurboDSV41EngineWithLMHead._export_param()   fsdp_turbo_dsv41_impl.py:305  ← 覆盖点
  └─ await self.rollout.update_weights(per_tensor_param)   engine_workers.py:804
       ├─ _execute_method("update_weights_from_ipc")       vllm_rollout.py:229   # 让引擎侧先起 receiver
       └─ await BucketedWeightSender.async_send_weights(gen)  vllm_rollout.py:236-241
            └─ async for name, weight in produce(gen)      bucketed_weight_transfer.py:145  ← 真正拉起来的地方
```

**这张图的读法:箭头 = "谁把 generator 往下拉一格"。** `get_per_tensor_param()` 是**普通函数返回惰性
generator**,调用它时一行专家权重都没算;真正的 all-gather / H2D / 逐专家切分发生在最后一行
`async for` —— 也就是 **sender 的循环里,和 ZMQ 传输串行、在训练的同步关键路径上**。

这解释了 `SYNC-PROFILE` 两行日志的语义(`bucketed_weight_transfer.py:122-135`):

| 字段 | 计时位置 | 含义 |
|---|---|---|
| `export_s` | `await src.__anext__()` 前后 | **生产下一块参数**:H2D + all-gather + `.contiguous()` 切分 |
| `flush_s` | `send_pyobj` + `recv` 前后 | 把桶推给 receiver 并**等它 ack**(接收+装载) |

`logs/..._104558.log`(默认路径):`4702 tensors / 104.96 GiB / 213 buckets | export 904–1009s, flush 13.2s`
→ **瓶颈量在 export 这一侧,传输本身只占 1.3%**。

---

## 2. `_export_param` 的输入/输出契约

```python
# fsdp_turbo_dsv41_impl.py:305
def _export_param(self, name, param) -> Generator[tuple[str, torch.Tensor], None, None]
```

| | 内容 |
|---|---|
| **输入 `name`** | `convert_weight_keys` 之后的参数名。本模型是 `...layers.{i}.ffn.experts.gate_up_proj` / `...ffn.experts.down_proj` —— 注意是 **`ffn.`** 不是 `mlp.`(`_FUSED_EXPERT_SUFFIXES` 定义在 `:303`) |
| **输入 `param`** | 模块的**活参数对象**:FSDP2 下是 `DTensor`(placement 含 `Shard(0)` 覆盖专家维);也可能是普通 tensor / 已复制(此时走通用路径) |
| **输出** | 惰性 generator,元素 = `(hf_name, device_tensor)`:`hf_name` 是 vLLM 认的**逐专家 checkpoint key**,张量是**已物化的普通 NPU 张量**(不再是 DTensor) |
| **数量关系** | **1 个 state_dict 条目 → 1 个(普通参数)或 384×3 / 48×3 个(fused 专家参数)**。这是全场张量数膨胀的唯一来源(94 → 4702) |

基类版本在 `fsdp/transformer_impl.py:977-987`:DTensor 就 `param.to(device).full_tensor()`,
否则原样 yield —— 即"**通用路径 = 全量 all-gather**",这正是快速路径的 fallback 目标。

### 2.1 命名映射(`fsdp/utils.py:90 split_fused_expert_tensor`)

```
[384, 4608, 5120]  .ffn.experts.gate_up_proj        # :128-138
   → chunk(2, dim=1) → gate/up 各 [384, 2304, 5120]
   → ...experts.{first_expert_id + id}.w1.weight    (gate)
   → ...experts.{first_expert_id + id}.w3.weight    (up)
[384, 5120, 2304]  .ffn.experts.down_proj           # :140-145
   → ...experts.{first_expert_id + id}.w2.weight
```

`first_expert_id` = `tensor[0]` 的**全局**专家号 —— 这是整条链路的正确性抓手:vLLM 只有拿到全局
专家号才能把它映射回自己的本地专家。所以 `_export_param` 的每一次 yield 都必须把"这块的起始全局号"
传对(默认路径传块起点,快速路径传 `rank × 48`)。

`.mlp.experts.*` 那两支(`:105-122`)是 Qwen 系的命名,本模型不走。

---

## 3. 默认路径:块级 all-gather = 每 rank 全量(慢)

```python
# fsdp_turbo_dsv41_impl.py:346-349
for first_expert_id in range(0, param.shape[0], local.shape[0]):     # 384/48 = 8 次
    block_tensor = param.narrow(0, first_expert_id, local.shape[0])
    materialized = block_tensor.to(get_device_id(), non_blocking=True).full_tensor()
    yield from split_fused_expert_tensor(name, materialized, first_expert_id=first_expert_id)
```

三个动作的语义(**这张表的读法:行 = 一步,右列 = 它在通信/H2D 上干了什么**):

| 步骤 | 语义 | 代价 |
|---|---|---|
| `param.narrow(0, k*48, 48)` | 切出全局第 k 块(DTensor 视图;因为 `48 == local.shape[0]`,这一块恰好对应**某一个 rank 的分片**) | 零拷贝视图 |
| `.to(get_device_id())` | 把本地分片从 CPU 搬到 NPU(`offload_policy` 下参数常驻 CPU) | **H2D**,每块一次 |
| `.full_tensor()` | DTensor 的 **all-gather**:把这一块的 8 份分片聚成全量副本 | **HCCL**,每块一次 |

**8 次块级 all-gather 的净效果 = 一次 [384, …] 全量 all-gather**,唯一好处是把峰值 HBM 从
18.1 GB 压到一个块(48×4608×5120×2 B = **2.11 GiB**,docstring 里的 "2.26 GiB" 是按十进制 GB 说的)。
代价是 `export_s` 里叠了 8 轮 H2D + 8 轮 HCCL + 每层 1152 次逐专家 `.contiguous()`。

**"7/8 是浪费"的准确含义**:每个 rank 都完整物化了 384 个专家,然后把这 1152 个 key(每层)全部推给
它配对的那**一个**引擎 rank;而引擎侧 EP8 只持有 48 个专家,其余到岸即被丢弃。8 个 rank 各自重复
一遍 → 每 rank 导出 104.96 GiB。

> **为什么默认要做这件"蠢事"**:docstring(`:308-314`)写明 —— 全量 18 GB gather 在 colocate 下
> **放不下**(引擎权重还在同一张卡的 HBM 上),只能分块控峰值。这是历史实现,不是设计意图。

> **⚠️ 一个容易看错的分支语义**:`_export_local_experts` 内部守卫失败时(`:363`)退的是
> `super()._export_param` —— 即**基类的全量 all-gather**,不是退回上面那个块循环。

---

## 4. 快速路径:本地分片直发,零 collective(快)

```python
# fsdp_turbo_dsv41_impl.py:351-368
local = param.to_local()                                     # 本 rank 的 48 个专家,已在本地
shard_mesh_dims = [d for d, p in enumerate(param.placements)
                   if isinstance(p, Shard) and p.dim == 0]
if len(shard_mesh_dims) != 1 or local.shape[0] * mesh.size(shard_mesh_dims[0]) != param.shape[0]:
    yield from super()._export_param(name, param)            # 不成"单维连续切分" → 退回全量路径
    return
first_expert_id = param.device_mesh.get_coordinate()[shard_mesh_dims[0]] * local.shape[0]
materialized = local.to(get_device_id(), non_blocking=True)
yield from split_fused_expert_tensor(name, materialized, first_expert_id=first_expert_id)
```

- **没有任何 collective**:只有一次本地 H2D(`local.to(device)`),48×3 = 144 个 key 直接进桶
- `first_expert_id = rank × 48`:由 mesh 坐标算全局号,与引擎侧"按 rank 连续切分"的专家布局对齐
- **这张表的读法**:行 = 守卫,右列 = 什么情况会触发退回(退回 = 走基类全量,安全但慢)

| 守卫 | 位置 | 触发条件(→ 退回通用路径) |
|---|---|---|
| 名字/类型 | `:333` | 不是 `_FUSED_EXPERT_SUFFIXES` 之一 / 不是 DTensor / 不是 3 维 |
| 分片形态 | `:337-340` | `local.shape[0] <= 0` / 专家维不能被整除 / **`local.shape[0] == param.shape[0]`(即专家维根本没被切,已复制)** |
| mesh 结构 | `:358-361` | 不是"**恰好一个** mesh 维按 dim 0 切" / 该维不恰好覆盖整个专家维 |

### 4.1 正确性前提:1:1 配对

ZMQ 端点按**节点内 local rank** 命名(`vllm_rollout.py:139-141`):

```
ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock
```

trainer rank i 的 sender ↔ 引擎 rank i 的 receiver,**两侧都沿专家维按 rank 连续切分**
→ "我要的那 1/8"恰好就是"本地这 1/8",**其余 7/8 在发出去之前就被丢掉了**。

- 启动脚本 `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh:71-76` 用
  `GEN_EP == EP_SIZE` 守住该前提,不满足就自动把开关降成 0 并告警
- **错了藏不住**(docstring `:324-327` + 实证):引擎会装载错误的专家,
  `training/rollout_probs_diff_mean` / `..._pearson_corr` 会从 ~1e-2 / ~0.98 **断层**到 O(1) / ~0.5

---

## 5. 下游:两级 unfuse → 分桶 → ZMQ → 引擎加载

### 5.1 两级 unfuse 是互补而非重复

| 级 | 位置 | 职责 |
|---|---|---|
| ① | `fsdp_turbo_dsv41_impl.py:349/368` 调 `split_fused_expert_tensor` | 把**本 rank 实际要发的**专家切成逐专家 key,带正确的 `first_expert_id` |
| ② | `fsdp/utils.py:150-174` `unfuse_moe_params`(由 `transformer_impl.py:1043` 套在外面) | 兜底:对**走基类路径**的 packed 参数(Qwen/GPT-OSS)用 `first_expert_id=0` 切 |

对 DSV41 路径,①已经把名字变成 `.w1.weight`、张量降成 2 维,②里
`split_fused_expert_tensor` 返回 `None`(它只认 3 维 + 旧后缀)→ **原样透传**。反之若走了基类 fallback,
①没切,②就用 `first_expert_id=0` 切全量张量 —— **两条路都能得到正确的全局编号**。

### 5.2 分桶与握手(`bucketed_weight_transfer.py`)

```
_init_socket()  :208   REQ socket bind 到 ipc://...
_init_buffer()  :219   申请 512 MB(update_weights_bucket_megabytes)设备缓冲,
                       把 reduce_tensor(handle) 经 ZMQ 发给 receiver,等 ack      # CUDA/IPC 路径
                       use_shm 时改为建共享内存段并传 {name,size}(NPU 无 IPC 时)
循环           :145    for 每个 (name, tensor):
                 :155    offset 按 element_size 对齐
                 :159    放不下 → flush:send_pyobj(bucket_meta) + recv()  ← 逐桶同步握手
                 :170    单张量就超桶 → _direct_send_large_weight(:263)走 IPC handle 直发
                 :185    buffer[offset:...].view(dtype).view(shape).copy_(weight)
末桶           :190    send_pyobj({..., is_last: True}) + recv(),打印 SYNC-PROFILE sender
```

### 5.3 引擎侧接收与装载(`vllm_rollout/utils.py:249 update_weights_from_ipc`)

| 步 | 位置 | 内容 |
|---|---|---|
| step 1 | `:263-311` | 布局回退:NPU 上把 Ascend 的转置推理布局(`w13`/`w2`)**还原成 checkpoint 布局**(`:281-284`),否则专家装载器对不上 |
| step 2 | `:313-344` | `BucketedWeightReceiver` 逐桶把缓冲**零拷贝 view** 成 `(name, tensor)`(IPC 路径)`→ on_bucket_received → _update_weights` → vLLM `load_weights` |
| step 3 | `:347-372` | 全部桶收完后统一跑一次 `process_weights_after_loading`(非幂等,只能跑一次;把专家再转回推理布局) |

`SYNC-PROFILE receiver:` 行(`bucketed_weight_transfer.py:366-371`)给 `views / load / device sync` 三段;
实测 fast 路径 `views 0.1s, load 11.3s` → **接收侧本来就不是瓶颈**。

---

## 6. 数字自校验(用 model config 实算,与实测逐项吻合)

`config.json`(`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer`):
`hidden=5120, moe_intermediate_size=2304, n_routed_experts=384, num_hidden_layers=4, vocab=129280`,
专家权重 bf16。

**表读法:行 = 一项可独立计算的量,"实测"列 = 日志里的对应数字,最后一列 = 是否吻合。**

| 量 | 算式 | 算得 | 实测 | ✓ |
|---|---|---|---|---|
| gate_up 单层 | 384×4608×5120×2 B | 16.87 GiB | — | |
| down 单层 | 384×5120×2304×2 B | 8.44 GiB | — | |
| **专家合计** | 25.31 × 4 层 | **101.25 GiB** | — | |
| **单块峰值** | 48×4608×5120×2 B | **2.11 GiB** | docstring "2.26 GiB"(十进制) | ✓ |
| **非专家(每 rank 全量)** | 104.96 − 101.25 | **3.71 GiB** | 反推值 | |
| 张量数(默认) | 4×384×3 + 94 | **4702** | `4702 tensors` | ✓ |
| 张量数(快速) | 4×48×3 + 94 | **670** | `670 tensors` | ✓ |
| 字节(默认) | 101.25 + 3.71 | **104.96 GiB** | `104.96 GiB` | ✓✓ |
| 字节(快速) | 101.25/8 + 3.71 = 12.66 + 3.71 | **16.37 GiB** | `16.37 GiB` | ✓✓ |

**这两条字节数都是"精确落到实测小数点后两位"**,而它们共用同一个 3.71 GiB 非专家项 ——
说明**快慢两条路的唯一差别就是"发几份专家"**,不存在别的隐藏开销。三个比值因此可以互相解释:

| 比值 | 值 | 为什么不是 8× |
|---|---|---|
| 字节 | 104.96 / 16.37 = **6.41×** | 非专家那 3.71 GiB 两份都有,稀释了 8× |
| 张量数 | 4702 / 670 = **7.02×** | 同上(94 个非专家两路都发) |
| 耗时 | 1032.7 / 18.7 = **55×** | 额外来自:省掉 8 轮 HCCL all-gather、省掉 CPU-offload 的重复 H2D、host 峰值 1.3 TB 下降后不再换页 |

---

## 7. 可观测性与复现命令

```bash
# 1. 看某次跑的同步画像(任何日志通用)
grep -n "SYNC-PROFILE" <log>                 # sender: tensors / GiB / buckets / export / flush
grep -o "timing_s/[a-z_]*:[0-9.]*" <log>     # update_weights 在 step 里的占比

# 2. 全量同步的某一步(真实 4 层,默认路径 —— 仅用于复核,不要日常跑)
cd /workspace-verl/verl && LOCAL_EXPERT_EXPORT=0 STEPS=2 \
  bash scripts/dsv41_consistency/sh/run_real4_grpo.sh

# 3. 快速同步(real4 迭代标配)
cd /workspace-verl/verl && LOCAL_EXPERT_EXPORT=1 STEPS=2 \
  bash scripts/dsv41_consistency/sh/run_real4_grpo.sh

# 4. 核对模型的专家形状(本文件所有算式的前提)
python3 -c "import json;c=json.load(open('<MODEL_PATH>/config.json'))['text_config'];\
print(c['hidden_size'], c['moe_intermediate_size'], c['n_routed_experts'], c['num_hidden_layers'])"
```

对照锚点(两份日志即 §6 表格的来源):

| 日志 | 路径 | sender | export | update_weights |
|---|---|---|---|---|
| `logs/..._104558.log` | 默认 | 4702 / 104.96 GiB / 213 桶 | 904–1009s | 1032.65s |
| `logs/..._124957.log` | 快速 | 670 / 16.37 GiB / 30 桶 | 0.1–0.9s | 18.75 / 17.83s |

---

## 8. 事实核对与待办(2026-09-24)

### 8.1 `MODEL_PATH` / 权重目录命名(核对过程与结论,已闭环)

- `scripts/dsv41_consistency/sh/run_real4_grpo.sh` 与 examples 脚本默认
  `MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real`;
- **核对时(2026-09-24 03:0x)曾发现该目录不存在**:当时 `weights/` 下是
  `DeepSeek-V4.1-Flash-4layer`(108 GB)、`-4layer-expert32-random`、`-4layer-random`
  (父目录 mtime = 2026-09-24 01:37,会话结束后被改名过)。当时的交叉验证依据:
  该目录内 `SLICE.md` 第一行自述 **"Slice: DeepSeek-V4.1-Flash-4layer-real"**,
  层 [0,1,2,3]、384 专家、**逻辑大小 105.88 GiB(= 113683600480 B)**、张量 4972 个
  → 就是同一份真实切片(2026-09-23 12:49 那次跑解析到的也是这个名字);
- **2026-09-24 03:20 已由用户改名回规范名,现核对通过**(三份权重目录的名/内容一致):

  | 目录 | 大小 | layers | 专家 | 用途 |
  |---|---|---|---|---|
  | `DeepSeek-V4.1-Flash-4layer-real` | 108 GB | 4 | 384 | 真实权重(生产一致性/real4 迭代) |
  | `DeepSeek-V4.1-Flash-4layer-scaled` | 106 GB | 4 | 384 | scaled 随机权重(384 专家) |
  | `DeepSeek-V4.1-Flash-4layer-scaled32` | 14 GB | 4 | 32 | scaled32(快速迭代) |

- **留下的判据(下次再遇到"脚本说路径不存在")**:先 `du -sh` + 读目标目录 `config.json` 的
  `text_config.{num_hidden_layers,n_routed_experts}` 与 `SLICE.md` 自述名,**按内容认权重、
  不按目录名认** —— 本次就是靠 SLICE.md 与 105.88 GiB 逻辑大小确认的。

### 8.2 交叉验证:SLICE.md 的 105.88 GiB ↔ 实测 104.96 GiB

两者差 0.92 GiB,量级合理(ckpt 逻辑大小含不随 state_dict 导出的项,如 buffer / 非语言塔部分;
另 `unfuse_moe_params` 只发 vLLM 认的 key)。**这条差值是"导出 = 参数子集"的旁证**,不是异常。

### 8.3 本篇未改动任何代码

纯读码 + 算账 + 路径核对;涉及文件清单为零(`git status` 里只有上一篇遗留的
`run_real4_grpo.sh` / `worklog_dsv41_real4_prod_consistency.md` 修改与本篇新增)。
后续若要真正"再快一档",候选(按性价比)见
[`worklog_dsv41_real4_prod_consistency.md`](worklog_dsv41_real4_prod_consistency.md) §7:
`get_per_tensor_param_shard`(delta 引擎那条**分片导出**的 API)已经在框架里,
理论上能让 naive 路径也走"本地分片 + 接收侧重组",但那是生产代码改动,需要过 autograd 与
一致性回归,不在本次范围内。