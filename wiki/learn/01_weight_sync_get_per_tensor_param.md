# 学习笔记 01:`BaseEngine.get_per_tensor_param` —— 训推算子之间的"重量接口"

> 读码基线:`/workspace-verl/verl`(commit `5412301b` 之后的树,2026-09-24)。
> 相关实测记录:[`../worklog_dsv41_weight_sync_export.md`](../worklog_dsv41_weight_sync_export.md)
> (把本文第 4.1 节的 FSDP 路径放大到"105 GiB / export 900s / 快慢两条路 55×"的现场)。
>
> 本文回答四件事:**① 调用链怎么走到它 ② 输入输出契约是什么 ③ 内部机制怎么实现
> ④ 五个后端各自的物化策略差在哪(重点 `FSDPEngine`)**。

---

## 0. 一句话

`get_per_tensor_param` 是**训练引擎(BaseEngine)→ 推理引擎(vLLM/SGLang)的唯一重量级接口**:
把"训练侧此刻的权重"(可能被 FSDP/TP/PP/EP 切得七零八落、可能被打包成 fused 张量、
可能还在 CPU 上)翻译成**一串 `(HF 名字, 普通张量)`**,让推理引擎按名字装载。

它是一条**契约方法**(`base.py:151` 里只有 docstring + `raise NotImplementedError`),
**同一个调用点、五种物化策略** —— 这正是它值得单独读的原因:接口不变,
但"怎么把参数从并行布局里拿出来"是各后端差异最大的一处。

---

## 1. 调用链(自顶向下)

### 1.1 主干:一次 step 里的 `update_weights`

```
ray_trainer.fit()                                                   verl/trainer/ppo/ray_trainer.py
 └─ with marked_timer("update_weights", ...):                       :1715-1716
     self.checkpoint_manager.update_weights(self.global_steps)
     └─ CheckpointEngineManager.update_weights                      checkpoint_engine/base.py:505
         │
         ├─【naive:colocate 同进程】backend == "naive"              :513-515
         │   ray.get(actor_wg.update_weights(mode="naive"))
         │    └─ ActorRolloutRefWorker.update_weights               workers/engine_workers.py:755
         │        ├─ effective_mode 解析(显式 mode 覆盖 config)     :755
         │        ├─ rollout.resume(tags=["weights"])               :782   # 引擎从 sleep 里醒来
         │        ├─ per_tensor_param, peft_config =
         │        │    self.actor.engine.get_per_tensor_param(      :786   ★★★ 本文主角
         │        │        layered_summon=..., base_sync_done=True)
         │        └─ await self.rollout.update_weights(gen, ...)    :804
         │            └─ VLLMRollout.update_weights                 vllm_rollout.py:210
         │                ├─ _execute_method("update_weights_from_ipc")   :229   # 引擎侧先起 receiver
         │                └─ BucketedWeightSender.async_send_weights(gen) :236-241
         │                     └─ async for name, w in produce(gen)  bucketed_weight_transfer.py:145
         │                        ← **惰性 generator 在这里才被消费**(export 计时就在这)
         │
         └─【非 naive:跨机/跨进程】其他 backend                        :517-555
             abort → 建 PG → release_kv_cache →
             ray.get(actor_wg.update_weights(mode=backend)          :534-537
                     + rollout.update_weights(...))
              └─ 同一个 ActorRolloutRefWorker.update_weights 入口,
                 但第 758-766 行走的是另一支:
                   ├─ delta_sharded:
                   │    engine.get_per_tensor_param()          ← 首步 seed(全量)
                   │    engine.prime_delta_snapshots()         ← 钉住 diff 基线
                   │    engine.get_per_tensor_param_delta_shard()  ← 之后每步
                   │    (delta_checkpoint_engine.py:586-600 驱动,worker 不参与决策)
                   └─ 其他(nccl/hccl/nixl/mooncake/kimi):
                        per_tensor_param, _ = engine.get_per_tensor_param()   :764
                        await checkpoint_engine.send_weights(per_tensor_param) :765
```

**读这张图要抓住三点:**

1. **`get_per_tensor_param()` 只是"拿到一个惰性生成器",真正的计算在消费它的时候发生。**
   naive 路径的消费点在 `bucketed_weight_transfer.py:145`;delta 路径在 seed 阶段
   (`delta_checkpoint_engine.py:588`)。所以它不是"一个函数调用 = 一次导出",而是
   **"把导出这件事的时机交给消费者"** —— 这是它能在 CPU-offload 下把峰值内存控制在
   一张张量的原因(见 §4.1)。
2. **同一个方法在两条链路上语义不同**:naive 路径它产出的是 `named_tensors` 线格式;
   delta 路径它只用于**首步 seed**(之后改走 `..._delta_shard`)。
3. 非 naive 路径下 **worker 不决定同步时机**,由 checkpoint engine 的状态机驱动
   (`delta_checkpoint_engine.send_weights(engine, ...)` 直接吃 engine 对象,不接生成器)。

### 1.2 方法族与消费者

| 方法 | 定义 | 产出 | 消费者 | 线格式 |
|---|---|---|---|---|
| `get_per_tensor_param` | `base.py:151`(抽象) | `(name, 全量 HF 张量)` generator + peft_config | naive sender / 所有 checkpoint engine 的 **seed** | `named_tensors` |
| `get_per_tensor_param_shard` | `base.py:163`(抽象) | `(name, 本地 flat bf16 分片, ShardSpec)` | `prime_delta_snapshots`(`base.py:189`,基类具体实现)+ delta 引擎 | 进程内 |
| `get_per_tensor_param_delta_shard` | `base.py:207`(抽象) | `(slots, dtype_str, counts, hf_idx, hf_val, gather_group)` —— **已是最终 HF 坐标** | delta 引擎 steady 阶段 | `delta_flush` |

三者是**同一件事的三个粒度**:全量 / 本地分片 / 增量。`prime_delta_snapshots` 在基类里
就是"消费一遍 shard 版、把每个 rank 的分片快照存下来"(`base.py:203-205`)——
所以**任何实现了 shard 导出的后端都白拿 delta 能力**。

`ShardSpec`(`workers/engine/spec.py:59`)是这三者之间共享的**声明式描述**:
`full_shape + mesh + placements`(直接抄 DTensor 的 `device_mesh`/`placements`),
由 `derive_dtensor_placement`(`spec.py:202`)推出本 rank 的 `(place, contributes, gather_group)`,
再由 `translate_flat_indices`(`spec.py:177`)把"分片内 flat 下标"翻译成"全局 flat 下标"。
**整个过程是纯数学,没有任何 collective。**

---

## 2. 输入输出契约

### 2.1 签名

```python
# 基类:只有契约,没有实现
#   verl/workers/engine/base.py:151
def get_per_tensor_param(self) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
    raise NotImplementedError

# 各后端自行加 kwargs(调用点按后端传):
#   FSDP      fsdp/transformer_impl.py:989   (self, layered_summon=False, base_sync_done=False, **kwargs)
#   Megatron  megatron/transformer_impl.py:1033  (self, base_sync_done=False, **kwargs)
#   VeOmni/Torchtitan/Automodel  (**kwargs)
```

**输入**:没有显式张量输入 —— 隐含输入是**引擎当前的 module 状态**(当前 step 训完的权重)。
两个常见 kwargs 是行为开关,不是数据:

| kwarg | 含义 | 谁在用 |
|---|---|---|
| `base_sync_done` | LoRA 场景:`True` = 基座已同步过,本次只发 adapter | `engine_workers.py:786/797`(SGLang 的"先基座后适配器"两步) |
| `layered_summon` | 逐层 summon(省显存,慢) | `engine_workers.py:786`,来自 rollout config |

### 2.2 输出

```python
( generator_of(name, tensor),  peft_config_or_None )
```

| 项 | 说明 |
|---|---|
| `name` | **HF checkpoint 命名空间**下的名字(`model.layers.3.self_attn.q_proj.weight` / `...experts.17.w1.weight`)。推理引擎按这个名字查它的 `params_dict` |
| `tensor` | **已物化的普通张量**,脱离了 DTensor / TP / PP / EP 布局。通常在 accelerator 上,量大时可能仍在 CPU(shm 路径)。**dtype 不保证 bf16** —— `bucketed_weight_transfer.py:146-151` 明确注释:*"不要在这里强制 cast,因为 moe gate 这类参数必须保持 fp32;接收侧需要时自己转"* |
| `peft_config` | `dict | None`。LoRA 时为 peft 配置(`peft_type/task_type/r/lora_alpha/target_modules/...`,枚举成员不是字符串) |

### 2.3 调用者必须知道的三个不变式

1. **惰性,但每个 rank 都要迭代完。** 多数后端在生成器内部做 collective(FSDP 的
   `full_tensor()`、veomni 的 `broadcast`),**少消费一个元素 = 别的 rank 挂死**。
2. **顺序。** 全量路径单机内顺序无所谓(按名字装载);但 **shard/delta 路径要求跨 rank
   同序**(delta 引擎按 lockstep 逐参数 gather),所以各后端的 shard 版都显式构造
   "每个 rank 都产出同样序列"的结构(Megatron 用零计数空条目占位,torchtitan 让
   expert stack 整栈出场避免"只命名本地专家")。
3. **有副作用。** 实现里普遍有 offload 进出(`load_*_to_gpu` / `offload_*_to_cpu`)、
   `state_dict()` 遍历、甚至 `merged_lora_context`。**要在正确的时机调用**
   (训练态、rollout 已 sleep/wake 到位),不要在中途插入别的集合通信。

---

## 3. 内部机制:五后端共用的"三段式"

不管哪个后端,`get_per_tensor_param` 内部都是这三段(顺序可能交错):

```
① 取出参数引用        state_dict() / bridge conversion tasks / 专用 converter
② 命名转换            convert_weight_keys(训练名 → HF 名)或 bridge 的 to-HF
③ 物化 + 打包         逐张量 摆脱并行布局 → 拼接/切分 → yield
```

差异全在第 ③ 段。**表读法:行 = 后端;列 = 它把"怎么把参数从并行布局里拿出来"做成了什么样。**

| 后端 | 类 | ③ 物化策略 | 打包/切分 | 代价 |
|---|---|---|---|---|
| **FSDP** | `FSDPEngine` `fsdp:106` | 逐张量 `.to(device).full_tensor()` **惰性全量 all-gather** | `_export_param` hook(子类可覆盖)+ `unfuse_moe_params` 拆 fused 专家 | 每 rank 物化全量;峰值 = 一张张量 |
| **Megatron** | `MegatronEngine` `megatron:164` | `bridge.export_hf_weights(module, conversion_tasks)` —— **TP/PP 融合由 bridge 负责** | bridge 的 conversion tasks(FP8 有特化路径) | 依赖 Megatron-Bridge |
| **VeOmni** | `VeOmniEngine(FSDPEngine)` `veomni:99` | 若有 converter(`DeepseekV4`)→ `converter.export_weights`;否则 state_dict + **EP 广播重组** | `get_moe_param_handler` 按 `expert_id_base` 拼 | EP 下每 rank 广播拼全量专家 |
| **TorchTitan** | `TorchTitanEngine` `torchtitan:96` | module 是**多 chunk 列表**(PP),逐 chunk `full_tensor()` | expert stack **整栈 gather 后按 slot 拆** | 整栈是峰值分配点 |
| **Automodel** | `AutomodelEngine` `automodel:70` | `param.full_tensor()` 最直白 | 无 | 无 offload 细节 |

---

## 4. 重点:`FSDPEngine`

### 4.1 `get_per_tensor_param`(`fsdp/transformer_impl.py:989`)逐段

| # | 行 | 做什么 | 为什么 |
|---|---|---|---|
| 1 | `:1000-1001` | `_is_peft` / `_skip_staging` 判定:FSDP2 且非 peft → `_skip_staging=True` | FSDP2 的 `state_dict()` **只收集 DTensor 引用**,真正物化由后面的 `.to(device).full_tensor()` 惰性完成;FSDP1 的 `(SHARDED_)STATE_DICT` 走 unshard 机制,必须在 GPU 上 |
| 2 | `:1002-1003` | 只有 `not _skip_staging` 才 `load_fsdp_model_to_gpu(module)` | 避免 CPU-offload 下"把整模型搬上卡"这种与 policy 冲突的操作(#5995) |
| 3 | `:1011-1026` | LoRA 三分支:`merge=False` → `collect_lora_params`;`merge=True` → `merged_lora_context` 内物化 | merge 分支注释写明:`state_dict()` 张量**别名活存储**,必须在 context 内物化,否则会静默发出"没合并 adapter 的基座权重" |
| 4 | `:1028` | `params = self.module.state_dict()` | 拿到 `{name: DTensor}` |
| 5 | `:1030` | `convert_weight_keys(params, module)` | 训练侧名 → HF 名 |
| 6 | `:1033-1034` | **立刻** `offload_fsdp_model_to_cpu(module)` | 关键:引用已拿到,数据还没物化 → 先把显存让出来,后面逐张量再拉回来。**这就是"惰性"的收益** |
| 7 | `:1040-1042` | `per_tensor_param = (entry for name, param in params.items() for entry in self._export_param(name, param))` | **两层生成器**:外层遍历参数,内层是**子类 hook** `_export_param`(见 4.2) |
| 8 | `:1043` | `unfuse_moe_params(...)` | 第二道拆包(面向 Qwen/GPT-OSS 的 packed 专家);对已被 hook 拆好的逐专家 key 是透传 |
| 9 | `:1045-1065` | QAT 分支:走 `QATQuantizer.quantize_with_fusion`,把权重以 **CPU 目标设备** 输出 | 量化版本单独一条路 |
| 10 | `:1067-1068` | `return per_tensor_param, peft_config_dict` | peft 从 dataclass 转 dict 交出去 |

> ⚠️ **第 6 步与第 7 步的配合是这套设计最精巧也最容易读错的地方**:`offload` 发生在
> "拿到引用"之后、"物化"之前。所以生成器被消费时,每个 `param.to(device)` 是从 CPU 拉回
> **一张**张量 —— 峰值显存 ≈ 单张张量,而不是整个模型。

### 4.2 `_export_param`:留给子类的"打包 hook"

```python
# 基类实现:fsdp/transformer_impl.py:977-987  —— 通用路径 = 全量 all-gather
def _export_param(self, name, param):
    if isinstance(param, DTensor):
        yield name, param.to(get_device_id(), non_blocking=True).full_tensor()
    else:
        yield name, param
```

这是整个 FSDP 系里**唯一的"参数打包"扩展点**。当前有两个覆盖:

| 覆盖者 | 位置 | 做了什么 |
|---|---|---|
| `FSDPTurboDSV41EngineWithLMHead` | `fsdp_turbo_dsv41_impl.py:305` | fused 专家张量按块 all-gather(默认)或**只导本 rank 分片**(`VERL_DSV41_LOCAL_EXPERT_EXPORT=1`)→ 这就是 `worklog_dsv41_weight_sync_export.md` 全文那条路径 |
| (基类) | `transformer_impl.py:977` | 全量 all-gather,其余所有参数走这里 |

**继承链**(读覆盖关系必备):

```
BaseEngine                                   base.py:30
└─ FSDPEngine                                fsdp/transformer_impl.py:106
   ├─ FSDPEngineWithLMHead                   fsdp/transformer_impl.py:1151
   │   ├─ FSDPTurboEngineWithLMHead          fsdp/fsdp_turbo_impl.py:24
   │   │   └─ FSDPTurboDSV41EngineWithLMHead fsdp/fsdp_turbo_dsv41_impl.py:301   ← _export_param 覆盖
   │   └─ FSDPEngineWithValueHead            fsdp/transformer_impl.py:1602
   └─ VeOmniEngine                           veomni/transformer_impl.py:99      ← 复用 FSDP 的 shard/delta
```

### 4.3 `get_per_tensor_param_shard`(`:918`)—— 给 delta 引擎用的分片导出

```
:927-929  FSDP1 才 staging(同 4.1 第 2 步的理由)
:930-931  state_dict() + convert_weight_keys
:937      ShardSpec.from_param(param)      ← 直接抄 DTensor 的 mesh/placements
:942      p.to(device, non_blocking=True)  ← 本地分片拉到卡上
:943-944  浮点统一转 bf16(线上协议就是 bf16)
:945-946  to_local() → reshape(-1)         ← **扁平一维**,方便按 flat 下标做 diff
返回      (_gen(), None)                   ← 注意:peft 一律 None("Non-LoRA base path only")
```

对比全量版,**这里没有 `full_tensor()` —— 只有本地分片,零 collective**。这就是 delta 引擎
"没有 rank 需要持有全模型快照"的基础。

### 4.4 `get_per_tensor_param_delta_shard`(`:963`)+ `_hf_delta_entry`(`:950`)

```python
gen, _ = self.get_per_tensor_param_shard()
return hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry), None
```

- `hf_delta_export`(`workers/engine/utils.py:250`;快照在 `:286 prime_delta_snapshots`,
  身份参数条目在 `:238 _hf_entry_identity`)是**后端无关**的执行器:逐参数算
  `当前 bf16 值 vs 快照` 的变化元素,翻译成全局 HF 坐标,产出
  `(slots, dtype_str, counts, hf_idx, hf_val, gather_group)`;每个 rank 出同样的序列
  (不贡献的 rank 出零计数条目,保持 lockstep)。
- `_hf_delta_entry` 是**每参数**的钩子。FSDP 版只处理"身份参数"(名字与 HF 同名、坐标直译),
  遇到需要 converter 的 spec **直接抛 `NotImplementedError`** —— 并明确写着"converter spec
  属于声明它的那个引擎"(即 veomni)。

---

## 5. 五个后端:主要差异对照

**表读法**:同一行 = 一个后端在**同一套契约**下的实现选择;空白 = 该后端没有实现/不需要。

| | FSDP | Megatron | VeOmni | TorchTitan | Automodel |
|---|---|---|---|---|---|
| 类位置 | `fsdp:106` | `megatron:164` | `veomni:99`(继承 FSDP) | `torchtitan:96` | `automodel:70` |
| `get_per_tensor_param` | `:989` | `:1033` | `:626` | `:631` | `:421` |
| 全量物化 | `.full_tensor()` 逐张量 | `bridge.export_hf_weights` | converter 优先,否则 state_dict + EP 广播 | 逐 chunk `full_tensor()` | `param.full_tensor()` |
| 名字转换 | `convert_weight_keys` + `unfuse_moe_params` | bridge conversion tasks | `get_moe_param_handler` / converter | `_to_hf_named_params` | `convert_weight_keys` |
| packed 专家 | `_export_param` hook(可覆盖)+ 逐专家 key | bridge 负责 | 按 `expert_id_base` 拼 + broadcast | **整栈 gather 再按 slot 拆** | 无 |
| `..._shard` | `:918` | `:1084` | `:586` | `:594` | ❌ 未实现 |
| `..._delta_shard` | `:963` | `:1116` | 继承 FSDP | `:574` | ❌ |
| LoRA | ✅ 三分支 | ✅ | ❌ TODO | ❌ TODO | ❌ |
| QAT | ✅ | ✅ | | | |
| 独有约束 | FSDP1/FSDP2 staging 差异 | **PP 空 shard 占位**;vanilla_bridge 不支持 delta;FP8 特化 | EP 手工切分 → `ShardSpec.place` 显式覆盖 | **shard 导出不支持 PP**(`_assert_shard_export_supported:582`:各 stage 持不同切片,序列无法跨 rank 一致) | — |

### 5.1 三条"差异的来源"归纳

所有差异都能归到三类:

1. **并行布局不同** → 物化方式不同。FSDP 只需 all-gather;Megatron 要融合 TP/PP(还可能 FP8);
   TorchTitan 的 module 是 PP chunk 列表;VeOmni 的 EP 是**手工切分**,DTensor placements
   不足以描述,所以要用 `ShardSpec.place` 这个显式覆盖字段。
2. **参数打包方式不同** → 拆包位置不同。Qwen 系是 `.mlp.experts.gate_up_proj` 3D 打包;
   DSV4.1 是 `.ffn.experts.gate_up_proj`(且训练侧和 ckpt 的命名习惯不同:w1/w3/w2);
   TorchTitan 的 expert stack 必须**整栈出场**否则 EFSDP 持有的专家会被漏掉;
   VeOmni 走自己的 MoE handler。
3. **有没有重型导出设施** → 自己写还是委托。Megatron 全部委托给 Megatron-Bridge;
   VeOmni 在 DeepseekV4 上委托给 `converter.export_weights`;其余各家自己写循环。

### 5.2 两个值得注意的"防御性设计"

- **Megatron 的零计数空 shard**(`:1096-1098`):PP 下某 rank 不持有这个参数,但**必须**
  产出同样序列的一个条目(空张量 + 零计数),否则 delta 的 lockstep gather 错位。
- **TorchTitan 的 `_assert_shard_export_supported`**(`:582-592`):PP 直接抛异常并解释原因
  ("每个 stage 持模型的不相交切片,导出顺序无法像 delta 引擎要求的那样跨 rank 一致"),
  而不是让它静默错位。**这是"宁可报错不要静默错"的好例子。**

---

## 6. 读码/改动时的踩坑清单

1. **返回的是 tuple,不是生成器。** `gen, peft = engine.get_per_tensor_param()`。
2. **别在生成器外面等全部张量。** 全量版每张量内含 collective,而且 offload 已在生成器
   之前完成 —— 一次性 `list(...)` 会把整模型重新拉回显存,等于把惰性设计抹掉。
3. **不要强制统一 dtype。** `bucketed_weight_transfer.py:146-151` 的注释是硬约束:
   moe gate 等参数必须保持 fp32,由接收侧按需 cast。
4. **少消费一个元素 = 分布式挂死**(collective 在生成器里)。
5. **`base_sync_done` 只对 LoRA 有意义**,别在非 LoRA 场景传 `True` 期待"跳过基座"。
6. **改了 `_export_param`(DSV41 的做法)就同时改变了 shard 版吗?不。** `..._shard` 走的是
   完全独立的实现(`:918`),不经过 `_export_param`。**两者要对齐需要各自改**(这是本次
   工作流里 `LOCAL_EXPERT_EXPORT` 只影响 naive 全量路径、delta 路径天然没有这个问题的原因)。
7. **行号会漂。** 引用时把代码片段一起抄下来(本文即如此)。

---

## 7. 与其他文档的关系

- 机制 → 实测的落地:[`../worklog_dsv41_weight_sync_export.md`](../worklog_dsv41_weight_sync_export.md)
  (FSDP `_export_param` 的两种路径、105→16.37 GiB、1032.7s→18.7s 的现场与数字自校验)。
- 训推一致性工作的全景:[`../worklog_dsv41_real4_prod_consistency.md`](../worklog_dsv41_real4_prod_consistency.md)。
- 权重同步的历史决策(G13 首次认定 export 是瓶颈):[`../worklog_dsv41_rl.md`](../worklog_dsv41_rl.md)。