# 02 —— V4.1 索引器写缓存与 `[-1,-1]`：E9 是怎么定位出来的

> 类型：学习笔记（以代码为纲）。所有行号是 2026-10-08 当时的工作副本，引用一律带 `文件:行号`
> 并抄下当时的片段——行号漂了片段还在。
> 时间线版的实验记录在 [`../worklog_dsv41_top64_grpo.md`](../worklog_dsv41_top64_grpo.md) §5.7.8。
> 仓库根：`/workspace-verl/vllm-ascend-v41-private`（下称 **fork**）与 `/workspace-verl/verl`。

## 0. 结论与当前状态

**结论**：`torch.ops._C_ascend.npu_scatter_nd_update_sk` **没有"跳过 `[-1,-1]` 行"的语义**。
V4.1 的 metadata builder 把"这个 token 不该写缓存"表达成一行坐标 `[-1, -1]` 并原样喂给这个算子；
算子把这行的坐标按有符号整数求和成线性行号（`-65`），再 `static_cast<uint64_t>` 当成元素偏移去
`DataCopyPad` → 目标地址变成 plane 基址**下面**几千到二十几万字节。page 轴是 view 时该地址落在
映射外 → `MTE accesses an invalid GM address` + vector core exception（**E9**）；page 轴连续时该地址
仍在映射内 → **不报错，静默改写别人的内存**。

| 项 | 状态 | 证据 |
|---|---|---|
| 根因判定 | ✅ 完成 | 单卡 6 例矩阵 + 生产形状复现（§4.2、§5.4） |
| 方案 A（python 侧过滤 marker 行） | ✅ 已打并验证；**可用 `DSA_SCATTER_DROP_MARKERS=0` 临时关掉**（做对照实验时必须关，否则过滤会掩盖现象） | 同一套 GRPO argv：未打补丁 3/3 死，打了补丁 3 个请求全部返回、日志里 `MTE/vector core` 命中 0 |
| 方案 C（kernel 里跳过负行号） | ⚠️ **源码已改，二进制没变，行为未变** | 见 §6，别当成已修好 |
| "为什么用户单独跑 vllm 不报错" | ✅ 已实测：**marker 一样出现（288 次），只是落点仍可映射 → 静默写脏**；翻转这条轴的是 `--enable-sleep-mode`，不是 expandable_segments / prefix caching / mem_util | §7 第 3 条的三组矩阵 |

## 1. 报错长什么样，以及它缺了哪块信息

GRPO 第 4 次启动（10-08 03:09）之后，8 个 engine worker 同时抛（`verl/logs/grpo_top64_steps1_20261008_030936.log`）：

```
error code = 0x800000 ... errorStr: MTE accesses an invalid GM address or the cross-device memory
access times out. fixp_error0 info: 0x6000054 ... [Error]: The vector core execution is abnormal.
kernel name: ScatterNdUpdateSk_38f6c0f42e6dd25f1ba132d7ecf7990f
```

有用的部分只有一个：**kernel 名**。其余全是"设备已经死了"的事后描述。

没用的部分（也是这次的关键教训）：Python traceback 指向的**不是**出事的那次调用。即使设了
`ASCEND_LAUNCH_BLOCKING=1`，fork 的 runner 用自建 stream + event 做流水（`worker/model_runner_v1.py`
的 `device_metadata.py:143 release → torch_npu/npu/streams.py:151 record`），异常是在**下一个同步点**
被"发现"的：

```
File ".../vllm_ascend/worker/model_runner_v1.py", line 3286, in _model_forward
File ".../vllm_ascend/worker/device_metadata.py", line 143, in release
File ".../torch_npu/npu/streams.py", line 151, in record
RuntimeError: LaunchRecordTask:... aclrtRecordEvent ... error code is 507035
```

所以**不能靠 traceback 定 op**，必须换招：① 用 kernel 名反查源码；② 在主机侧把每次调用的参数打印出来。

## 2. 定位路线（按实际执行顺序，每步都标了"排除了什么"）

| 步 | 做了什么 | 得到什么 | 排除了什么 |
|---|---|---|---|
| 1 | kernel 名 `ScatterNdUpdateSk_38f6c0f4….o` 在 fork 里 `find` → 存在，mtime 09-30 04:13 | **跑的就是这份源码编出来的 kernel**，不是系统里另一份 | "构建产物与源码不一致"这类（E2 的老坑） |
| 2 | 只起引擎、dummy 权重、GRPO 配方的 argv 原样（`scripts/run_vllm_dsv41_rl_dummy.sh`），10 分钟到第一个真实 prefill | 复现成功：04:39 那次照样死，栈在 `indexer.py:109` 附近 | 训练侧、Ray、DCP 权重、权重同步内容 |
| 3 | 同脚本 A/B：`RL_ENABLED/BATCH_INVARIANT=false`（05:00） | **照样死** | `rl_config`、batch-invariant（当时唯一没排除的"RL 特有开关"） |
| 4 | 写主机侧探针（`scripts/dsa_scatter_probe.py`，用 `--worker-extension-cls` 注入 worker），在**调用前**检查 `indices` 对 `cache` 的 shape/stride | 04:55 那次抓到：坐标全在界内（`rows[-1,1] cols[-1,1] skipped=3/5`），且**只有出事那次**同时具备"有 marker"+"是 view"两个特征；同轮 bf16 scatter 是 `contiguous + skipped=0/5` → 活下来 | "越界索引"这个第一反应。**注意探针此时只做 bounds 检查，没检查"负数"**，这是下一步才补上的视角 |
| 5 | 读算子源码三层（aclnn → tiling → kernel），`grep -i skip` 只命中注释 | 负坐标在算子里没有任何概念 | 剩下唯一解释方向：marker 就是被当真实坐标用了 |
| 6 | 单卡微复现，把两个变量正交开（`scripts/test_scatter_nd_update_sk_skip.py --matrix`），并在 plane 前后各放 16384 元素哨兵 | 6 例全中预期：marker 才是因，view 只决定"炸/静默"（§5.4） | "view 本身有问题"（view+无 marker 是 OK 的） |
| 7 | 按 kernel 的公式手算 marker 的落点 | 预测 `−8320`（contig）/ `−254080`（view），与哨兵实测**逐字节吻合** | 一切"大概是因为"的说法——现在它是算术，不是猜测 |
| 8 | 回查 marker 从哪来（`dsa_v41.py:769-798` + `compressed_slot_mapping`） | `compress_ratio=2` ⇒ **一半行天然是 marker**，不是偶发脏数据 | "是不是某个 batch 才有" |
| 9 | 回查为什么以前没暴露 | profile run 在 `dsa_v41.py:540-542` 整层早退；探针第一次打印在 `Application startup complete` 之后 | "以前跑过生成没报错"这个前提——历史日志里 `finish_reason` 命中数为 0，**从来没有一次生成成功过** |

## 3. 这条链上的代码，逐个讲

### 3.1 调用链（从 `forward` 到报错那一行）

```
model.py:1339  DeepseekV41Model.forward          逐层调用 decoder layer
model.py:1260  layer(...)                        一个 block：mHC 混合 → self_attn → FFN
model.py:891   DeepseekV41Attention.forward      壳：output=empty_like(hs); torch.ops.vllm.dsa_v41_forward(...)  [model.py:753]
dsa_v41.py:52  dsa_v41_forward                   从 forward_context 取回 attn 模块 → attn.v41_impl.forward(...)
dsa_v41.py:533 AscendDSAV41Impl.forward          ← 主控
   :540-542   attn_metadata is None → output.zero_(); return      【profile/dummy run 的早退通道】
   :543      _get_layer_metadata                     取这层的 swa / long_kv / index_k / compressor_state 四份元数据
   :549 → :310  multistream_preprocess               Q 与 KV 双流重叠；:362 写 SWA 缓存（scatter #1）
   :298 → :378  _write_compressed_source             压缩层的写
   :418 → indexer.py:94  update_keys                 :109 写 int8 index-K（scatter #2 ← E9 就在这）
                                                       :110 写 fp16 scale   （scatter #3，那次运行没走到）
   :432                                              写 bf16 long-KV       （scatter #4，同上）
   :550 → :438 _select_sparse_indices → indexer.py:116 select → :213 npu_quant_lightning_indexer_v3（粗排 TopK）
   :551 → :472 _forward_attention → :500 npu_sparse_flash_mla（真注意力）
   :562 → :302 _project_output                      O-proj 写进调用方给的稳定 buffer
```

一层里"写缓存"最多 4 次 scatter。`compress_ratios`（`config.json`）=
`[0,0, 2,2,…×18, 1,1,…×20, 0,0,0]` → 层 2–19 是 ratio=2 的压缩层。
探针看到的顺序正好对上：`call#1-3` 是 `:362` 的 SWA 写（层 0、1、2），`call#4` 是 `indexer.py:109` → **死**。

### 3.2 `DeepseekV41Indexer.update_keys`（`indexer.py:94-114`）

**作用**：把"索引卡片"（index K）算出来并写进缓存，供后面的粗排 `select` 读。
**输入**：`latent`（compressor 池化出来的每组向量，`[组数, width]`）、`slots`（**这个函数不产生** slots，它是
`indexer_metadata.cache.slot_mapping[:positions.shape[0]]`，见 `dsa_v41.py:415`）、`cos/sin`。
**做的事**：`wk` 投影 → `k_norm`（`:98`）→ 只旋转尾部 `rope_width` 维（`:99-105`，`partial_slice=[width-rope_width, width]`）
→ `npu_dynamic_quant` 转 int8 + 每组 scale（`:107`）→ **两次 scatter**（`:109` 写 int8 K、`:110` 写 fp16 scale）。
**为什么写两份**：int8 是可逆性很差的量化结果，读的时候必须配 scale，所以 scale 是**同 page 里的另一个 plane**。

```python
94  def update_keys(self, latent, slots, cos, sin):
96      if not self.owns_k or latent.shape[0] == 0:      # 只有"这层负责生产 K"才写
97          return
107     quantized, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
108     k_cache, scale_cache = self.k_cache.kv_cache[0]   # 一个 layer 的 kv_cache 是 (K, scale) 元组
109     scatter_cache_sk(k_cache, slots, quantized)       # ← E9
110     scatter_cache_sk(scale_cache, slots, scale.unsqueeze(-1).to(torch.float16))
```

`k_cache` 这个 plane 的形状从 `:73-87` 的 spec 来：

```python
AscendMLAAttentionSpec(block_size=cache_config.block_size, num_kv_heads=1, head_size=self.width,  # 128
                       dtype=torch.int8, tokens_per_state=compress_ratio,                          # 2
                       storage_block_size=(cache_config.block_size // compress_ratio),             # 128//2 = 64
                       scale_dim=1, scale_dtype=torch.float16)
```

`block_size // compress_ratio` 就是探针看到的 `64` 行/页，`head_size=128` 就是每行 128 个 int8，
`scale_dim=1` 就是形状里那个 `1`。

### 3.3 `compressed_slot_mapping` 与 builder：`[-1,-1]` 从哪来

**`dsa_v41.py:177-184`**——把"原始 token 的物理 slot"换成"已完成压缩组的 slot"：

```python
177 def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
180     """... Negative/padded slots and incomplete compression groups never produce a write."""
183     valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
184     return torch.where(valid, slot_mapping // ratio, -1)
```

**输入**：一维 slot（每个 token 一个整数，无效为负）。**输出**：同长的一维，**无效位仍是 -1**。
注意 docstring 的意图——"never produce a write"是**约定**，不是实现。

**`dsa_v41.py:763-800`（builder 内）**把一维 slot 变成算子要的二维坐标，并把无效行**两个坐标都**写 -1：

```python
769  valid = active_slots >= 0
770  if compressed and ratio == 2:
776      valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
777      valid &= torch.arange(num_input_tokens, ...) < valid_end
779      valid &= positions.remainder(2) == 1          # ← 每 2 个 token 只有 1 个负责写这一组
780  physical = active_slots.clamp_min(0)
781  self._slot_mapping_2d[:n, 0].copy_(torch.where(valid, physical // storage_block_size, -1))   # 页号
792  self._slot_mapping_2d[:n, 1].copy_(torch.where(valid, physical %  storage_block_size, -1))   # 页内行号
```

`:779` 这一行是**根因的源头**：ratio=2 时有效集合是**交错**的（每隔一个 token），所以无效 marker 不可能靠
"切前缀"绕开——`device_op.py` 里 V4 那招（只喂 `slots[:tokens]` 前缀，见 §3.8）在这里天然失效。
`:596-597` 还说明了另一个细节：`_slot_mapping` / `_slot_mapping_2d` 是**预填 -1 的常驻 buffer**
（`torch.full((max_tokens, 2), -1)`），pad 出来的行本来就是 marker。

### 3.4 `scatter_cache_sk`（`dsa_v41.py:207-247`，**含本次改的前后两版**）

改之前（当时的原文，注意那句 docstring 就是错的）：

```python
212 """Store rows using builder-prepared coordinates and V4's Ascend op.
216    ... ``npu_scatter_nd_update_sk`` preserves that stride and
217    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4."""
219 cache = cache.squeeze(-2)                        # (页, 行, 1, 维) → (页, 行, 维)
220 indices = slot_mapping[: values.shape[0]]        # 只取 values 那么多行
221 updates = values.to(cache.dtype).contiguous()
222 torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates)
```

- **输入**：`cache` 是**某个 plane**（4 维，倒数第 2 维恒为 1）、`slot_mapping` 是 `[n,2]` 的 int32 坐标、`values` 是 `[n, dim]`。
- **输出**：无返回值，原地写 `cache`。
- **为何错**：`:217` 那句是**假的**（§5 全部实测都是在证伪它）。它把 marker 原样交给算子，算子把它当坐标。

改之后（方案 A，`dsa_v41.py:236-247`）在调用前把 marker 行滤掉：

```python
240  if indices.numel():
242      keep = (indices >= 0).all(dim=-1) if indices.dim() == 2 else indices >= 0
243      if not bool(keep.all()):                # ← 这一次 .item() 是 DtoH 同步
244          indices, updates = indices[keep], updates[keep]
245  if indices.shape[0] == 0: return
247  torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates.contiguous())
```

代价：每次 scatter 一次同步（压缩层每层 3 次 × 18 层）。这与同文件 `:187-204` 的 `_request_counts`
（docstring 明确 "without synchronizing the NPU"）的设计意图相反，**所以它是诊断件，不是终态**。

### 3.5 算子内部：一次 scatter 的四个阶段

**① torch binding（`csrc/torch_binding.cpp:1909-1917`）**——把 `var.strides()`（**元素**为单位）当属性传下去：

```cpp
1909 void npu_scatter_nd_update_sk(at::Tensor& var, const at::Tensor& indices, const at::Tensor& update) {
1914     at::IntArrayRef var_stride = var.strides();
1915     EXEC_NPU_CMD(aclnnScatterNdUpdateSk, var, indices, update, var_stride);
```
schema 在 `:3544-3549`：`npu_scatter_nd_update_sk(Tensor(a!) var, Tensor indices, Tensor update) -> ()`。

**② aclnn 包装（`op_host/op_api/aclnn_scatter_nd_update_sk.cpp:88-103`）**——只做三件事：把 view 的
storage shape 修正回来、把 `indices/updates` 变连续、进 AICore 队列。**没有任何过滤**：

```cpp
 93  // kernel直接原地更新varRef，通过 strides 属性处理非连续（首轴可非连续），无需对 varRef 做 Contiguous
 94  varRef->SetStorageShape(varRef->GetViewShape());
 95  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
```

**③ tiling（`op_host/scatter_nd_update_sk_tiling.cpp`）**——两件事与本案有关。

一是**行号掩码**（`:84-90`，注意它用的是 `GetStorageShape()`，`:470-481` 的 `ParseScatterShapes` 里
`indexDim = indices 最后一维 = 2`、`scatterLength = 后面各维乘积 = 128`）：

```cpp
84  uint64_t strides = 1;
85  for (int64_t i = indexDim - 1; i >= 0; --i) { indicesMask_[i] = strides; strides *= varShape.GetDim(i); }
```
→ `indicesMask = [64, 1]`（64 = 每页行数）。

二是**判断 page 轴是不是 view**（`:238-277`）：

```cpp
239 firstDimStrideRows_ = (indexDim_ > 1) ? indicesMask_[0] : 1;          // 64
240 uint64_t stride0Expected = scatterLength_ * firstDimStrideRows_;      // 128*64 = 8192 元素
254 int64_t stride0 = stridesPtr->GetData()[DIM_0];                       // 131072（探针实测值）
273 if ((uint64_t)stride0 > stride0Expected) { isViewStride0_ = 1; varStride0Elements_ = stride0; }
```
→ int8 plane 判定为 view，`varStride0Elements_=131072`；bf16 SWA plane 的 `stride0=65536` 元素恰好等于
`128*512`，判定为连续。**这一分支就是 E9 的"炸/静默"分叉点。**

**④ kernel（`op_kernel/`）**。入口只 dispatch 到两条路（`op_kernel/scatter_nd_update_sk.cpp:52-62`）：
tiling key 10/11/20/21 → HP kernel，30 → LargeIndex kernel（`no_sort.h`/`scatter_nd_update.h`/`linear_index.h`
在这份 build 里不可达）。

先算线性行号（`op_kernel/arch22/scatter_nd_update_common.h:98-124`）：

```cpp
103 Duplicate<int>(indicesLocal, 0, rows);
108 for (uint64_t i = 0; i < indexDim; ++i) {
114     Gather(addTmpLocal, indicesOriginLocal, rangeCasted, ...);   // 取 coord[i]
117     Muls(addTmpLocal, addTmpLocal, (int)indicesMask[i], rows);   // × mask[i]
119     Add(indicesLocal, indicesLocal, addTmpLocal, rows);          // 累加，**有符号**
```
`[-1,-1]` → `(-1)*64 + (-1)*1 = -65`。这里没有、也没有地方做负数检查。

再把行号变成字节偏移并写出（`scatter_nd_update_common.h:62-81` + `scatter_nd_update_hp.h`）：

```cpp
62  __aicore__ inline uint64_t ComputeViewedRowOffset(uint64_t linearIndex, uint64_t firstDimStrideRows,
63                                                    uint64_t varStride0Elements, uint64_t scatterLength) {
68      uint64_t i0 = linearIndex / firstDimStrideRows;      // 无符号除法！
69      uint64_t rest = linearIndex - i0 * firstDimStrideRows;
70      return i0 * varStride0Elements + rest * scatterLength; }
74  template <bool isViewStride0> ResolveOutOffset(...) { isView ? ComputeViewedRowOffset(...) : linearIndex*scatterLength; }

// hp.h（改动前的行号）：
222 for (uint64_t k = 0; k < batchRows; ++k) {
223     uint64_t linearIndex = static_cast<uint64_t>(indicesLocal_.GetValue(idx));   // ← -65 变成 2^64-65
224     uint64_t gmOut = ResolveOutOffset<isViewStride0>(linearIndex, ...);
226     DataCopyPad(outputGm_[gmOut], buf[ubOffset], pout);                          // ← 写出去
```

`static_cast<uint64_t>` 是致命的第二步：第一步把 `-1` 变成 `-65`（还只是"错的行号"），第二步把 `-65`
变成一个**巨大的无符号数**，于是"错的行号"升级成"错的地址"。

## 4. 为什么 marker 的地址恰好是那两个数

设 `mask0=64`（每页行数）、`scatterLength=128`（每行元素数）。marker 行 `[-1,-1]` → `linear = -65` →
`u = 2^64 - 65`。

| page 轴 | 偏移公式 | 结果（mod 2⁶⁴） | 相对 plane 基址 | 后果 |
|---|---|---|---|---|
| 连续 | `u * 128` | `2^64 − 8320` | **−8 320 字节**（int8） | 仍在映射内 → **静默改写 plane 前面的邻居** |
| view | `i0*131072 + rest*128`，`i0 = u//64 = 2⁵⁸−2`、`rest = 63` | `2^75 − 2^18 + 8064 ≡ 2^64 − 254080` | **−254 080 字节** | 掉出映射 → `MTE invalid GM address` → E9 |

`2^75 mod 2^64 = 0` 是那一步的关键：view 分支里 `i0*varStride0Elements` 直接溢出取模，落到一个
"离基址 248 KB"的位置；而连续分支只落到"离基址 8 KB"的位置。**同一份错误数据，两种死法，
区别只在 page 轴的 stride。** 这两个数不是推演——§5.4 的哨兵实测到 `plane_base-8320`，
§5.3 的复现实测到 view 分支的设备异常。

> **这一节后来被真引擎的数据修正过一次口径**（§7 第 3 条）：stride 决定的是"**离基址多远**"
> （8 KB 还是 248 KB），而**炸不炸**取决于"那个地址是否在本进程的映射里"。三组引擎配置里
> int8 plane 都是 view、落点都是 `-254080 B`，但只有开 `--enable-sleep-mode` 的那组炸——
> 因为那条路径下 KV 池张量单独预留 VA、起点 16 MB 对齐，下面就是空洞；不开时 plane 只是
> 一个大块内部的切片，下面 248 KB 仍可读。所以"stride 决定生死"要读成"stride 决定距离，
> 分配方式决定那块距离是否映射内"。

## 5. 复现与取证：三个工具怎么用

### 5.1 bounds 探针（只在真引擎里跑）

`scripts/dsa_scatter_probe.py`：包一层 `dsa.scatter_cache_sk`，在**主机上**把 `indices` 对
`cache.shape/stride` 校验后再放行；用 `ProbeWorkerExtension`（继承 verl 的
`vLLMColocateWorkerExtension`，必须继承而非委托，否则 verl 那个零参 `super().__new__(cls)` 会炸）
经 `--worker-extension-cls` 注入。开关：`DSA_SCATTER_PROBE=1`（`_MAX_REPORTS` 用
`DSA_SCATTER_PROBE_MAX`，`DSA_SCATTER_PROBE_TOUCH=1` 会额外去**读**一遍要写的行，用来区分
"坐标错"和"这块内存压根没映射"）。它当时抓到并让推理继续走下去的关键一行：

```
[dsa-scatter-probe] call#4 from indexer.py:109 update_keys cache=shape=(33357, 64, 1, 128)
  strides=(131072, 128, 128, 1) dtype=torch.int8 ... values=(5, 128) rows[-1,1] cols[-1,1] skipped=3/5
```
`call#5`（`:110` 的 scale 写）在整份日志里从未出现——**死在第一次带 marker 的 scatter 里**。

探针现在还会对**每个含 marker 的调用**打印落点算术（`_marker_forecast()`），这一行把"为什么这组配置炸、
那组不炸"从叙述变成可比对的数：

```
marker_dest[view]=-254080B->0x12d713521f80 plane_storage=[0x12d713550000,0x12d903fb0000)
  in_own_storage=False in_mapped_block=NO
```

`in_own_storage` 是纯算术、可信；`in_mapped_block` 来自 `torch.npu.memory.memory_snapshot()`，
**在 CuMem/池化分配下不可信**（不炸的运行也打印 `NO`），别拿它当证据（§7 第 4 条）。
另外 `DSA_SCATTER_PROBE_STRICT=0` 让"越界"只警告不中断——否则探针自己先把实验停掉了，
就看不到算子本来会做什么（§7 那张矩阵就是这么跑出来的）。

### 5.2 引擎独立复现

`scripts/run_vllm_dsv41_rl_dummy.sh`：把 GRPO 配方交给 vLLM 的 argv/env 原样搬过来 + dummy 权重 +
发 3 个请求（短解码 / 跨 chunk 的长 prefill / n=4 批量），**无 trainer、无 Ray、无 DCP**，10 分钟到
第一个真实 prefill。A/B 开关：`RL_ENABLED` / `BATCH_INVARIANT` / `BLOCK_SIZE` / `DO_SLEEP_WAKE` /
`WORKER_EXT` / `LOAD_FORMAT`。

### 5.3 单卡微复现（本案的决定性实验）

`scripts/test_scatter_nd_update_sk_skip.py`。设计要点三个：

1. **两个变量正交**：`--layout contig|view` × `--skip no|yes`，别的一个不动；
2. **哨兵**：plane 前后各铺 16384 个 `SENTINEL=3` 的元素（`:144-147`），因为"plane 内容全对"**并不能
   证明没写坏东西**——marker 的落点在 plane 外面。这一条是本案最容易漏的取证点；
3. **整 plane 比对**：期望值先算成整块张量再 `torch.equal`（`:257-263`），而不是只看写过的那几行，
   这样"落到 plane 里另一个错误行"的静默错也能抓到。

`--engine` 用探针抓到的真实形状（33357 页 × 64 × 128 int8，stride 131072）；`--fix drop|sink` 用来
验证两种修法的可行性；`--matrix` 一个 case 一个进程（设备异常会污染整个 context，不能同进程复用）。

### 5.4 实测结果（10-08，Ascend910，单卡）

```
layout  skip   fix    结果
contig  no     -      OK
contig  yes    -      WRONG RESULTS: guard below: 128 byte(s) overwritten; first at plane_base-8320 elements (value=7)
view    no     -      OK
view    yes    -      FAULT: ... vector core exception / AclrtSynchronizeDeviceWithTimeout 507035   ← E9
view    yes    drop   OK: 3 rows written, guard intact
view    yes    sink   OK: 3 rows written, guard intact
```
`--engine` 形状下第 4 行同样 FAULT，第 5/6 行同样 OK。**结论：marker 是因；contig/view 决定它静默还是响亮。**

### 5.5 跑单卡自定义算子的两个环境坑（都实测过）

- 必须先 `bootstrap_custom_op_env(include_vendor_lib=True)`（`vllm_ascend/utils.py:410-418`，fork 自己的
  单卡算子测试也是这个套路），否则报
  `aclnnScatterNdUpdateSk ... not in libopapi.so` —— 那是**算子没找到**，不是 E9，别混。
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 后再 `import torch; import torch_npu; import vllm_ascend.vllm_ascend_C`：
  这个容器里 torch 的 device-backend 自动加载会崩，显式导入才通。
- `PYTHONPATH` 要**追加**不要覆盖（`/usr/local/Ascend/cann-9.1.0/python/site-packages` 里有 `acl`），
  且 cwd 不要在 `/workspace-verl`（那里的 `vllm/` 目录会遮蔽真的包 →
  `ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)`）。

### 5.6 配置轴矩阵（回答"为什么那套配置不报错"）

`scripts/run_vllm_dsv41_rl_dummy.sh` 现在有 6 个镜像开关，可以把"用户的 plain serve"和"GRPO 配方"
这两套 argv 在同一条命令上互换：`SLEEP_MODE / EXPANDABLE / PREFIX_CACHING / MEM_UTIL /
MAX_NUM_BATCHED / MAX_SEQS`（`--enable-sleep-mode` 与 `--no-enable-prefix-caching` 已从硬编码改成条件生成）。
一次三组的驱动脚本当时放在 `/tmp/e9_axis_matrix.sh`（每组都打 `marker 调用数 / 设备异常数 /
首个落点打印`，并在组间 `pkill -9 -f VLLM::` + `sleep 25` 等设备回收）。跑法：

```bash
DSA_SCATTER_DROP_MARKERS=0 DSA_SCATTER_PROBE=1 DSA_SCATTER_PROBE_STRICT=0 \
SLEEP_MODE=0 EXPANDABLE=1 PREFIX_CACHING=1 MEM_UTIL=0.9 MAX_NUM_BATCHED=8192 \
MAX_MODEL_LEN=8192 MAX_SEQS=16 RL_ENABLED=false \
  bash scripts/run_vllm_dsv41_rl_dummy.sh      # ≈ serve 形状；GRPO 形状把这几个开关换回默认值
```

结果与解读在 §7 第 3 条（时间线版：worklog §5.7.9）。

## 6. 修复状态

| 方案 | 位置 | 状态 |
|---|---|---|
| A：python 侧过滤 marker 行 | `dsa_v41.py:236-247` | **已打、已验证**（引擎独立复现 3/3 通过，0 次设备异常）。副作用：每次 scatter 一次 DtoH 同步。**它是诊断件**：能证明因果，不该作为终态留在生产路径 |
| B：每 plane 留一页 sink | 需改 `models/deepseek_v41/cache_config.py:137-160` + `worker/model_runner_v1.py:5181-5213`，并保证 block pool 不发那页 id | 单卡上 `fix=sink` 已验证**行为可行**；工程改动最大，未做 |
| C：kernel 跳过负行号 | `arch22/scatter_nd_update_hp.h:222-243`（`DoBatch`）、`:246-255`（`ScatterTileSlice`），判 `linearIndex < 0` 就跳过 | **源码已改、重编已跑完（`Successfully installed vllm_ascend-…d20261008`），但单卡行为没变**（view+marker 仍 FAULT、contig+marker 仍写到 `plane_base-8320`） |

C 未生效的证据与下一步排查（**别跳过，这份笔记的下一步就在这里**）：

1. `vllm_ascend/_cann_ops_custom/.../kernel/ascend910_93/scatter_nd_update_sk/` 下的 `.o` **mtime 更新到
   09:31，但 hash 名字与 09-30 完全一致**，且整个仓库里没有新 hash 的 `.o`（`find -name 'ScatterNdUpdateSk_*.o'`
   → 16 个 hash 全是旧的）⇒ **我的头文件改动没有进二进制**。
2. 最可能是 09-30 那次 E3 的同族问题：`csrc/build` 的 Ninja/CMake 缓存没把 `op_kernel/arch22/*.h` 当依赖
   （改的是 `.h`，不是 `.cpp`）。试 `touch csrc/moe/scatter_nd_update_sk/op_kernel/scatter_nd_update_sk.cpp`
   重编；再不行就 `rm -rf csrc/build` 重编（E3 的教训：`CMakeCache.txt` 里 `AICPU_CUST_OBJ_TARGETS`
   是 CACHE INTERNAL，只增不减）。
3. 也要排除"运行时读的不是这份 `_cann_ops_custom`"：`ASCEND_CUSTOM_OPP_PATH` 由
   `bootstrap_custom_op_env` 前置到工作副本，但 `csrc/cmake/scripts/custom/install.sh` 也可能把包
   装进 toolkit 的 `opp/vendors/…`。验证办法：编完后**再看一次 §5.3 的矩阵**（行为是唯一可信的证据，
   文件时间戳不是）。
4. `LargeIndexKernel`（tiling key 30，int64 大缓存）**本来就天然跳过 marker**：
   `scatter_nd_update_large_index.h:100-103` 的
   `if (linearIndex >= startInt64_ && linearIndex < endInt64_)` 用有符号比较，负行号直接落在范围外。
   这既解释了为什么这 bug 在 int64 路径上从不露头，也说明"`< 0` 就跳过"是这个算子体系里被接受的写法。

顺带一个必须记录的口径冲突（改 kernel 之前先解决它，否则改完还会有人写回来说"算子会跳过"）：

| 位置 | 说法 |
|---|---|
| `vllm_ascend/attention/dsa_v41.py:216-217`（改前） | "treats the builder's `[-1, -1]` coordinates as skipped rows, matching V4" ← 与实测矛盾 |
| `vllm_ascend/device/device_op.py:45-48` | "…does **not** filter negative slots, and **neither fast operator is assumed to skip them**" ← 与实测一致 |

`device_op.py:42-70` 的 `scatter_cache` 用的是**另一套约定**：它只喂 `slots[:tokens]`（有效行必须是前缀），
并且要求 `cache.view(-1, width)` 能成视图（`:81-85` 显式拒绝非连续首轴，否则退回
`torch_npu.npu_scatter_nd_update_`）。V4.1 的 marker 是交错的，前缀这招用不了，所以 V4.1 只能靠
"算子会跳过"这个不存在的语义——这就是这个 bug 的结构性成因。

## 7. 为什么"启动没报错 / 服务看着正常"不能当证据

1. **profile run 根本不走这条路**：`dsa_v41.py:540-542` 在 `attn_metadata is None` 时整层 `zero_()` 返回。
   探针实测也一致：第一次打印在 `Application startup complete` 之后（日志行 501→503、549→551）。
2. **历史上没有任何一次真实生成成功过**：`verl/logs/` 全部日志里 `finish_reason` 命中数 = 0；
   09-30 各轮死在 E5/E8。所以"以前跑过没事"这个前提是空的。
3. **同一 bug 有两种表现，分岔点是"落点是否被映射"，不是"是不是 RL 配置"**（10-08 11:25 实测矩阵）。
   只改显存相关参数、代码完全相同、并且把方案 A 的过滤关掉（`DSA_SCATTER_DROP_MARKERS=0`）：

   | 组 | sleep-mode | expandable_segments | marker 调用 | 设备异常 | int8 plane storage 起点 | 起点 16 MB 对齐 |
   |---|---|---|---|---|---|---|
   | GRPO 形状（mem .6 / batch 2048 / mlen 2048 / seqs 256 / prefix off） | **1** | 0 | 8（只活到层 2） | **158** | `0x12d940000000` | **是** |
   | 用户 serve 形状（mem .9 / batch 8192 / mlen 8192 / seqs 16 / prefix on） | 0 | **1** | **288** | **0** | `0x12d713550000` | 否 |
   | serve 形状但不带 expandable | 0 | 0 | **288** | **0** | `0x12d713550000` | 否 |

   三组打印出来的落点算术**完全一样**（`marker_dest[view]=-254080B → plane storage 起点之下 188544 B`），
   所以"触发了没有"没有区别；区别只在**那 188 KB 可不可读**：

   * 开 `--enable-sleep-mode` 时 KV 池张量由 CuMemAllocator **单独预留 VA**，起点正好落在 16 MB 边界上，
     下面就是空洞 → MTE 越界 → 设备异常、8 worker 同死；
   * 不开时 KV 池是缓存分配器里几十 GB 大块**内部的切片**（起点只 4 KB 对齐），下面 188 KB 仍在映射内
     → 不报错，把邻居写脏。

   `expandable_segments` 被 R1/R3 这对照**排除**了（两者只差它，结果同为"不炸"）；
   `prefix_caching / mem_util / max_num_batched / max_num_seqs` 也都不是这条轴。
   ⇒ **"我的服务起来了、能出词、没报错"在这个 bug 下等价于"每次压缩写入都在写脏"**（矩阵里 288 次），
   而不是引擎健康的证据。用户观察到的复读输出与该静默形态相容，但**还没证明是它的后果**：
   要证就在同一份 serve 配置下用 `DSA_SCATTER_DROP_MARKERS=0/1` 各跑一次、比同一个 prompt。
4. **一个取证口径的坑（我自己踩了）**：探针里 `in_mapped_block` 用
   `torch.npu.memory.memory_snapshot()` 判断"落点是否在本进程映射内"，**这个字段不可信**——
   它看不见 CuMem/池化分配的块，两个"不炸"的运行也打印 `NO`。
   判据只有"有没有设备异常"这一条（`in_own_storage` 是纯算术，可以用；`in_mapped_block` 不要用）。
   时间线版在 [`../worklog_dsv41_top64_grpo.md`](../worklog_dsv41_top64_grpo.md) §5.7.9。

## 8. 这套定位方法里可复用的几条

1. **能把复现范围砍多小就砍多小**：从"40 分钟的 GRPO"砍到"10 分钟的引擎"再砍到"20 秒的单卡算子微复现"，
   每一步都让 A/B 变便宜，本案最后三个结论全部出自单卡。
2. **异常名只告诉你 kernel，不告诉你参数**：需要"调用前把参数拿到主机上检查"的探针，而不是更多 traceback。
3. **异步设备异常会指错地方**（`ASCEND_LAUNCH_BLOCKING=1` 也救不了自建 stream/event 的流水）。
4. **"没报错"和"没写坏"是两件事**：哨兵/guard 这类**边界外取证**要主动设计进去；本案最关键的
   一条证据（`plane_base-8320`）只有哨兵能看见。
5. **两个可疑变量必须正交**，不能靠"这次有 marker 那次没有"的相关性下结论。
6. **算子类 bug 先查"约定在哪一层写的"**：本案的判定点是同一 fork 里两处互相矛盾的 docstring
   （`dsa_v41.py` vs `device_op.py`），实测站在后者那边。
7. **改了源码不等于改了二进制**：以行为为准（§6 现在正卡在这一步）。

## 相关

- 时间线实验记录：[`../worklog_dsv41_top64_grpo.md`](../worklog_dsv41_top64_grpo.md) §5.7.8（E9）、§5.7.1/§5.7.2
- 工具：`/workspace-verl/scripts/{run_vllm_dsv41_rl_dummy.sh,dsa_scatter_probe.py,test_scatter_nd_update_sk_skip.py,ask_dsv41_serve.sh}`
- 权重同步侧的姊妹事故（E8）：[`../worklog_dsv41_top64_grpo.md`](../worklog_dsv41_top64_grpo.md) §5.7.4/§5.7.7
- 训推一致性取证（本案的静默改写分支会影响它的结论）：[`../worklog_dsv41_real4_prod_consistency.md`](../worklog_dsv41_real4_prod_consistency.md)
