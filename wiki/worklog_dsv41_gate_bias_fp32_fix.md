# DeepSeek-V4.1 训推一致性：**修掉 fp32 路由器纠偏 bias 的 dtype 不对称**（工作记录，2026-09-22）

> 本文是 [`worklog_dsv41_real_weights_4layer.md`](worklog_dsv41_real_weights_4layer.md) §12.2「方向 1」的实施与验证记录：
> 把诊断（§6 的 H2 专项）变成**训练栈里的修复 + 可复现回归**。计划与事实表在
> `/mnt/share/m00899630/plans/dsv41-real-weights-4layer/`，本篇只写"怎么改的、踩了什么、验成什么样"。
>
> 状态：**已完成**（代码 + CPU 侧逐位校验 + 真实切片 smoke/probe 回归 + 离线 identification + 8 卡 GRPO 冒烟，全部通过）

---

## 0. 一句话结论

`layers.<i>.ffn.gate.bias`（引擎侧 `e_score_correction_bias`）在训练侧从 **bf16 Parameter** 改成
**fp32 buffer**：它既不被 `prepare_deepseek_v41_model_for_fsdp` 统一转 bf16（那里只遍历 parameter），
也不进任何 FSDP 参数组（FSDP2 要求每组一种 original dtype），于是训练侧持有的就是 checkpoint 的
**逐位 fp32 原值**。名字、状态字典键、引擎侧映射（`.ffn.gate.bias` → `.ffn.gate.e_score_correction_bias`）
都不变，**引擎不动**。

回归判据（计划 §12.2.4）：`in.all` 地板 ≤0.005、`model.norm` ≤0.006、argmax ≥0.96，且**不再需要**
`VERL_DSV41_KEEP_FP32_PARAMS` 才成立。

**回归结果（真实 4 层切片，384 专家，len=200，无任何 env 开关）**：baseline σ **0.2573 → 0.0967**、
输入全钉死后 σ **0.1882 → 0.0202**、MoE 地板 **0.0316–0.0449 → 0.0037/0.0042/0.0038/0.0034**、
`model.norm` **0.0735 → 0.0062**、argmax **0.819 → 0.970**；离线 identification 从"bf16 100%"
翻成"**fp32 100%**"。数字与修复前 `=gate` 效果预演**逐项相同**。详见 §5。

---

## 1. 为什么要改、为什么是这个参数

上游结论（真实 4 层切片，384 专家，len=200）：

| 配置 | baseline σ | 输入全钉死后 σ | 钉死后 MoE 地板 | `model.norm` | argmax |
|---|---|---|---|---|---|
| 现状（训练侧 bf16） | 0.2573 | 0.1882（**1.4×**） | 0.0316–0.0449 | 0.0735 | 0.819 |
| 只把 `gate.bias` 还原 fp32 | **0.0967** | **0.0202** | **0.0034–0.0042** | **0.0062** | **0.970** |
| 只把 `hc_*`+`attn_sink` 还原 fp32（阴性对照） | 0.2467 | 0.1882（无变化） | 同上 | 0.0735 | 0.829 |

即：**一个 fp32 参数解释了 ~9× 的"输入钉死地板"与 2.7× 的 baseline σ**；它只进
`(scores + bias).topk(...)` 的选择、不进路由权重，所以它的舍入是**纯离散**扰动（两侧喂逐位相同的输入时，
20–62% 的 token 会换掉 top-6 专家集合）。**可改，且改法是确定的：训练侧持有 fp32。**

---

## 2. 方案选择（为什么是 buffer）

要求：① 训练侧持有 checkpoint 的 fp32 值；② 不进 FSDP 组（否则踩 `FSDP expects uniform original
parameter dtype`）；③ 名字不变（引擎侧 WeightsMapper 与权重同步都按 `layers.<i>.ffn.gate.bias` 匹配）；
④ 训练/推理/保存/同步四条路径都不静默降级。

| 方案 | 做法 | 否决/采纳的理由 |
|---|---|---|
| **A. fp32 buffer（采纳）** | `Gate.bias`/`bias_vl` 改成 `register_buffer(..., persistent=True)` | 参数遍历（转 dtype）、FSDP 参数组、都天然跳过 buffer；名字不变；`persistent=True` 让它仍是"checkpoint 状态"→ 进 `state_dict()`，同步与保存都不丢。代价：加载侧要认识 buffer（本次改的就是这一处） |
| B. 拆成独立子模块 | `gate.correction_bias.*` 单独成组 + fp32 mp_policy | 名字变了 → 加载映射、引擎侧 WeightsMapper、同步全都要跟着改（"引擎不动"这条底线破了） |
| C. `ignored_params` | 让 FSDP 忽略该参数 | 只能按**模块**粒度忽略（`FSDPPlanConfig.ignored_modules` → `get_ignored_modules` 收集其 `parameters(recurse=True)`），忽略 `gate` 会把 `gate.weight` 也一起变成不受管、不切分的普通参数（它是要训练的），改动面更大且风险更高 |
| D. Gate 整组 fp32 | `gate.weight` 也存 fp32 + 该组 mp_policy fp32 | 数值上等价（forward 本来就 `weight.float()`），但会给同步/保存引入**新的** dtype 差异（引擎侧 `gate.weight` 是 bf16），与"两栈持有同一个值"的目标背道而驰 |

**采纳 A**，并补一条判据（写进 `Gate` 的 docstring 与 `prepare_deepseek_v41_model_for_fsdp`）：
*"checkpoint 里 fp32 且梯度不流经的参数"* → buffer；`hc_*`/`attn_sink` 虽然也是 ckpt fp32，但它们
**在 forward 的乘法里**、梯度会流（是真正要训练的参数），实测对训推差距零贡献，所以不动。

---

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py` | `Gate.bias`/`bias_vl`：`nn.Parameter(torch.empty(...,fp32))` → `register_buffer(..., torch.zeros(..., fp32), persistent=True)`（`bias_vl` 在 vision 关闭时注册 `None`，行为与原来 `= None` 一致：`named_buffers`/`state_dict` 都会跳过 None）；类 docstring 写明"为什么是 buffer" |
| `FSDPTurbo/fsdp_turbo/models/deepseek_v41/adapter.py` | `prepare_deepseek_v41_model_for_fsdp` docstring：补上"无梯度的 fp32 checkpoint 值用 buffer 承载"这条约定（**函数体不用改**：它只遍历 `parameters()`，buffer 天然不被 cast） |
| `verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py` | ① 新增 `checkpoint_buffer_meta(module)`：用 PyTorch 语义（persistent = checkpoint 状态）挑出要读的 buffer；② `read_dsv41_checkpoint_state_dict(..., buffer_meta=None)`：buffer 按**自己的 dtype** 载入（fp32→fp32 逐位，不再舍入），`missing` 也覆盖 buffer；③ 新增 `place_checkpoint_buffers_for_load(...)`（见 §4 E1）与 `move_buffers_to_device(...)`（见 §4 E4）；④ `_build_fsdp_module` / `_materialize_dsv41_parameters` 串起来；⑤ 引擎侧加载后加**逐位校验**（`checkpoint buffers loaded: N (bit-exact ...)`，不通过就报错——因为 DCP 静默丢弃，见 E2） |
| `verl/scripts/dsv41_consistency/{dump_trainer_stages,probe_module_inputs,graft_trainer_stages}.py` | 同一个 build/load 路径，同步接上 `checkpoint_buffer_meta` / `place_checkpoint_buffers_for_load` / `move_buffers_to_device`；`dump_trainer_stages.py` 额外把加载后的 buffer 值写进 dump（`checkpoint_buffers`）、打印 sha1 指纹、以及 `load check ... bit-exact` 一行 |
| `verl/scripts/dsv41_consistency/verify_gate_bias_fp32.py` | **新增**：CPU 侧回归检查（provenance + identification，§5.1/§5.3） |
| `verl/scripts/dsv41_consistency/sh/run_fix_regression.sh` | **新增**：一条龙跑 smoke + probe + 汇总 + verify |
| `verl/scripts/dsv41_consistency/sh/run_fix_grpo_smoke.sh` | **新增**：8 卡 GRPO 冒烟（1 步，含权重同步；带本机 gloo 网卡修正，见 E9） |

---

## 4. 遇到的问题与解决（逐个记）

### E1（阻断）：`ValueError: Multiple devices found` —— DCP 的广播加载只认一个 device

**现象**：改完直接跑训练侧冒烟，rank 1–7 全部在 `set_model_state_dict` 处挂掉：

```
File ".../torch/distributed/checkpoint/state_dict.py", line 592, in _load_model_state_dict
    raise ValueError("Multiple devices found")
```

**定位**（在 harness 里加了一行 DCP 视角的设备清单，就是它给出的答案）：

```
DCP device scan: cpu: 90 e.g. ['model.embed.weight', ...], npu:0: 4 e.g. ['model.layers.0.ffn.gate.bias', ...], meta: 8 e.g. ['...experts.gate_up_proj', ...]
```

`_load_model_state_dict` 会遍历模型状态里所有 `dim>0` 的张量收集 device 集合，`meta` 被排除
（那是给 deferred experts 用的），**剩下必须只有一个**。现状是"参数在 cpu（FSDP2 的
`cpu_offload=True` 把参数放在 host，100 GB 专家就是这么载的）+ 我新加的 buffer 在 npu"→ 两个 device。

为什么以前的 buffer 没撞上：模型里其它 buffer（kv cache、rope 表）**全是 `persistent=False`**，
`_iterate_valid_model_state` 按 `_non_persistent_buffers_set` 跳过它们；我这个是 persistent，
所以**第一次**被 DCP 看见。

**解决**：新增共享助手 `place_checkpoint_buffers_for_load(module, buffer_meta, log)`——加载前把
checkpoint buffer 对到**参数的 device**（offload 开着就是 cpu，关掉就是 npu，跟随策略，不写死），
加载后再由两侧本来就有的"buffer 跟随计算设备"循环搬回 NPU（这条循环现在是必需的：见 E3）。
调用点 4 处（引擎 1 + harness 3），每处一行。

### E2（重要的认知修正）：DCP 的 `strict=True` **不会**报缺失——它静默丢弃

读源码发现 `_broadcast_state_dict` 的语义是**反的**：

```python
    if strict:
        if missing_keys := (local_state_dict_keys - global_keys):
            for key in missing_keys:
                local_state_dict.pop(key)      # 丢掉，不报错
```

也就是说"忘了给 buffer 喂值"这件事**不会**被 DCP 拦下来（我原本以为 strict 会报
`Missing key`）。既然错误是静默的，验证就必须是**逐位比对**而不是"没报错即通过"——
所以有了 §5.1 的 provenance 检查（sha1 比 ckpt 原始字节）。同理，`read_dsv41_checkpoint_state_dict`
里的 `missing` 现在把 buffer 也算进去（checkpoint 真缺这个张量时在**读取阶段**就报出来）。

### E3（认知修正）：`cpu_offload=True` + `assign=True` 会把载入的张量**替换**成 host 张量

`_load_model_state_dict` 里：因为专家参数在 meta 上，`assign=True` 会被打开，最后的
`model.load_state_dict(..., assign=assign)` 是**整体替换**而不是 `copy_`；配合 `cpu_offload=True`，
被载入的 buffer 落地就是 **cpu 张量**。所以加载后"把 buffer 搬到计算设备"的循环（引擎与 harness
里原本就有，注释写的是"buffers ... must follow the parameters' compute device"）现在是
**正确性必需**，不是可选的清理。

### E4（阻断，第二次）：`RuntimeError: Attempted to call 'variable.set_data(tensor)', ... incompatible tensor type` —— **我自己的诊断代码造成的**

**现象**：E1 修好后加载成功，随即死在加载之后的"把 buffer 搬到计算设备"循环：

```
File ".../dump_trainer_stages.py", line 506, in main
    buffer.data = buffer.data.to(device)
RuntimeError: Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.
```

**根因（排除法 + 复现）**：我先怀疑"跨设备赋值"，但 CPU 侧复现给出了明确答案：

```
cpu <- meta     : FAILED -> RuntimeError: ... incompatible tensor type      <-- 就是这个
cpu <- float64  : OK
cpu <- fp32     : OK
```

即这条报错由 **meta 目标**触发，而不是 cpu↔npu。而 `device` 为什么会是 meta？因为我在冒烟脚本里
加的诊断打印写了 `for device, keys in inventory.items():`——**把主流程的 `device` 变量名覆盖成了字典键**
（`inventory` 的键是 `"cpu"`/`"npu:0"`/`"meta"`，最后一轮就是 `"meta"`）。于是
`move_buffers_to_device(model, "meta", log)` 真的把 16 个运行时 buffer 搬到了 meta 上，forward 随即失败。
**教训**：诊断代码不要复用主流程变量名（现在改成 `where`）。

**解决**：新增共享助手 `move_buffers_to_device(module, device, log)`（引擎 + 3 个 harness 脚本共用），
用 **`setattr(owner_module, local_name, tensor.to(device))`** 替换注册的 buffer：`setattr` 会更新
`_buffers[name]` 并保留 persistence（CPU 单测确认：`state_dict()` 仍含该键，`persistent=False` 的兄弟项不受影响）。
它不是"为跨设备而写"，而是"加载会**替换**这些张量对象，所以按注册表替换回去是语义正确的做法"，
并且不依赖 `set_data` 的类型规则（那条规则至少会拒绝 meta 目标）。运行验证：日志 `moved 4 buffers to npu:0`，
随后 len 64/200 两个 forward 正常跑完。

### E5（自伤之一，已修）：指纹日志读的是**加载前**的 tensor 引用

`assign=True` 会**替换** buffer 对象，所以加载前抓的引用在加载后仍是旧的（初值全 0）——
我第一次跑出来 `sha1=53ea2cb7... first=[0.0, 0.0, 0.0, 0.0]` 四个层完全一样，一度以为"加载没生效"。
现在改成**加载后按名字重新读**（`dict(model.named_buffers())`），并加了一行
`load check <name>: checkpoint->model bit-exact`——因为 DCP 不会替我们报这个错（E2），这个检查必须自己写。

### E6（自伤之二，已修）：`AttributeError: module 'dump_trainer_stages' has no attribute 'move_buffers_to_device'`

harness 里 `probe_module_inputs.py` / `graft_trainer_stages.py` 通过兄弟模块 `import dump_trainer_stages as dt`
复用构建逻辑，所以我一开始写的是 `dt.move_buffers_to_device(...)`。但 `dump_trainer_stages` 里那些
`verl.workers.engine...` 的 import 是**写在 `main()` 函数体内**的（harness 的既有约定：让 `import torch_npu`
之类在进程内先就绪），于是模块顶层并没有这个名字 → probe 一进循环就 AttributeError（8 个 rank 同时）。
改成直接从 impl 模块 import。

### E8（自伤之三，**被 GRPO 冒烟抓到**）：`NameError: name 'device' is not defined`

`_materialize_dsv41_parameters` 里原本有一行 `device = torch.accelerator.current_accelerator()`，
我在把"搬 buffer"改成共享助手时把这行**内联**进了调用参数，于是后面那句
`refresh_dsv41_expert_metadata(module, device)` 就悬空了：

```
File ".../fsdp_turbo_dsv41_impl.py", line 512, in _materialize_dsv41_parameters
    refresh_dsv41_expert_metadata(module, device)
NameError: name 'device' is not defined
```

**为什么 harness 没抓到**：三个 harness 脚本走的是自己那份 build/load 流程（与引擎同源但是**副本**），
`_materialize_dsv41_parameters` 只在**引擎**（生产路径）里执行 → **引擎侧改动必须靠 GRPO 冒烟兜底**，
这条经验值得记住（也正是计划 §12.2.5 强调"8 卡 GRPO 能起"的原因）。修法：把 `device = ...` 恢复成独立一行。

### E9（环境，非本次改动引入）：GRPO 起来时 `Unable to find interface for: [141.61.29.117]`

本机 `/etc/hosts` 里 `node-29-117` 仍指向旧 IP（`141.61.29.117`，实际 `80.5.25.117`）→ 凡是用**主机名**
建 gloo 的 process group 都会失败（既有 E1′，`worklog_dsv41_real_weights_4layer.md`）。
torchrun 直连 `127.0.0.1` 的 harness 不受影响，**GRPO 这类 Ray 路径必须显式指定**：
`MASTER_ADDR=127.0.0.1` + `GLOO_SOCKET_IFNAME=enp48s3u1u1`（`enp48s3u1u1` 实测存在且承载 `80.5.25.117`，
`/proc/net/dev` 可查）。已写进 `sh/run_fix_grpo_smoke.sh`。

### E7（工具维护）：`restore_fp32_parameters`（`VERL_DSV41_KEEP_FP32_PARAMS`）语义变化

修完之后 `gate.bias` 不再是 parameter，`=gate`/`=1` 这两个模式**再也找不到目标**（只会剩
`hc_*`/`attn_sink` 这类连续项，而那正是阴性对照 `=continuous`）。函数保留（`=continuous` 仍有用），
docstring 里写明"gate 一档已被修复取代，跑出来应与 baseline 相同"，避免以后有人拿它当 A/B。

---

## 5. 验证

### 5.1 CPU 侧（不需要 NPU，秒级）

1. **helper 形状**：stub 模块上 `checkpoint_buffer_meta` 只返回 `layers.0.ffn.gate.bias`
   （`bias_vl=None`、`persistent=False` 的 scratch 都被正确跳过）。
2. **加载逐位**：对真实 4 层切片调用加载器，4 个 `layers.<i>.ffn.gate.bias` 全部
   **`torch.equal(ckpt, loaded) == True`**（fp32）；同一份代码若按老路径 cast 成 bf16，则
   |Δ|max ≈ 0.031、mean ≈ 0.010–0.023，**384/384 行全被改动**（这就是被抹掉的信息量）。
3. `verify_gate_bias_fp32.py`：provenance（sha1）+ identification（用训练侧自己记录的 top-6
   反查 fp32/bf16 哪个值在手里），结果见 §5.3。

### 5.2 NPU 侧（真实 4 层切片，8 卡）

命令：`bash scripts/dsv41_consistency/sh/run_fix_regression.sh`（smoke + probe 一条龙，无任何 env 开关）。

**训练侧冒烟（`dump_trainer_stages.py`，len 64/200）——加载证据（日志原文）：**

```
[dump-trainer] model built in 31.9s (98 parameters, 4 checkpoint buffers: ['model.layers.0.ffn.gate.bias', ...])
[dump-trainer] aligned 4 checkpoint buffers to cpu for the load (e.g. ['model.layers.0.ffn.gate.bias', ...])
[dump-trainer] DCP device scan: cpu: 94 e.g. ['model.embed.weight', ...], meta: 8 e.g. ['...experts.gate_up_proj', ...]
[dump-trainer] load check model.layers.{0..3}.ffn.gate.bias: checkpoint->model bit-exact
[dump-trainer] parameters materialized in 38.1s
[dump-trainer] moved 4 buffers to npu:0 (e.g. ['model.layers.0.ffn.gate.bias', ...])
[dump-trainer] checkpoint buffer model.layers.0.ffn.gate.bias: dtype=torch.float32 shape=(384,) sha1=366ca30d3dea7338 first=[9.844348, 9.8523512, 9.8603544, 9.8223391]
[dump-trainer] len=64:  next_logprobs[1:4]=[-18.0034, -9.4226, -18.9964] logits absmax=21.13 stages=57
[dump-trainer] len=200: next_logprobs[1:4]=[-18.0032, -9.3936, -19.0028] logits absmax=21.16 stages=37
```
（修复前的同一行是 `parameters`102 个、logprob `[-17.9408, -9.5695, -19.1052]`、`absmax=21.11`——量级不变、数值按预期变化。）

**参数个数变化可核对**：`102 → 98` = 4 个 `gate.bias` 离开 parameter 集合（每层 1 个），
buffer 侧正好多出这 4 个 ✓（切片是 4 层；生产 40 层就是 40 个）。

**provenance（`verify_gate_bias_fp32.py`，CPU，逐位）**：

| 张量 | ckpt dtype | 训练侧 dtype | ckpt sha1(fp32) | 训练侧 sha1 | sha1(若 bf16) | 一致 |
|---|---|---|---|---|---|---|
| `layers.0.ffn.gate.bias` | float32 | float32 | `366ca30d3dea7338` | `366ca30d3dea7338` | `aefe8b25a3b09333` | **YES** |
| `layers.1.ffn.gate.bias` | float32 | float32 | `eaf2c3f16d452189` | `eaf2c3f16d452189` | `ebd6975b6e22e9ae` | **YES** |
| `layers.2.ffn.gate.bias` | float32 | float32 | `d1d613ca6e462796` | `d1d613ca6e462796` | `83eabfbadfd42869` | **YES** |
| `layers.3.ffn.gate.bias` | float32 | float32 | `641757cb55d8b649` | `641757cb55d8b649` | `100104f7403890bf` | **YES** |

→ 训练侧持有的就是 checkpoint 的 fp32 原值（逐位），而按老路径走 bf16 会是右列那些 sha1。

**一致性（probe，`summarize_probe.py` 全表见 `/tmp/summarize_probe_real4_fix.txt`）：**

| 配置 | baseline σ | in.all σ（改善） | in.all.ffn 地板 | model.norm（in.all） | argmax（in.all） |
|---|---|---|---|---|---|
| **len=200** 修复前（bf16） | 0.2573 | 0.1882（1.4×） | 0.0316–0.0449 | 0.0735 | 0.819 |
| len=200 修复前 + `=gate`（效果预演） | 0.0967 | 0.0202 | 0.0037/0.0042/0.0038/0.0034 | 0.0062 | 0.970 |
| **len=200 修复后（无 env）** | **0.0967** | **0.0202（4.8×）** | **0.0037/0.0042/0.0038/0.0034** | **0.0062** | **0.970** |
| **len=64** 修复前（bf16） | 0.2417 | 0.1964（1.2×） | 0.0090–0.0377 | 0.0754 | 0.841 |
| len=64 修复前 + `=gate` | 0.0492 | 0.0178 | — | — | — |
| **len=64 修复后（无 env）** | **0.0492** | **0.0178（2.8×）** | **0.0034–0.0045** | **0.0062** | **0.952** |

**修复后的数字与"只把 `gate.bias` 还原 fp32"的效果预演逐项相同**（σ、地板、norm、argmax 全部对上），
也就是说：**曾经的 A/B 开关现在是默认行为**。同时"把输入钉死"的收益从 1.4× 回到 4.8×，
钉死后的地板回到核噪声量级（0.0034–0.0042）——机制形态与随机权重那套（§11.3）一致了。

残余路由翻转（自然输入 vs 钉死输入，len200 7/4/9/10 of 200）仍在，但那是**输入差**经 MoE 放大的老机制：
翻转对误差能量的占比 0.64/0.80/0.90/0.84，与既有结论一致。

### 5.3 identification：训练侧手里到底是哪个值（离线，CPU）

方法：拿训练侧**自己记录的** gate 输入（probe dump 的 `inputs["model.layers.<i>.ffn"]`，
baseline 变体里它就是 gate 真正吃到的张量）重算 top-6 两遍，只换 bias 取值，与训练侧**自己记录的**
top-6（`stages["model.layers.<i>.ffn.gate[1]"]`）比命中率。**修复前/后**（同一份工具、同一批 dump 结构）：

| len | layer | tokens | match fp32 | match bf16 | flips | flip rate | slot overlap |
|---|---|---|---|---|---|---|---|
| 64 | 0 | 64 | 39.1% → **100.0%** | 100.0% → 39.1% | 23 | 0.3594 | 0.9271 |
| 64 | 1 | 64 | 65.6% → **100.0%** | 100.0% → 64.1% | 11 → 13 | 0.1719 → 0.2031 | 0.9661 → 0.9609 |
| 64 | 2 | 64 | 20.3% → **100.0%** | 100.0% → 21.9% | 38 → 37 | 0.5938 → 0.5781 | 0.8620 |
| 64 | 3 | 64 | 21.9% → **100.0%** | 100.0% → 29.7% | 38 → 36 | 0.5938 → 0.5625 | 0.8880 |
| 200 | 0 | 200 | 38.0% → **100.0%** | 100.0% → 38.0% | 81 | 0.4050 | 0.9125 |
| 200 | 1 | 200 | 62.0% → **100.0%** | 100.0% → 62.0% | 35 → 36 | 0.1750 → 0.1800 | 0.9692 → 0.9683 |
| 200 | 2 | 200 | 21.5% → **100.0%** | 100.0% → 22.0% | 119 → 122 | 0.5950 → 0.6100 | 0.8533 → 0.8542 |
| 200 | 3 | 200 | 24.0% → **100.0%** | 100.0% → 27.5% | 114 → 104 | 0.5700 → 0.5200 | 0.8925 → 0.9017 |

（"→"前 = 修复前 dump，后 = 修复后 dump。）**100% 那一列从 bf16 换成了 fp32**，这就是"训练侧手里的值变了"
的直接证据；翻转率/槽位重叠只是 bias 对与输入的属性，两次基本不变（小差异来自修复后基线输入本身变了）。

### 5.4 判据对照（计划 §12.2.4）

| 判据 | 目标 | 修复后实测 | 结论 |
|---|---|---|---|
| `in.all` 钉死后 MoE 地板 | ≤0.005 | 0.0034–0.0042 | ✅ |
| `model.norm`（in.all） | ≤0.006 | 0.0062 | ✅（= 效果预演的同一个值；计划里那条 ≤0.006 就是它的写法） |
| argmax（in.all） | ≥0.96 | 0.970 | ✅ |
| **不再需要** `VERL_DSV41_KEEP_FP32_PARAMS` | 是 | 全程无 env | ✅ |

补一条更强的判据：**修复后数字与效果预演逐项相同**（§5.2 表）。

### 5.5 生产路径冒烟（8 卡 GRPO，`run_fix_grpo_smoke.sh`）

目的：harness 只覆盖 build/load/forward，**引擎路径**（FSDP 包装 → 优化器 → 权重同步到 vLLM-Ascend →
rollout → 训练步）要另外验证——新版 `state_dict()` 里多了 buffer，它的名字必须仍然命中引擎侧的
`.ffn.gate.bias` → `.ffn.gate.e_score_correction_bias` 映射。用的是 4 层 32 专家的 scaled fixture，
`trainer.total_training_steps=1`。

**结果：跑通（第 1 步完整走完，无异常）**。actor 与 ref 两侧的加载日志：

```
[fsdp_turbo_dsv41] materializing 98 parameters + 4 checkpoint buffers from .../DeepSeek-V4.1-Flash-4layer-scaled32
[fsdp_turbo_dsv41] aligned 4 checkpoint buffers to cpu for the load (e.g. ['model.layers.0.ffn.gate.bias', ...])
[fsdp_turbo_dsv41] checkpoint buffers loaded: 4 (bit-exact against the checkpoint)      ← 新增的引擎侧校验
[fsdp_turbo_dsv41] parameters materialized in 28.0s
```

第 1 步指标（节选）：`training/rollout_probs_diff_mean 0.0108`、
`training/rollout_actor_probs_pearson_corr 0.9753`、`rollout_corr/kl 0.0744` —— 与随机 32 专家的既有量级
（0.010 / 0.979 / kl≈0.11 一族）一致，**没有出现 O(1) / ~0.5 的错位断层**（若同步落错张量就会出现）。
`timing_s/update_weights 89.7s`（整权重同步路径，**这次它把 fp32 bias 一起送过去了**）、
`timing_s/gen 88.8s`、`update_actor 5.0s`、`step 205.9s`。

两点说明：
- `actor/grad_norm 0.0` / `actor/pg_loss 0.0` 是**随机 fixture 的退化**（n=2 两个样本拿到相同 reward −1、
  advantage 全 0），不是本修复引入的问题；
- scaled fixture 的 `gate.bias` 两侧都是 0，所以这次冒烟验的是**通路**（加载/同步/训练步），
  数值改进的证据在 §5.2/§5.3（真实切片）。

过程中另见 E8（本次冒烟抓到的引擎侧 `NameError`）与 E9（主机名/gloo，环境问题）。

---

## 6. Caveat 与遗留

### 6.1 Caveat

1. **bias 不再是 Parameter**（这正是它能保持 fp32 的原因）。梯度本来就不流经它（`weights` 取自无偏 `scores`），
   所以对当前训练语义等价；但**如果将来要实现 noaux_tc 的"无梯度纠偏更新"规则（`bias += γ·sign(load_err)`），
   要看它写在哪里**——写 buffer 也行，只是不能被"优化器遍历 parameter"顺带覆盖。本仓库目前没有这类逻辑。
2. **同一个类里还有 28 个 ckpt-fp32 / 训练-bf16 的参数**（`hc_*` ×6、`attn_sink`，每层 7 个×4 层）**没有动**：
   实测对训推差距零贡献（§6.3 阴性对照），而且它们在 forward 的乘法里、**梯度会流**（是真正要训练的参数），
   改成 buffer 会破坏训练。要彻底对齐它们需要 FSDP "每个 param group 各自 dtype"的能力，属于另一个工程。
3. **4 层切片 ≠ 40 层生产模型**（candidate_source=2、engram 关、层 3 开候选预筛）。`gate.bias` 机制是**逐层局部**的，
   层数更多只会让翻转更多，故 40 层下的绝对值预计**不小于**本处测得值；生产绝对 σ/kl 仍要整模型复跑（计划方向 3）。
4. **delta 同步路径**（`get_per_tensor_param_shard`，`checkpoint_engine/delta_checkpoint_engine.py` 专用）
   会把**所有**浮点张量 cast 成 bf16 再传，其中包括这个 bias。该路径不在本 recipe 上（GRPO 走
   `get_per_tensor_param`，已核实），所以没改；**若将来启用 delta 同步，这里要按参数 dtype 放行 fp32**，否则
   纠偏 bias 会在同步通道上被重新舍入。
5. **权重同步的量**：修复后 fp32 bias 会随 `state_dict()` 一起同步到引擎（名字不变 → 引擎 WeightsMapper 照旧命中）。
   数值上这与引擎自有值相同（都是 ckpt fp32），但这也意味着**训练侧保存的 checkpoint 现在会带这个 buffer**
   （DCP 保存含 persistent buffer），恢复时不会再退回 0。

### 6.2 遗留

- **`=gate` 档的"变为空操作"没有单独复跑**：E7 说它现在找不到目标（`gate.bias` 不是 parameter 了），
  跑出来应当与 baseline 逐位相同；这一条是代码读出来的结论，**没有**再花 10 分钟 NPU 时间做一次确认
  （identification 已经从"训练侧手里是哪个值"这个更直接的角度证明过了）。
- **8 卡 GRPO 冒烟**：见 §5.5（本机主机名解析问题需要 `MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=enp48s3u1u1`，
  与既有 E1′ 同源，已写进 `run_fix_grpo_smoke.sh`）。
- **整模型（40 层）复跑**：拿生产绝对 σ/kl（计划方向 3）。
- **RL 端到端 A/B**（同一份数据"修 vs 不修"，看 `rollout_corr/kl`、`pearson`、reward、grad norm）：
  现在的预期变了——修复后两栈在**首轮 rollout 之前**就已经一致（不再有"引擎 fp32 / 训练 bf16"的差），
  所以 A/B 的意义变成"验证指标回到与专家数无关的形态"，而不是"要不要上 TIS"。
- **层 1 参数指纹**（`dump_engine_stages.py` 的 `PARAM_PATTERNS` 已补 `layers.1.`，但现有 `engine_real4/params.json`
  是补之前生成的，层 1 只覆盖 4 个参数）——要补齐需重跑一次引擎 dump（~40–70 min），与本修复正交。

---

## 7. 复现命令

```bash
# 全部：smoke + probe + 汇总 + CPU 校验（~20–25 min，8 卡）
bash scripts/dsv41_consistency/sh/run_fix_regression.sh

# 只跑其中一步
STEPS=smoke bash scripts/dsv41_consistency/sh/run_fix_regression.sh
STEPS=probe bash scripts/dsv41_consistency/sh/run_fix_regression.sh

# CPU 侧：先建 stub/加载器检查（§5.1 的 1–2 步），再跑回归检查
python3 scripts/dsv41_consistency/verify_gate_bias_fp32.py --lengths 64,200
```
