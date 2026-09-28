# 用「更新后的 FSDP-Turbo / vllm-ascend」重测数值 —— 工作记录（第 2 轮，2026-09-28）

> 对象：与第 1 轮相同（`wiki/work_log/pearson_0999/worklog_p0999.md`）：真实 4 层切片
> （`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real`）、8 卡 GRPO、
> 指标 `training/rollout_actor_probs_pearson_corr`。
>
> 本轮任务（用户 2026-09-28 指示）：**「我更新了 FSDP-turbo 和 vllm-ascend 的代码，请用最新代码重新看下数值」**。
> 状态：**已完成**（新栈跑通并出数；BI-off 机制判别实验因外部 SIGKILL 未完成，见 §6.3）。

---

## 0. 最终结论（一句话）

**把整个引擎换掉（自研 fork 的 V4.1 实现 → 上游 vLLM v0.30.0 + vllm-ascend main 的 V4.1 实现），
并把训练侧更新到最新（上游 refactor + 合并修复），这个指标没有变化：**

| 运行 | steps | tokens | pooled pearson | mean_step | med\|d\| | std | >0.5% | >1.0% | hi-p rel |
|---|---|---|---|---|---|---|---|---|---|
| **新栈**（vLLM 0.30.0 + vllm-ascend main + 修复后 FSDP-Turbo） | 7 | 14336 | **0.99456** | **0.99459** | 0.0502 | 0.2376 | 5.36 | 1.11 | 0.0061 |
| 09-27 基线（fork 引擎 + 旧训练栈） | 10 | 20480 | **0.99464** | **0.99458** | 0.0494 | 0.2315 | 5.26 | 1.07 | 0.0054 |

差 **0.00008**（pooled）与 **0.00001**（单步均值），远小于 run-to-run 噪声（±0.0005）；
分布统计（`med|d|`、`std`、尾部占比、高 p 相对误差）全部同带。

**这就是第 1 轮结论的独立验证**：指标的地板不是「fork 的引擎实现差」，而是
「训练与推理是两套独立 bf16 内核栈」——**换一整套引擎内核，地板一动不动**。
所以第 1 轮的落地建议不变：**不要用 pearson 当验收指标，用 RL 侧的 TIS 消化这一量级**。

> 副产物（本轮的主要工作量其实在这里）：**更新后的代码本身是跑不起来的**，
> 有 3 个环境/合并层面的阻塞点（§3），全部定位并修复后才得到上面的数字（§4）。

---

## 1. 侦察：这次到底「更新了什么」

### 1.1 时间线（证据：`git log` / `git reflog --date=iso`）

| 时间（UTC） | 仓库 | 事件 |
|---|---|---|
| 09-10 23:40 | vllm-ascend | 用户分支 `sp_feature` 最后一次提交 `a30e1e5c3`（"add sequence parallel dSPark support"） |
| 09-16 11:27 | vllm | `git clone` → `main`(3bb7826214) → `a97dacb71`(0.28.1rc1-dev) |
| **09-17 08:51** | vllm | **`git checkout v0.27.0`**（为旧 vllm-ascend 分支做适配） |
| 09-22~09-27 | verl / FSDPTurbo | 第 1 轮一致性工作（ALIGN 门控、gate-bias fp32、padding 探针、TIS…） |
| **09-28 07:42** | **vllm-ascend** | **`checkout: moving from sp_feature to main_origin/main`** → `main_main` = `79f4a63f8`（upstream main） |
| 09-28 08:53 | FSDPTurbo | `43de40b`/`37e246a`/`a251f94`（上游 refactor）× |
| 09-28 06:48 | FSDPTurbo | `7cbc46f "support_verl"`（**含 4 处未解决的冲突标记**，见 §3.1） |
| 09-28 09:06-09:07 | vllm-ascend | 用户合入两个修复分支（`jack/fix/engram-node-local-edp` + `jack/fix/engram-bf16`，见 §4.5） |

### 1.2 三个仓库的起点状态

| 仓库 | 路径 | HEAD | 说明 |
|---|---|---|---|
| FSDPTurbo | `/workspace-verl/FSDPTurbo` | `7cbc46f`（branch `dev_dsv41_one`，工作区 clean） | **坏提交**（§3.1） |
| vllm-ascend | `/workspace-verl/vllm-ascend-v41-private` | `79f4a63f8`（branch `main_main` = upstream main） | 要求 vLLM v0.30.0（§3.2） |
| vllm | `/workspace-verl/vllm` | `4bdc8a788`（tag `v0.27.0`，detached） | **与新 vllm-ascend 不匹配**（§3.2） |
| verl | `/workspace-verl/verl` | `60290071` | 未变（第 1 轮的脚本/补丁都在） |

---

## 2. 本轮做的所有改动（总表）

| # | 对象 | 改动 | 性质 | 位置/证据 |
|---|---|---|---|---|
| 1 | vllm | `git checkout v0.30.0`（`ced6857afa`）+ 手工把 `_version.py` 改成 `0.30.0` | 环境 | §4.1 |
| 2 | 切片 ckpt | 影子目录 `/tmp/p0999/ckpt_v41`：`deepseek_v4.1*` → `deepseek_v41*`，其余 36 项 symlink | 数据（非破坏） | §4.2 |
| 3 | FSDPTurbo | 解开 **4 处冲突标记**（语义全取 `dev_dsv41` 侧） | 代码 | §4.3 |
| 4 | FSDPTurbo | 补回 `build_deepseek_v41_model` 的 `engram_meta_init` / `row_sharded` 兼容（有 engram 层即报错） | 代码 | §4.3 |
| 5 | FSDPTurbo | 补回 `prepare_deepseek_v41_model_for_fsdp` 的 **meta 安全搬运** | 代码 | §4.4 |
| 6 | vllm-ascend | 解掉 `tests/ut/models/test_engram.py` 的 `UU` 冲突（两边 import 都留） | 代码 | §4.5 |
| 7 | vllm-ascend | **重编 C++ 扩展**（清掉 09-18 的脏 CMake 缓存） | 环境 | §4.6 |
| 8 | verl | `get_auto_config_with_vllm_fallback` 白名单加 `deepseek_v41(_text)` 等 | 代码 | §4.7 |
| 9 | verl | recipe：`VLLM_VERSION` 默认 `0.30.0`；`enable_engram`/`engram_storage` 改为 `VERL_DSV41_ENGINE_LEGACY=1` 门控 | 代码 | §4.7 |
| 10 | 工具 | 新增 `/tmp/p0999/an_r2_compare.py`（逐 token 分解对比，供本轮判读） | 工具 | §5 |

---

## 3. 三个阻塞点（诊断与证据）

### 3.1 阻塞点 1：FSDP-Turbo 的 `7cbc46f` 不是可运行的提交

**现象**：

```bash
$ python3 -c "import fsdp_turbo.models.deepseek_v41"
  File ".../fsdp_turbo/models/deepseek_v41/model.py", line 1035
    >>>>>>> 62c7a92 (support_verl)
SyntaxError: invalid decimal literal
```

**证据**：`git grep -n -E "^(<<<<<<<|=======|>>>>>>>)"` 命中 4 处：

| 文件:行 | 上游侧（HEAD） | 用户侧（`62c7a92`） |
|---|---|---|
| `model.py:999-1013` | （无） | `Gate` 类 docstring：说明 correction bias 必须是 fp32 buffer |
| `model.py:1026-1035` | `nn.Parameter(torch.empty(..., float32))` | `register_buffer("bias"/"bias_vl", torch.zeros(..., float32))` |
| `model.py:1109-1129` | `MoE.forward(x, image_mask)` 2 参 | 多一个 `router_fp32=None`（`ALIGN_ROUTER_FP32` 入口） |
| `experts.py:202-210` | `use_eager=False` + 无条件 `.float()` | `use_eager=True` + `if not MOE_BF16_ACT: .float()` |

**因果（读 git 图）**：`7cbc46f` **只有一个父提交**（`a251f94`）——不是 merge commit，而是
「把带冲突标记的工作区原样提交」。标记里的 `62c7a92` 不在本仓库对象库（来自另一个 clone，
与 `origin/dev_dsv41@bd9a464 "add debug 0928"` 同源）。两条线的分叉点 = `5ae9723`（09-16）：

- `dev_dsv41_one`（HEAD）：有**上游 refactor**（`43de40b` 共享 backbone、`37e246a` 解耦公共模块），
  但**没有** gate-bias fp32 修复，**没有** `engram_meta_init/row_sharded` API；
- `dev_dsv41`（用户线，`bd9a464`）：有全部 ALIGN 门控 + gate-bias 修复 + verl 需要的 API，
  但没有上游 refactor。**09-27 campaign 跑的就是这条线**（所以冲突按它解，见 §4.3）。

**第二层问题（合并丢代码）**：verl（未改动）调用

```python
build_deepseek_v41_model(tokenizer=..., engram_meta_init=True,
    engram_storage_backend="row_sharded", use_sparse_flash_attn=False,
    experts_meta_init=True, config_path=..., max_seq_len=...)
```

而 HEAD 的签名**没有** `engram_meta_init`（只在 docstring 里被提到——docstring 走了用户侧、
代码走了上游侧），`Literal` 也不含 `"row_sharded"` → 即使语法修好，模型构建仍 `TypeError`。

### 3.2 阻塞点 2：vllm-ascend main 需要 vLLM v0.30.0，本机是 v0.27.0

**证据 A（模块存在性）**：把 `vllm_ascend/**/*.py` 里所有 `from vllm.X import …` 收齐（330 个模块），
逐个 `git cat-file -e <rev>:vllm/…` 查存在性：

| rev | 缺失模块数 | 例子 |
|---|---|---|
| `v0.27.0`（当时的 checkout） | **11** | `vllm.config.engram`、**`vllm.models.deepseek_v41.common.engram`**、`vllm.models.deepseek_v41.common.mm_preprocess`、`vllm.v1.attention.ops.pcp`、… |
| `main`（09-16 的 main） | 1 | `vllm.v1.core.sched.batch_job_aware_scheduler`（非关键路径） |

**决定性的一条**：V4.1 模型本体（含 config 类）现在**随 vLLM 发行**（`vllm/models/deepseek_v41/`），
v0.27.0 里根本没有。

**证据 B（官方配对）**：`vllm-ascend/Dockerfile:34` → `ARG VLLM_TAG=v0.30.0`。

**证据 C（版本分支）**：新代码里只剩一个版本目标：
`vllm_version_is("0.29.0")` ×8（旧 fork 是 `vllm_version_is("0.27.1")` ×37）。

### 3.3 阻塞点 3：切片 config 的 `model_type` 拼写是旧写法

| 对象 | `model_type` |
|---|---|
| 真实 40 层 ckpt `/mnt/share/DeepSeek-V4.1-Flash-bf16/config.json` | `deepseek_v41`（text=`deepseek_v41_text`） |
| 4 层切片（本轮用） | **`deepseek_v4.1`**（text=`deepseek_v4.1_text`），旧模板生成 —— `SLICE.md` 已列为已知偏差 |
| 新栈认的 | `deepseek_v41`（`vllm/transformers_utils/configs/deepseek_v41.py` + `_CONFIG_REGISTRY`） |

**连带**：verl 的 `verl/utils/model.py:87 get_auto_config_with_vllm_fallback()` 白名单写死了
`("deepseek_v4",) ("deepseek_v4.1",)`，并 `import vllm_ascend.patch.platform.patch_deepseek_v41_config`
——新 vllm-ascend 里**该模块已不存在**（`git cat-file -e` 失败），于是 fallback 链每一步都断。
**这就是 09-28 第一次试跑失败的直接原因**：

```
File "/workspace-verl/verl/verl/utils/model.py", line 96, in get_auto_config_with_vllm_fallback
File "/workspace-verl/vllm/vllm/transformers_utils/config.py", line 316/337, in parse
ValueError: The checkpoint you are trying to load has model type `deepseek_v4.1` but Transformers
            does not recognize this architecture.
```

---

## 4. 修复明细（做了什么、为什么）

### 4.1 vLLM → v0.30.0

```bash
cd /workspace-verl/vllm && git checkout v0.30.0     # ced6857afa, 2026-09-21
```

`_version.py` 是构建产物（`.gitignore:/vllm/_version.py`），checkout 不更新它（仍是 `0.28.1rc1…`），
手工改三行（`__version__` / `__version_tuple__` / `__commit_id__`）→ `0.30.0`，
因为 `vllm_ascend/utils.py::vllm_version_is()` 在没有 `VLLM_VERSION` 环境变量时会读 `vllm.__version__`。

**验证**：`import vllm` → `/workspace-verl/vllm/vllm/__init__.py`，`__version__ == 0.30.0`；
`vllm.transformers_utils.config.get_config('/tmp/p0999/ckpt_v41')` → `DeepseekV41Config`。
**回退**：`git checkout v0.27.0` + 恢复 `_version.py`。

### 4.2 影子 ckpt 目录（不改 `/mnt/share`）

```bash
SRC=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real
DST=/tmp/p0999/ckpt_v41     # 36 个 symlink + 重写的 config.json（共 37 项）
```

只改三个字符串：`deepseek_v4.1` → `deepseek_v41`（+ `_text` + `_vision`）。
训练侧 `_model_args_kwargs_from_hf_config` 只按 `text_config` 的**字段**读取、不看 `model_type`，
所以对训练无影响；引擎侧则是必须。

### 4.3 FSDP-Turbo：4 处冲突 + adapter API

**冲突解法（全取 `dev_dsv41` 侧，理由逐条）**：

| # | 采用 | 理由 |
|---|---|---|
| 1 | 用户侧 docstring | 文档化「bias 必须是 fp32 buffer」的原因 |
| 2 | `register_buffer("bias"/"bias_vl", torch.zeros(..., float32))` | `prepare_deepseek_v41_model_for_fsdp` 会把**所有**浮点 parameter 转 bf16 → parameter 版让 384 专家的 correction bias 掉到 bf16（离散 top-k 扰动）；且 verl 的 `fsdp_turbo_dsv41_impl.checkpoint_buffer_meta` 本来就按 buffer 加载 |
| 3 | 用户侧 `MoE.forward(..., router_fp32=None)` | `Block.forward`（**未冲突**，`model.py:1294`）已经在传第 3 个位置参数 `getattr(self.ffn_norm, "last_fp32", None)`；取上游 2 参签名会 `TypeError` |
| 4 | `use_eager=True` + `if not MOE_BF16_ACT: .float()` | 保留 `MOE_BF16_ACT` 门语义；`use_eager` 只影响 `native_eager_forward`，该函数**全仓库无调用点**（recipe 走 `ep_plan.dispatcher=fused` → `dispatcher.py::experts_forward`），对数值无影响 |

**adapter API 兼容**（`build_deepseek_v41_model`）：

```python
    engram_meta_init: bool = True,                                  # 新增（兼容 verl 调用）
    engram_storage_backend: Literal["host_offload", "elastic_buffer", "row_sharded"] = "host_offload",
    ...
    if engram_storage_backend == "row_sharded":
        # 上游 0769938「keep host-offload Engram only」删掉了 row-sharded 实现；
        # 只有模型确实没有 Engram 层时它才惰性（4 层切片正是如此，ckpt 无 engram 表）。
        if model_args.engram_layer_ids:
            raise ValueError(...)      # 有表就报错，绝不静默换实现
        engram_storage_backend = "host_offload"
```

`engram_meta_init` 在新设计下**恒真**（`HostOffloadEmbedding.weight` 一律 `device="meta"`），
所以只作为「接口兼容、无副作用」参数接受。

**验证**（CPU，无需 NPU）：

```
engram_layer_ids: ()
model built OK: DeepseekV41ForCausalLMAdapter
param count: 56.353 B
```

> 长期解（本轮未做）：把 `dev_dsv41` 的 row_sharded Engram
> （`distributed/embedding_parallel.py`、`training/engram_optimizer.py` 等）真正移植回上游线。
> `git diff --stat HEAD..dev_dsv41` = 48 文件 / +4550 −2334，且上游是**有意**删除的，
> 所以本轮用「接口兼容 + 有表即报错」。

### 4.4 FSDP-Turbo：第二处「合并丢代码」—— meta 安全搬运

第 2 次启动在**训练侧**就崩了：

```
File ".../adapter.py", line 377, in prepare_deepseek_v41_model_for_fsdp
    module._apply(lambda tensor: tensor.to(device=device), recurse=False)
NotImplementedError: Cannot copy out of meta tensor; no data!
```

上游版只跳过 `HostOffloadEmbedding`；而 `experts_meta_init=True` 让**路由专家的
`gate_up_proj`/`down_proj` 也在 meta 上**。已按 `dev_dsv41` 改回通用版：

```python
    def _to_device_or_keep_meta(tensor):
        return tensor if tensor.is_meta else tensor.to(device=device)
    for module in model.modules():
        module._apply(_to_device_or_keep_meta, recurse=False)
```

**离线复刻 verl 的整条构建链**（`_build_dsv41_module`）验证：

```
prepare OK; meta tensors remaining: 8
  ['model.layers.0.ffn.experts.gate_up_proj', 'model.layers.0.ffn.experts.down_proj', …]  # 4 层 × 2
param dtypes: {'torch.bfloat16': 98}
gate biases: [('model.layers.0.ffn.gate.bias', 'torch.float32', False), …]
```

（8 个 meta 张量正是「等加载器落地」的部分；98 个 parameter 全 bf16；gate bias 确认 fp32 buffer。）

另外用脚本核对了 **verl 里全部 11 个 `from fsdp_turbo… import`**：无缺失符号。

### 4.5 vllm-ascend：两个 patch 的收尾

用户合入 `jack/fix/engram-node-local-edp` + `jack/fix/engram-bf16`（reflog：09:06 合第一个 →
09:07:25 `reset` 回 `main_origin/main` → 09:07:29 合第二个，HEAD = `0e76d62ab`）。
**实际生效的是工作区**，而工作区 = HEAD + reset 前那次 merge 遗留的 staged 改动
（`git diff --cached`：`parallel_state.py` +100、`engram/parallel.py` ±39、
`patch_engram_config.py` −53、`patch/__init__.py` −37、`models/deepseek_v41/model.py` ±9 等）。

**留下的唯一未解决冲突**：`tests/ut/models/test_engram.py` 的两行 import 二选一
（`engram_gate` vs `resolve_dp_shared_memory`）。两边符号**都存在且都被该测试使用**
（`engram_gate` 用于 :161，`resolve_dp_shared_memory` 用于 :77-79）→ 正确解法是**两条都留**，
已解掉并 `git add`（索引不再是 unmerged）。

**复核**：`vllm_ascend/` 内无冲突标记（唯一 `=====` 命中是注释分隔线）；`import vllm_ascend` 正常；
两个 patch **没有**动 `AscendConfig` schema（`enable_engram`/`engram_storage` 仍不存在）
→ §4.7 把这两个键门控掉的做法仍然正确。两个 patch 只动 engram（切片 `engram_layer_ids=()`
→ 对本 recipe 数值应当惰性；出数结果与此一致）。

### 4.6 vllm-ascend C++ 扩展重编（本轮最耗时的一步）

第 3 次启动：训练侧全通过（8 卡各 ~10 GB），**引擎 worker 起核崩**：

```
RuntimeError: Worker failed with error '_OpNamespace' '_C_ascend' object has no attribute 'npu_hc_pre_v3'
  … vllm_ascend/worker/model_runner_v1.py:4347 _dummy_run → _model_forward
  … vllm_ascend/models/deepseek_v41/vl_model.py:246 forward
```

**诊断**：

| 事实 | 证据 |
|---|---|
| 源码里有 `npu_hc_pre_v3` | `csrc/torch_binding.cpp:1506/3405` 定义+注册；`torch_binding_meta.cpp:2171` 注册 meta |
| 编译产物里没有 | `vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` mtime **09-18 02:17**（fork 时代），而 `csrc/torch_binding.cpp` mtime **09-28 07:42** |
| 调用**无 fallback** | `models/deepseek_v41/model.py:851 hc_pre()` 直接调 `npu_hc_pre_v3`（对比 `rms_norm_cast()` :839 有 `enable_custom_op()` 门） |
| 同一 .so 另一 op 签名也过期 | `inplace_partial_rotary_mul(..., negate_sin=False)` 多了一个参数 |
| .so 不在 git | `.gitignore:33 *.so` → checkout 不更新，**必须重编** |

**第 1 次重编失败（脏 CMake 缓存）**：

```
CMake Error at cmake/symbol.cmake:253 (add_custom_command):
  $<TARGET_OBJECTS:vllm_quant_lightning_indexer_metadata_obj>
  Objects of target "vllm_quant_lightning_indexer_metadata_obj" referenced but no such target exists.
Call Stack: CMakeLists.txt:145 (gen_cust_aicpu_kernel_symbol)
```

`csrc/build/CMakeCache.txt:483` 的 `AICPU_CUST_OBJ_TARGETS` 仍列着 **fork 才有的 op**
（新树里已改名 `quant_lightning_indexer_v2_metadata`），而 `build_aclnn.sh` 复用 `csrc/build`。

**处理与验证**：

```bash
cd /workspace-verl/vllm-ascend-v41-private
mv csrc/build  csrc/build.stale-0918      # 不删，可回退
mv csrc/output csrc/output.stale-0918
MAX_JOBS=64 pip install -e . --no-deps --no-build-isolation
# → Successfully installed vllm_ascend-0.19.1rc2.dev2461+g0e76d62ab.d20260928   （约 13 分钟）

python3 -c "import vllm_ascend.vllm_ascend_C"   # 必须显式 import，op 才注册
#   torch.ops._C_ascend.npu_hc_pre_v3                        ✓
#   torch.ops._C_ascend.inplace_partial_rotary_mul…negate_sin ✓
#   torch.ops._C_ascend.npu_rms_norm_cast                     ✓
```

### 4.7 verl 侧小改

| 文件 | 改动 | 理由 |
|---|---|---|
| `verl/utils/model.py::get_auto_config_with_vllm_fallback` | KeyError 白名单加 `("deepseek_v4.1_text",) ("deepseek_v41",) ("deepseek_v41_text",)`；旧补丁模块 import 保留在 `try/except ImportError` 并注明「vLLM ≥ 0.30 已无此模块」 | 新栈下 transformers 仍不认识 `deepseek_v41`，但 vLLM 0.30 的 `_CONFIG_REGISTRY` 认识 |
| `examples/grpo_trainer/run_deepseek_v41_grpo_fsdp_turbo_npu.sh` | ① `VLLM_VERSION` 默认 `0.27.1` → `0.30.0`；② `enable_engram`/`engram_storage` 改为 `VERL_DSV41_ENGINE_LEGACY=1` 门控 | 新 `AscendConfig` 是 `extra="forbid"`，旧键直接报错；旧配对可用 env 开关继续跑 |

（两处 `bash -n` 通过。）

---

## 5. 跑通与数值

### 5.1 第 4 次启动（跑通）

引擎正常起核（`Worker_TP0..7_EP0..7`），两步 rollout 完成并写 dump：

| step | 单步 `rollout_actor_probs_pearson_corr` |
|---|---|
| 0 | **0.9972280** |
| 1 | **0.9957978** |

- dump：`/tmp/p0999/batch_latest-baseline/batch_step{0,1}.pt`
- 速度：训练 **135.6 s/it**（09-27 时 2 步要 ~15 min）；权重同步 `export 0.1s / flush 11.6s`
- 结束时的 `RuntimeError: DataLoader worker … killed by signal: Killed` 出现在**指标与 dump 都落盘之后**
  （`torch/library.py::_del_library` 的 weakref 退出钩子），属收尾阶段的被杀，不影响数据。

### 5.2 正式对比（7 步 vs 10 步，同口径）

为与 09-27 的 10 步池化基线对齐，再跑 `STEPS=10`（`TAG=latest-baseline10`）；
**第 6-7 步被 SIGKILL**（`main_ppo` Killed；宿主 cgroup 无内存上限、`dmesg` 无 OOM 记录，
判断为其他租户造成的系统级内存压力），拿到 **7 步 / 14336 token**。

**逐 token 池化（`scripts/dsv41_consistency/pool_dumps.py`）**：

| 运行 | steps | tokens | pooled pearson | mean_step | med\|d\| | std | >0.5% | >1.0% | var>0.5% | hi-p rel |
|---|---|---|---|---|---|---|---|---|---|---|
| `batch_latest-baseline10`（新） | 7 | 14336 | 0.99456 | 0.99459 | 0.0502 | 0.2376 | 5.36 | 1.11 | 74.6 | 0.0061 |
| `batch_baseline`（09-27） | 10 | 20480 | 0.99464 | 0.99458 | 0.0494 | 0.2315 | 5.26 | 1.07 | 74.6 | 0.0054 |

**逐步单值（从 dump 复算的 probs-pearson）**：

```
新栈 n=7  mean=0.99459   0.99667 0.99236 0.99339 0.99338 0.99443 0.99633 0.99554
旧栈 n=10 mean=0.99458   0.99505 0.99517 0.99188 0.99441 0.99458 0.99327 0.99465 0.99730 0.99513 0.99431
```

**逐 token 分解（`/tmp/p0999/an_r2_compare.py`，两栈各自的自证）**：

| 量 | 新栈（4096 tok，2 步） | 09-27 基线（20480 tok，10 步） |
|---|---|---|
| `dlogp` mean / std / med\|d\| / p99 / max | −0.0290 / 0.2495 / 0.0500 / 1.052 / 3.545 | −0.0268 / 0.2315 / 0.0494 / 1.025 / 2.487 |
| 自证 `KL` vs `0.5σ²` | −0.0290 vs +0.0311（比值 −0.93） | −0.0268 vs +0.0268（比值 −1.00） |
| `Var(Δp)` / `2Var(p)` vs 实测 `1−pearson` | 0.00324 vs 0.00345 | 0.00568 vs 0.00536 |
| `p²Δlogp²` 里 `|dlogp|>0.5` 的占比 | 74.6% | 74.6% |

两条都满足「`|KL| ≈ 0.5σ²`」（零均值高斯噪声，无系统性偏置），
且都满足第 1 轮 §5.1 的恒等式 `1−pearson ≈ Var(Δp)/(2Var(p))`。

### 5.3 判读

- **pooled 差 0.00008、单步均值差 0.00001** → 远小于 run-to-run 噪声（±0.0005）；
- min/max 区间重合（新 0.99236–0.99667，旧 0.99188–0.99730）；
- 所有分布统计（`med|d|`、`std`、尾部占比、高 p 相对误差）同带；
- 2 步那次读到的 0.9972/0.9958 是**抽样的运气**（同样的 harness、同样的 prompts，
  但采样出的 response 每次不同；10 步跑的均值立刻回到 0.9946）。

**→ 结论：新栈 = 旧栈，指标没动。**

---

## 6. 与读码预判的对照 + 机制判别

### 6.1 子代理读码给出的「应当生效」的变化

| # | 位置 | 变化 | 预判 |
|---|---|---|---|
| 15 | **MoE router 输入精度** | 旧 fork 无条件调融合 op（router 输入 = fp32 精确）；main 用 `enable_custom_op()` 门控，而 `VLLM_BATCH_INVARIANT`（= `rl_config.enable_batch_invariant=true`）下该门为 False → 退回 bf16 `post_attention_layernorm` 再 `.float()` | 引擎挪到与训练侧相同舍入点 → **改善** |
| 13 | batch-invariant `reduce_sum` | 「任意非 None dim / 强制 dtype」→「仅最后一维、fp16/fp32/bf16」 | 漂移 |
| 14 | fused MoE combine | `shared_output + fused_output` → `torch._foreach_add` | 漂移 |
| — | sampler / logprob | `vllm_ascend/sample/` 逐字节相同；`logprobs_mode` 仍 `raw_logprobs` | 无变化 |
| — | mHC（`npu_hc_pre_v2/v3`） | 只是重命名（同一 C++ body/同一 aclnn） | 无变化 |

### 6.2 实测：**预判未兑现**

三条"应当动"的改动，在本 recipe 上都**没有观测到效果**（§5.2 的数字与 09-27 一致）。
两种可能：(a) 它们确实惰性；(b) 15 的改善与 13/14 的漂移恰好抵消。

### 6.3 机制判别实验：**本轮未完成**（外部 SIGKILL，两次尝试）

设计：`VERL_DSV41_ALIGN_NOBI=1`（引擎 `rl_config.enable_batch_invariant=false`）
→ `enable_custom_op()` 变 True → 引擎 router 走融合 `rms_norm_cast`（fp32 精确，即 fork 行为）。
若数字**仍不动** → §6.2 的 (a)（这些改动确实惰性）；若**动** → (b)（15 的改善与 13/14 的漂移抵消）。

```bash
MODEL_PATH=/tmp/p0999/ckpt_v41 TAG=latest-nobi STEPS=6 VERL_DSV41_ALIGN_NOBI=1 \
  bash scripts/dsv41_consistency/sh/run_p0999_ab.sh
```

**实际结果：两次都没跑出数据。**

| 尝试 | 时间 | 结果 |
|---|---|---|
| 1 | 11:13 启动 | **启动阶段被 SIGKILL**（`main_ppo ... Killed`），0 步、0 dump。resolved config 已核实 `rl_config.enable_batch_invariant=false` 正确传入 |
| 2 | 11:30 启动（STEPS=4） | 用户叫停（"不要启动了"）→ 已 `TaskStop` + `cleanup_ray.sh` |

**为什么判断是外部杀进程**（而非 OOM）：宿主 `free` 显示 2454 GB 总量、仅 28 GB 使用、2077 GB 可用；
`dmesg | grep -i oom` 无记录；cgroup 无内存上限（`memory.max` = 无限制）；`ulimit -v/-m` 均 unlimited；
`/sys/fs/cgroup/memory.events` 不存在。同一现象也出现在 §5.2 的 10 步跑（第 6-7 步 Killed）。
这台机器是**多租户共享**（前几轮也记录过"其他租户占卡/占内存"），最可能是别的租户的 `pkill -f`
误杀，或集群侧看门狗。

**未完成的影响**：只影响机制归因的最后一格（(a) vs (b)），**不影响主结论** ——
§5.2 的两组数（新栈 vs 09-27 基线）都是完整可比数据，且 09-27 旧引擎上 BI-off 也测过（0.99402，无变化）。
若要补这一格：机器空闲时重跑上面的命令即可（约 25 min，6 步足够）。

---

## 7. 结论与建议

1. **指标地板被独立验证**：把引擎从「自研 fork 实现」整体换成「上游 vLLM 0.30 + vllm-ascend main 实现」，
   pearson 仍然落在 0.9946。第 1 轮「地板 = 两套独立 bf16 内核栈，不共用内核就没有收益」成立。
2. **不要用 `rollout_actor_probs_pearson_corr` 当验收指标**（第 1 轮 §7 的建议不变）；
   用 TIS（`algorithm.rollout_correction`）的 `rollout_is_ratio_fraction_high` / `rollout_is_eff_sample_size`
   与训练稳定性做验收。
3. **更新代码时要带上三件事**（否则跑不起来）：
   - vLLM 必须与 vllm-ascend 分支配对（main → **v0.30.0**）；
   - **重编 `vllm_ascend_C` 扩展**（换分支/改 csrc 之后必做，且要先清脏的 `csrc/build` 缓存）；
   - 切片/ckpt 的 `model_type` 必须是新拼写 `deepseek_v41*`。
4. **FSDP-Turbo 的 `dev_dsv41_one` 合并需要收尾**：本轮解了 4 处冲突 + 2 处"被覆盖掉的用户代码"
   （gate bias buffer、meta 安全搬运、adapter API）。**这些修复目前只在工作区（未提交）**，
   建议 commit 到 `dev_dsv41_one`（或重新做一次干净的 merge）。
5. 若要继续追这个指标：唯一的路仍是「两侧共用同一套内核」（第 1 轮 §6.3）；
   本轮进一步说明：**换引擎实现 ≠ 共用内核**——除非训练侧改用引擎的融合核（或反之）。

## 7.1 本轮未完成 / 待办

| # | 事项 | 说明 |
|---|---|---|
| 1 | BI-off 机制判别（§6.3） | 两次尝试均被外部 SIGKILL/叫停；机器空闲时重跑约 25 min |
| 2 | FSDP-Turbo 修复入库 | 4 处冲突 + 2 处被覆盖代码目前只在 `/workspace-verl/FSDPTurbo` 工作区（`git diff` 可见），建议提交或重做 merge |
| 3 | row_sharded Engram 的长期解 | 若要真正恢复 `dev_dsv41` 的 row-sharded 实现，需移植 `embedding_parallel.py` + `engram_optimizer.py`（上游有意删除，工程量 48 文件量级） |
| 4 | 满 10 步复测 | 本轮拿到 7 步（共享机被外部杀）；要更紧的置信区间可重跑 |

## 7.2 交付物清单

| 类别 | 路径 |
|---|---|
| 本记录 | `wiki/work_log/pearson_0999/worklog_p0999_r2.md` |
| 新栈 dump（7 步） | `/tmp/p0999/batch_latest-baseline10/` |
| 新栈 dump（2 步，顺便验证） | `/tmp/p0999/batch_latest-baseline/` |
| 09-27 基线 dump（10 步） | `/tmp/p0999/batch_baseline/` |
| 影子 ckpt | `/tmp/p0999/ckpt_v41/`（36 symlink + 新拼写 config.json） |
| 逐 token 分解脚本 | `/tmp/p0999/an_r2_compare.py`（新增）、`scripts/dsv41_consistency/pool_dumps.py`（沿用） |
| 运行日志 | `logs/p0999-latest-baseline10.log`、`logs/p0999-latest-baseline.log` |
| 引擎重编日志 | `/tmp/p0999/build_vllm_ascend2.log`（成功那次）、`/tmp/p0999/aclnn_build.log`（失败那次） |

---

## 8. 复现命令汇总

```bash
# ---------- 0) 环境（本轮改过的） ----------
cd /workspace-verl/vllm && git checkout v0.30.0            # vllm 与 vllm-ascend main 配对
#   （_version.py 是构建产物，需手工改成 0.30.0，否则 vllm_version_is 读到旧版本号）
cd /workspace-verl/vllm-ascend-v41-private && git status   # 确认两个 patch 在工作区
mv csrc/build csrc/build.stale-0918; mv csrc/output csrc/output.stale-0918   # 清脏 CMake 缓存
MAX_JOBS=64 pip install -e . --no-deps --no-build-isolation                 # 重编扩展（~13 min）
python3 -c "import vllm_ascend.vllm_ascend_C; import torch; print(torch.ops._C_ascend.npu_hc_pre_v3)"

# ---------- 1) 影子 ckpt（新拼写） ----------
python3 - <<'EOF'
import json, pathlib
src = pathlib.Path('/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-4layer-real')
dst = pathlib.Path('/tmp/p0999/ckpt_v41'); dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != 'config.json': (dst / f.name).symlink_to(f)
c = json.loads((src / 'config.json').read_text())
def fix(d):
    if isinstance(d, dict):
        if isinstance(d.get('model_type'), str): d['model_type'] = d['model_type'].replace('deepseek_v4.1', 'deepseek_v41')
        for v in d.values(): fix(v)
    elif isinstance(d, list):
        for v in d: fix(v)
fix(c); (dst / 'config.json').write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
EOF

# ---------- 2) 跑 A/B（本轮口径） ----------
cd /workspace-verl/verl
MODEL_PATH=/tmp/p0999/ckpt_v41 TAG=latest-baseline10 STEPS=10 \
  bash scripts/dsv41_consistency/sh/run_p0999_ab.sh
# BI-off 机制判别
MODEL_PATH=/tmp/p0999/ckpt_v41 TAG=latest-nobi STEPS=6 VERL_DSV41_ALIGN_NOBI=1 \
  bash scripts/dsv41_consistency/sh/run_p0999_ab.sh

# ---------- 3) 分析 ----------
python3 scripts/dsv41_consistency/pool_dumps.py /tmp/p0999/batch_latest-baseline10 /tmp/p0999/batch_baseline
python3 /tmp/p0999/an_r2_compare.py /tmp/p0999/batch_latest-baseline10 /tmp/p0999/batch_baseline
bash scripts/dsv41_consistency/sh/cleanup_ray.sh          # 清 ray/vllm 残留（pkill -f 会杀自己，用脚本）

# ---------- 4) 回退 ----------
cd /workspace-verl/vllm && git checkout v0.27.0           # 回旧 vllm（配合 vllm-ascend sp_feature）
cd /workspace-verl/vllm-ascend-v41-private && git checkout sp_feature
mv csrc/build.stale-0918 csrc/build; mv csrc/output.stale-0918 csrc/output   # 若要回旧扩展
cd /workspace-verl/FSDPTurbo && git diff                  # 查看本轮未提交的修复
```

---

## 9. 本轮踩到的坑（供下次省时间）

1. **`git status` clean 不代表代码能跑**：FSDP-Turbo 的"冲突标记"是被**提交进 HEAD** 的
   （`7cbc46f` 是单父提交）。判断方法：`git grep -n -E "^(<<<<<<<|>>>>>>>)"` + `ast.parse`。
2. **合并会静默丢功能**：`7cbc46f` 里 4 处冲突之外，还有 2 处"用户侧代码被上游侧覆盖"
   （gate bias buffer 的语义、`prepare_…_for_fsdp` 的 meta 安全搬运、adapter 的 engram API）。
   判据是「verl 调用签名 vs 新 FSDP-Turbo 签名」以及**离线跑一遍构建链**（CPU 即可，秒级）。
3. **editable 安装的优先级**：`sys.meta_path.append(_EditableFinder)` 是**追加在 PathFinder 之后**，
   所以 **`PYTHONPATH` 赢**（可用 `git worktree` + `PYTHONPATH` 试别的版本，不动用户 checkout）。
4. **`.so` 是构建产物、不进 git**：换 vllm-ascend 分支后必须重编 `vllm_ascend_C`，
   且 `csrc/build/CMakeCache.txt` 里的 op 清单会残留旧 op → 必须先清 build 目录，
   否则报 `Objects of target "…" referenced but no such target exists`。
5. **`import vllm_ascend` 不注册自定义 op**：要显式 `import vllm_ascend.vllm_ascend_C`
   （引擎是在 `platform.py:130` 按需 import）。
6. **rank0 的单步读数会被抽样运气骗**：2 步就敢下结论会踩坑（§5.3）；
   本轮的判读一律用 ≥7 步池化 + 分布统计。
7. 运行末段的 `DataLoader worker … killed by signal: Killed` 是**收尾阶段的被杀**
   （指标/dump 已落盘）；10 步那次 `main_ppo` 被 SIGKILL 发生在第 6-7 步，
   宿主 cgroup 无上限、dmesg 无 OOM → 其他租户造成的系统级内存压力，重跑即可。