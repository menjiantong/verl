# DeepSeek-V4.1 运行失败库（错误 / 错误栈 / 修复 分类速查）

> 用途：训练/推理运行中遇到的错误按类归档，出问题时先在本库检索。
> 维护：新错误解决后追加对应分类，格式固定为 **症状（原文日志行/栈）→ 根因 → 修复（命令）→ 验证 → 来源**。
> 创建：2026-10-13（用户指令："把所有遇到的错误、错误栈、修复方式记录成一个运行失败的库，分类记录"）。
> **2026-10-13 晚 补全版**：按用户要求"补齐"——新增 F-04 / F-24 / F-25 / F-32 / F-33 / F-42 / F-56 / F-57 / F-62 / F-63 / F-64，
> 及 §8 一致性 harness 类（F-70…F-76）；原有条目补齐原始日志行与可复制的修复命令。**当前共 41 条**。
> 相关工作记录：`wiki/worklog_dsv41_taskA_gsm8k_rl.md`（任务 A，§2 排障全史 8 次尝试 6 根因）、
> `wiki/worklog_dsv41_top64_grpo.md`（E1–E12 错误表）、`wiki/worklog_dsv41_sft.md`、`wiki/dsv41_consistency_playbook.md` §7（harness 14 坑）。

**索引**

| 类 | 条目 |
|---|---|
| 1. 依赖/同步类（栈克隆、文件缺失） | F-01 `_version.py` 缺失；F-02 `libvllm_ascend_kernels.so` 缺失；F-03 `_cann_ops_custom/**` 缺失；F-04 engram tiling 误报 |
| 2. 算子/内核类 | F-10 aclnnHcPre；F-11 npu_sparse_flash_mla NaN；F-12 fused permutation；F-13 CANN info 文件 |
| 3. 显存/内存/磁盘类 | F-20 attention chunk E11；F-21 SFT 专家态 OOM；F-22 RL update 峰值；F-23 chunk 256 余量；F-24 engram 缓存尺寸；F-25 本地盘满 |
| 4. 正确性/静默错误类 | F-30 RL EFSDP=2 权重同步损坏；F-31 never-EOS；F-32 收尾噪声（无害）；F-33 SFT 保存 Adapter 崩溃（无害） |
| 5. 通信/死锁类 | F-40 跨机 all_gather_object 死锁；F-41 gloo 跨机广播死锁；F-42 `No backend type … cpu` |
| 6. 环境/工具类 | F-50 npu-smi 非登录 shell；F-51 libhccl；F-52 acl 模块；F-53 pkill 自匹配；F-54 ssh 双 cat 吃 stdin；F-55 sort locale；F-56 CANN 告警（无害）；F-57 集群建好后 pkill "ray::" |
| 7. 进程/资源类 | F-60 Z 僵尸；F-61 清场流程；F-62 从机整树暴毙（心跳丢失）；F-63 slave 加入失败；F-64 raylet fd 耗尽 |
| 8. 一致性 harness 类 | F-70 切片前缀；F-71 钩子类名；F-72 router 层标识；F-73 TP token 切块；F-74 压缩位编号；F-75 cast 口径；F-76 dump 生产加载三件套 |

---

## 1. 依赖 / 同步类（**新机克隆旧栈时高发**；完整方法论见 memory `dsv41-npu-stack-sync-so-deps`）

共性根因：运行态里有一批**不入 git 的生成物/编译产物**（setuptools_scm 生成文件、自定义算子包、伴随 .so），tar/文件级同步时被漏掉；且部分缺口要跑到很深阶段（引擎 spawn / 首算子调用）才爆。

### F-01 `vllm/_version.py` 缺失 → `InvalidVersion: Invalid version: 'dev'`

- **症状**（run #10，03:42）：
  ```
  File ".../verl/workers/rollout/vllm_rollout/vllm_async_server.py", line 74
      _VLLM_VERSION = version.parse(vllm.__version__)
  packaging.version.InvalidVersion: Invalid version: 'dev'
  ```
- **根因**：`vllm/version.py` 尝试 `from ._version import __version__`，失败即 fallback `"dev"`；`_version.py` 是 setuptools_scm 生成文件、不入 git。119/126 的树在历次 "rm -rf 包目录 + tar 解压" 同步中丢了它。
- **修复**（以 131 为母本）：
  ```bash
  ssh …119 'cat > /workspace-verl/vllm/vllm/_version.py' < /workspace-verl/vllm/vllm/_version.py   # 126 同
  ```
- **验证**：`python3 -c 'import vllm; print(vllm.__version__)'` → `0.30.1.dev0+gced6857af.d20261013`；`version.parse` 通过（login shell + run 环境）。
- **来源**：taskA worklog §5.1；md5 `f659c379…`（三机一致）。

### F-02 `vllm_ascend/libvllm_ascend_kernels.so` 缺失 → `WorkerProc initialization failed`

- **症状**（run #11，04:30；两机、可重试复现）：
  ```
  (vLLMHttpServer …) (Worker_TP0_EP0 …) ERROR multiproc_executor.py:944 … TypeError: 'NoneType' object is not callable
  … camem.py:97  init_module(python_malloc_fn, python_free_func)
  … WorkerProc.wait_for_ready → raise e from None → Exception: WorkerProc initialization failed
  ```
  决定性前置信（真正根因，表面栈只是后果）：
  ```
  camem.py:67 WARNING Failed to import vllm_ascend_C:libvllm_ascend_kernels.so:
  cannot open shared object file: No such file or directory. Sleep mode will be disabled.
  ```
- **根因**：`vllm_ascend_C.so`（RUNPATH `$ORIGIN:$ORIGIN/lib:…`）NEEDED `libvllm_ascend_kernels.so`，该库未同步 → dlopen 失败 → `camem.py` 顶层 `except ImportError` 把 `init_module` fallback 成 `None` → 引擎 load_model 一进 memory-pool 就 TypeError。
- **修复**：推 `vllm_ascend/{,lib/}libvllm_ascend_kernels.so`（md5 `5acd0c78`，811640 B），**顶层与 lib/ 两个位置都要**。注意：同一条 ssh 里连续两个 `cat > f` 只有第一个吃到 stdin（见 F-54）。
- **验证**：login shell 下 `import vllm_ascend.device_allocator.camem` → `camem_available: True`、`init_module: <built-in function init_module>`；或 `ldd <so> | grep "not found"` 无输出。
- **来源**：taskA worklog §5.3。

### F-03 `_cann_ops_custom/**`（自定义算子包）缺失 → `aclnnHcPre … not in libopapi.so`

- **症状**（run #12，05:01；camem 一关已过、引擎 spawn 后崩）：
  ```
  multiproc_executor.py:437 raise RuntimeError: Worker failed with error
    'aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so, or libopapi.so not found.'
  multiproc_executor.py:314 Worker proc VllmWorker-1 died unexpectedly (exit code: None), shutting down executor.
  ```
- **根因**：`aclnnHcPre` 是 vllm-ascend **自定义算子**（`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/{op_proto,op_api,op_impl}/**`，由 `op_api/lib/libcust_opapi.so` 提供，经 `vllm_ascend_C.so` 的 RUNPATH 解析）。119/126 上该目录只有 `.gitkeep`（gitignore 构建产物未同步）。同批还漏了 `include/**`（11 文件）与 `vllm_ascend/_version.py`（584 B）。
- **修复**：
  ```bash
  cd /workspace-verl/vllm-ascend-v41-private/vllm_ascend
  tar cf - _cann_ops_custom include _version.py | ssh …119 \
    'cd /workspace-verl/vllm-ascend-v41-private/vllm_ascend && tar xf -'
  ```
- **验证**：`find _cann_ops_custom -type f | wc -l` = 1304；`libcust_opapi.so` md5 `23045ddd`；算子 smoke：
  `ctypes.CDLL(…/libcust_opapi.so)` 加载 ok 且 `hasattr(lib,'aclnnHcPre')/('aclnnHcPreGetWorkspaceSize')` 均 True。
- **预防（整类问题）**：克隆/同步栈后跑全树对照：
  - filelist：`find . -type f ! -path "*__pycache__*" ! -path "*/.git/*" | LC_ALL=C sort` 两端 diff（**两端必须同 locale**，见 F-55）；
  - 内容：全树 `md5sum` 后比对；
  - .so：`readelf -d <so> | grep NEEDED`、`ldd <so> | grep "not found"`。
  脚本存于 131：`/tmp/treecmp.sh`、`/tmp/md5cmp.sh`。已知无关差异（不用管）：131 的 `csrc/build/**` CMake 中间物、egg-info、`vllm/pyproject.toml`、verl 的 logs/wiki、FSDPTurbo docs/tests。
- **来源**：taskA worklog §5.4。

### F-05 旧版 dump 脚本残留（同步遗漏）→ `ModuleNotFoundError: No module named 'torch'` + 默认版本漂移

- **症状**（2026-10-13，120 上连跑两次崩）：`dump_engine_stages.py → from vllm import LLM → env_override.py:150 import torch
  → ModuleNotFoundError: No module named 'torch'`；日志头部显示 `+ export VLLM_VERSION=0.27.1`（应为 0.30.0）。
- **根因**：120 的 `/workspace-verl/verl/scripts/dsv41_consistency/sh/run_dump_engine.sh` 是**旧版**（镜像后 119 又改过）：
  ① 无 `export PATH=/usr/local/python3.11.15/bin:$PATH`（登录 shell 本身不带该 python 路径）→ `exec python3` 落到系统
  python（无 torch）；② `VLLM_VERSION` 默认还是 0.27.1（fixture 时代）→ 拉错补丁集。
- **修复**：把一致性脚本目录**整体同步**到 120（`tar czf - scripts/dsv41_consistency | ssh …'tar xzf -'`），md5 对账三机一致。
- **教训（整类）**：**同步遗漏不止在"不入 git 的生成物"**（F-01/02/03）——**被频繁修改的脚本目录同样要同步**；
  跨机跑任何 harness 前先 md5 对账该 harness 的脚本。
- **来源**：2026-10-13 实踩（120）；一致性 worklog §6.6 副产品。

### F-04 Engram tiling 误报 `rows are not tiled across ranks at 0`（FSDP16/EP8 双节点布局）

- **症状**：新机首次双机（FSDP16/EP8）启动，state-dict 广播结束后立即中止：
  ```
  [fsdp_turbo_dsv41] **rows are not tiled across ranks at 0**（随后 abort）
  ```
  （2026-10-08 16:58 双机首跑即此；run 脚本 header 有专段描述。）
- **根因**：FSDP16/EP8 = **两个节点本地 EP 组**，每组各持全表行区间（合法重复）；旧的 tiling 校验按"行区间唯一"假设读，误报。
- **修复**：`verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py::_load_host_engram_tables` 改为**复本感知**校验（判据 `copies_per_range`）。
  ⚠ **两台节点都必须带此修复**（/workspace-verl 是每机本地盘）——否则单侧带修复也不行。
- **验证**：`grep -c copies_per_range <impl>` > 0；run_grpo_2node.sh 的 precheck 会对缺修复的节点打 WARNING。
- **来源**：memory `verl-dsv41-engram-tiling-replica-fix`；`worklog_dsv41_direct_shard_load.md:33`；run_grpo_2node.sh 头注释。

---

## 2. 算子 / 内核类

### F-10 `aclnnHcPre` 系列 —— 见 F-03（归属自定义算子包缺失；注意它**不在** CANN 自带 `libopapi.so` 里，`grep libopapi.so` 徒劳，要去 `_cann_ops_custom` 找）。

### F-11 `npu_sparse_flash_mla` 长上下文 NaN（任务 B，2026-10-11/12 定位中）

- **症状**：长上下文（>~700 token 起、1500 附近明显）attention 输出 NaN。逐函数 NaN 扫描（len=1500）：
  `multistream_preprocess`(0) → `compressor.norm`(0) → `_write_compressed_source`(0) → `_select_sparse_indices`(0) → **`_forward_attention` = 第一个 NaN（6.14M/7.67M 元素）**；
  trainer 同长度 0 NaN；len=700 引擎全 0 → **NaN 在 700~1500 间爆发于 layer2/3（compressor 层）**。
  `_forward_attention` = `torch.ops._C_ascend.npu_sparse_flash_mla(...)`（compressed 路径 cmp_block_table/cmp_mask_mode=3/cmp_ratio）或 `backend.qsmla`。
- **排除项**：sleep/wake、`FREE_CACHE_ENGINE=False` 均无影响（无损/仍乱码 → 无罪）。
- **状态**：最小复现待做（隔离内核本身 vs `dsa_v41` 的 cmp 元数据组装）。与任务 A 的 RL 乱码同根（长上下文 NaN→采样塌缩）；也解释引擎 dump 早先 `decode-vs-prefill [nan]`。
- **来源**：taskA worklog §4 / 一致性 worklog §3.5。

### F-12 `fused permutation is not available`

- **症状**：DeepSeek-V4 GRPO 启动报 `fused permutation is not available`（同样触发条件亦可表现为需要 TE 的算子缺失）。
- **根因**：本环境**未安装 transformer_engine(TE)**；配置里 `moe_permute_fusion=True` 需要它。
- **修复**：llm_kwargs/配置里 `moe_permute_fusion=False`（与官方 recipe 的 llm_kwargs diff 即这一处）。
- **验证**：重新启动，permute 走非融合实现，不再报错。
- **来源**：memory `verl-mindspeed-moe-permute-fusion-fix`。

### F-13 `CANN toolkit info file does not exist`

- **症状**：verl 启动即报 CANN toolkit info 文件不存在。
- **根因**：CANN 9.x 改了 info 文件命名——合并布局下 `aarch64-linux/` 只有新命名 `ascend_all_cann_install.info` / `ascend_ops_install.info`，无旧名 `ascend_toolkit_install.info`（见 memory `verl-npu-training-env`）。
- **修复**：修 `verl/.../device.py` 兼容新命名（按新文件探测版本）。
- **验证**：启动不再报错、能取到 CANN 版本。
- **来源**：memory `verl-cann-install-info-compat-fix`。

---

## 3. 显存 / 内存 / 磁盘类

### F-20 关闭 FSDPTurbo 注意力 chunk/recompute → step 1 update OOM（E11）

- **症状**：step 1 actor update，`ops/cpu/sparse_attention.py::_indexed_sparse_attention_queries` 的 `torch.gather` 单次要 **7.2–7.3 GiB**（卡上已 alloc ~48–51/61 GiB），多 rank 同时 OOM（FSDP16 也复现）。
- **修复**：保持 `FSDP_TURBO_ATTN_RECOMPUTE=1` + `FSDP_TURBO_ATTN_CHUNK=64`（默认）。恢复旧行为仅在缩 update 形状时（`RECOMPUTE=0 CHUNK=100000000`）。
- **来源**：memory `verl-dsv41-e11-attn-chunk-restore`；run 脚本 2026-10-09 注释。

### F-21 SFT（FSDP32）每卡常驻专家态 OOM

- **症状**：EP8/efsdp=1 时每卡常驻专家 params+grads+优化器态 ≈44 GB → OOM。
  **判据**：step2 死在 `grouped_matmul.py` backward（差 +182MiB 一类）→ 先查 efsdp 是否为 2。
- **修复**：`engine.turbo_config.distributed.expert_fully_shard_parallel_size=2`（峰 55.1→30.4 GB）。
  ```bash
  # 脚本已默认（run_sft_dsv41_2node.sh 的 SFT_ARGS_COMMON）；手工核对：
  grep expert_fully_shard_parallel_size /workspace-verl/scripts/run_sft_dsv41_2node.sh   # 期望 =2
  ```
- **⚠️ 反向坑**：**RL 侧 efsdp 必须=1** —— efsdp=2 会让 RL 权重 sync 走未验证回退路径（见 F-30）。SFT/RL 不同参！
- **来源**：memory `dsv41-rl-npu-workstream` §24；`wiki/worklog_dsv41_sft.md`；SFT skill §1/§3。

### F-22 RL actor update 内存峰值（与 response 长度）

- **实测**：每层 ~1.1 GB saved-for-backward 累积、backward 全程不释放；峰值≈Σ层驻留+base，与 response 弱相关。安全区=resp≤1024（alloc 峰 38 GB / reserved 54.9 存活）；resp=2048 需再挤 5–10 GB（杠杆：cpu_offload、Ulysses SP=2）。
- **工具**：`FSDP_TURBO_MEM_DEBUG=1` + `FSDP_TURBO_MEM_DEBUG_RANKS`（`fsdp_turbo_dsv41_impl.py::_build_dsv41_module` 内探针）。
- **来源**：memory `dsv41-rl-npu-workstream` §23④。

### F-23 attention chunk 256 余量不足

- **症状**：12:55 run 的 actor update OOM“差一点”：rank 2 要 1.54 GiB、free 仅 1.33 GiB（53.81/61.27 已 alloc，含共卡 vLLM worker ~17.4 GiB）。
- **修复**：chunk **256 → 64**（每 chunk 分配 ~0.4 GiB，清掉 0.2 GiB 缺口）。**两节点必须同值**。
- **来源**：run 脚本 2026-10-09 注释。

### F-24 engram 缓存尺寸不匹配 `expanded size (4) must match (8)`

- **症状**：SFT `micro>4` 时崩（`engram.py:235`）：
  ```
  expanded size (4) must match (8)  （形状断言）
  ```
- **根因**：`ModelArgs.max_batch_size` 默认 4，管着 engram hash cache/压缩器状态，未按 micro batch 放大。
- **修复**：env `VERL_DSV41_MAX_BATCH_SIZE=$MICRO_BSZ`（从 engine 一路穿到 ModelArgs；`adapter.py` 三处 + `fsdp_turbo_dsv41_impl.py::_dsv41_max_batch_size()`）。**SFT 启动器已自动设置**，默认 4 不变。
- **验证**：`grep VERL_DSV41_MAX_BATCH_SIZE scripts/run_sft_dsv41_2node.sh`；micro=8 可跑过 step1。
- **来源**：`worklog_dsv41_sft.md:106`。

### F-25 本地盘满（ray spill / ckpt 堆积）

- **症状**：SFT 多次跑后从机本地盘逼近满（ray 报 spilling 相关错误；每机 ckpt `/workspace-verl/ckpt_sft_dsv41` 约 825GB）。
- **修复**：共享盘备份核实后 `rm -rf /workspace-verl/ckpt_sft_dsv41`；导出流程用 symlink 零拷贝。
- **验证**：`df -h /workspace-verl` 空闲回升；重跑不再 spill。
- **来源**：SFT skill §3。

---

## 4. 正确性 / 静默错误类（**不报错但结果坏** / 或表面吓人实则无害）

### F-30 RL `EFSDP_SIZE=2` → 权重同步静默损坏 → reward 恒 -1

- **症状**：reward 恒 -1；**RL 侧输出 dump 实证为多语种乱码词沙拉**（不是循环、不是 never-EOS）；serve 用同一导出目录正常答题 + SFT 引擎加载导出 loss=0.681（两端文件本身正常）。
- **根因**：为省显存给 RL 新加的 `EFSDP_SIZE=2` 让权重 sync 走未验证回退路径 → vLLM 收到的权重损坏。
- **修复**：`EFSDP_SIZE=1`（历史 proven）。修复后：reward max +1.0 / advantages -0.0124 / grad_norm 0.35 / response_len 500（不再顶格）/ pearson 0.9798。
  ⚠ **注意 `run_grpo_2node.sh` 内历史默认是 `EFSDP_SIZE=${EFSDP_SIZE:-2}`**——直接裸跑主启动器会踩雷；`launch_grpo_master.sh` 已显式覆盖为 1，请用它启动。
- **辨识要点**：与 F-31（never-EOS）区分——两者都表现为 reward 恒 -1，但 F-31 是 response 顶格（clip_ratio=1.0、min=max=cap），F-30 是乱码输出。
- **来源**：memory `dsv41-rl-npu-workstream` §24②。

### F-31 never-EOS（SFT 前）→ reward 恒 -1 + response 100% 顶格

- **症状**：512/1024/2048 三档、几百条样本**零 EOS**（response_length min=max=cap、clip_ratio=1.0、grad_norm 0）；EOS 配置/prompt 渲染/`ignore_eos=False` 全排除。
- **根因**：384→32 专家裁剪致生成退化/循环（判别实验：离线单样本生成看文本）。
- **修复**：**SFT 热启动**（GSM8K 1 epoch/116 步）教学"终止+Answer"→ 前后对比 1/32→29/32 正常收尾。之后 RL reward 出现正常分布（2026-10-13 run #13：step1 mean **-0.59**、[-1,+1]）。RL 用 `.../weights/DeepSeek-V4.1-Flash-top32-sft116` 权重。
- **来源**：memory `dsv41-rl-npu-workstream` §23②/§24①；SFT skill §0。

### F-32 日志尾部 `DataLoader worker … killed by signal: Killed` + `ERR99999`（**收尾噪声，结果有效**）

- **症状**（2026-10-13 1 步验证跑 `logs/grpo_top64_steps1_20261013_102932.log`，**step:1 指标已产出之后**）：
  ```
  Traceback (most recent call last):
    File ".../ray/_private/worker.py", line 1066, in main_loop
    File ".../torch/utils/data/_utils/signal_handling.py", line 73, in handler
      _error_if_any_worker_fails()
  RuntimeError: DataLoader worker (pid 260329) is killed by signal: Killed.
  [ERROR] 2026-10-13-10:57:40 (PID:244569, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
  ```
- **判读**：**收尾噪声**——发生在 `Training Progress: 100%|1/1` 之后（清理阶段 worker 被 SIGKILL，Ray 退出时重抛；`ERR99999` 只是 CANN 外层包装）。**不影响任何 step 结果**。
- **如何确认良性**：① 之前已有 `step:N - …` 指标行 + `Training Progress: 100%`；② Traceback 只出现在收尾段、无更早同类栈。
- **来源**：本 run 日志；一致性 worklog §6.4。

### F-33 SFT ckpt 保存尾部 `AttributeError: '…Adapter' object has no attribute 'config'`（**权重已存成功**）

- **症状**：保存 global_step 时末尾崩：
  ```
  AttributeError: 'FSDPDeepseekV41ForCausalLMAdapter' object has no attribute 'config'
  ```
- **根因**：dsv41 是 FSDPTurbo nn.Module（无 HF config 属性）；保存元数据段仍按 HF 模型取 `model.config`。
- **修复**：`fsdp_checkpoint_manager.py` HF config 段加 `getattr(model,'config',None)` 守卫（dsv41 跳过 HF config、保留 tokenizer、留显式日志 "skipping HF config save"）。
- **验证**：权重分片 `model_world_size_*_rank_*.pt` **在崩溃前已写成功**（数量=world size）；日志含 skip 行。
- **来源**：`worklog_dsv41_sft.md:80`。

---

## 5. 通信 / 死锁类

### F-40 131+128 作 2 节点对 → model-load 阶段跨机集合通信死锁

- **症状**：model-load 卡死跨机 `all_gather_object`；双方 py-spy 均 idle 于分布式调用。与 2026-10-08 "cross-machine gloo broadcast deadlock" 同型。
- **结论**：**不要用 131+128 组 2node**（各 8 卡独立可用）。已验证可用的双机组合：**119+120**（2026-10-13 多次跑通）。
- **来源**：taskA worklog §4.5。

### F-41 gloo 跨机广播死锁（主机名解析旧 IP）

- **症状**：进程组建立在主机名上时，解析到旧 IP → 跨机广播死锁。
- **修复**：`GLOO_SOCKET_IFNAME=<NIC>`（及 `HCCL_SOCKET_IFNAME`）显式指定网卡；run 脚本 set_env 已含（`enp48s3u1u1`）。
- **来源**：memory `verl-dsv41-npu-grpo-launch-env`。

### F-42 `RuntimeError: No backend type associated with device type cpu`（两个出处，勿混）

- **症状 A（RL optimizer_step / clip grad-norm）**：
  ```
  RuntimeError: No backend type associated with device type cpu
  ```
  **根因**：FSDPTurbo `clip_grads.compute_grad_norm`——CPU-offload 的 DTensor 梯度分片在主机，而进程组是 hccl。
  **修复**：`_reduction_device/_all_reduce_on_group_device`（FSDPTurbo `:82-114`：CPU 标量先搬组设备再 all_reduce），**已在树内**。
- **症状 B（权重导出路径）**：EP 不是 8（== GEN_EP）时 `LOCAL_EXPERT_EXPORT` 自动关闭、回退 all-gather，遇 CPU-offloaded DTensor 即报同句错。
  **修复**：**EP_SIZE 固定 =8 且 GEN_EP=8**（run_grpo_2node.sh 已固定，见其头注释）。
- **来源**：`worklog_dsv41_top64_grpo.md:599`（E6）/§5.6；`dsv41_memory_and_parallel_archive.md:184`；run_grpo_2node.sh 头注释。

---

## 6. 环境 / 工具类（含"无害噪声"识别）

### F-50 `npu-smi info` 在非 login shell 下静默失败（输出为空）

- **表现**：`ssh … 'npu-smi info'` 无输出（缺 `libc_sec.so`），导致 `grep -c "Process id"` 恒为 0 → **误判"NPU 无占用"**。
- **修复**：远端一律 `bash -lc "npu-smi info …"`。教训：NPU 占用观测要用两种手段交叉（npu-smi + 进程 CPU/py-spy）。
- **来源**：memory `verl-dsv41-rebase-20261012`（roster 注）；2026-10-13 实踩。

### F-51 `import torch_npu` 报 `libhccl.so: cannot open shared object file`

- **根因**：shell 没 source CANN 环境（`/etc/profile` 的 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 未生效）。
- **修复**：`bash -ls`（login shell）或手动 source set_env。**（2026-10-13：`dsv41_deploy/` 的 `preflight.sh` 与 4 个
  `launch_*.sh` 已内置 CANN 自举——`import torch_npu` 失败时自动 source `/usr/local/Ascend/ascend-toolkit/set_env.sh`，
  非 login shell 直跑不再踩；119 实跑验收 ALL OK）**
- **来源**：taskA worklog §5.3 验证坑①；部署包实跑验收。

### F-52 `from acl.rt import memcpy` → `No module named 'acl'`

- **根因**：`PYTHONPATH=…` **前缀式**赋值把 CANN 的 python 路径整个覆盖掉了。
- **修复**：追加式 `export PYTHONPATH="/workspace-verl/vllm:…:${PYTHONPATH:-}"`（run 脚本即此写法）。
- **来源**：taskA worklog §5.3 验证坑②。

### F-53 `pkill -f <模式>` 自匹配把自己的 shell 杀掉（exit 144）

- **表现**：命令行里含待杀模式串（如 `pkill -f watch_steps10`）时，pkill 匹配到执行中的 `bash -c` 自身 → 自杀 → 退出码 128+16。
- **修复**：用不自匹配的模式（`pkill -f 'watch_steps1[0]'` 字符类技巧）或按已知 PID kill。`pgrep -f` 同理（曾把自身 pid 当目标）。
- **来源**：2026-10-13 实踩（两次）。

### F-54 同一条 ssh 里连续两个 `cat > f` 只成立一个

- **表现**：`ssh … 'cat > a && cat > b' < bigfile` → 第二个文件为空（stdin 被第一个 cat 耗尽；md5 为空文件 `d41d8cd9…`）。
- **修复**：每个待传文件单独一条 ssh，或 tar 管道。
- **来源**：taskA worklog §5.3。

### F-55 `comm`/`diff` 文件清单两端 locale 不同 → 结果错乱

- **表现**：`comm: file 2 is not in sorted order`，diff 结果整列跑偏（曾输出 "missing=2005" 的假象）。
- **修复**：两端统一 `LC_ALL=C sort` 后再 `comm`。
- **来源**：2026-10-13 实踩。

### F-56 `/etc/profile: line 29: …cann-9.0.0/…/set_env.sh: No such file or directory`（**无害，忽略**）

- **症状**：每条非交互 shell / ssh 命令开头固定出现该行（常连带 pynvml FutureWarning）。
- **根因**：`/etc/profile` 引用旧版本 CANN（9.0.0）路径；本机实际 CANN 在 `/usr/local/Ascend/cann-9.1.0`（合并布局）。
- **处置**：**忽略**，不要为此改 /etc/profile。写检查脚本时建议过滤（`grep -v -e "line 29: /usr/local/Ascend" -e pynvml`）。
- **来源**：memory `verl-npu-training-env`；日常实见。

### F-57 集群建好后再 `pkill -f "ray::"` → raylet 连带死亡

- **症状**（2026-10-09 06:20 两次实踩）：
  ```
  The raylet exited immediately because one Ray agent failed,
  agent_name = runtime_env_agent/dashboard_agent
  driver: Failed to connect to Ray cluster（或 ConnectionError）
  ```
  raylet.err 时间与 wrapper 启动秒级对齐（日志名 `_062006`）。
- **根因**：ray 用 setproctitle 把代理/worker 显示名改成 `ray::DashboardAgent` / `ray::RuntimeEnvAgent` / `ray::WorkerDict`，`pkill -f "ray::"` 会匹配它们；杀掉活代理 → raylet fate-share 退出。
- **修复/纪律**：清扫**只能在 `ray start` 之前**做。run_grpo_2node.sh / run_grpo_top64.sh 已内置该纪律（RAY_ADDRESS 存在时跳过清扫；`GRPO_FORCE_SWEEP=1/GRPO_SKIP_SWEEP=1` 可控）。
- **来源**：`worklog_dsv41_direct_shard_load.md:12`；run_grpo_top64.sh 头注释。

---

## 7. 进程 / 资源类

### F-60 Z 态僵尸（`<defunct>`）杀不掉也没必要杀

- **表现**：`kill -9` 无效；`ps` 显示 `<defunct>`、STAT 含 `Z`、PPID=1（容器 PID1 不收割）。`pgrep -af vllm` 会以**裸名**列出它们（无 cmdline）。
- **实例**：2026-10-13 120 上有 6 个昨天（10-12）的 `[vllm] <defunct>`；不占 NPU/内存（`bash -lc "npu-smi info"` 无对应进程行）。
- **处理**：**无害，忽略**。清场检查时用 `ps -eo stat,comm | grep -v "^Z"` 只看活进程。
- **来源**：多次实踩（2026-10-13）。

### F-61 清场标准流程（换 run / 停止全部时）

```bash
# 每机各跑一遍
ray stop --force
sleep 2
pkill -9 -f "VLLM::" ; pkill -9 -f "main_ppo" ; pkill -9 -f "vLLMHttpServer"
pkill -9 -f "ray::" ; pkill -9 -f "raylet"        # ⚠ 仅当集群已停/启动前（F-57）
sleep 3
ps -eo stat,comm | grep -E "VLLM::|WorkerDict|main_ppo|raylet" | grep -cv "^Z"   # 期望 0
bash -lc "npu-smi info" | grep -c "Process id"                                   # 期望 0（须 login shell，见 F-50）
```
启动器（run_grpo_2node.sh / run_sft_dsv41_2node.sh）已内置"**ray start 之前**清扫残留"的逻辑；
启动前完整性检查用 `scripts/dsv41_deploy/preflight.sh`（卡占用、PYTHONPATH、import、生成物）。
- **来源**：2026-10-13 多次清场（含用户两次指令）。

### F-62 从机整棵树突然死亡（心跳丢失）——两种已见根因

- **症状 A（121，2026-10-11）**：worker group 创建后 3 分钟，121 上**全部 ray 进程同时变僵尸**（`[raylet] <defunct>` ×2 代）；
  GCS 记 `DEAD … missing too many heartbeats`；driver `ActorDiedError`。
  排查排除项：主机 dmesg 干净、cgroup 无限制且 `oom_kill 0`、无 core、fd 仅 115、无外来 ray 进程。
  根因（时间线实证）：同刻**邻居容器启动 16 卡作业**，其清理/抢占动作（典型 `pkill -9 -f python`）把同机全部 python 进程灭掉（本容器进程全是 python、sshd 独活，特征唯一）。**处置：弃用该机，从机换机。**
- **症状 B（126，2026-10-13）**：dump run 期间 `node has died, missing too many heartbeats`；随后 `RayTaskError(OutOfMemoryError)`。
  根因：126 宿主被邻居压满（host used 1.7T；16 卡被邻居各占 ~41G）。**处置：弃用。**
- **通用排查**：`dmesg | tail`、`free -g`、`bash -lc "npu-smi info"`、`ps -eo stat | grep -c Z`；区分"邻居干的" vs "自己 OOM"（F-22/23 是自己）。
- **来源**：taskA worklog §2 #2/#3；2026-10-13 实踩（P1 dump run 连崩两次）。

### F-63 slave 加入失败 / master 日志停在 `nodes=1/2`

- **症状**：master 等待循环一直 `[2node] nodes=1/2 after Xs`；或 slave 侧 "joined" 后立刻掉线（该机 raylet 此前崩溃过）。
- **根因**：slave 上残留旧 Ray session（raylet 死后未清理）导致新 raylet 起不来/注册失败。
- **修复**：在 slave 上 `ray stop --force`（清残留 session）后**重跑 `launch_grpo_slave.sh <master_ip>`**；master 等待窗口默认 60 分钟，无需重启 master。
- **验证**：master `ray status` Active ≥2 且 NPU total=16（或日志 `[2node] slave present`）。
- **来源**：2026-10-13 实踩（120：清 ray 后重启 slave 成功）。

### F-64 head raylet fd 耗尽（run 卡在 "Connected to Ray cluster" 不动）

- **症状**：driver 卡 "Connected to Ray cluster" 后**14 分钟零输出**；无 actor 进程、NPU 全闲；`raylet.err` 满屏：
  ```
  E1012 …… "File descriptor limit reached. Retrying."   （每 2s 一轮）
  ```
  `/proc/<head-raylet>/fd` = **1024/1024**（软限值）；从机 raylet 仅 58/1024（**head 特有**）。
- **根因**：镜像软限 1024，head raylet 在数分钟内被 gRPC 撑满。
- **修复**：启动器顶部 `ulimit -n 65536`（硬限 1073741816，可提；两个启动器自带、raylet/workers 全继承）。
- **验证**：重启后 head raylet fd 1274 < 65536、无 fd 错误行、driver 正常推进。
- **来源**：taskA worklog §2 #1；run_grpo_2node.sh:70-78 注释。

---

## 8. 一致性 harness 类（**排查工具自身的坑**；深度展开见 `wiki/dsv41_consistency_playbook.md` §7，实操手册同文件）

> 这类问题发生在"训推一致性对拍工具链"里（不是训练本身）。速查 7 条：

### F-70 切片失败 `KeyError: layers.2.self_attn.compressor.norm.weight`

- **根因**：切片层号必须是 **0 起连续前缀**（pick_names 不做层号重排）；engram 层要排除 + config 联动重写（`compress_ratios` / `kv_source_layer_ids` / `index_source_layer_ids` / `candidate_source_layer_id` / `engram_layer_ids=[]` / `_dummy_no_engram`）。
- **修复**：`--layers 0,1,2,3` 前缀 + `/tmp/make_top32_slice.py` 模板一趟出包。
- **来源**：playbook §3；一致性 worklog §6。

### F-71 引擎 dump 只有模块级量、注意力内部量全缺（钩子静默失效）

- **根因**：rebase 后注意力实现类改名 `DeepseekV41EagerAttentionImpl` → **`AscendDSAV41Impl`**，import 失败被 `except` 吞掉 → 整条注意力内部钩子从未生效。
- **修复**：新名优先、旧名回退（`dsv41_dump_extension.py`）；钩 `multistream_preprocess / _write_compressed_source / _select_sparse_indices / _forward_attention`。
- **来源**：一致性 worklog §3.4；playbook §7。

### F-72 `compare_routing: no engine router capture`（router 无层标识）

- **根因**：MoE router 不在 `named_modules`（无 prefix、无 layer_idx），dump 抓不到带层号的键。
- **修复**：hook_begin 扫描模块属性，给 router 打 `_dsv41_probe_path` 层路径章；wrapper prefix 三级回退（prefix→layer_idx→probe_path）。
- **来源**：playbook §7。

### F-73 路由对比形状 64 vs 8（TP 切 token）

- **根因**：引擎 MoE router 输入按 TP **切 token**（rank0 = 前 len/8 连续块，最近邻验证），trainer 是全量。
- **修复**：比较前按 `ref = raw_t[:other.shape[0]]` 对齐；top-k 用 set-match 口径（顺序差≠集合差）。
- **来源**：playbook §7。

### F-74 压缩索引对比 0% match（长序列）

- **根因**：编号基准不同——trainer `attn_opN.topk_indices` 压缩位 = **S+idx**；engine `_select_sparse_indices` = **0 起**。
- **修复**：`id = v - S` 换算后再比（`compare_sparse_indices.py` 内置）。
- **来源**：playbook §4/§7。

### F-75 `L0 attn_norm` 差 0.0017"第一处差异"（**口径假象**）

- **根因**：trainer dump 存的是 **cast 前 fp32**；按 bf16 语义比较后两侧**逐位相同**（引擎融合核正确）。
- **修复（纪律）**：比较模块输出前**必须 cast bf16**（全表同口径复算）。
- **来源**：一致性 worklog §6.2；playbook §7。

### F-76 `dump_trainer_stages.py` 起不来（三件套）

- **症状**：① `ValueError: Engram requires the model tokenizer`；② `ImportError: cannot import name 'FSDPTurboEngine'`；③ engram 表把 366.8GiB 搬 device OOM。
- **修复**：① 传 `AutoTokenizer.from_pretrained(rl_tokenizer)`；② 真实类名 `FSDPTurboDSV41EngineWithLMHead`（shim 走生产 `_materialize_dsv41_parameters`）；③ 接入生产 host-offload engram 加载路径。
- **来源**：playbook §3/§7。

### F-77 `ray start --head` 报 session 不匹配 / 8265 被占（**外部遗留 Ray GCS 抢端口**）

- **症状**：`AssertionError: Session name session_... does not match persisted value b'session_<更早的时间>'`；
  日志里 gcs_server 另有 `Failed to start the grpc server. The specified port is 6379 ... Address already in use`；
  页面还报 `Failed to bind to 127.0.0.1:8265`（dashboard）。
  **特征：本机 `ps` 里看不到任何 ray/redis 活进程，`rm -rf /tmp/ray` 也无效**（2026-10-13 128 上两连踩）。
- **根因**：6379（/8265）被**本 PID 命名空间之外**的遗留 Ray GCS 占用（邻居/宿主容器同网络命名空间；
  探测：`python3 -c 's.connect(("127.0.0.1",6379)); s.send(b"PING\r\n")'` 回的不是 `+PONG` 而是 HTTP/2 帧 = gRPC）。
  我们自己的 gcs bind 失败后，Python 侧连到那个外部 GCS → 读到旧 session 名 → assert。外部进程不可见、杀不掉。
- **修复**：换端口绕过——master/slave 两侧同时 `RAY_PORT=6390`（或任何空闲端口）再启动；
  先验证空闲：`python3 -c 'socket.bind(("0.0.0.0",6390))'`。dashboard 8265 冲突只是 WARN，不影响训练。
- **来源**：2026-10-13 任务2 启动（128+129）。

### F-78 新机 vLLM worker 起不来：`OSError: libatb.so: cannot open shared object file`（129 上首见）

- **症状**：worker init 阶段 `torch.ops.load_library(.../torch_npu/lib/libop_plugin_atb.so)` →
  `libatb.so: cannot open shared object file` → `WorkerProc failed to start` → `WorkerProc initialization
  failed ... background process`（8 个 worker 齐挂）。**preflight 的 `import torch_npu` 检查不一定能拦住**
  （atb 插件是懒加载，导入期不触发）。
- **根因**：torch_npu 的 ATB op 插件需要 `libatb.so`，其在 `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib`；
  该路径**靠 /etc/profile 注入 LD_LIBRARY_PATH（仅登录 shell）**。新容器/非登录 shell（ssh 一次性命令、raylet）
  拿不到 → worker 挂。131 等老机容器 sshd 环境自带该路径所以从未暴露。
- **修复**：`run_grpo_2node.sh` 的 `set_env()` 无条件把上述 ATB lib 目录并入 `LD_LIBRARY_PATH`
  （幂等、两机同改，2026-10-13）；launcher/raylet 继承后 worker 正常。
- **来源**：任务2 第二次启动（128+129）18:02 首炸。

### F-79 torch_npu：对**巨型非连续 plane 做任意高级索引**会整表拷贝 → 24GB OOM（2026-10-13）

- **症状**：引擎内对 paged cache plane（`[num_blocks, 128, 1, D]`，24-51 GiB、layer-outermost 非连续视图）
  做 `plane[blk, off, 0]`（blk/off 为任意大小的索引张量）→ `torch.OutOfMemoryError: Tried to allocate ~24 GiB`——
  **请求大小恒等于整个 plane 的体积**，与索引张量多小无关（K 切到 32、甚至 dim0 只取 2 个元素都会触发）。
- **独立复现**（/tmp/probe_plane_big*.py，131 单卡）：构造 48GB storage 的 `storage[:, 0]` 视图（24GB 非连续），
  `plane[u]`（u 仅 2 元素）→ 请求 24.00 GiB；同代码在 512MB plane 上不复现（小 copy 被 allocator 吞掉）。
- **修复**：**逐块视图 gather**——`plane[b]`（标量选取=视图，廉价）→ `plane[b].squeeze(-2)[off[m]]`（小视图上的单轴索引，
  只付实际行数）→ mask 赋值回输出。唯一块数在真实场景很小（压缩缓存 ≤ seq/ratio/128 块；窗口 ≤ 几块）。
  实测：正确（与 contiguous oracle torch.equal=True）、内存 +0.2GB。
- **来源**：任务1 注意力核 eager 替换（VA_V41_ATTN_EAGER），两次 dump OOM。
- **教训**：这个栈上**别对巨型非连续张量用花式索引**（哪怕只取几行）；先 `plane[b]` 视图分解。

### F-80 引擎侧补丁只改主控机、从机引擎跑旧码（"验证跑"白跑）

- **症状**：训推一致性/引擎补丁改在 **131（master）**，但双机 RL 的 **vLLM 引擎跑在 slave（119/129）**。
  slave 的 `vllm_ascend/…/dsa_v41.py`（无 `VA_V41_ATTN_EAGER`）、`dsa_v1.py`（无 `FP32_PARTIAL`）、
  `vllm_async_server.py`（无 `PREFILL_LOGPROBS`）、FSDPTurbo `model.py`（无 `HC_BF16`）——**全部旧版**，
  两轮"全配方验证跑"（19:0x pearson 0.9581 / 20:21 metrics NaN）实际都在旧引擎码上跑。
- **发现方式**：md5 对账（`md5sum` 两端对比四个文件），diff 显示本地为严格超集（含补丁注释行）。
- **修复**：`scp` 四文件到 119 + 两端 md5 复核一致（2026-10-13 晚）；重跑验证。
- **纪律（重要）**：**任何引擎/模型侧代码改动，验证跑之前必须对"引擎所在节点"做 md5 对账**；
  双机 RL 里引擎在 slave、trainer 跨两机，两边树都必须同步。清单：`vllm_ascend/{dsa_v41,dsa_v1}.py`、
  FSDPTurbo `models/deepseek_v41/model.py`、`verl/.../vllm_async_server.py`（+ 各自同目录的关联文件）。
- **旁支**：同轮"NaN 指标"疑为本坑的次生现象或独立随机（旧码上两跑仅 env 不同），重跑后按新数据判读。

---

## 附：一条错误出现时的速查顺序

1. **先看最深栈**：`grep -a "Traceback" -A40 <log> | tail`（ray 聚合日志常把 root cause 截在中段）。
2. 若是 **ImportError/找不到 .so/.so 符号** → 查 §1/§2（八成是同步遗漏：对照 131 母本跑 `treecmp/md5cmp` + `ldd`）。
3. 若是 **OOM** → 查 §3（对照当前 chunk/recompute/EFSDP 配置与实验记录）。
4. 若是 **reward 恒 -1 一类的"静默坏"** → 查 §4（先 dump 输出文本判定 F-30 vs F-31）。
5. 若是 **卡住无输出** → §5（跨机通信）+ py-spy dump 关键进程；"Connected to Ray cluster 后不动"先查 F-64（fd）。
6. 若是 **节点/进程整片消失** → §7（F-62 邻居击杀 / F-63 加入失败）。
7. 工具异常（npu-smi 空、shell 行为怪、日志尾假报错）→ §6 + F-32。
8. **一致性排查工具链本身的坑** → §8，或直接翻 `wiki/dsv41_consistency_playbook.md` §7。
