# DSV41 双仓 rebase 工作记录（FSDPTurbo + vllm-ascend，2026-10-12）

> 任务来源（用户 2026-10-12）：
> 1. **119 新机器连通性测试**；
> 2. **FSDPTurbo / vllm-ascend main 均已前进很多，需要 rebase 并保留我们的修改**。
>    vllm-ascend 特别说明：*只有最近两个 fix 是我们的*；倒数第 3~4 的提交是用户手动解冲突合入的 PR，
>    **上游主线现在已包含它** —— 因此策略 = reset 到较靠前位置 → 拉最新 main → 只重放我们的两个 fix（cherry-pick 或 patch）；
>    3. **输出当前所有可用机器**。
> 记录要求：执行命令、错误日志、关键证据、关键设计/修改方案。

---

## 0. 结论速览

| 仓库 | 最终状态（2026-10-12 14:40） | 备份 |
|---|---|---|
| FSDPTurbo | `dev_dsv41_one` = `dev_dsv41_one_rebased` = **`31fd979`**（= upstream main `4b6f2a6` + 我们全部 delta） | `backup_dev_dsv41_one_20261012` = `7714bef`（"fix reward"，旧 tip） |
| vllm-ascend | `local_run` = `local_run_rebased` = **`0568da9db`**（= upstream main `2fa829f95` + 我们的 fix） | `backup_local_run_20261012` = `b2da8fb47`；旧 `.so` 在 `/tmp/vllm_ascend_C.backup-20261012.so` |
| vllm-ascend 重编 | ✅ `Successfully installed vllm_ascend-0.19.1rc2.dev2749+g0568da9db`（`.so` 14:28，`torch.ops._C_ascend` 算子可解析） | — |

---

## 1. FSDPTurbo rebase（本轮主战场）

### 1.1 背景与准备

- 我们的 delta = `git diff a251f94..7714bef`（9 文件 **+793/-95**，含 "fix reward" 等提交），
  已导出为补丁 `/tmp/our_delta_fsdpturbo.patch`（55KB，14:19）。
- 上游 main tip = `4b6f2a6 "feat(deepseek-v41): optimize NPU training with fused operators"`
  （核心变化：deepseek_v41 的**融合算子**——fused permute、`.activation` 的 `silu_mul/packed_swiglu`、
  sparse attention 的 compressed-KV/TND 签名、`_depth_std` 初始化规则等）。
- 操作：从 `origin/main@4b6f2a6` 建 `dev_dsv41_one_rebased` → `git apply -3 /tmp/our_delta_fsdpturbo.patch`
  → **5 个文件 11 处冲突**，逐个人工缝合。

```bash
cd /workspace-verl/FSDPTurbo
git checkout -b dev_dsv41_one_rebased origin/main        # 4b6f2a6
git apply -3 /tmp/our_delta_fsdpturbo.patch              # 3-way，冲突见下
```

### 1.2 冲突逐项与解法（原则：**上游新结构 + 保留我方修改语义**）

| # | 文件（冲突数） | 上游侧（新） | 我方侧（delta） | 解法与依据 |
|---|---|---|---|---|
| 1 | `training/clip_grads.py` (3) | 逐设备 `non_blocking` 拷贝语义 | 我们的**流式 clip**（不再整份 fp32 物化，13GB→0 瞬态） | 保留流式实现，回插上游语义 `if local_g.device != dev: local_g = local_g.to(dev, non_blocking=True)`（证据：现文件 173/268/283/286 行）。这也是 staged stat 比原 delta **多 12 行**的原因。 |
| 2 | `models/deepseek_v41/model.py` (1) | `from .activation import silu_mul` | 我们引入 `MOE_BF16_ACT`、`DeepSeekV41Experts` | **imports 并集**：`from .activation import silu_mul` + `from .experts import MOE_BF16_ACT, DeepSeekV41Experts`（现 29-30 行）。 |
| 3 | `ops/cpu/sparse_attention.py` (3) | **新签名**：`use_sparse_flash_attn / compressed_key_value / compressed_topk_indices / compress_ratio / window_size / layout / cu_seqlens` + prologue 里 compressed-KV 索引偏移与 concat | 我们的 `_ATTN_DEBUG` 块（`FSDP_TURBO_ATTN_DEBUG=1` 逐 chunk 打印显存曲线）+ `ALIGN_ATTN_FP32`（QK 分数 / PV 概率两处 fp32 分支，`VERL_DSV41_ALIGN_ATTN=1`） | 取上游**签名 + 前导逻辑**；随后接我方 `_ATTN_DEBUG` 块；分值/PV 计算保留我们的 `if ALIGN_ATTN_FP32 / else` 双分支（else 与上游逐位一致）。格式化取上游单行。 |
| 4 | `models/deepseek_v41/experts.py` (4) | `_use_fused_permute` / `_reorder_router_weights` 新辅助；激活改 `.activation.silu_mul / packed_swiglu`；权重乘改**非原地**（注释 "Do not mutate an activation output that a fused backward may save"）；combine 改 `fused=fused_permute` | `MOE_BF16_ACT` / `ALIGN_MOE_COMBINE` 两个诊断门（`VERL_DSV41_MOE_BF16` / `VERL_DSV41_ALIGN_MOE`）；`use_eager=True`；`device=` 参数 | 取上游结构与激活算子；我们的门**织入**：`if not MOE_BF16_ACT: gate_up = gate_up.float()`；权重乘三支 `ALIGN_MOE_COMBINE→pass / MOE_BF16_ACT→act dtype / else→fp32`（全部非原地，服从上游新约束）；combine 三支都改 `fused=fused_permute`（服从 `_use_fused_permute` 注释"两对 permutation 必须同一后端"的契约——dispatch fused 而 combine eager 会破坏 unpermute 簿记）。`use_eager=True` 保留（见 1.3 取证：无生产调用点，数值无影响）。 |
| 5 | `models/deepseek_v41/adapter.py` (1) | 新增 `_IMAGE_MARKERS / _MHC_PARAMETERS / _depth_std` 初始化辅助 | （此区域我们只有空行差异） | 取上游块（`_depth_std` 被同文件 457/466/480/490 行引用，必须保留）。 |
| — | `run.sh`、`expert_parallel.py`、`tensor_swap_manager.py`、`config2.yaml` | — | — | **干净应用**。注：`config2.yaml` 是我方**新增文件**（delta 显示 "new file mode"，上游没有），内容 146 行，`dispatcher: "fused"`。 |

### 1.3 关键取证：`use_eager=True` 有没有数值影响？→ 没有（但保留我方值）

- 全仓库调用点检索：
  ```bash
  grep -rn "native_eager_forward" fsdp_turbo/ | grep -v "\.pyc"
  # → 只有 def（experts.py:200）与陈旧 __pycache__，无调用点
  ```
- 动态分发机制（`expert_parallel.py:137-154`）：`dispatcher.startswith('custom_')` 时
  `method = getattr(self, method_name)` —— 即 **只有配置写 `custom_native_eager_forward` 才会走到这个函数**。
  该配置只出现在上游示例 `examples/deepseek_v41/config.yaml` / `config_tnd.yaml`；
- **我们的生产脚本强制 `fused`**：
  ```bash
  grep -rn "dispatcher" /workspace-verl/scripts/*.sh
  # run_sft_dsv41_1node16.sh:127  "+engine.turbo_config.distributed.ep_plan.dispatcher=fused"
  # run_sft_dsv41_2node.sh:122    "+engine.turbo_config.distributed.ep_plan.dispatcher=fused"
  ```
  → `get_experts_forward_fn(..., fused=True)` → `dispatcher.py::experts_forward` → `dispatch_mlp_combine`
  （自带 permute/alltoall/grouped_matmul 实现，**不调用** `native_eager_forward`）。
- 结论：`experts.py::native_eager_forward`（以及其中的 `use_eager` 标志、以及我们织进去的
  MOE_BF16/ALIGN_MOE 门）在**生产 `dispatcher=fused` 路径下不参与计算**。
  保留 `use_eager=True`（延续我方 delta，避免再次改行产生冲突；数值上两者都不影响生产）。
  *这也解释了 09-27 一致性实验中 MOE 门"无增益"的结论——该路径本就未生效；对齐实验的有效面在
  fused dispatcher 与 `ops/cpu/sparse_attention.py`（后者确为 trainer 实际执行路径，ALIGN_ATTN 门有 40% update_actor 耗时证据）。*

### 1.4 验证（全部通过）

```bash
grep -rn "^<<<<<<<\|^>>>>>>>" --include="*.py" --include="*.yaml" --include="*.sh" . | grep -v ".stale"   # → clean
python3 -m py_compile <9 个改动文件>                                    # → 全过
python3 -c "import fsdp_turbo.models.deepseek_v41.experts, ...adapter, ...ops.cpu.sparse_attention, ...training.clip_grads"
# → OK ×4（真实 import，非仅 py_compile）
git diff --cached --stat    # vs 上游 tip：9 文件 +800/-78
git diff a251f94..7714bef --stat   # 原 delta：9 文件 +793/-95（差异仅 clip_grads +12，来源见 1.2#1；sparse_attention 少 21 行=上游已携带部分改动）
```

- 符号抽查：`MOE_BF16_ACT`/`ALIGN_MOE_COMBINE`（experts.py:18/31）、`device=device`（80/89）、
  `ALIGN_ATTN_FP32`/`_ATTN_DEBUG`/`compressed_key_value`（sparse_attention 24/48/135）、
  `non_blocking=True`（clip_grads 173/268/283/286）、union imports（model.py 29-30）全部在位。

### 1.5 提交与分支归位

```bash
git add -A && git commit -m "rebase(deepseek-v41): re-apply our delta onto upstream main (4b6f2a6)"
# → 31fd979
git checkout dev_dsv41_one && git reset --hard 31fd979
```

- `dev_dsv41_one` → `31fd979`；`dev_dsv41_one_rebased` 同点；`backup_dev_dsv41_one_20261012` → `7714bef`（旧 tip，保留）。
- 提交信息完整记录了 5 处冲突的解法与验证结论（可 `git show 31fd979` 复查）。

---

## 2. vllm-ascend（私有仓）rebase + 重编

### 2.1 状态与分支

- 最终 tip：**`0568da9db` "fix"**，其父为上游 main **`2fa829f95`**（`[CI][Misc] Run Qwen3-32B QuaRot Eagle3 nightly…`）。
  即 = 上游最新 main + 我们重放后的 fix。
- 分支映射：
  - `local_run` = `local_run_rebased` = `0568da9db`；
  - `backup_local_run_20261012` = `b2da8fb47`（旧 tip，回退用）。
- 我们 fix 的**在树证据**（marker 行修复，`vllm_ascend/attention/dsa_v41.py`）：
  ```
  275: # DSA_SCATTER_DROP_MARKERS=0 restores the pre-fix call ...
  277: _DROP_MARKER_ROWS = os.environ.get("DSA_SCATTER_DROP_MARKERS", "1") != "0"
  323: torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates.contiguous())
  ```
  文档化取证也在 docstring 里：builder 的 `[-1,-1]` marker 行会被kernel按 `uint64` 重解释坐标而
  写入 `plane_base - 254080`（device fault），故 **host 侧先丢 marker 行**（默认开）。
  > ⚠️ 交互注意：GRPO 启动器 `run_grpo_2node.sh` 显式传 `DSA_SCATTER_DROP_MARKERS=0`
  >（见其 master 分支的启动行与头注释 "kernel-side marker fix"）——即**生产方式实际走 kernel 侧守卫**
  >（本次重编取的上游 `scatter_nd_update_hp.h` 等价负索引守卫），python 丢行路径是 A/B 备选。
  > 两个路径现在都在树内、都有保护；重编 .so 已包含 kernel 守卫。
- 其他重放细节（上会话，结论保留）：kernel `scatter_nd_update_hp.h` 取**上游版**（上游已有等价的
  负索引 int32+if-guard 守卫）；`test_engram.py` 相关 cherry-pick 与上游重写冲突 → 取 HEAD + `--skip`
  （我们的语义已被上游覆盖）；`dsa_v41.py` 冲突为 import 块 → 并集解决。
- **残留 delta 实测（2026-10-12 14:45）**：
  ```bash
  git diff 2fa829f95..0568da9db --stat
  #  vllm_ascend/attention/dsa_v41.py | 45 ++++++++++++++++++++++++++++++++++++----
  #  1 file changed, 41 insertions(+), 4 deletions(-)
  ```
  即用户说的"最近两个 fix"里，**kernel 侧那一处已彻底融进上游（diff 为空）**，我们相对最新 main
  只多出 `dsa_v41.py` 的 host 侧 marker 丢行逻辑（+41/-4）。
- 旧构建产物已挪至 `csrc/build.stale-1009/`（untracked）。

### 2.2 重编与验证

```bash
cd /workspace-verl/vllm-ascend-v41-private
MAX_JOBS=64 pip install -e . --no-deps --no-build-isolation     # 日志 /tmp/vllm_ascend_rebuild.log
# → Successfully installed vllm_ascend-0.19.1rc2.dev2749+g0568da9db
#   （旧安装 0.19.1rc2.dev2427+gc1e3442d0.d20261009，Can't uninstall 属正常：editable 覆盖）
ls -la vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so   # 14:28 新 .so（旧版备份 /tmp/vllm_ascend_C.backup-20261012.so）
python3 -c "
import torch, torch_npu, vllm_ascend.vllm_ascend_C
for n in ['npu_scatter_nd_update_sk','npu_sparse_flash_mla','npu_sparse_flash_mla_metadata']:
    print('OK', getattr(torch.ops._C_ascend, n))
"
# → OK ×3
```

- 注意点（踩过）：`dir(torch.ops._C_ascend)` **不枚举**惰性注册的算子（返回只有 dunder），
  要用 `getattr(torch.ops._C_ascend, name)` 逐名解析才算验证。

---

## 3. 机器 roster（2026-10-12 14:35 实测）

探针命令（本地 131 直接跑；远端必须 `bash -lc`，否则见下方错误日志）：

```bash
ssh -p 50001 root@80.5.25.<h> 'bash -lc "hostname; npu-smi info"'
```

| 机器 | 地址 | NPU（16 卡/台） | 主机内存 | 判定 |
|---|---|---|---|---|
| **node-29-131**（本机/master 位） | 80.5.25.131 | **16/16 忙**：16×`python3`（外部作业，45–54 GB/chip，PIDs 2864624–639 一簇） | 2454GB，用 173GB | **被外部占用，不可用于本项目**（历史纪律：勿杀） |
| **node-29-128**（原 slave 位） | 80.5.25.128 | — | — | **sshd down**：`ssh: connect to host 80.5.25.128 port 50001: Connection refused`（14:35 复测仍拒连）→ 需基础设施恢复 |
| **node-25-121** | 80.5.25.121 | **16/16 空闲**（HBM 仅 ~2.9–3.1GB 驱动基线） | 用 65GB | ✅ 可用（/mnt/share 已挂：180T 盘 17T 空余；torch 2.10.0+cpu / torch_npu 2.10.0.post4） |
| **node-29-126** | 80.5.25.126 | **9/16 空闲**（chip 0,2,3,4,5,6,7 被外部作业占，~47GB/chip；chip 1,8–15 空闲） | 用 51GB | ⚠️ 部分可用（9 卡） |
| **node-29-119**（新） | 80.5.25.119 | **16/16 空闲**（同上基线） | 用 25GB | ✅ 可用（连通性 ✅；/mnt/share 已挂；环境与 121/126 同版） |

- **119 连通性测试（任务 1）**：ssh（`bash -lc`）✅、`npu-smi` ✅（16 卡全部 OK/idle）、
  `/mnt/share` ✅（172.27.1.1:/Share，91% 已用，17T 可用）、python 环境 ✅（`/usr/local/python3.11.15`，torch 2.10.0+cpu / torch_npu 2.10.0.post4，与 121/126 一致）。
- 备用（未探）：本 roster 外的机器未逐个验证。

### 错误日志（本轮遇到并已解）

| 现象 | 根因 | 处置 |
|---|---|---|
| 远端 `npu-smi: error while loading shared libraries: libc_sec.so: cannot open shared object file` | 非登录 shell 未带 CANN 运行库环境 | `ssh … 'bash -lc "…"'` 包一层 |
| `ssh: connect to host 80.5.25.128 port 50001: Connection refused` | 128 的 sshd 未运行 | 无法远程自愈，待恢复（或改用 121/119 组双机，见 §4） |
| roster 解析 `chips=0` | 正则把第 3 列 `0 / 0` 当 HBM 字段 | 改为直接匹配 `(\d+)\s*/\s*65536` |
| `dir(torch.ops._C_ascend)` 空 | torch.ops 命名空间惰性注册不参与 `dir` | `getattr` 逐名解析（见 §2.2） |

---

## 4. 后续 / 待办

1. **GSM8K RL 热启动训练**：数据已备好（`gsm8k_main_train_rl.parquet` 7473 行 / test 1319），
   启动命令已备，**等待用户指令**；原 slave 128 不可用 → 若即刻开跑需改选双机组合
   （例如 121+119 或 126 的 9 卡 + 单机方案，需按 efsdp/EP 几何重新核算）。
2. **rebase 后首次生产回归**：建议用一次短线 SFT/GRPO 冒烟验证 FSDPTurbo `31fd979` + vllm-ascend
   `0568da9db` 的组合（重点：fused permute 路径、sparse attention compressed-KV 分支、weight sync）。
3. 遗留（历史）：E3b 探针接入 host-offload engram 加载器；引擎 decode/prefill 算子级归因。

## 5. 2026-10-13 分支迁移与恢复（用户操作）

- 用户/同事在 131（及 130）恢复了旧栈（未 rebase）代码："131 130我先恢复了原来的代码"。
  实测 131 分支现状（02:27–02:48 reflog）：
  | 仓 | 恢复后（当前用） | rebase 线保存分支 | 纯 rebase 提交 | 备份分支（未动） |
  |---|---|---|---|---|
  | FSDPTurbo | `dev_dsv41_one`=`7714bef`（重建自 my_origin，"fix reward"） | `dsv41_one_rebase_main_1012`=`9a13229`（已推 my_origin，02:27:12） | `dev_dsv41_one_rebased`=`31fd979` | `backup_dev_dsv41_one_20261012`=`7714bef` |
  | vllm-ascend | `local_run`=`b2da8fb47` | `local_run_rebase_main_1013`=`2c52519ef`（已推 my_origin，02:47:56） | `local_run_rebased`=`0568da9db` | `backup_local_run_20261012`=`b2da8fb47` |
  | vllm | checkout `ced6857afa`（旧树） | —（rebase 新树 `8cc9aa5ad3`） | — | — |
- 用户指令：**不要 push**（此后未经要求不推任何仓；`*_rebase_main_101x` 两分支已在 my_origin，如不需要可删）。
- 同日 03:42 起 RL run 已在 119/126 基于旧栈启动（详见 taskA worklog §5；"130"不在本项目机器表内，未接触）。

---

*关联文档：`worklog_dsv41_consistency_v2.md`（训推一致主线）、`worklog_dsv41_sft.md`（SFT）、
`work_log/pearson_0999/worklog_p0999_r2.md`（ALIGN 门实验与 use_eager 分析）、
`dsv41_memory_and_parallel_archive.md`（显存/并行账）。*
