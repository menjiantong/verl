# 任务 A 工作记录：GSM8K 数据 RL 全流程（2026-10-12 起）

> 目标（用户 2026-10-12 指令）：跑完基于 GSM8K 的 RL 全过程，对比 reward 相对 DAPO 是否更好；
> **先观察 10 步**，OK 则续跑 50 步；否则排查。A 优先级最高。
> 规则：执行命令、错误栈、日志行、代码修改全部记录（本文件为任务 A 专属 wiki）。

---

## 0. 环境与机器

- 数据/模型（/mnt/share，两机共享）：
  - train `/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_train_rl.parquet`（7473 行）
  - val `/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_test_rl.parquet`（1319 行）
  - model `/mnt/share/m00899630/dsv41/weights/DeepSeek-V4.1-Flash-top32-sft116`（SFT-116 导出）
- 代码同步（三机 md5 对账通过）：新增通用脚本 `scripts/sync_to_host.sh <ip>`（tar-over-ssh，
  含 FSDPTurbo/ vllm_ascend/ 重建 .so / vllm/ verl/ scripts/）。
- **机器状态（当日变化很大，实时为准）**：
  | 时刻 | 131 | 128 | 119 | 120 | 121 | 126 |
  |---|---|---|---|---|---|---|
  | 14:35 | 16/16 外部占 | sshd down | 空闲 | （未加入） | 空闲 | 9 空闲 |
  | 15:30 | 16/16 外部占 | sshd down | 本项目 master | 14 空闲 | **被邻居 16 卡占满（~60GB/die）** | 本项目 slave |
  | 17:45 | **空闲** | **恢复、空闲** | 本项目 master | 空闲 | **邻居已走、空闲** | 本项目 slave |

---

## 1. 启动命令（第 8 次，当前在跑）

master（119）——脚本文件方式下发（避免多级引号地雷；`ulimit` 由启动器自理）：
```bash
# /tmp/launch_master.sh 内容（下发：ssh … 'bash -lc "bash -s"' < /tmp/launch_master.sh）
ulimit -n 65536 2>/dev/null || true
export PATH=/usr/local/python3.11.15/bin:$PATH
cd /workspace-verl
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_train_rl.parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/gsm8k_main_test_rl.parquet
export MODEL_PATH=/mnt/share/m00899630/dsv41/weights/DeepSeek-V4.1-Flash-top32-sft116
export EFSDP_SIZE=1
export ROLLOUT_IS=token
nohup bash scripts/run_grpo_2node.sh master 5 > /tmp/grpo_gsm8k_master.log 2>&1 &
```
slave（126）——**固定空闲 die**（邻居曾抢占，pick_devices 选择不可信时人工 pin）：
```bash
# /tmp/launch_slave126.sh：export ASCEND_RT_VISIBLE_DEVICES=1,8,9,10,11,12,13,14（mem_get_info 预验证 61GiB 空闲）
nohup bash scripts/run_grpo_2node.sh slave 80.5.25.119 > /tmp/grpo_gsm8k_slave.log 2>&1 &
```
关键实测数字：每步 ≈ 392–440s（gen≈284s 主导，dapo 20 步 run 的 timing_s/step 统计）。

---

## 2. 排障全史（8 次尝试，6 个独立根因，全部有证据）

### #1 挂死：head raylet fd 饱和（1024/1024）
- 现象：driver 卡 "Connected to Ray cluster" 后 14 分钟零输出；无 actor 进程；NPU 全闲。
- 证据：`/proc/<raylet>/fd` = **1024**（软限制值）；`raylet.err` 满屏
  `E1012 … "File descriptor limit reached. Retrying."`（每 2s 一轮，14:50 起）；driver py-spy 栈
  `main_ppo.py:93 → ray.get(...)`；从机 raylet 仅 58/1024（head 特有）。
- 修复：`scripts/run_grpo_2node.sh` 与 `run_grpo_top64.sh` 顶部加 `ulimit -n 65536`（硬限 1073741816，可提）；
  重启后 head raylet fd 1274 < 65536，0 条 fd 错误。

### #2 从机（121）整树暴毙：邻居作业"扫杀 python"
- 现象：worker group 创建后 3 分钟，121 上全部 ray 进程（raylet+agents+workers）同时变僵尸
  （`[raylet] <defunct>` ×2 代）；GCS 记 `DEAD … missing too many heartbeats`；driver `ActorDiedError`。
- 排除项：主机无 OOM（dmesg 干净）、cgroup 无限制且 `oom_kill 0`、无 core、fd 仅 115、
  无外来 ray 进程（ps 私有 PID ns）。
- 结论（时间线吻合）：同时刻（15:12）121 的**另一个容器启动 16 卡作业**，其清理/抢占动作（典型
  `pkill -9 -f python` 类）把同机所有 python 进程灭掉——全树进程均 python、sshd 独活，特征唯一。
  15:18 起邻居 16 卡作业（16×~56.9GB）在 121 上逐卡铺开，15:29 占满 16 die。
- 处置：弃用 121；从机换 126。

### #3 121 邻居占卡 → 引擎构建 NPU OOM
- 证据：`torch.OutOfMemoryError: NPU out of memory. Tried to allocate 24.00 MiB (NPU 3; 61.28 GiB total; 5.53 GiB allocated; 10.04 MiB free)`；同刻 npu-smi 121 全卡 59–60GB 被外来进程占。
- 处置：换 126，并**人工 pin `ASCEND_RT_VISIBLE_DEVICES=1,8,9,10,11,12,13,14`**（先 `torch_npu.npu.mem_get_info` 逐卡验证 61GiB 空闲再拉）。

### #4/#5 vLLM 引擎启动崩（三层，逐层剥开）
1. `adapt_patch() → patch_qwen3_5 → ops/gdn.py:23 → from fla_npu.ops.ascendc import …` **ModuleNotFoundError: No module named 'fla_npu'**
   - fla_npu 全机不存在、PyPI/华为云镜像无、repo 未声明依赖（上游隐含）。
   - **修1**（vllm-ascend `e565028e2`）：patch/worker/__init__.py 三个 qwen patch 改 try-import；
     仅当 `exc.name == "fla_npu"` 才跳过（其它缺模块照常 raise）。
2. 剥开后：`aclgraph_utils.py → from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens` **ImportError**（vllm 树太旧）
   - 依据 vllm-ascend `.github/vllm-main-verified.commit`（pin），vllm 树升级：checkout **8cc9aa5a**
     （本地对象库已有；本地构建类改动 stash 保存）。vllm 为 `.empty` 纯 python 构建（无 .so，升级免编译）。
3. 再剥开：`patch_step3p5.py:67 from vllm.model_executor.models.step3p5 import FP32ReplicatedLinear` **ImportError**
   - 该行处于 `if vllm_version_is("0.30.0"):` 门；门函数**优先读 env `VLLM_VERSION`**；
     启动器久留 `export VLLM_VERSION=0.30.0` 强行把门判成"老 vllm 分支"。
   - 新 vllm 三处都是主线形态（`aux_output_config` 在、`device_index` 已改名、`GateLinear` 在）→
     **修2**：删启动器该 export（改 `unset VLLM_VERSION` + 注释），让真实版本（0.30.1.dev0+…）驱动门。
- 验证手法：本机（131）`PYTHONPATH=… python3 -c "import vllm_ascend.patch.worker"` 逐层复现/复验，
  远端 126 同测 `REMOTE BOOT-SITES OK`。

### #6 剥到第二处 fla_npu：`register_ascend_customop`
- 栈：`worker.py → register_ascend_customop → from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention → fla_npu`
  （非 310P 硬件的 else 分支，无条件导入）。
- **修3**（vllm-ascend `557b411cd`）：同风格 try-import；缺 fla_npu 则跳过注册
  （GDN 仅 Qwen3.5/Kimi 用，DeepSeek-V4.1 不用；上行 fallback 兜底）。本机 `register_ascend_customop()` 复验通过。

### #7 训练侧注意力调用崩：**rebase 合并缺口（我方）**
- 栈：`actor_rollout_compute_log_prob → … → model.py:1101 indexed_sparse_attention → dispatch_op →
  ops/npu/sparse_attention.py:642 → indexed_sparse_attention_torch(… layout=layout, cu_seqlens=cu_seqlens)`
  → `TypeError: indexed_sparse_attention_torch() got an unexpected keyword argument 'layout'`。
- 根因：上游给 `_indexed_sparse_attention_queries`（私有）与**注册版** `indexed_sparse_attention_torch`
  都加了 `layout/cu_seqlens`；rebase 时该区域无冲突、取了我们的旧签名 → 两处调用方（npu 包装器）全部传新参 → 崩。
- **修4**（FSDPTurbo `214db7c` on `dev_dsv41_one`）：注册包装器补 `layout="BSND", cu_seqlens=None` +
  BSND 校验（与上游逐字对齐），保留我们的 compressed-KV 预合并与 chunk 路径。CPU 冒烟：
  BSND 正常产出、TND 正确 `NotImplementedError`。

### #8（当前）状态
- 17:36 集群形成（16 NPU），worker group → 权重/engram 加载中。
- 预期链路：引擎加载（~10min）→ rollout warmup → **生成 8×8 样本（gen≈284s）** → log_prob（#7 死点，已修）→
  advantage → update → **step:1 指标（含 reward）**。

---

## 3. 记录要点（供后续任务复用）

- 远端命令一律 `bash -lc`（非登录 shell 缺 CANN 环境：`libc_sec.so` 报错）；**引号复杂时用
  `ssh … 'bash -lc "bash -s"' < /tmp/script.sh` 下发脚本文件**（多级引号两次踩雷）。
- 观察者：`/tmp/watch.sh`（20s 粒度 raylet fd/存活 + 主机内存）部署在 119/126；
  注意 `pgrep -c python3` 不算 python3.11 的 worker（用 `ray::WorkerDict` / cmdline 匹配）。
- 卡上诊断以 **`torch_npu.npu.mem_get_info(dev)`** 为准（npu-smi 是按 card 编址，非 die 编址，易误读）。
- vllm 版本字符串是构建期戳（`…+gced6857af.d20261009`），**与树内容无关**；
  `dir(torch.ops._C_ascend)` 不枚举惰性算子（用 getattr）。

## 4. 第 8 次 run 的判决与三轮判别实验（2026-10-12 傍晚-夜）

### 4.1 现象：全 -1 + 乱码（第 8 次 run，119+126）
- step:1-3 全为 `critic/score/mean:-1.0`、`response_length/mean:962-976`（大量贴 1024 上限）、
  `rollout_actor_probs_pearson_corr:nan / diff_mean:nan`、`grad_norm:nan`。
- **agent dump（`/tmp/agent_dump.txt.<pid>`，VERL_AGENT_DUMP）实证引擎输出=乱码词沙拉**
  （`"Mus odingruwerom..."`——与 efsdp=2 权重损坏同型；另有样本直接 EOS，rlen=1）。
- **关键逻辑**：step 1 时训练权重 == 引擎初始权重（都=SFT116），**sync 即使坏了也不该影响 step1**，
  除非引擎"直载后又被改坏"。→ 指向引擎侧自身或 sync 把引擎权重改坏。

### 4.2 判别轮 1：引擎栈自身（serve 直测）→ **引擎栈无罪**
- 120 上以当前栈直服 SFT116（8 卡 TP8）：
  - 坑1：`vllm serve` cwd=/workspace-verl 时 `import vllm` 命中间名仓库目录（namespace 包遮蔽，
    `ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)`）→ **必须 cd 到非仓库目录启动**。
  - 坑2：新校验 `enable_return_routed_experts only supported by V2 runner`（旧栈因 0.30.0 版本门跳过）→ 测试版去掉该 flag。
  - 坑3：自写启动脚本覆盖 PYTHONPATH 丢了 CANN pyACL（`No module named 'acl'`）→ 必须**追加式** `${PYTHONPATH:-}`。
- **结果：生成完全正常**（rl_endthink temp0 = 2/2 正确；temp0.9 = 1/3 但文本连贯；"Answer: 3" 等）。
- 判别轮 2：**RL 配置复刻**（rl_config.enabled/enable_batch_invariant/mc2 fullmesh_v2/关 npugraph）→ 同样正常。
  ⇒ 引擎推理栈 + RL 配置无罪。

### 4.3 判别轮 3：LOCAL_EXPERT_EXPORT 的**自然 A/B**（第 9 次 run 的意外收获）
- 第 9 次 run 里 master(119) 收到 `LOCAL_EXPORT=0`（全量导出：**4724 tensors, 99.70 GiB, 203 buckets |
  export 836.2s**——即 G13 注记的 ~920s 级 slow path 实测），而 126 从机未收到该 env（走 LOCAL=1：
  1364 tensors, 25.87 GiB | export 1.0s）→ 两个 node-local 引擎副本正好一全量一本地。
- **结果：两侧 agent dump 均乱码**（119 全量侧乱码；126 本地侧乱码）。
  ⇒ **LOCAL export 不是（唯一）腐坏源**；嫌疑收窄到公共下游：**接收/装载** 或 **colocate sleep-wake**。

### 4.4 判别轮 4（进行中）：`VERL_SKIP_WEIGHT_SYNC=1`
- 新调试开关（verl `workers/rollout/vllm_rollout/utils.py`，env 门控，默认关）：
  收包并回 ACK（保持 ZMQ 握手）但**跳过 `_update_weights`** → 引擎保持自载权重。
- **判读**：step1 生成恢复正常 → sync 路径腐坏（继续在接收/装载打桩定位）；
  仍乱码 → **colocate sleep-wake 路径**腐坏引擎权重（新栈回归）。
- 命令：master `VERL_SKIP_WEIGHT_SYNC=1 VERL_SYNC_PROFILE=1`（119，3 步）+ slave 同 env（126，pin die）。

### 4.4b 判别结果与后续轮次（21:0x 更新）

| 轮 | 实验 | 关键证据 | 判决 |
|---|---|---|---|
| 9 | LOCAL_EXPORT 自然 A/B（119 全量 4724t/99.7GiB/836s vs 126 本地 1364t/25.87GiB/1.0s） | 两侧 agent dump 均乱码 | LOCAL 导出无罪（全量与本地都坏） |
| 10 | `VERL_SKIP_WEIGHT_SYNC=1`（收包不回填）| **receiver 日志 `load 0.0s` 实证 skip 生效**（正常 8.6s）；生成仍乱码 | **权重同步/回填彻底无罪**（引擎保持自载 SFT116 仍坏） |
| iso | serve+`--enable-sleep-mode` 隔离：生成 A → sleep?level=1 → wake_up → 生成 B（同一 prompt/temp0） | **A==B 逐字符相同**（"Answer: 72" 正确且一致） | **纯 sleep/wake 往返无损**（无训练侧共存时） |
| 11 | **`FREE_CACHE_ENGINE=False`**（recipe 改 env 驱动 `${FREE_CACHE_ENGINE:-True}`，A-5 运行中） | — | 待判：若转好 → **colocate 共存（引擎睡着时训练侧同卡活动）是腐坏点** |

**额外发现（NZ 守卫）**：serve 不带 `rl_config` 时 `wake_up` 直接 500：
`FRACTAL_NZ mode is enabled... precision issues in the RL scenarios. Please set weight_nz_mode=0`
→ vllm-ascend 自己对 NZ×RL 有精度告警（RL 已强制 nz=0）；NZ 路径是历史雷区（见下条"训练侧"）。

**当前嫌疑排序**：① colocate 共存（引擎 `sleep(level=1)` 后只 offload "weights" 池；
   非池内 device 常驻物（DSA Hadamard、元数据等）由训练侧在同卡大规模分配时可能被踩）；
   ② rollout 适配层（vLLMReplica/async server）。

### 4.4c 根因锁定（2026-10-12 夜，多轮判别后）

**A 的乱码根因 = 引擎 V4.1 稀疏注意力在长上下文处产出 NaN → 采样塌缩成 token 0。**

证据链（全部实测）：
1. RL dump 的 `RESP-LAST-IDS` = **[0,0,0,0,0,0]**（token 0=BOS 连发）——不是"权重随机"而是"分布塌缩"。
2. 引擎 dump（4 层 fixture）len=1500：**layers.2.self_attn 是第一个 NaN 算子**
   （nan=7,674,880；其 input_layernorm/compressor.norm 全 0）；trainer 同长度 **0 NaN**；
   len=700 引擎也全净 → NaN 在 700~1500 间随长度爆发（layer2/3=带 compressor 层）。
3. 修复注意力内部钩子（扩展里 `DeepseekV41EagerAttentionImpl` 已改名 `AscendDSAV41Impl`，
   导入失败被静默跳过——钩子从未生效；已修并推送）后逐函数定位（len=1500）：
   ```
   multistream_preprocess  nan=0      ← q/kv 投影+rope 干净
   compressor.norm         nan=0
   _write_compressed_source nan=0
   _select_sparse_indices  nan=0      ← 索引选择干净
   _forward_attention      第一批 NaN ← ★ 出生点（输出 6.14M/7.67M NaN）
   ```
   `_forward_attention` 内部 = **`torch.ops._C_ascend.npu_sparse_flash_mla`**（含 compressed 路径：
   cmp_block_table/cmp_sparse_indices/cmp_mask_mode=3/cmp_ratio）或 `backend.qsmla`。
4. **与旧栈对比**：10-11 的 TIS-20 run（旧 vllm-ascend）`response_length/mean≈962-976`、有 +1 奖励
   → 旧引擎在 1000 token 级不 NaN ⇒ **长上下文 NaN 是 rebase（新引擎栈）引入的回归**。

**为什么 serve 测试没发现**：serve 生成 ≤256 token，未达 NaN 阈值长度。

### 4.4d 已被排除的假设（每项都有实验）
| 假设 | 实验 | 结论 |
|---|---|---|
| 引擎栈自身坏 | serve（含 RL 配置复刻）直生成 | 正常（temp0 2/2 对） |
| 权重同步/回填 | `VERL_SKIP_WEIGHT_SYNC=1`（receiver load 0.0s 实证生效） | 仍乱码 → 无罪 |
| LOCAL 专家导出 | 自然 A/B（全量 99.7GiB vs 本地 25.87GiB） | 两侧都坏 → 无罪 |
| sleep/wake 舞蹈 | serve + sleep?level=1 + wake_up 隔离（A==B 逐字符相同）；`FREE_CACHE_ENGINE=False` | 均无损/仍乱码 → 无罪 |

**下一步（待办）**：① 用最小合成输入隔离 `npu_sparse_flash_mla` 内核本身 vs dsa_v41 的 cmp 元数据组装
（谁在小长度 700→1500 间踩雷）；② 若内核：对照上游仓库该内核近期 commit；③ 若元数据：
查 `AscendDSAV41MetadataBuilder`/cmp_block_table/cmp_residual 在长序列下的边界；④ 修后重跑本任务。

### 4.5 场外记录
- **131+128 组合被证实为不可用对**：model-load 阶段卡死在跨机集合通信（all_gather_object，
  py-spy 双方均 idle 于分布式调用；与 2026-10-08 记录的 "cross-machine gloo broadcast deadlock" 同型）。
  **两只机各 8 卡独立可用**，但不要作为 2node 对（至少当前栈）。
- 131 侧外部 16 卡作业 17:40 前后退场；128 sshd 恢复；121 被邻居 16 卡占满（当天持续）。

---

## 5. 第 10 次 run：回退旧栈（未 rebase）直接启动（2026-10-13 凌晨）

> 用户指令（03:3x）："126 119 目前机器空闲的，抓紧启动下verl gsm8k的RL全流程。基于未rebase的分支" + "直接启动"。
> 131 侧代码已由用户/同事恢复为旧栈（dev_dsv41_one=7714bef / local_run=b2da8fb47 / vllm=ced6857af；分支迁移明细见 rebase worklog §5）。

**启动前核对（119/126，md5 全部匹配旧栈 /tmp/oldstack_\*）**：
- FSDPTurbo：adapter `42b3eaaa`（=7714bef）、clip_grads `a0f684f9`、experts `f90aaec7`、sparse_attention `13ecdd64`、model `23f3f099`
- vllm-ascend：dsa_v41 `a00e6ca0`（=b2da8fb47）、`.so 4d3f7e77`、`_build_info.py`（A3）；vllm 树 `ced6857afa`
- `scripts/run_grpo_2node.sh` `ff7c5b86`（含 `VLLM_VERSION=${VLLM_VERSION:-0.30.0}` —— 旧栈必需）
- 注：机器上 `git log` 头不反映实际内容（tar 文件级同步未动 .git），判据以上面 md5 为准。

**启动（与 10-12 晚同款）**：
- master(119)：`ssh …119 'bash -lc "bash -s"' < /tmp/launch_master_old.sh` → 03:42:22 precheck ok，ray head 起；
  driver `python3 -m verl.trainer.main_ppo … trainer.total_training_steps=10`（pid 1383083）。
- slave(126)：`< /tmp/launch_slave_old.sh`（ASCEND_RT_VISIBLE_DEVICES=1,8,9,10,11,12,13,14）→ 03:48:41 joined（16 NPU）。
- run log：`/workspace-verl/verl/logs/grpo_top64_steps10_20261013_034327.log`；03:50 数据过滤完成（7473 行）。

**监控**：131 后台 watcher `/tmp/watch_steps10.sh`（60s 粒度 → `/tmp/a3_watch_steps10.log`；每 ~10min 快照日志到 `gsm8k_logs_from_119_126/`）。
**风险**：126 raylet 持续报 /tmp 盘 95% 警告（~204GB 可用，仅 spilling 才受影响）；上次 01:14 的 driver 外部 SIGKILL 原因未明，watcher 继续盯 `Killed` 行。

### 5.1 03:42 run 首崩与修复（04:0x–04:2x）

- **现象**：03:42 run 到 `TaskRunnerV1` 初始化时崩：
  `verl/workers/rollout/vllm_rollout/vllm_async_server.py:74 _VLLM_VERSION = version.parse(vllm.__version__)`
  → `packaging.version.InvalidVersion: Invalid version: 'dev'`（TaskRunnerV1 在 126；driver 04:02 打印完整栈后退出）。
- **根因**：旧栈 `vllm` 树缺 **setuptools_scm 生成文件 `vllm/_version.py`**（该文件不入 git；
  `version.py` 的 import 失败后 fallback `__version__="dev"`，而新版 `vllm_async_server` 会对它跑 `version.parse`）。
  131 上有该文件（`/workspace-verl/vllm/vllm/_version.py`，588B，`0.30.1.dev0+gced6857af.d20261013`），119/126 的树在历次
  "rm -rf 包目录 + tar 解压" 同步中丢失了它。
- **修复**：把 131 的 `_version.py` 推到两机同名路径（md5 `f659c379…` 三机一致）：
  ```bash
  ssh …119 'cp -f /tmp/grpo_gsm8k_master.log /tmp/grpo_gsm8k_master_0342_crash.log && cat > /workspace-verl/vllm/vllm/_version.py' < /workspace-verl/vllm/vllm/_version.py
  ssh …126 同款（并保留 slave 日志 …_0342_crash.log）
  ```
- **验证**（两机，login shell + run env）：
  `import vllm` → `0.30.1.dev0+gced6857af.d20261013`；`version.parse` ok；`import vllm_ascend` ok；
  崩溃模块 `from verl.workers.rollout.vllm_rollout import vllm_async_server` → `_VLLM_VERSION=0.30.1.dev0+…`，EXIT=0。
- **崩溃日志已拉回 131**：`gsm8k_logs_from_119_126/119_master_0342_crash.log`（1388 行）、
  `119_runlog_0342_crash.log`（1346 行）、`126_slave_0342_crash.log`（29 行）。

### 5.2 修复后重启（04:14–04:16，第 11 次）

- master(119)：04:14 precheck ok → Ray head；slave(126)：04:15 joined（2 nodes / 16 NPU）；
  04:15:43 driver 起：`grpo_top64_steps10_20261013_041543.log`（steps=10，pid 1468568）。
- watcher 指向新日志重挂（`/tmp/watch_steps10.sh` → `/tmp/a3_watch_steps10.log`）。
- 待观察：worker group → 加载（direct shard load）→ rollout（gen≈284s）→ **step:1（reward）**。

### 5.3 04:30 camem 二次崩溃与修复（libvllm_ascend_kernels.so 缺失），04:41 重启（第 12 次）

- **现象**：04:30:12 vLLM EngineCore 崩（119 `vLLMHttpServer pid=1515792`/`Worker_TP0_EP0 pid=1518138`；126 04:35:43 重试同款 `pid=248762`）：
  `multiproc_executor.py:808 wait_for_ready → raise e from None → Exception: WorkerProc initialization failed …`，栈被截。
- **完整根因栈**（拉全后）：
  `vllm_ascend/worker/worker.py:842 load_model → with context, set_current_vllm_config(...)`
  → `camem.py:282 use_memory_pool → camem.py:107 → camem.py:97 init_module(python_malloc_fn, python_free_func)`
  → **`TypeError: 'NoneType' object is not callable`**。
- **决定性 warning**（run log）：`camem.py:67 WARNING Failed to import vllm_ascend_C:libvllm_ascend_kernels.so:
  cannot open shared object file: No such file or directory. Sleep mode will be disabled.`（两机、11x）。
  根因：119/126 的 `vllm_ascend_C.cpython-311-aarch64-linux-gnu.so`（`4d3f7e77`，01:08 同步）NEEDED
  `libvllm_ascend_kernels.so`（RUNPATH `$ORIGIN:$ORIGIN/lib:…`），但该库未随同步带过去；camem.py 顶层
  `except ImportError` 把 `init_module` fallback 成 None → 引擎 load_model 一进 memory-pool 就 TypeError。
  131 现役树（03:57-03:58 重编旧栈）有完整 `vllm_ascend/{,lib/}libvllm_ascend_kernels.so`（`5acd0c78`）。
- **修复**：推 kernels 库到两机 `vllm_ascend/` 顶层 + `lib/`，md5 `5acd0c789348becdc5c39ca6381cd88b` 三机一致
  （注意：同一条 ssh 里两个 `cat > file` 只能吃到一次 stdin，第二份要单独再推一次）。
- **验证**（两机 login shell + run env 追加式 PYTHONPATH）：
  `import torch_npu; import vllm_ascend; import vllm_ascend.device_allocator.camem` →
  `camem_available: True`、`init_module: <built-in function init_module>`（符号兼容，dlopen 成功）。
- **验证方法两坑**（供排查复用，均非 run 问题）：① 非 login shell 缺 CANN → `import torch_npu` 报 `libhccl.so: cannot open`；
  ② `PYTHONPATH=…` 前缀式会覆盖 CANN 的 python 路径 → `camem.py: from acl.rt import memcpy → No module named 'acl'`；
  必须 `export PYTHONPATH="…:${PYTHONPATH:-}"` + `bash -ls`。
- **清理**：ray stop --force + pkill（119: 63 进程 → 余 8 个 `Z` 僵尸；126: 45 → 余 6 个僵尸；PPID=1 无法收割、无害）；NPU chips-busy=0。
- **崩溃证据保全**：119 `/tmp/grpo_gsm8k_master.log` → 另存 `…_0430_camem.log` → 拉回
  `gsm8k_logs_from_119_126/119_master_0430_camem_crash.log`（1,517,831 B）。
- **重启（第 12 次，04:41–04:46）**：master(119 `pid=1538969`) → Ray head；slave(126 `pid=254326`) → 04:46:30 joined
  （2 nodes / 16 NPU）；driver 04:41:22 起：`grpo_top64_steps10_20261013_044122.log`；watcher 重挂（`pid=399332`）指向新日志。
- **教训**：文件级同步（tar/解压）后必须核对**动态库依赖**：`readelf -d <so> | grep NEEDED` + `ldd`，不能只比主 .so md5；
  131 树的 `vllm_ascend/lib/`、`_cann_ops_custom/…` 等伴随目录一并同步。

### 5.4 05:01 自定义算子（HcPre）缺失致 rollout worker 崩 + 全树对照补齐，05:03 重启（第 13 次）

- **现象**（run #12 `grpo_top64_steps10_20261013_044122.log`，05:01:47）：vLLM rollout worker 死：
  `EngineCore … multiproc_executor.py:437 raise RuntimeError: Worker failed with error 'aclnnHcPre or aclnnHcPreGetWorkspaceSize
  not in libopapi.so, or libopapi.so not found.'` → `multiproc_executor.py:314 Worker proc VllmWorker-1 died unexpectedly
  (exit code: None), shutting down executor`。camem 一关已过（无 sleep-mode warning；worker group 正常建立）。
- **根因**：`aclnnHcPre` = vllm-ascend **自定义算子**（`_cann_ops_custom/vendors/custom_transformer/`：`op_proto/`、`op_api/lib/libcust_opapi.so`、
  `op_impl/`…；由 `vllm_ascend_C.so` 的 RUNPATH `$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib` 解析）。
  119/126 上 `_cann_ops_custom/` 只有 `.gitkeep` —— **整个算子包（1303 文件 / 71MB）没同步**（该目录是 gitignore 构建产物）。
- **全树对照**（新增判据脚本 `/tmp/treecmp.sh`（filelist）+ `/tmp/md5cmp.sh`（内容），131 为基准）：
  | 树 | 119/126 结论 |
  |---|---|
  | vllm_ascend/ 子目录 | 缺 `_cann_ops_custom` 1303 + `include/` 11 + `_version.py` 1（131=2005 vs node=691）|
  | vllm-ascend 全树 | 除 csrc/build/**（131 构建中间物）、egg-info、已知 .so 外，Python 全一致 |
  | FSDPTurbo | 文件集除 docs/tests/examples 9 个（node 多），**运行相关内容差异=0** |
  | vllm | 全一致（仅 pyproject.toml 不同，安装期文件） |
  | verl | 核心全一致（差异仅 logs/、wiki、dsv41_consistency 2 脚本） |
  | scripts | 全一致 |
- **修复**：tar 推 `_cann_ops_custom/ include/ _version.py` → 两机 `find _cann_ops_custom -type f` = 1304；
  `libcust_opapi.so` `23045ddd`（=131）、`_version.py` `fcda8364`（=131）。
- **验证**（两机 login shell）：`ldd libcust_opapi.so` 无 not-found；`ctypes.CDLL` 加载 ok；`hasattr(lib,'aclnnHcPre')/('aclnnHcPreGetWorkspaceSize')` 均 True。
- **重启（第 13 次，05:03）**：master(119 `pid=1613328`) → Ray head；slave(126 `pid=267207`)。
- **教训（升级版）**：克隆旧栈到新机后，除 filelist 对照外必须做 **md5 内容对照 + .so 依赖（ldd/NEEDED）验证 + 算子包符号 smoke**；
  被 gitignore 的生成物/编译产物是重灾区（`_version.py`、`libvllm_ascend_kernels.so`、`_cann_ops_custom/**`、`include/**` 已先后抓到）。

### 5.5 run #13 全流程贯通：step:1 reward 数值出现（2026-10-13 05:2x–05:40，里程碑）

修复 custom ops 后 run #13（`grpo_top64_steps10_20261013_050331.log`）一路推进，**首次越过此前全部阻塞点**：

- 05:20 引擎 spawn（`VLLM::Worker_TP×8`），无 `WorkerProc initialization failed`、`sleep-mode` 警告 0 次、`aclnnHcPre` 0 次 → **三项修复（camem/kernels/custom-ops）全部生效**；
- 权重加载：`npu-smi`（须 `bash -lc`）HBM 46–47 GiB/卡；worker 栈实测 `execute_model → deepseek_v41 forward → dsa_v41_forward → scatter_cache_sk`（真前向）；
- **SYNC-PROFILE**（actor→rollout 权重同步）：sender 1364 tensors / 25.87 GiB / 51 buckets，receiver load ~9 s、total ~13 s ×16 ranks；`replica.sleep` 正常返回；
- Training Progress 条出现：1/10 [08:05] → 2/10 [14:38<57:28, ~431–486 s/it]；
- **step:1 关键数字**（reward 有数值，与 10-10 "恒 -1" 决裂）：
  - `critic/rewards/mean -0.59375`（min -1.0 / max +1.0）；`critic/score/mean -0.59375`
  - `response_length/mean 99.59`（max 1024 / min 14）、clip_ratio 0.03125、`response/aborted_ratio 0.0`
  - `training/rollout_actor_probs_pearson_corr 0.9604`、`rollout_corr/rollout_is_mean 0.9137`、`log_ppl_diff 0.2468`
  - `timing_s/gen 353.4 / old_log_prob 8.8 / update_actor 90.4 / update_weights 16.5 / step 485.4`；`perf/throughput 1.724`
  - `actor/perf/max_memory_allocated_gb 37.5 / reserved 54.7`；`cpu_memory_used_gb 1671.8`
  - `global_seqlen/mean 836.9`、`training/num_turns/mean 2.0`（每 prompt 2 turns）
- 错误计数 0（`aclnnHcPre|Traceback|Killed` 全部 0）。后续 2–10 步继续按 ~7–8 min/步观察 reward 趋势（任务 A：与 DAPO 对比）。

### 5.6 用户指令停止 run #13 + 建"运行失败库"（2026-10-13 06:1x）

- **用户指令**：① 停止运行，把所有进程杀掉；② 把遇到的错误/错误栈/修复方式建成分类错误库（内存、算子、显存等），方便以后查询。
- **停止与清场**：watcher 停止（`pid 408271`）；119/126 `ray stop --force` + `pkill VLLM::/main_ppo/vLLMHttpServer/ray::/raylet` →
  两机**非僵尸活进程 = 0**（余 PPID=1 的 Z 僵尸，无害）、`npu-smi`（须 `bash -lc`）无占用。
- **run #13 停止时状态**：`6/10 [39:17<25:06, 376.67s/it]`（第 6 步 metrics 未及落盘即停止）。reward 前 5 步（落盘）：
  | step | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | `critic/rewards/mean` | **-0.59375** | -0.90625 | -0.90625 | -0.875 | -0.84375 |
  前 5 步均值 **-0.825**；resp_len mean 96–178（预算 1024 有富余、无顶格）。
- **任务 A 对比结论（GSM8K vs DAPO 数据）**：DAPO 基线 = 10-11 `grpo_top64_steps50_20261011_083935.log`
  （数据 `dapo_math_train_512.parquet`，同为 SFT 权重后、同 GRPO 配置）：
  - DAPO：前 10 步均值 **-0.9906**、全 34 步均值 -0.9715、最好单步 -0.875（resp 顶格 1024 截断所致）；
  - **GSM8K 前 5 步均值 -0.825（最好单步 -0.59）→ 明显优于 DAPO**（口径：数据难度/预算适配差异需计入——DAPO 长 CoT 常触 1024 上限）。
- **错误库（交付物②）**：新建 **`wiki/dsv41_run_failures_catalog.md`** —— 7 大类（依赖/同步、算子/内核、显存/内存、
  正确性/静默错误、通信/死锁、环境/工具、进程/资源），条目 F-01…F-61，每条=症状（原文日志/栈）→根因→修复（命令）→验证→来源；
  附"错误出现时的速查顺序"。本日三个同步遗漏案例（F-01/02/03）与 npu-smi 非登录 shell（F-50）、pkill 自匹配（F-53）等实踩坑全部入档。
- 后续备注：`trainer.save_freq=-1`（不存 ckpt），如续跑 10 步需从头重启（三处修复已在 119/126 持久化，重启即用）。

### 5.7 环境镜像分发：119（跑通基准）→ 131/128/120（2026-10-13 06:4x，用户指令）

- **用户指令**："把环境同步到其他容器上一份，比如131等。保证环境相同。"
- **基准选择**：**119**（唯一实测跑通全流程的字节态；与 126 关键 md5 完全一致）。基准涵盖:三处修复后的完整态
  （`vllm/_version.py`、`libvllm_ascend_kernels.so`+`lib/`、`_cann_ops_custom/**`+`include/**`+`vllm_ascend/_version.py`）+
  实测跑通的 .so 组合（`vllm_ascend_C`=4d3f7e77 + kernels 5acd0c78 + libcust_opapi 23045ddd）。
- **方法**：`/tmp/mirror_from_119.sh <target>`（tar czf over ssh；路径=scripts、FSDPTurbo、vllm-ascend-v41-private/{vllm_ascend,pyproject.toml}、
  vllm/{vllm,pyproject.toml}、verl/{verl,examples/grpo_trainer,scripts}；排除 `.git/__pycache__/*.pyc/core_*/logs/wiki`——
  **wiki 排除以保护 131 上最新工作记录**）。130 不可达（No route to host），121 邻居占用未动。
- **launch 启动器分发**：`/tmp/launch_master_old.sh`（`9d936dcc`）/`launch_slave_old.sh`（`d199bdbd`）推齐 119/126/128/120（131 为源已有）。
- **验收（`/tmp/verify_mirror2.sh`，全树逐文件 md5 对照，基准 119，6002 个运行文件）**：
  | 目标 | missing | extra | content-diff |
  |---|---|---|---|
  | 131 | 0 | 0 | 0 |
  | 128 | 0 | 0 | 0 |
  | 120 | 0 | 0 | 0 |
- **extra 隔离**（"保证相同"需要同构）：128 有 203 个、120 有 418 个源机没有的文件（rebase 时代残留：FSDPTurbo/ops/npu 新 .py、
  vllm-ascend custom-ops 开发中间物 .h/.o/.cpp、vllm .py 等）→ `mv` 到各机 `/tmp/extra_isolated/`（**可逆、未删**），逐个复核后隔离。
- **环境 smoke（三台 bash -ls + run 环境）全过**：`torch 2.10.0+cpu / torch_npu 2.10.0.post4`；
  `vllm 0.30.1.dev0+gced6857af.d20261013`；`camem_available: True`；`custom-ops HcPre: True`；`/mnt/share` 数据/权重可达。
- **系统层差异备注**：131 的 `npu-smi` = **26.2.rc1.b050**，其余（119/126/128/120）= **26.0.rc1**（驱动/工具链版本差，属机器安装层、
  软件镜像无法对齐；日常用 119/126 跑训练，131 备用时留意）。
- **结论**：131/128/120 三台达到与 119 **"可直接启动"字节级一致**；今后新机入列照 `mirror_from_119.sh` + `verify_mirror2.sh` 流程办理。
