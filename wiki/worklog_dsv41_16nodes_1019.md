# DeepSeek-V4.1 全量 763B（去 engram）· 16 机 256 卡 GRPO — 2026-10-19

## 0. 任务与机器
- 机器：80.5.21.101~114、116（15 台，容器 `dsv41-mjt-1019`）+ **80.5.25.120**（`ssh -p 50002`，容器内）
- 每台 16 卡 Ascend 910C（64GB），共 256 卡；主机 RAM 2.4~3TB
- 工作目录：`/mnt/share/m00899630/workspace/dsv41-verl-copy`（避免与 `dsv41-verl` 其他 agent 冲突）
- 启动脚本：`.../dsv41-verl-copy/verl/my_scripts/dsv41_grpo_16node_new.sh`

## 1. 模型与 noengram 权重
- 原始：`/mnt/share/DeepSeek-V4.1-Flash-bf16`，实测 **763.21B 参数**（384 routed experts，engram 196.93B，bf16 1.53TB）
- 新建：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-bf16-noengram`
  - 275 个文件 = 274 个软链（全部指向原始权重）+ 1 个 config.json
  - config.json 由**原始 config 派生**：`engram_layer_ids=[]`、`engram_num_embeddings=[]`，`n_routed_experts` 保持 **384**
  - ⚠️ 未直接拷贝 `DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3/config.json`：该 config 是 **32 专家**（top32 模型），与 384 专家全量权重不兼容；除 engram 字段外，两份 config 仅 `n_routed_experts` 不同，故从原始 config 清空 engram 字段等价且正确
- 去 engram 后参数量：566.28B（专家 557.17B + 其他 9.11B），bf16 约 **1.13TB**
- `VERL_DSV41_DIRECT_SHARD_LOAD=1` 已核实：加载计划只包含模型参数，engram 的 196GB 分片（model-00008）不会被读取

## 2. 并行策略（16 机 256 卡）
| 维度 | 值 | 依据 |
|---|---|---|
| NNODES × CPN | 16 × 16 = 256 | 16 台 × 16 卡 |
| 训练 FSDP_SIZE | 256 | dense 参数全分片 |
| 训练 EP_SIZE | 64 | 专家 64 路分片；每 rank 8.7B 专家参数（host offload） |
| 训练 EFSDP_SIZE | 1 | 与 GEN_EP 对齐以启用 LOCAL_EXPERT_EXPORT |
| TP_SIZE（训练） | 1 | FSDPTurbo 训练不用 TP |
| rollout GEN_TP | 64 | 可整除 64 个 attention heads；每 replica 64 卡（跨 4 机），共 4 replica |
| rollout GEN_DP | 1 | 每机 16 卡，16 % 64 ≠ 0，>1 不合法 |
| rollout GEN_EP | 64 | 配置断言要求 = GEN_TP × GEN_DP |

- **跨节点引擎**：`RolloutReplica` 中 `world_size=TP*DP*PP=64`，`gpus_per_replica_node=min(16,64)=16`，`nnodes=4`；node_rank>0 的 server 走 vLLM headless。这是代码明确支持的 multi-node replica 路径。E22 的失败是“跨机 DP hybrid-LB”，本方案是纯 TP，不触发。
- **权重同步**：`GEN_EP(64) == EP_SIZE(64)` 且 `EFSDP_SIZE=1` → `LOCAL_EXPERT_EXPORT=1`，trainer rank i 的 ep 坐标 = `i % 64`，engine TP rank = `i % 64`，逐卡配对，只传本 rank 的 8.7B 专家分片，避免 1.13TB 全量 all-gather。
- **显存**：引擎每卡权重 1.13TB/64 ≈ **17.7GB**，`ROLLOUT_GPU_MEM_UTIL=0.6`（38.4GB 预算）→ KV 约 15-20GB；训练侧 offload_policy 全 offload，host RAM 每节点约 1.1TB < 2.4-3TB。
- **KL 关闭**（USE_KL=0）：否则 ref 模型再占一份 1.13TB host/param，120 的 2.4TB RAM 会爆。
- **batch**：TRAIN_BSZ=32 × SAMPLE_N=8 = 256 序列 = 256 卡每卡 1 条，满足 `TRAIN_BSZ*SAMPLE_N % (NNODES*CPN)==0`。

## 3. 启动命令（待执行）
- master（101 容器内）：`bash .../dsv41_grpo_16node_new.sh master 50`
- join（其余 15 台）：`bash .../dsv41_grpo_16node_new.sh join 80.5.21.101`
- 环境检查：16 台 NPU 全部空闲（每 chip 约 3.1GB 基线）；120 已清理旧 ray/vllm 残留；版本 ray 2.59.0 / Python 3.11.15 一致；**driver 120=26.0.rc1 vs 101=26.2.rc1**（跨机 HCCL 风险点）

## 4. 监控记录

### 2026-10-19 22:51-22:54 启动
- 22:51:33 master 启动（101 容器）：
  `docker exec -d dsv41-mjt-1019 bash -lc "cd .../dsv41-verl-copy/verl && nohup bash my_scripts/dsv41_grpo_16node_new.sh master 50 > logs/16node_master_101.nohup.log 2>&1"`
- 22:52:27 其余 15 台 join（102-114、116 容器内 + 120 容器内），均使用 `join 80.5.21.101`（不 pkill）
- 22:53 Ray 状态：**Active 16 nodes**，`256.0 NPU` 全部 reserved；各节点 16 NPU / 640 CPU
- 22:53:12 master 进入 `main_ppo`，配置回显确认：FSDP=256, TP(train)=1, EP=64, EFSDP=1, GEN_TP=64, GEN_DP=1, GEN_EP=64, USE_KL=0, bsz=32×8，`rollout_is=token`
- 22:54 数据集过滤完成（train 7473 / val 1319），开始创建 worker group `actor_rollout`
- 主日志：`.../verl/logs/DeepSeek-V4.1-Flash-bf16-noengram-20261019_145312.log`
- 非致命告警：120 的 raylet 报 `/tmp/ray/... over 95% full, available 688GB / 15251GB`（对象存储默认约 186GB，暂不影响；持续观察）

### 2026-10-19 23:00-23:05 健康巡检（5 分钟一次）
- 巡检脚本：`depoly_log/inspect_1019.sh`（Ray/driver + 日志增长 + 16 台 NPU）；cron `*/5`，第一个 step 后改 `*/10`
- 23:00 快照：`active_nodes=16/16`、`driver_pids=1`、日志持续增长（3360B/125s）、无 Traceback/OOM/HCCL 错误
- 阶段：256 个 WorkerDict 已全部 Gloo 连通，FSDPTurbo 正在构建模型/加载权重
- NPU：16 台均 16 chip，max_chip≈6.5GB（基线 3.1GB，worker 已开始分配）；120 的 npu-smi 需补 `driver/lib64` 到 LD_LIBRARY_PATH（已修正巡检脚本）
- 114 曾瞬时只报 13 chip，复查 16 chip 全 OK（查询抖动，非故障）
- 已知 warning（非致命）：`expert_parallel.py:43 Cannot create tensor with internal format`（重复 243×，ND 格式回退）；AscendC attention/KDA 不可用 → eager 回退

### 2026-10-19 23:05 【崩溃】全量模型 rollout 无法放下 — TP 上限 8
**现象**：23:05 driver 退出，NPU 回落到基线；日志 3.5MB 停在 vLLM engine worker 初始化。
**根因栈**（`logs/DeepSeek-V4.1-Flash-bf16-noengram-20261019_145312.log:13470+`）：
```
Worker_TP33_EP33 ERROR multiproc_executor.py:944 WorkerProc failed to start.
  ... vllm_ascend/worker/worker.py:843 load_model
  ... vllm_ascend/worker/model_runner_v1.py:4383 get_model
  ... vllm_ascend/ops/linear.py:132 process_weights_after_loading
      layer.weight.data.view(layer.n_local_groups, layer.o_lora_rank, -1)...
RuntimeError: shape '[0, 1024, -1]' is invalid for input of size 524288
```
**根因**：
- `vllm-ascend/models/deepseek_v41/model.py:376` 与 `ops/linear.py:477`：`n_local_groups = o_groups(8) // tp_size`。
- GEN_TP=64 → `n_local_groups=0` → wo_a 3D reshape 失败。**vLLM 引擎 TP 必须 ≤ 8（且整除 o_groups=8）**。
- TP=8 时每个 replica 用 8 卡装完整 noengram 权重 1.13TB → **141GB/卡**，> 64GB HBM，物理上放不下。
- `verl/workers/config/rollout.py:319-323`：vllm/sglang/trtllm 的 `pipeline_model_parallel_size > 1` 直接 `NotImplementedError`，无法用 PP 切层。
- vllm-ascend 无通用权重 `cpu_offload_gb`（仅 engram 表有 host offload）。
**结论**：全量 384 专家 763B（去 engram 566B）在当前软件栈上无法作为 rollout 模型运行。与 EP/FSDP/显存参数无关，是 TP 分组结构 + PP 未实现的硬限制。
**待决策**：(A) 打补丁支持 PP=4（TP8×PP4=32 卡/replica，权重 35GB/卡；但 verl PP 同步路径未验证、generic weight sync 预计每步数十分钟~小时）；(B) 改用 top32 68B 模型（已验证，可立即跑）；(C) 先把全量模型裁剪到 top64/top128 再跑；(D) 只做全量训练、不做 rollout。

### 2026-10-20 00:10 决策：A（全量 + PP=4 补丁）
用户确认：**目标固定为全量 384 专家模型，不启动小模型，自主决策**。

**代码修改（dsv41-verl-copy）**：
1. `verl/verl/workers/config/rollout.py`：vLLM 的 `pipeline_model_parallel_size > 1` 不再 `NotImplementedError`，改为 warning；sglang/trtllm 仍禁止。
2. `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`：vLLM args 增加 `"pipeline_parallel_size": self.config.pipeline_model_parallel_size`（RolloutReplica 的 world_size=TP*DP*PP、nnodes=world/16、node_rank>0 headless 逻辑已具备）。
3. 启动脚本拓扑：`GEN_TP=8, GEN_DP=1, GEN_PP=4, GEN_EP=8`（32 卡/replica，8 replicas）；训练 `EP=64, EFSDP=2`（专家 128 路分片，edp=2）；`ROLLOUT_GPU_MEM_UTIL=0.8`（引擎权重 35.4GB/卡，预算 51GB）。
4. `LOCAL_EXPERT_EXPORT` 因 `GEN_EP(8) != EP_SIZE(64)` 自动关闭 → 走 generic export（全量 all-gather），这是 PP 方案的已知代价，待实测同步耗时。

**重启**：00:11 清理 16 台 Ray/vLLM 残留 → 00:12 master + 15 join → 00:13 Ray 16/16、driver 已连接，进入 worker group 创建。

### 2026-10-20 00:20 【PP 第二处】PPMissingLayer + 模型 forward PP 支持
**现象**：v2 在 engine 初始化时崩：
```
vllm_ascend/models/deepseek_v41/model.py:1264
  if layer.layer_idx + 1 in self.aux_hidden_state_layers:
AttributeError: 'PPMissingLayer' object has no attribute 'layer_idx'
```
根因：模型 forward 未按 PP stage 处理，`self.layers` 中非本 stage 的层是 `PPMissingLayer`（nn.Identity），且 forward 忽略 `intermediate_tensors`、无条件做 hc_collapse/norm。

**代码修改（`vllm-ascend/vllm_ascend/models/deepseek_v41/model.py`）**：
1. forward 开头：`intermediate_tensors is not None` 时读取 `hidden_states` 与 `pre_mix`（跨 stage 传递超连接状态），仅 first rank 做 embed 和 pre_mix 初始化。
2. 层循环：`isinstance(layer, PPMissingLayer)` 直接 `continue`。
3. 非 last rank 返回 `IntermediateTensors({"hidden_states": ..., "pre_mix": ...})`；只有 last rank 做 `hc_collapse` + norm。
4. `_make_empty_intermediate_tensors` 增加 `pre_mix` 缓冲区（[batch, hc_mult] fp32），与返回 keys 对齐。
- `py_compile` 通过。

**v3 重启**：00:23 master → 00:24 Ray 16/16，进入 engine 初始化。

### 2026-10-20 00:32 【PP 第三处】KV cache 空组 StopIteration
**现象**：v3 engine 初始化在 `determine_available_memory` 阶段崩：
```
vllm_ascend/worker/worker.py:672 _scale_kv_cache_memory_for_multi_group
  → vllm/v1/core/kv_cache_utils.py get_kv_cache_groups
  → vllm_ascend/patch/platform/patch_kv_cache_utils.py:385 _ascend_get_packed_kv_cache
  → vllm_ascend/models/deepseek_v41/cache_config.py:93 group_cache_specs
  → kv_cache_interface.py:1245 is_uniform_type
  first_spec = next(iter(kv_cache_specs.values()))
StopIteration
```
根因：PP 某个 stage 没有本地的 `CircularBufferSpec`（compressed state）或 SWA 组，`UniformTypeKVCacheSpecs.from_specs({})` 空 dict 触发 StopIteration。

**代码修改（`vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py`）**：`group_cache_specs` 仅在 `mla_names` / `state_names` / SWA 组非空时调用 `from_specs`。
**v4 重启**：00:34 master → 00:35 Ray 16/16。

### 2026-10-20 00:42 【PP 第四处】无 full-MLA 层的 stage KV 配置除零
**现象**：v4 engine 初始化在 `get_deepseek_v41_kv_cache_config` 崩：
```
vllm_ascend/models/deepseek_v41/cache_config.py:148
  capacity = max(available_memory // sum(page_sizes), 0)
ZeroDivisionError: integer division or modulo by zero
```
根因：PP 某 stage 没有 full-MLA 层（只有 compressed/index/SWA 层）。`group_cache_specs` 的 `padded` 只从 `layer_tuples` 构建，而 `layer_tuples` 由 `full` 层驱动；`full` 为空 → `padded` 为空 → groups/page_sizes 为空；KV config 再除零。

**代码修改（同文件）**：
1. `group_cache_specs`：`layer_tuples` 为空时，保留本地所有 spec 到 `padded`（按各自 unpadded page size），并为 SWA-only stage 单独建组。
2. `get_deepseek_v41_kv_cache_config`：`page_sizes` 为空时，按每个本地 spec 独立建 KV tensor；除零保护 `sum(page_sizes) or 1`。
**v5 重启**：00:43 master → 00:44 Ray 16/16。

### 2026-10-20 00:51 【PP 第五处】各 stage num_blocks 不一致断言
**现象**：v5 在 `generate_scheduler_kv_cache_config` 断言失败：
```
vllm/v1/core/kv_cache_utils.py:2353
  assert all([cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs])
AssertionError
```
根因：PP 各 stage 的本地 page_sizes 不同（有的 stage 无 full-MLA 层），各自算出的 `num_blocks` 不同；该函数要求同一 engine 内所有 worker 的 `num_blocks` 相同。

**代码修改（`vllm_ascend/models/deepseek_v41/cache_config.py`）**：`get_deepseek_v41_kv_cache_config` 在算出本地 capacity 后，对 engine 内默认进程组做 `all_reduce(MIN)`，让所有 stage 取最受限容量；`torch.distributed` 未初始化时回退本地值。
**v6 重启**：00:52 master → 00:53 Ray 16/16。

### 2026-10-20 01:01 【PP 第五处·修正】改用 num_gpu_blocks_override
v6 仍在 `generate_scheduler_kv_cache_config` 的 num_blocks 一致性断言失败（all_reduce(MIN) 未生效，可能 worker 默认进程组未初始化/不可用）。
**修改**：撤回 all_reduce，改用 vLLM 原生 `--num-gpu-blocks-override`：启动脚本新增 `NUM_GPU_BLOCKS=4096`，并在 master/single 启动参数中传
`+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=${NUM_GPU_BLOCKS}`。`may_override_num_blocks()` 会让同 engine 所有 worker 的 num_blocks 都是 4096。
**v7 重启**：01:02 master → 01:03 Ray 16/16。

### 2026-10-20 01:11 【PP 第六处】空投影组导致 bytes_per_block=0
**现象**：v7 过了 num_blocks 断言，但核心 `get_kv_cache_configs` 在 override 分支除零：
```
vllm/v1/core/kv_cache_utils.py:2713
  avail_mem // bytes_per_block
ZeroDivisionError
```
根因：`_project_kv_cache_groups_to_worker` 把全局组投影到某 stage 时，即使该 stage 没有组内任何层，也仍 append 一个 `worker_layer_names=[]` 的空组；`_get_kv_cache_bytes_per_block` 对空层列表求和得 0，所有组都空时 max=0 → 除零。

**代码修改（`vllm/vllm/v1/core/kv_cache_utils.py`）**：投影后 `worker_layer_names` 为空的组直接 `continue`，不再保留。
**v8 重启**：01:13 master → 01:14 Ray 16/16。

### 2026-10-20 01:21 【PP 第七处】override 分支 bytes_per_block=0
v8 过了投影空组问题，但 override 分支 `logger.info(avail_mem // bytes_per_block)` 仍除零（某 stage 的投影组只含零尺寸 compressed/index spec）。
**代码修改（`vllm/vllm/v1/core/kv_cache_utils.py`）**：`bytes_per_block <= 0` 时跳过 override 缩放（保留 `avail_mem`），不除零。
**v9 重启**：01:23 master → 01:24 Ray 16/16。

### 2026-10-20 01:32 【里程碑】v9 引擎初始化成功，8 个 PP replica 就绪
- 日志证据（`DeepSeek-V4.1-Flash-bf16-noengram-20261019_172452.log`）：
  - `INFO:...:create worker group dict_keys(['actor_rollout'])`
  - 8 × `vLLMReplica`；`EngineCore` 启动、`enforce eager`、`LoRA dynamic loading enabled` 等仅 WARNING
  - `INFO:...:checkpoint engine manager initialized`
  - `[sleep-debug] sleep_replicas: 8 replica(s)`
- `ps` 确认 driver `python3 -m verl.trainer.main_ppo`（pid 961759）与 256 × `ray::WorkerDict` 存活；无 Traceback/OOM。
- **意义**：前面 7 处 PP 兼容补丁全部生效：TP=8×PP=4 的 4 机 replica 已能完成 KV cache config、显存 profile、引擎启动。
- 下一阶段：trainer→engine 权重同步（generic export，1.13TB 量级）→ 首次 rollout → step 1。
- 巡检脚本 driver 检测修正为 `verl.trainer.main_ppo|ray::TaskRunnerV1`（Ray actor 进程名不含 main_ppo）。

### 2026-10-20 01:40 【权重同步】HCCL 单卡多进程端口冲突
**现象**：v9 引擎 init 成功后，首次 `actor_rollout_update_weights` 崩：
```
HCCLUtils.cpp:140 hcclCommInitRootInfoConfig ... error code is 1
Communication_Error_Bind_IP_Port(EI0020): Failed to enable listening for the NPU network adapter socket.
  The IP address fd00::105 and port 65536 have already been bound.
  Solution: ... configure the port number using HCCL_NPU_SOCKET_PORT_RANGE
Value 10 for environment variable RankIpFamily is invalid. Expected value: 2.
```
根因：trainer WorkerDict 与 vLLM engine worker 同卡多进程，共用 `HCCL_*_SOCKET_PORT_RANGE`，engine 先绑定后 trainer 的 FSDP all-gather 新建 HCCL comm 冲突；另有 IPv6 rank info（RankIpFamily=10）被拒绝。

**代码修改**：
1. 启动脚本：trainer 段加宽为 `HCCL_HOST_SOCKET_PORT_RANGE=60000-60150`、`HCCL_NPU_SOCKET_PORT_RANGE=61000-61150`。
2. `vllm_async_server.py launch_servers`：engine server actor 的 env 覆盖为独立段 `60200-60350` / `61200-61350`（trainer 与 engine 不再抢同一端口段）。
**v10 重启**：01:41 master → 01:42 Ray 16/16。

### 2026-10-20 01:51 【权重同步·续】强制 RankIpFamily=IPv4
v10 端口冲突消除，但跨机 PP 的 functional collective 仍报 `Value 10 for environment variable RankIpFamily is invalid. Expected value: 2`（NPU RoCE IPv6 fd00:: 被 KFC 拒绝）。
**修改**：启动脚本与 engine server env 均设置 `RankIpFamily=2`、`HCCL_RANK_IP_FAMILY=2`。
**v11 重启**：01:53 master → 01:54 Ray 16/16。

### 2026-10-20 02:08 【根因·宿主机】101 的 NPU RoCE IP 缺失
- `hccn_tool` 全量扫描：**101 只有 dev0 配了 IP（10.1.225.2），dev1~15 全部未配置（1/16）**；其余 14 台均为 16/16 且方案规整。
- 方案（成对节点）：101/102 用 10.1.225/224 段，后缀 11/21；103/104 用 31/41；105~108 用 227/226 段；109~112 用 229/228；113/114 用 231/230；115/116 的 116 为 41。
  - dev 偶数 → `10.1.X.(base+dev/2)`，dev 奇数 → `10.1.(X-1).(base+dev/2)`，网关 `.1`。
- 已在 101 宿主机用 `hccn_tool -i N -ip -s address ... netmask 255.255.255.0` + `-gateway -s gateway` 配置 dev0~15，校验 **16/16**（10.1.225.11~18 / 10.1.224.11~18）。
- 这正是跨机 HCCL 报 `RankIpFamily=10 / fd00::` 的根因：未配置 IP 的 device 回落到 IPv6 link-local 被 KFC 拒绝。
**v12 重启**：02:09 master → 02:10 Ray 16/16。

### 2026-10-20 02:19 v12：NPU IP 修复后首次 `ready to fit`，无 HCCL 错误
- 证据：`DeepSeek-V4.1-Flash-bf16-noengram-20261019_181050.log`
  - 18:16:12 `reward loop manager initialized`
  - 18:19:26 `sleep_replicas: done` → `Training from scratch` → **`all initialize finished, ready to fit`**
  - 全日志无 `RankIpFamily` / `HCCL function error` / Traceback（此前 RankIpFamily=10 的根因是 101 的 NPU IP 缺失，已修复）
- `VLLM::Worker_PP` 进程 250-270% CPU，101 宿主 RAM 783GB / 102 411GB → 首个 fit 的权重同步/引擎装载进行中；npu-smi 在高负载下查询无返回（不代表故障）
- 当前等待：首次 weight sync 完成 → rollout 生成 → step 1

### 2026-10-20 02:31 首次 weight sync 进行中（非卡死）
- `ray list tasks`：大量 `WorkerDict.actor_rollout_update_weights` 状态 **RUNNING**，运行时长约 218-231s（持续增长）；`vLLMHttpServer.collective_rpc` 也 RUNNING。
- vLLM worker：HBM 81%（device 0，约 53GB，权重已载入），Aicore 0%（尚未开始生成）；进程 1 running + 820 sleeping threads，累计 CPU 高（权重处理）。
- WorkerDict：78% CPU，`futex_wait_queue_me`（等待 engine ack，属同步背压）。
- 日志在 `ready to fit` 后无新内容，但任务在推进 → 1.13TB generic export 需要较长时间。
- 判定：**不是空跑/死锁**，首次 weight sync 正在传输；继续 5 分钟巡检。

### 2026-10-20 02:33 同步持续（无 bucket 级日志）
- `/proc/<pid>/io` 的 rchar/wchar 在 2 分钟内完全不变，但 **权重走共享内存 bucket（ZMQ 只传 metadata）**，socket I/O 不能反映进度。
- VLLM worker utime/stime 在 5s 内增长 ~13.9s（约 277% CPU），说明仍在计算（导出 all-gather / bucket 拷贝），非死锁。
- `ray list tasks` 仍显示 27 个 `actor_rollout_update_weights` RUNNING；无 Traceback/OOM/HCCL error。
- `VERL_SYNC_PROFILE=1` 只在同步结束时打印，目前无输出；继续等待首次 sync 完成。

### 2026-10-20 02:37 同步仍在推进（RUNNING task 数下降）
- `active_nodes=16/16`、`driver_pids=2`、无 Traceback/OOM/HCCL。
- `actor_rollout_update_weights` RUNNING 任务数 27 → **25**（部分 rank 已完成）。
- VLLM worker utime+stime 3s 内增长约 8.3s（≈277% CPU），HBM 81%，Aicore 0% → 仍在权重同步/装载，未开始生成。
- 日志增长主要来自 120 raylet 告警；主内容仍停在 `ready to fit`。

### 2026-10-20 02:41 同步继续（~22 分钟）
- `active_nodes=16/16`、`driver_pids=2`、无 Traceback/OOM/HCCL。
- VLLM worker 3s 内 utime+stime +8.37s（≈279% CPU）；HBM 81%，Aicore 0%。
- `actor_rollout_update_weights` RUNNING 25（`ray list tasks` 默认截断，不能作为严格进度）。
- 判定：仍在 CPU 密集的首次权重导出/装载，无死锁迹象；继续等 step。
- 巡检 cron 更新：npu max_chip=0 视为查询失败，改用 CPU 增长 + HBM usages + task 状态判断，避免误报空跑。

### 2026-10-20 02:46 【崩溃+根因】FSDP 全局 all-gather 跨平面超时；101 NPU 出向被 ACL 限制
**崩溃**：18:46:47（sync 约 27 分钟后）
```
RuntimeError: ... current working operator name is HcclAllGather
ERR00100 PTA call acl api failed
Communication_Error_Get_Socket(EI0006): socket times out.
  101 device0 fails to connect to dstRank 105/109/113/116/120 device0
```
**NPU 平面连通性实测**（`hccn_tool -i 0 -ping`）：
- 同平面：101↔102/103/104 ✅
- 101→105/106/109 ❌；105→101(.11/.3) ❌
- 105→102 ✅、105→109 ✅、109→102 ✅、113→102/109 ✅、120→116 ✅
- **101 改回原 IP 10.1.225.2 后：101→105 ✅，105→101.2 ✅**；101 的其它 device（.3 等）出向跨平面 ❌（入向可达）
- 结论：**101 的交换机端口只放行原 IP 10.1.225.2 的跨平面出向**，其余 15 个 device 被 ACL 阻断；HCCL 无 TCP 回退（libhccl 只有 HCCS/PCIe/RoCE）。101 不能作为 16 卡 NPU 节点参与跨平面集合通信。
- 其余 15 台（102-114,116,120）NPU 平面全通。

**方案调整（用户目标=全量模型，自主决策）**：使用 102-114,116,120 共 **15 机（240 卡）**，101 仅保留控制/不参与训练。
- 训练：FSDP=240；EP=24 × EFSDP=2 = 48 路专家分片（edp=5；384/48=8 experts/rank）。
- 引擎：GEN_TP=8 × GEN_PP=10 = 80 卡/replica（5 机，40 层/10 级=4 层/级），240/80=3 replicas；每卡权重 1.13TB/80≈14.1GB，util=0.5。
- batch：TRAIN_BSZ=30 × SAMPLE_N=8 = 240（每卡 1 条）。
**v13 重启**：02:52 master 102 → 02:53 Ray 15/15。

### 2026-10-20 03:07 【崩溃·硬阻塞】PP 与 V4.1 long-KV 跨层共享不兼容
**好消息**：15 机 v13 的引擎初始化和**权重同步成功**：
```
SYNC-PROFILE sender: 2798 tensors, 35.97 GiB, 70 buckets | export 65.9s, flush 1.6s
SYNC-PROFILE receiver: 72 buckets, 2798 tensors | load 0.5-3.0s
SYNC-PROFILE step1/2/3: total 69-71s
```
（PP=10 下每 rank 只导出 35.97GiB，generic export 约 70s，不再是瓶颈）

**崩溃**（19:07:08）：
```
Worker_PP1_TP0_EP0 ERROR
  File vllm_ascend/attention/dsa_v41.py:535, in _get_layer_metadata
    long_kv = metadata[self.long_kv_source_prefix] ...
KeyError: 'language_model.model.layers.2.self_attn.long_kv_cache'
```
**根因**：V4.1 的 long-KV 是**跨层共享**的：
- `kv_source_layer_ids=[2,8,14,20]`
- 层 2-7 用 source 2；8-13 用 8；14-19 用 14；**层 20-39 共 20 层都用 source 20**
- PP=10 把 source 和 consumer 分到不同 stage（PP1 的层 4-7 需要 PP0 的层 2 KV），当前 stage 的 `attn_metadata` 里没有该 source 的 cache/metadata → KeyError。
- 任何 PP>1 切分都会跨 stage；唯一对齐的切分是 `[0-7],[8-13],[14-19],[20-39]`，但最后一级 20 层在 TP8 下需 **70GB/卡** 权重，超过 64GB，不可行。
- 结论：**PP 与 long-KV 共享在模型结构上冲突；TP>8 又被 o_groups=8 限制**（见 02:50 记录）。

**当前状态（09:32 巡检）**：driver=0、actors=0、tasks=0，日志停在 03:38，**无 step**；15 台 Ray 节点空转。5 分钟巡检 cron 已停。
**待决策**：top64/top128 裁剪、TP>8 部分组补丁、跨 stage KV 复制、或交厂商/上游。
