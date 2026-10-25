# DeepSeek-V4.1 全量 566B（去 engram）· 8 机 / 16 机可行性分析与脚本 — 2026-10-25

## 0. 结论摘要

- 模型：原始 `/mnt/share/DeepSeek-V4.1-Flash-bf16` 为 **763.21B 参数 / 1421.7GiB bf16**；去掉 2 层 Engram（196.93B）后 **566.28B / 1054.8GiB（约 1.13TB）**。
- **8 机和 16 机都能“装下”训练侧**：按 8 机 128 卡、EP=64×EFSDP=2（128 路专家分片）计算，每 rank 4.35B 参数，param+grad+Adam ≈ 70.8GB host，16 rank/节点 ≈ **1.13TB host**（机器 2.4TB，余量约 1.2TB）。16 机每节点内存与 8 机相同（专家分片仍 128 路，多出的节点是 edp/rollout 副本）。
- **rollout 是约束项，且与总机数无关**：
  - 每机本地 replica（TP8×DP2、EP16）：每卡权重 72.9GB > 61.27GB HBM，装不下。
  - PP 方案：verl 当前对 vLLM `pipeline_model_parallel_size>1` 直接 `NotImplementedError`；即使打补丁，V4.1 的 long-KV 跨层共享（`kv_source_layer_ids=[2,8,14,20]`，layer 20 被 20-39 层共用）只允许 `[0-7],[8-13],[14-19],[20-39]` 的切分，最后一级 20 层在 TP8 下需 **65.9GiB/卡** > 61.27GiB，仍放不下。
  - **唯一可行：跨机 DP** —— TP8×DP8=64 卡/replica（跨 4 机），EP=TP×DP=64；每卡专家 557.17B/64 + dense 9.11B/8 ≈ **20.9GB**，util=0.6 时 KV 约 15GB，能放下。8 机 = 2 个 replica；16 机 = 4 个 replica。
- 跨机 DP 的历史阻塞 E22 已在代码层定位并修复：verl 给非 headless 节点传 `data_parallel_start_rank` 会让 vLLM 推断 `hybrid_lb=True`，随后拒绝远端 `--headless`。去掉该键后实测 `hybrid_lb=False`、`dp_rank` 仍正确推断。
- 05:55 复查：**118/119/120/121/122/123/126/127/128 九台可达**（117/124/125/129/130/131 容器未起）。8 机已具备条件；16 机仍需 16 个地址（117-131 只有 15 个，且不得用 80.5.21），暂不可行。
- `/mnt/share` 05:55 复查为 **189T 总 / 178T 用 / 11T 可用（95%）**（此前 817GB 已被释放）；单份 1054.8GiB bf16 权重可存，但含优化器的全量 checkpoint 仍很大，脚本保持 example 默认 `save_freq=-1`。

## 0.1 更新（05:45）：EFSDP=1 + EP=128（最终采用）

按“EFSDP=1、增大其他并行”的思路，把专家分片全部放到 EP 维，而不是 EFSDP 维：

- **训练**：`EP_SIZE=128, EFSDP_SIZE=1`。每 rank `557.17B/128=4.35B`（与 EP64×EFSDP2 相同），
  param+grad+Adam ≈70.8GB/rank，host 仍 **1.13TB/节点**；4 机（64 卡）时 EP 最多 64 →
  8.7B/rank、2.23TB/节点，仍不可行，所以 8 机仍是最小规模。
- **rollout**：`TP=8 × DP=16 = 128 卡/replica`；8 机 = **1 个 replica**（128 卡），
  16 机 = **2 个 replica**。每卡专家 `557.17B/128` + dense `9.11B/8` ≈ **11.4GiB**
  （util 0.6 → KV 约 25GB，余量比 EP64 方案更大）。
- **关键收益**：`GEN_EP(128) == EP_SIZE(128)` 且 `EFSDP_SIZE=1`，example 的守卫会保留
  `VERL_DSV41_LOCAL_EXPERT_EXPORT=1`；sync 只传本 rank 的 3 个专家（约 8.5GB），
  加上 dense 全量 all-gather（9.11B×2B=18.2GB），**预计 30-70s/次**，不再是 generic
  export 的 ~10 分钟。配对关系：trainer 的 ep 坐标 = rank%128，engine 的 EP rank 也是
  rank%128（`init_ascend_model_parallel` 的 `(dp,pp,pcp,tp)` reshape 后每组连续 128），
  两边 expert 范围都是 `[3i, 3i+3)`，1:1 对齐。
- **新风险**：EP=128 的 MoE all-to-all 跨 8 机（训练与 vLLM 均未端到端实测）；
  vLLM DP=16（每机 2 个 local DP rank ×8 机）也未实测。必须先跑 1-2 step smoke。
- 脚本已更新：`MY_ENV_EP_SIZE=128, MY_ENV_EFSDP_SIZE=1, MY_ENV_GEN_DP=16, MY_ENV_GEN_EP=128`。
  下文 2.1/2.2 的 EP64×EFSDP2 账仍成立，作为对照。

## 1. 模型与权重证据

命令（118 上执行）：

```bash
du -sh /mnt/share/DeepSeek-V4.1-Flash-bf16
# 1.4T
python3 - <<'PY'  # 读 model.safetensors.index.json，按文件大小求和
# num tensors 48496, files 268, total 1421.7 GiB
PY
cat /mnt/share/DeepSeek-V4.1-Flash-bf16/config.json
```

- config：`num_hidden_layers=40`、`n_routed_experts=384`、`engram_layer_ids=[1,14]`、`engram_num_embeddings=[384006168,384016682]`、`engram_head_dim=256`。
- Engram 参数：`(384006168+384016682)*256 = 196.61B`（加上投影/gate 约 196.93B，见 `worklog_dsv41_16nodes_1019.md`）。
- 去 engram 权重目录 `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-bf16-noengram`：
  - `du -sh` 仅 8.7M —— 274 个软链指向原始权重 + 1 个改过的 `config.json`；
  - `config.json` 的 `engram_layer_ids=[]`、`engram_num_embeddings=[]`，`n_routed_experts` 保持 384；
  - `model.safetensors.index.json` 仍列 268 个文件/48496 张量（含 engram 分片），但 `VERL_DSV41_DIRECT_SHARD_LOAD=1` 的加载计划只含模型参数，**196GB 的 engram 分片不会被读**（10-19 已验证，见 worklog 1019 §1）。
- 因此训练/推理实际加载量：`1421.7 - 366.2 = 1055.5GiB ≈ 1.03TiB` bf16；按参数算 `566.28B*2B = 1132.6GB = 1054.8GiB`，两者一致。

## 2. 显存/内存账

### 2.1 训练侧（8 机 128 卡 / 16 机 256 卡）

拓扑：`FSDP_SIZE = NNODES*16`；`EP_SIZE=64`；`EFSDP_SIZE=2` → 专家分片 `64*2=128` 路。

| 项目 | 公式 | 数值 |
|---|---|---|
| 专家参数/rank | 557.17B / 128 | 4.353B |
| bf16 权重/rank | 4.353B × 2B | 8.71GB |
| bf16 梯度/rank | 同上 | 8.71GB |
| fp32 master/rank | 4.353B × 4B | 17.41GB |
| Adam m / v | 2 × 4.353B × 4B | 34.82GB |
| dense 参数/rank | 9.11B / FSDP_SIZE | 0.14GB（8 机）/0.07GB（16 机） |
| **host 合计/rank** | 8.71+8.71+17.41+34.82+0.14 | **≈70.8GB** |
| **host 合计/节点** | ×16 rank | **≈1.13TB** |

- 机器内存 2454GB，1.13TB ≈ 47%，余量约 1.2TB；与当前 top32 单机长跑（每节点约 1TB）同量级。
- HBM：训练时只 materialize 当前层（每 rank 6 专家/层 × 70.8MB ≈ 0.42GB，EFSDP 暂存约 0.85GB）+ 激活（batch 1×seq1664，ffn recompute）+ vLLM sleep 常驻元数据（top32 实测约 12.7GB）。估算 < 25GB/61.27GB。
- 4 机不可行的原因：EP=32×EFSDP2=64 路 → 8.7B/rank → 139GB/rank ×16 = **2.23TB/节点**，加 OS/引擎余量超过 2.4TB。因此 **8 机是训练侧最小规模**。

### 2.2 rollout 侧（约束项）

三种拓扑的每卡权重：

| 方案 | 每 replica 卡数 | 每卡权重公式 | 每卡权重 | 结论 |
|---|---|---|---|---|
| 每机本地 replica（TP8×DP2，EP16） | 16 | 557.17/16 + 9.11/8 + vision | **≈72.9GB** | >61.27GB，直接失败（还没算 KV） |
| PP4 对齐切分（TP8×PP4） | 32 | 最后一级 20/40 层：1054.8×20/40/8 | **65.9GiB** | >61.27GiB；且 verl 禁止 PP>1、long-KV 跨 stage KeyError |
| **跨机 DP（TP8×DP8，EP64）** | **64** | 557.17/64 + 9.11/8 + vision | **≈20.9GB（19.5GiB）** | **可放下**；util0.6 → KV 约 15GB |

- 跨机 DP 的 dense/vision 在每个 DP 副本复制 8 份（+18%），总权重约 64×20.9=1.34TB/replica，仍远小于 2.4TB host；引擎权重全在 HBM。
- 8 机 128 卡 → `num_replicas = 128 / (8*8) = 2`；16 机 256 卡 → 4 个 replica。**节点数只增加副本，不改变每卡权重**。
- 参考实测：10-19 全量模型 15 机、PP=10（每个 engine rank 只拥有 4 层）时首次 sync `35.97 GiB/sender, export 65.9s, total 70s`。本次 EFSDP=2 会关闭 LOCAL_EXPERT_EXPORT 走 generic export，而 PP=1 时每个 engine rank 拥有全部 40 层，导出量按层数线性放大 **约 10×** → 预计 `~360GiB/sender、~10 分钟/次 sync`（未实测）。这是本方案最大的性能风险；若不可接受，需要给 `_export_local_experts` 增加 EFSDP 维 all-gather（EFSDP=1 时 host 2.23TB/节点，超过 2.18TB 空闲内存，不可行）。

## 3. E22 根因与修复（跨机 DP 的前提）

历史结论“跨机 DP hybrid-LB 拒绝远端 headless”在代码中的精确链路：

1. `verl/workers/rollout/vllm_rollout/vllm_async_server.py:408-419` 在 `data_parallel_size>1` 时向每个节点传 `data_parallel_start_rank = node_rank*local`。
2. `vllm/vllm/engine/arg_utils.py:2265-2269`：
   ```python
   if self.data_parallel_start_rank is not None and not headless:
       self.data_parallel_hybrid_lb = True   # 非 headless 的 node0 被推断为 hybrid LB
   ```
3. `vllm/vllm/v1/engine/utils.py:1266-1267`：
   ```python
   remote_should_be_headless = (
       not parallel_config.data_parallel_hybrid_lb
       and not parallel_config.data_parallel_external_lb)
   ```
4. verl 对 `node_rank>0` 固定 `run_headless`（`vllm_async_server.py:496-499`），于是 `utils.py:1349-1361` 抛
   `Remote engine N must not use --headless in external or hybrid dp lb mode`。

数值实证（118，真实模型 config，不加载权重）：

```
RESULT with_start_rank    hybrid_lb=True  external_lb=False dp_rank=0 dp_local=2 dp_size=8
RESULT without_start_rank hybrid_lb=False external_lb=False dp_rank=0 dp_local=2 dp_size=8
```

修复：`vllm_async_server.py` 的 `dp_args` **不再传 `data_parallel_start_rank`**（保留 `data_parallel_size/local/address/rpc_port`）。`dp_rank` 仍由 `arg_utils.py:2286` 按 `node_rank*local_world_size//(TP*PP)` 推断；单机 DP2（local==size）行为不变。

配套：`my_scripts/dsv41_grpo_2node_new.sh` 的多机分支允许 wrapper 用 `MY_ENV_EP_SIZE / MY_ENV_EFSDP_SIZE / MY_ENV_GEN_TP / MY_ENV_GEN_DP / MY_ENV_GEN_EP` 覆盖拓扑（不设置时保持历史双机默认 EP8/EFSDP2/GEN_TP8/GEN_DP=CPN/8）。备份：`my_scripts/old/vllm_async_server.py.bak_20261025`、`my_scripts/old/dsv41_grpo_2node_new.sh.bak_20261025`。
- 另：trainer 的 HCCL 端口段放宽到 `60000-60150/61000-61150`，engine server actor 固定 `60200-60350/61200-61350`（`vllm_async_server.py:1470-1485`），避免 10-19 v9 实测的 trainer/engine 同卡抢端口 `Communication_Error_Bind_IP_Port(EI0020)`。

## 4. 8 机 / 16 机结论

> 下表是 EP64×EFSDP2 对照；**最终采用 EP128×EFSDP1**（见 0.1）：8 机 = 1 个 128 卡 replica、16 机 = 2 个 replica，每卡 11.4GiB，sync 预计 30-70s。

| | 8 机 128 卡 | 16 机 256 卡 |
|---|---|---|
| 训练 FSDP | 128 | 256 |
| 训练 EP×EFSDP | 64×2=128（edp=1） | 64×2=128（edp=2） |
| 每 rank 参数 | 4.35B | 4.35B |
| host/节点 | ≈1.13TB | ≈1.13TB |
| rollout replica | 2 × (TP8×DP8=64 卡) | 4 × 64 卡 |
| 每卡引擎权重 | ≈20.9GB | ≈20.9GB |
| 结论 | 能装下（需跨机 DP + E22 修复） | 能装下；只增加吞吐/副本 |

- **16 机不会比 8 机更省单卡显存**：专家参数只在 128 路内分片，增加节点只会产生 edp=2 和更多 rollout 副本。8 机已经是训练 host 内存的最小可行规模。
- 若只追求“能跑”，8 机即可；16 机用于更大 batch/吞吐（batch 256 vs 128）。

## 5. 脚本与用法

新增：

- `my_scripts/full_noengram_common_20261025.sh`：模型/数据/长度/offload/一致性公共 env（master/join 两端都 source）。
- `my_scripts/1025_W3_longrun_full_noengram_8node_v1.sh`
- `my_scripts/1025_W3_longrun_full_noengram_16node_v1.sh`

8 机示例（8 台容器内分别执行）：

```bash
# 头节点
bash /mnt/share/m00899630/workspace/dsv41-verl/verl/my_scripts/1025_W3_longrun_full_noengram_8node_v1.sh master 500
# 其余 7 台（有租户的机器也用 join，不 pkill）
bash /mnt/share/m00899630/workspace/dsv41-verl/verl/my_scripts/1025_W3_longrun_full_noengram_8node_v1.sh join 80.5.25.118
```

16 机把 `8node` 换成 `16node`，其余 15 台 join。

## 6. 节点与磁盘状态（2026-10-25 05:05 UTC 左右）

- 端口 50002 探测（05:55）：**OK 118/119/120/121/122/123/126/127/128 共 9 台**；FAIL 117/124/125/129/130/131（容器未起）。9 台 NPU 全部空闲（每 chip HBM 2.9-3.3GB，busy=0），无 VLLM/ray/driver 进程。
- 各节点 `free -g`：118 used 21G/avail 2330G、119 20G/2345G、120 21G/1622G、121 12G/2433G、122 11G/2433G、123 430G/1899G、126 19G/2345G、127 17G/2419G、128 28G/2392G；均无 VLLM/ray/driver 进程，NPU 基线 HBM 2.9-3.3GB/chip。
- `/mnt/share`（05:55）：**189T 总，178T 已用，11T 可用（95%）**。单份 1054.8GiB bf16 权重可存，但含优化器的全量 checkpoint 更大；example 默认 `trainer.save_freq=-1`（`run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh:365`），本次仍保持不存。

## 7. 风险与待验证

1. **跨机 DP + EP64 尚未端到端实测**：vllm-ascend 代码支持 DP（`prepare_finalize.py:452/502/566` 对 dp_size>1 做 all-gather/reduce-scatter；`distributed/parallel_state.py:61-70` 建 rank grid），10-19 也用 TP64/EP64 跨 4 机跑到引擎初始化；但 **TP8×DP8 的生成**尚未跑过。建议先 1-2 step smoke，盯 engine init、首次 sync、HBM/host。
2. 首次 engine 初始化要从 NFS 读 ~1.3TB/replica（8 机 1 个、16 机 2 个 replica）；10-19 全量模型引擎初始化约 30 分钟，需耐心巡检。EFSDP=1/EP128 已启用 LOCAL_EXPERT_EXPORT，sync 预计 30-70s/次（dense all-gather 18.2GB 为主）；generic ~10 分钟的风险不再适用，但 EP128 配对/性能仍需首个 sync 实测。
3. 原始 release（非 SFT）可能 never-EOS（同 top32 基线风险）；先看 step1 的 reward/response 是否正常。
4. E22 修复改动共享文件 `vllm_async_server.py`；单机 DP2 逻辑不变（local==size），但所有 DP>1 的跨机实验都依赖它。
5. 16 机当前不具备条件：15 个地址 + 9 台可达；需要额外主机且不能用 80.5.21。8 机可用 118(头)/119/120/121/122/123/126/127/128 中的 8 台（建议留 120 作备用，因其 avail 1622G 相对最小）。

## 8. 关键命令记录

```bash
# 模型大小/配置
du -sh /mnt/share/DeepSeek-V4.1-Flash-bf16
python3 -c '... model.safetensors.index.json -> 48496 tensors / 268 files / 1421.7 GiB'
cat /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-bf16-noengram/config.json

# 可达性（端口 50002）
scripts/ssh_exec.sh --test   # OK 118/119/120/123/128

# E22 数值实证
python3 - <<'PY'
from vllm.engine.arg_utils import EngineArgs
# EngineArgs(model=..., tp=8, dp=8, dp_local=2, nnodes=4, node_rank=0)
PY
# with_start_rank hybrid_lb=True; without_start_rank hybrid_lb=False
```
