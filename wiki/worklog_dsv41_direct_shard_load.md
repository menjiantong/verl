# worklog：双机 GRPO 攻坚与直载切片加载（2026-10-09）

> 背景：1node16 因 engram 在两份 EP 副本下 host 内存翻倍（trainer 786GB + engine 786GB）撞死，回退双机
> （FSDP16 / EP8 / GEN_TP8 ×2 节点内副本）。今天的核心目标：把 materialize（权重装载）从
> "rank0 读全量 state dict + 跨机 Gloo 逐张量广播"（实测 ~30MB/s，估算 1 小时以上）改成
> **"每个 rank 切片读自己那份"**（设计目标 1–3 分钟，跨机搬运归零）。

## 一、今日时间线与三次死亡

| 时刻 | 事件 | 证据 |
|---|---|---|
| 06:19 | 双机首启；wrapper 一启动 head raylet 即死（agent failed）→ 驱动 `ConnectionError` | `raylet.err`: "The raylet exited immediately because one Ray agent failed, agent_name = runtime_env_agent/dashboard_agent"，时间与 wrapper 启动秒级对齐（日志名 `_062006`） |
| 06:37 | 修复 pkill 后重跑；materialize 广播中 **.124 raylet 意外终止 #1**（进入广播 ~8 分钟） | driver 日志 `Raylet is terminated. Termination is unexpected` + `.131` 侧 `gloo Connection closed by peer [80.5.25.124]` |
| 07:12 | 重跑；**#2**（广播 ~18 分钟） | 同签名 |
| 07:56 | 首次带直载跑：**被拒回退**（融合专家不在 checkpoint 单张量名下，见 §三.2）；广播中 **#3**（~10 分钟） | 旧格式 `checkpoint read: 964 tensors (1532 …)` + rank0 转发行 `VERL_DSV41_DIRECT_SHARD_LOAD=1 rejected (checkpoint has no tensor named 'layers.0.ffn.experts.gate_up_proj')` |
| 08:36 | 专家支持补齐后重跑：**直载读取成功**（`slice-read directly (11.51 GiB this rank)`）；08:49 死在 meta 物化缺失（见 §三.3） | traceback: `_verify_direct_shards:537 actual.cpu()` → `NotImplementedError: Cannot copy out of meta tensor; no data!` |
| 08:58 | 修 assign 后重跑：**materialize 8.3 秒一次通过**、engram 载入 167s 完成，随后**死在权重同步导出**（见 §三.4，最深一次） | `get_per_tensor_param → module.state_dict() → param.detach()` → `RuntimeError: Attempted to set the storage of a tensor on device "cpu" to a storage on different device "npu:4"` |
| 09:33 | .124 被别的租户占用 → 改为**单机冒烟验证**（train_engine_smoke，8 卡 8-15 动态选卡）。第一次起在 chips 0-7，建图时 **OOM 秒挂**——真因是另一容器 09:33–09:36 之间抢占 chips 0-7（npu-smi 每卡 `VLLM::Worker_TP` 57GB，pid 在本容器 /proc 不可见）；报错签名极具误导性（`61.27 GiB total; 5.25 GiB already allocated; 7.73 MiB free`，PyTorch 自己只占 5.4GiB） | torchrun `ChildFailedError` @09:36:19；误判要点：**容量 61GB + torch 占用很小仍 OOM = 卡被外部占** |
| 09:47 | 冒烟第二次启动：`run_train_engine_smoke.sh` 加入与双机脚本同款 `pick_devices()` 动态选卡，落在 chips **8-15**（`[smoke] devices=8,9,10,11,12,13,14,15`）；日志 `train_engine_smoke_20261009_094750.log` | 直载全绿但死在首个 forward（见 §三.5） |
| 10:10 | 修复 CPU-offload 设备归属后第三次启动（chips 0-7，日志 `train_engine_smoke_20261009_101010.log`） | **10:22 `TRAIN-ENGINE SMOKE PASSED`，全链路通**（详见 §三.5 末尾） |
| 10:34 | 用户重启双机 50 步（master 10:34:13 / driver 10:35:14，ray 16 NPU 就绪，日志 `grpo_top64_steps50_20261009_103509.log`） | 直载 5.2s ✓、权重同步 ✓（关①通过），12:45 后死在 rollout 数据通路：`ModuleNotFoundError: No module named 'numpy._core.numeric'`（跨节点 numpy 版本不一致，`failure: 4` 循环，见 §六.1） |
| 12:06 | numpy 修复后重启（.131 升 2.4.6 对齐 .124）：直载 **11.7s**、engram **83.2s**、权重同步 ✓、rollout `running: 8, failure: 0` ✓、**step 1 首个 actor forward ✓（CPU-offload 修复实证）** | 12:45 死在 step 1 update **OOM = E11 复现**（我们关掉的分块/重算法被撞穿，见 §六.2）；两个启动脚本已改回内存安全默认 |

三次死亡**全部发生在 rank0（.124）持有全量 state dict + 广播的窗口内**（8/18/10 分钟），高度指向 .124 当时的资源压力（该机另有租户：NPU 8–15 曾被约 49GB/卡占用）。
**待办：在 .124 上 `dmesg -T | grep -iE "oom|killed process"` + `tail -30 raylet.err` 取实锤。**

## 二、三个启动期 blocker（均已修，未提交）

1. **wrapper 的 pkill 会杀死活着的集群**（`scripts/run_grpo_top64.sh`）
   - 机制：ray 用 setproctitle 改 argv，agent 进程名就是 `ray::DashboardAgent`/`ray::RuntimeEnvAgent`（`ray/autoscaler/_private/constants.py:138-140` 亦有记录），`pkill -9 -f "ray::"` 一命即中 → raylet 与 agent 同命退出。单机流程无碍（清扫在建集群之前）；双机流程集群先建、wrapper 后启 → 必炸。
   - 修复：wrapper 在 `RAY_ADDRESS` 非空时跳过清扫；清扫移入 `run_grpo_2node.sh` 的 `ray start` **之前**（主/从机各自执行）。
2. **engram tiling 校验不支持 EP 副本**（`fsdp_turbo_dsv41_impl.py:_load_host_engram_tables`）
   - FSDP16/EP8 → 两个 EP 组各持全表，行区间合法重复；旧校验误报 `rows are not tiled across ranks at 0`。
   - 修复：唯一区间覆盖 [0, rows) + 每区间恰被 `world/EP` 个 rank 持有。
3. **双机脚本加固**（`run_grpo_2node.sh`）：`pick_devices()` 动态选 8 张空闲卡（0-7 优先，占用顺延，`/dev/davinciN` 校验）；precheck 检查 engram 修复存在；等待循环要求 **16 NPU** 才放行。

## 三、直载切片加载（核心交付，开关 `VERL_DSV41_DIRECT_SHARD_LOAD=1`）

**位置**：`verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py`（`_plan_direct_shard_load` / `_read_direct_shards` /
`_load_direct_shards` / `_verify_direct_shards` / `_assemble_fused_expert_window`）；两个启动脚本默认导出该开关。

**数据流**：每 rank 用 `compute_local_shape_and_global_offset(full_shape, mesh, placements)`（DTensor 自身工具，含不均匀分片）算出窗口 → `safe_open.get_slice()` 只读该窗口 → 分片值构 `DTensor.from_local(..., shape=, stride=)`（**shape 与 stride 必须同时给**）→ `module.load_state_dict(local_state, strict=True, assign=True)`。

**三个硬问题与修复**：
1. **融合专家不在 checkpoint 单张量名下**：924 个参数含 40 层 ×2 的 `ffn.experts.gate_up_proj/down_proj`
   （旧读取器由逐专家 `w1/w2/w3.weight` 内存拼装：gate_up dim1 = [w1 ；w3]，down = w2；**广播载荷的 ~91GB/103GB 就是它们**）。
   → planner 识别融合名并按本 rank 窗口校验逐专家源存在；reader 走 `_assemble_fused_expert_window`（dim0=专家序、
   dim1 支持跨 w1/w3 边界、dim2 任意子区间、空窗口）；rank0 验证时**独立整表拼装** gate_up/down 各一层做 bit-exact 对比。
2. **meta 参数物化**：本模型参数为 meta 初始化（dense 建图时 `initialize_deepseek_v41_model` 跳过 meta；专家 deferred）。
   `assign=False` 对 meta 张量的 `copy_` 是**静默 no-op** → 装载后参数仍 meta（旧广播路径由 DCP 的 meta 机制用
   `assign=True` 收尾，无广播模式不触发）。玩具实验（2 进程 gloo + 真实 `fully_shard` + meta 参数）三方案对比：
   `assign=False` ✗ / `to_empty()` ✗ 崩溃 / **`assign=True` + DTensor 值 ✓（物化+bit-exact+前向可用）**。
   → 改为直接 `module.load_state_dict(..., assign=True)`（DCP 对 FSDP2 的 `fsdp_context` 是 nullcontext，已翻源码确认等价）。
3. **fail-closed 防护网**（均为有意设计）：
   - plan 阶段：嵌套 placements、`_StridedShard(split_factor≠1)`、缺名、计算 local shape ≠ 实际 `to_local().shape` → 打 reason 回退旧路径；
   - 装载后：40 个 buffer bit-exact、抽样参数（默认 3 个，`VERL_DSV41_DIRECT_SHARD_LOAD_VERIFY` 可调）rank0 全量读 vs 已载窗口、两个融合专家独立拼装对比；**验证步骤的 meta 守卫这次直接抓住了 §三.2 的事故**。

**已验证**：`/tmp/test_direct_shard_read.py`（不均匀 5|4 分片 + Replicate 全链）、`/tmp/test_fused_expert_window.py`（6 用例含跨界/空窗）；离线 2D-mesh `fully_shard` 复现确认 placements 为扁平 `(Replicate, Shard(0))` 且窗口切片与真实本地分片一致。

**4. 装载必须镜像 torch 的"分发协议"，不能替换真实参数**（2026-10-09 08:58 事故）：
- 第一版把**所有**参数（含真实参数的）都用 `DTensor.from_local(...)` 构造的新对象 + `assign=True` 替换；随后权重同步导出（`get_per_tensor_param → module.state_dict() → param.detach()`）抛
  `Attempted to set the storage of a tensor on device "cpu" to a storage on different device "npu:4"`——FSDPTurbo 对参数对象的内部引用被替换破坏。
- torch 官方协议在 `_distribute_tensors`（`torch/distributed/_state_dict_utils.py:600-637`，即 rank0 广播装载的收尾，旧路径一直在用）：
  **真实参数** → `ret = local_state; ret.to_local().copy_(full[slices])`（**原地写，参数对象不动**）；
  **仅 meta 参数** → `DTensor.from_local(局部切片, 现有DTensor的 mesh/placements/shape/stride)` 替换。
- 修复：`_load_direct_shards` 重写为该协议（真实→原地 copy；meta→按现有 spec 替换，替换集才走 `load_state_dict(assign=True, strict=False)`）。
- 玩具验证（/tmp/probe_protocol.py）：两用例均 ✓（meta 替换后 state_dict OK；真实原地写后参数对象同一且 state_dict OK）。

**实测**：直载运行时每 rank `11.51 GiB`（全场 ≈184GB，含 dp=2 的专家复制份）；旧广播实测 **~30MB/s**（enp48s3u1u1/80.5.25.x 管理网）。
单机 8 卡冒烟：`slice-read directly (12.46 GiB this rank)`、**materialize 10.0 秒**（旧路径单机 399s / 广播模式 27–90s）、engram 131.7s；验证行 `buffers loaded: 40 (bit-exact)`、`gate_up_proj/down_proj assembled independently and matched`、`5 sampled shard(s) bit-exact on rank0` 全绿。

**5. meta 替换的设备归属：CPU-offload 组必须落 CPU（2026-10-09 10:00 冒烟首撞，最难查的一个）**：
- 症状：直载+验证全绿、engram 载完后，**首个 forward** 死：`RuntimeError: FSDP parameters should be materialized on CPU when enabling CPU offloading ... Found: [('model.layers.0.ffn.experts.gate_up_proj', device(type='npu', index=0)), ('model.layers.0.ffn.experts.down_proj', ...)]`（torch `_fsdp_param_group._validate_cpu_offload_params`，首次 forward 才 lazy 校验）。
- 根因：**本 torch 的 `DTensor.from_local` 会把 local 张量静默搬到 mesh 设备**（`_api.py`："convert the local tensor to desired device base on device mesh's device type"）。直载的切片值是 CPU 的（safetensors），`from_local` 后 local 落在 npu → 替换进的 meta 专家参数成了 npu-local → CPU-offload 校验拒绝。
- 旧广播路径为何没事：DCP `StateDictOptions(cpu_offload=True)` → `_broadcast_state_dict` 在 `_distribute_tensors` 之后执行 `local_state_dict[key] = local_state_dict[key].cpu()`（DTensor 整体 `.cpu()`）→ 终态是「npu mesh + CPU local」。
- 修复：`_load_direct_shards(..., cpu_offload=bool(offload_policy))`，meta 替换在 cpu_offload 时 `.cpu()`（一行），镜像 DCP 行为。
- 验证（`/tmp/probe_cpu_local.py`，真 NPU + `fully_shard(offload_policy=CPUOffloadPolicy())` + meta 参数 + 首 forward）：v1=现代码（from_local(cpu)）→ param local **npu**、forward 报**同款生产错误**（reproduce ✓）；v2=`.cpu()` 修复 → local **cpu**、**forward OK，sum=496 数值正确**；v3=DCP `cpu_offload=True` 参考路径 → local **cpu**、forward OK（v2 与 v3 终态完全一致）。

**最终验收：单机冒烟 `TRAIN-ENGINE SMOKE PASSED`（2026-10-09 10:10–10:22，11.8 分钟，Top32 真实权重）**：
- `checkpoint read: 964 tensors slice-read directly (12.46 GiB this rank; no cross-rank broadcast)`（rank 切片直读，零跨机广播）
- 独立验证全绿：`gate_up_proj / down_proj assembled independently and matched (fused expert)`、`5 sampled shard(s) bit-exact on rank0`、`checkpoint buffers loaded: 40 (bit-exact)`
- `parameters materialized in 9.1s (direct shard load)`（**旧路径单机 399s**）
- `host-offload Engram tables loaded: 2 table(s), 45.8 GiB per rank in 12.6s (frozen)`（比上午 131.7s 更快，NFS 缓存热）
- **首个 forward 过 CPU-offload 校验**（本次修复主验证点）→ `train_batch done in 49.8s metrics={'smoke/log_probs_abs_mean': [13.56], 'grad_norm': 1.85}`（前向+反向+optimizer 全跑）
- **权重同步导出通过**（08:58 的致命关）：`sync payload: 1364 tensors in 0.8s; engram tables in payload: []` → `TRAIN-ENGINE SMOKE PASSED`
- 全程 `Traceback` 计数 = 0；`initialize() done in 626.2s`（建图 ~10min 是主要耗时）。

## 六、首次 2-node 全程推进 + 两个新 blocker（2026-10-09 中午）

### 1. 跨节点 numpy 不一致（已修：.131 升级对齐）
- 症状：rollout 数据通路全线失败 `ray.exceptions.RaySystemError: ... No module named 'numpy._core.numeric'`（TransferQueue `serial_utils.py` 反序列化），`pending: 4, running: 0, failure: 4` 每分钟循环。
- 根因：numpy 2.x 的 pickle 引用 `numpy._core.numeric`（1.x 里叫 `numpy.core.numeric`）；**.131 = 1.26.4、.124 = 2.x**，反向不兼容（2.x 读 1.x 的没问题，1.x 读 2.x 的炸）。
- 修复：用户将 .131 numpy 升到 **2.4.6**（12:01 装，配 torch 2.10/vllm 0.30.1 实测兼容；scipy 1.13.1 打 <2.3 警告但 `linear_sum_assignment` 可用）。

### 2. E11 复现：关掉注意力分块/重算 → step 1 update OOM（已回退默认）
- 症状：12:45（step 1 `actor_rollout_update_actor`）两节点多 rank 同时 `torch.OutOfMemoryError: Tried to allocate 7.2-7.3 GiB (61.27 GiB total; 48.5-51 GiB already allocated; ~6 GiB free)`。
- 位置：`FSDPTurbo/fsdp_turbo/ops/cpu/sparse_attention.py:110 _indexed_sparse_attention_queries` 的 `torch.gather`（栈：model.py:976 → sparse_attention → npu:523 → torch fallback）。
- 根因：早前按"保性能"把 `FSDP_TURBO_ATTN_RECOMPUTE=0 FSDP_TURBO_ATTN_CHUNK=1e8` 打开（关掉分块+重算）——**"FSDP=16 就没事"的判断是错的**：update 阶段同样撞墙（分块把 gather 的 int64 index/KV 限每块 ≤256 行；重算免掉 backward 每层 ~5GiB 驻留）。
- 修复：`run_grpo_2node.sh` / `run_grpo_top64.sh` 默认改回 `RECOMPUTE=1 / CHUNK=256`（= 代码默认值）。**注意 .124 的脚本副本要同步**（raylet env 是各节点脚本自己 export 的），否则从机还是 0/1e8。
- 备用排查：`FSDP_TURBO_ATTN_DEBUG=1` 每 chunk 打印 allocator 状态。

### 3. 深挖："update 期间把 vllm 权重挪出显存"的睡眠机制 **从未生效**（2026-10-09 下午，只读调查）
- **设计上就该发生**：`PPOTrainerSync.on_sample_end → checkpoint_manager.sleep_replicas() → replica.sleep() → server.sleep()`（init 时也 sleep 一次）；对应 vllm-ascend `worker.sleep(level=1)` → `CaMemAllocator.sleep(offload_tags=("weights",))` → **权重 D2H 备份到 pin memory + unmap 释放**（`vllm_ascend/device_allocator/camem.py`）；`--enable_sleep_mode` 已在 engine args、`free_cache_engine=True` 已在配置。update 结束由 `on_step_end → update_weights` 唤醒（resume weights → sync → resume kv）。
- **实测从未执行**：worker.sleep 成功必打 `Sleep mode (level=1) freed X GiB memory, Y GiB memory is still in use.`（worker.py:265+）；在 12:06/12:55 两轮主日志 + 全部残留 ray session 全树搜（通道已校验：`indexer_topk` 命中 7 文件）→ **`Sleep mode`/`CaMem` 0 命中**。即 init 的 sleep_replicas（`ready to fit` 前一行）和 update 前的 sleep 都是**静默 no-op**。
- 门控纸面排查：`server.sleep()`（vllm_async_server.py:872）只挡 `node_rank!=0` 与 `free_cache_engine=False`；launch 循环 `for node_rank in range(nnodes)`、vllm replica 的 nnodes=1 → 应有 rank0；配置为 True。→ 疑似某处提前 return / RPC 未落到 worker / ascend 客户端 sleep 路径断裂，**需插桩定位**（候选插桩点：`server.sleep` 872、`_sleep_hybrid` 1284、`replica.sleep` replica.py:269）。
- **影响**：update 期间 vllm 权重 17.4GB 常驻（npu-smi 实测 17358MB/卡）——等于白扔 17GB 余量。若修好，update 的 OOM 余量直接翻倍。当前短期对策 chunk=64（13:39 run，仍 OOM：基线升到 55.2GB 后连 96MiB 分配都放不下）；睡眠修复列为下一个课题。
- **深链路排查（第二轮，14:30+）**：全链 4 个必经 marker 日志**一个都不存在**：`Waiting for in-flight requests to complete before sleeping...`（EngineCore.sleep）/ `It took X seconds to fall asleep.`（executor sleep）/ `Sleep mode (level=1) freed ...`（NPU worker）/ 也无 "coroutine was never awaited" 警告。而各门控纸面全过：trainer_mode='sync' ✓（配置 dump）、server actor 名 `vllm_server_0_0`/`vllm_server_1_0` → **node_rank=0 实证** ✓、free_cache_engine=True ✓。
- **独立睡眠探针（14:44 全绿，决定性）**：`/tmp/probe_sleep_mode.py` 在空闲卡上单起 vllm-ascend（`DeepSeek-V4.1-Flash-4layer-scaled32` + `enable_sleep_mode=True` + 生产同款 `rl_config.enabled=true` additional_config）：**sleep(level=1) 1.8 秒释放 25.66 GiB**（worker 打出 `Sleep mode (level=1) freed 25.66 GiB memory, 0.85 GiB memory is still in use.`，free 34.75→60.41GiB）、**wake_up 1.2 秒恢复**、唤醒后生成正常。→ **引擎层完全健康，断点确证在 verl 调用链**。
  - 探针三个坑（供复用）：① 小模型 config 的 `model_type=deepseek_v4.1(_text)`（带点号）Transformers 不认 → 做符号链接目录 `/tmp/probe_model_sleep` 只改 config 为 `deepseek_v41(_text)`；② 必须 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（NPU 不能 fork 再初始化）且脚本要有 `__main__` 保护；③ `PYTHONPATH` 必须**追加**（CANN 的 acl 在里面），不能覆盖。
- **verl 调用链插桩（已加，`[sleep-debug]` 标记、warning 级，待下一轮 run 抓断点）**：① `trainer/ppo/v1/trainer_sync.py` on_sample_end 前后；② `checkpoint_engine/base.py` sleep_replicas 数量与完成；③ `workers/rollout/vllm_rollout/vllm_async_server.py` 的 `vLLMReplica.sleep`（drain 前/后、gather 后）；④ 同文件 `vLLMHttpServer.sleep`（入口打 node_rank/free_cache_engine/mode、门控跳过、engine sleep 返回后）。**注意 ③④ 在 server actor 里、两节点各有一份，.124 必须同步该文件**；①② 在 driver（仅 .131）。诊断完成后移除这些日志。
- **2026-10-09 15:00 两节点一致性核查（ssh -p 50001 root@80.5.25.124 已开通）**：impl md5 两侧完全一致（`89cd4afb08cb80a1635c0b8fc71d64ae`）；.124 的 `run_grpo_2node.sh` 默认已 1/64（`run_grpo_top64.sh` 旧值 0/1e8 已顺手对齐 **1/64**）；三个插桩文件已 scp 到 .124 并 py_compile 通过。**下一轮 run 即可抓 `[sleep-debug]` 断点日志。**
- **2026-10-09 15:06 从 .131 远程启动双机 run（记录用）**：正确的远程启动姿势（两个坑）：
  1. **.124 上经 ssh 启动必须钉 PATH**：`ssh -p 50001 root@80.5.25.124 'bash -lc "export PATH=/usr/local/python3.11.15/bin:$PATH && cd /workspace-verl && nohup bash scripts/run_grpo_2node.sh slave 80.5.25.131 > /tmp/slave_launch.out 2>&1 &"'` —— 该机非交互 ssh 的 `python3` 是 `/usr/bin/python3`(3.10, 无 torch_npu)，precheck 会报 `torch_npu/vllm_ascend import failed`；CANN env 也要 `bash -lc` 才有。
  2. **顺序必须先 master 后 slave**（旧集群还活着时 slave 会接到旧 head）：master 的 `ray stop --force + ray start --head` 会把旧集群推倒 → 等 head 起来后再跑 slave 加入；否则 slave 接到旧 head、master 重建后 slave 掉线，等待循环空转。
  本 run 日志 `grpo_top64_steps50_20261009_150803.log`，目的 = 抓 `[sleep-debug]` 断点（chunk=64 仍在，sleep 修好前 update 可能仍 OOM，不影响取证——睡眠链在 update 前执行）。
- **15:43 step-1 复现实证**（run `..._150803`）：`on_sample_end: entering, 2 replica(s)`（15:43:57）→ `sleep_replicas: 2 replica(s)` → **7 秒后** `sleep_replicas: done` → `sleep_replicas returned` —— 与 init 时完全同型：**入口/出口全在，`replica.sleep`/`server.sleep`/`Sleep mode` 全缺**（断点定位于 `sleep_replicas` 内 gather 与 `vLLMReplica.sleep` 之间）；随后 update 如期 OOM（`Tried to allocate 1.23 GiB; 54.43 GiB allocated; 541 MiB free` / `2.47 GiB; 54.64 GiB` 两 rank）。
- **15:56 重启 v3 探针 run**（同前过程：master 先、slave 后；cwd 必须在 /workspace-verl 再执行 master 脚本）：v3 在 `sleep_replicas` 打印**每个 replica 运行时类型 + `type(r).sleep` 的 module.qualname**，逐 replica 捕获异常，replica/server 侧 `print(flush=True)` 直写 stdout。init 时即可见答案（~16:10）。
- **用户 16:05 指示（监控范围升级）**：OOM 解决后重点盯 **训推一致性**（`rollout_probs_diff_mean` 健康 ~1e-2、`rollout_actor_probs_pearson_corr` 健康 ~0.98）与 **奖励曲线**（每 step reward 应上升）；异常则继续定位分析。
- **⚠️ 日志通道大坑（16:40 发现，推翻此前一半证据）**：**生产 run 的日志里 vllm 日志族（`vllm.*` 及 `logging.getLogger(__file__)` 的模块 logger）被整体过滤**——连引擎启动必打的 `All plugins in this group will be loaded`（INFO）都是 0 命中；只有 python-root 格式的行（`WARNING:2026-...`，如 `verl.checkpoint_engine` logger）和 **`print()` 直写 stdout** 可靠可见。**教训：用日志判 vllm 内部行为（`Sleep mode`/`It took X to fall asleep`/`Waiting for in-flight` 等全为 INFO 级 vllm logger）必须用 print 或改 logger，否则"缺失"是假象。**（此前"sleep 从未执行"的部分推断即受此污染。）
- **v3 实锤链条（16:32:19-25，print 证明）**：`on_sample_end` → `sleep_replicas(2)` → `replica.sleep: draining` → `drained; sleeping 1 server(s)` → **`server.sleep: node_rank=0 free_cache_engine=True mode=HYBRID`（两节点）** → `engine sleep call returned` → `all servers returned` —— **verl 侧全链跑通**；但 28 秒后（16:32:53）update OOM：`55.03 GiB allocated / 402 MiB free`（若引擎真睡掉 17.4GB，此刻应 ~38GB 不会 OOM）⇒ **断点最终锁定在 `engine.sleep(level=1)` 的内部（AsyncLLM→engine_core→EngineCore→executor→worker），且设备层未见释放**。下一步：全 run npu-smi 采样（直接看 VLLM::Worker_TP 在 sleep 窗口是否掉）作为设备层终审。
- **用户 16:52 总目标（长期指令，照此推进）**：**持续跟进直到跑通**，拿到 **reward 曲线**；若曲线与**训推一致**指标正常 → 放行 **50 步**；异常 → 排查分析后重跑。**做好记录。**
- **16:42 第四次 run 的取证器材**：`/tmp/npu_trace2.csv`（本地+.124 双列 npu-smi 采样，3s/9s 一拍；.124 远程采集脚本 `/tmp/proc_mem.sh`，需 `bash -lc` 才有 LD_LIBRARY_PATH）。sleep 窗口终审：`VLLM::Worker_TP` 17.4GB 掉否。
- **运维教训补记**：`pkill/pgrep -f <模式>` 只要模式的字符串出现在"本命令自身"的命令行里就会自杀 shell（exit 144）——此坑今日两次（ray:: 与 npu_sampler.sh）；用 `[.]` 括号技巧或按 PID 操作。
- **🏁 17:17 定案（采样器全程录像，推翻此前归因）**：
  ```
  17:17:32  on_sample_end 触发；35733MB（KV 满档）
  17:17:38  VLLM::Worker_TP = 207MB  ← 引擎真的睡了（35.7GB 全释放，此前"没睡"全是日志过滤造成的假象）
  17:17:51  trainer 4.2 → 12.9GB（update 起步）
  17:18:08  OOM：96MiB 分配失败，device-wide 55.05 GiB（其中 vllm 仅 207MB）
  ```
  ⇒ **OOM 与 vllm 无关：trainer 自己 update 阶段就吃到 ~55GB**（与全天所有 OOM 的 53.8-55.2 形状吻合）；**"单独训练不会 OOM"的早前判断作废——单独训练同样 OOM**。OOM 栈：`transformer_impl.py:1573 forward_step → model.py:1604 forward_backbone → model.py:1268 forward → model.py:1235 hc_post`（与 15:08 轮同址，hc_pre/hc_post 一对）。
  **下一步 = 给 trainer 的 update 峰值降载**（目标：把 ~55GB 压到 <53GB）：分析 hc_pre/hc_post 与层前向的内存模式 + 找 FSDPTurbo 的激活重算/分段开关（`turbo_config`/`fsdp_plan` 是否有 recompute 类选项），或收 update 形状（mini-batch/response 长度）。
- **🔧 17:40 update 峰值降载：MHC 的 hc_pre/hc_post 低显存改写（已实施，双节点同步）**：
  - 结构：DeepSeek-V4.1 的 MHC（`hc_mult=4`）每层做 `hc_pre`（[b,s,4,d]×[b,s,4]→[b,s,d]）与 `hc_post`（展开回 4 副本）。**非融合实现**（`use_fused_mhc` 默认 False）里 `hc_post` 的广播和会物化 **`[b,s,4,4,d]`＝16×hidden/层**（update OOM 现场 `model.py:1235 hc_post` 即此），`hc_pre` 物化 fp32 上采样 [b,s,4,d]。
  - 融合核路线：`use_fused_mhc=True` 需要 `mhc_pre_sinkhorn_premix/backward + mhc_post_backward` 4 个 CANN 自定义算子，本环境 `torch.ops.cann_ops_transformer` **只有 `mhc_post` 一个**（缺 3）→ 不可用。
  - **改用纯 torch 等价改写（`FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py`，`[mem-lean 2026-10-09]` 标记）**：
    - `hc_pre` → `torch.matmul(pre_mix.unsqueeze(-2), x).squeeze(-2)`（NPU 探针：前向**逐位一致**，梯度差 ≤2.4e-4）
    - `hc_post` → `... + torch.matmul(comb.transpose(-1,-2), residual)`（**注意：原广播实际配对是 res[i] 配 comb[i,j]，等价式必须转置 comb**；探针：前向差 0.125、梯度差 ≤0.25，值域 ~30 下的 bf16 舍入级）
    - 消除 16× 中间量与 fp32 上采样；反向只保存小输入。
  - 校验：`/tmp` NPU 探针（chip0）前后向 dual 对比；py_compile ✓；scp .124 ✓（grep mem-lean=2 ✓）。
  - **17:44 重启第五次 run 验证**（master→slave；采样器已续期 2h）。
- **第五次 run（17:28-18:05）结果：仍 OOM，但信息关键**：采样器显示 update 期显存 11 秒内 12→60.4GB 冲满；OOM 栈已**从前向 hc_post 移到 backward**（`_engine_run_backward`，`Tried to allocate 3.70 GiB`）。判读：3.70GiB ≈ **单层全参 all-gather buffer**（78B/40层×2B≈3.9GB），且峰值主项 = **40 层激活驻留**（~1.3GB/层 × 40），op 级改写只切掉了小项。.124 侧确认加载的是新版（mem-lean×2、pyc 17:26）。
- **🔧 真正的降载开关（已启用）：`memory.recompute`（层级激活重算）**：FSDPTurbo `MemoryConfig` 内置四件套（`recompute`/`chunk_batch`/`swap_activation`/plans，`fsdp_turbo_config.py:240-256`），由 `fsdp_turbo.py:47 apply_recompute_modules()` 在构图时按 `module_name_match` 包 torch.utils.checkpoint、`:146` 门控。布线：`fsdp_turbo_impl.py:33 _dict_to_dataclass(FSDPTurboConfig, engine_config.turbo_config)` → **可直接 CLI override**。已在 `run_grpo_top64.sh` 启动行加：
  `+...turbo_config.memory.recompute=true` 与 `'...memory.recompute_plan=["model.layers.{*}"]'`（示例脚本 `"$@"` 透传 ✓，bash -n ✓；slave 侧无需改——配置随 driver 序列化到各 rank）。
  **18:10 重启第六次 run 验证**（预期 update 峰值从 ~55-57GB 降到 ~35-45GB）。若仍紧 → 再加 `memory.swap_activation`（激活换主机内存，本机 2TB 空闲）。
- **第六次 run（18:14-18:50）：update 阶段非 OOM 崩溃 —— `recompute` 与层内嵌套 checkpoint 冲突**。链条：睡眠 ✓（35735→207MB）→ update 启动 → **`torch.utils.checkpoint.CheckpointError: A different number of tensors was saved during the original forward and recomputation`**（`torch/utils/checkpoint.py:1177 unpack_hook`）。根因：层内 sparse-attention 自带 checkpoint（5th run 日志里就有 `torch.utils.checkpoint: the use_reentrant...` 警告），层级 recompute 叠上造成嵌套 checkpoint 的张量保存数不一致。**`recompute` 路线放弃**（除非 vendor 修）。
- **🔀 改用 `memory.swap_activation`（激活换出到主机内存）**：实现完整（`fsdp_turbo.py:48 apply_swap_activation_modules`、`:151` 门控、`memory/swap_activation/{swap_activation,tensor_swap_manager}.py`——`swap_out_cur_and_release_prev` 逐层换出/换回）。无 autograd 魔法 → 不会有 nested-checkpoint 问题；本机 2TB 空闲主机内存兜底。启动行已改为：
  `+...memory.swap_activation=true` 与 `'...memory.swap_activation_plan={"apply_modules":["model.layers.{*}"]}'`（校验函数要求 plan 非空 apply_modules，已匹配）。**18:56 重启第七次 run**。
- **第七次 run（18:57）秒死：Hydra 语法**。`swap_activation_plan={"apply_modules":[...]}` 的**带引号 dict key 不被 Hydra 接受**（`no viable alternative at input '{"apply_modules"'`）；可工作的对照：dict 用**无引号 key**（fsdp_plan 例）或**点号直设嵌套字段**（`hook_modules` 例）。已改：`'...memory.swap_activation_plan.apply_modules=['\''model.layers.{*}'\'']'`。**19:07 第八次 run**。
- **第八次 run（19:07-19:45）：swap_activation 运行时崩溃 —— torch API 版本代差**。`tensor_swap_manager.py:139 AttributeError: module 'torch.accelerator' has no attribute 'Event'`——本 torch(2.10) 的 `torch.accelerator` 只有 current_stream/device 类，**缺 Event/Stream/stream()/default_stream**；vllm-ascend 的 `patch_torch_accelerator.py` 只补内存 API，不管这套。**修复：在同文件顶部加 4 行兼容垫片**（`if not hasattr → torch.npu.Event/Stream/stream/default_stream`，try-import torch_npu；import 验证 4 个属性全 True）。**19:52 重启第九次 run**（垫片已 scp .124）。
- **第九次 run（19:49-20:26）：swap 前向有效、后向累积 → 仍 OOM，且 swap 路线判死**。采样曲线：睡后 update 起步 4.2→13GB 并**保持 13GB 平台 ~10 秒**（swap 把前向驻留压到 ~2 层 ✓ 证明机制在跑），随后 **backward 15 秒内 13→60GB 冲满**、OOM（同签名 3.70GiB@52GB）。读码确认根因：`tensor_swap_manager` 的 `release_device_tensor` **只在 forward 的 save 路径被调用**，backward 经 `wait_cur_and_prefetch_next` 取回的 device 张量**无对应释放** → 逐层累积 ≈ 全部驻留。vendor 实现缺陷，不宜深改。
- **🏁 20:40 决策：弃 swap/recompute 路线，改用确定性的形状开关**。`run_grpo_top64.sh` 启动行改为 **`data.max_response_length=512`**（并移除两条 swap override）。数学：激活驻留≈随 token 线性 —— 序列上限 2048→1536，峰值 ~55GB → **~41GB < 53GB 墙，余量 ~12GB**；MHC matmul 改写（纯收益）保留。**代价：rollout 上限 512（训练语义变化，已向用户标注；后续可用 Ulysses SP=2 换回 1024——需引擎侧验证）。** **20:42 重启第十次 run**。
- **🎉 第十次 run 判决（20:42 启动，含 `data.max_response_length=512`）：21:05–21:20 update 破墙 + 权重同步全速。** 原始证据链：
  - `21:05:43 WARNING ...[sleep-debug] on_sample_end: entering, 2 replica(s)` → `21:05:51 [sleep-debug] sleep_replicas returned`（生成结束 → vLLM 入睡）；采样：睡态 `207MB/卡`、trainer update 峰值 **60572–60833MB/卡（~59.4GiB，92.8% of 65536MB）**，**OOM=0、Traceback=0** —— 形状开关奏效（前几轮同相位 55–60GB 即崩）。
  - `21:05:55 [rank15]:W1009 ... is_fx_tracing`（首步 update 的 torch.compile 预热）；随后 16 rank 全量 `clip_grads.py:94` 警告 = backward 已过 → clip/optimizer。
  - ~21:10 trainer 落回 **7481MB**、vLLM 苏醒 **15769MB**（权重回载）；py-spy 双栈实锤相位：trainer `update_weights → async_send_weights:190 → _export_param:861 → wait_tensor`、vLLM `receive_weights:382 → recv_pyobj` —— **actor update 完成，进入权重同步**。
  - **同步流动实锤（新探针）**：全接口扫描发现 **`lo` loopback 6972–7662MB/s 持续 20s+**（.124 同 pattern 4620MB/s）→ 权重桶经 zmq-localhost 流给本机 replica，**全速非卡死**。踩坑记录：跨节点权重**不走 enp48s3u1u1**（该口 0 流量）= 各节 trainer→本机 replica 全走 loopback；跨节点仅发生在 `_export_param` 的 FSDP all-gather（走 NPU ROCE，不可见于 host `/proc/net/dev`）。冒烟基线：`[smoke] sync payload: 1364 tensors in 0.8s`。
  - **判同步是否在流动：`awk '$1=="lo:"{print $2,$10}' /proc/net/dev` 两次 diff，GB/s 级即在流。**
- **📊 step:1 指标（21:22 出，21:22:50 打印）——链路全通、训推一致健康，但 reward 退化**：
  - 原始行（节选）：`step:1 - ... training/rollout_probs_diff_mean:0.05500035732984543 - ... training/rollout_actor_probs_pearson_corr:0.9597921371459961 - rollout_corr/kl:0.29126083850860596 - ... critic/score/mean:-1.0 - critic/rewards/mean:-1.0 - critic/advantages/mean:0.0 - ... response_length/mean:512.0 - response_length/clip_ratio:1.0 - ... timing_s/gen:173.3 - timing_s/update_actor:252.6 - timing_s/update_weights:876.4 - timing_s/step:1328.9 - actor/grad_norm:0.0`
  - **训推一致：健康**（pearson 0.9598 / kl 0.291 / diff_mean 0.055；历史全尺寸基线 0.90-0.92 / 0.38；错误配对失败模式为 ~0.5 / O(1)）。
  - **reward 退化根因链（三条证据）**：① `response_length mean=min=max=512.0、clip_ratio=1.0` = **100% 响应顶到上限、从未出 EOS**；② rl_tokenizer 的 chat 模板 `{% if add_generation_prompt %}<｜Assistant｜><think>{% endif %}` = **思维链模型、生成即开 `<think>` 块**——512 全部消耗在 think 块内，永远到不了 "Answer:" 行；③ `math_dapo` 提取不到答案 → 全 -1 → GRPO 归一化优势=0（`advantages:0.0`）→ `grad_norm:0.0` 无学习信号。**即：20:40 把 response 从 1024 砍到 512 保内存的直接代价 = 训练信号被砍没。**
  - **次级发现（builder 回落）**：run 日志 `(AgentLoopWorkerTQ pid=3027465) WARNING:...:No model-specific Continuous Token builder is registered for config.json model_type='deepseek_v41'; falling back to default.` —— `continuous_token_wiring.py::_MODEL_TYPE_TO_FAMILY` 精确匹配表只有 `"deepseek_v4"`，**没有 `"deepseek_v41"`**；top64 config.json 恰恰是 `deepseek_v41`（`DeepseekV41ForCausalLM`）。V4 专用 builder 注释明确：checkpoint 无可用 jinja 模板、V4 应走 `encode_messages`（与引擎 `vllm/tokenizers/deepseek_v4_encoding.py` 逐字节一致）。回落路径 = base builder 走 `apply_chat_template`（本地复现：`<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>`，**格式看起来合法且 think 已开**）→ 疑似非致命，待生成探针裁决。
  - timing 旁证：`update_weights 876s` = 未开 `VERL_DSV41_LOCAL_EXPERT_EXPORT` 的慢导出路径（G13 全尺寸 920s 档），step 1329s、进度条自估 50 步需 18h——50 步目标下必须开启该开关（G13 scaled32 验证 86.7→6.5s；全尺寸复验=本 run 的 3 项一致性指标对照）。
  - 计划：gen #3 醒窗期（~21:47）用**与回落渲染逐字节一致的 prompt** 发一条 max_tokens=2048 的探针生成，判定 ①模型在正确 think 格式下能否给出 `Answer:` ②需要多长预算 → 据此定 restart 的 `data.max_response_length` + `train_batch_size`（内存账：峰值≈23MB/rank-token、现 60804MB@2613tok；候选 resp1024/B4≈54GB、resp1536-2048/B2≈39-51GB、全部保留 n=8 组）。
- **21:50 重启（第 11 次 run，`grpo_top64_steps50_20261009_215451.log`）——两项变更同发**：
  1. **响应预算 512→2048、`TRAIN_BSZ=2`**（SAMPLE_N=8 → 16 seqs = 1 seq/rank；内存预测 5GB+21.4MB/tok×2189 ≈ 52GB，线性拟合来自本日实测 2613tok↔60.8GB）。HTTP 探测生成因服务器路由/token 未打通（"Forbidden: missing or invalid connection token" 不在本仓库、45595 端口 body 为空、且 80.5.25.131 被公司 HIS 代理拦截需 `--noproxy`）而放弃 → **改为"以 step 指标为探针"**：新版 step:1 的 `response_length/clip_ratio`、`critic/rewards` 直接判读 2048 是否够模型闭合 `Answer:` 行。
  2. **`VERL_DSV41_LOCAL_EXPERT_EXPORT=1` + `VERL_SYNC_PROFILE=1`**（写入 `run_grpo_2node.sh::set_env`，scp .124 且 md5 核对一致）。验收 = sync 末尾打印 `SYNC-PROFILE sender: ...`（G13 scaled32 对照 86.7→6.5s）+ step:1 三项一致性 vs 第 10 次基线（diff_mean 0.0550 / pearson 0.9598 / kl 0.291；配对错会跳 O(1)/~0.5）。
  - 坑复现：slave 首启 precheck 失败 `torch_npu/vllm_ascend import failed`——ssh 非登录 shell 的 `python3` 落到 3.10；修法 `bash -lc "export PATH=/usr/local/python3.11.15/bin:$PATH; ..."` 重发即通（与 09:2x 老坑同源）。
  - 采样器换代 `/tmp/npu_trace6.csv`（trace5 保留）。旧 run（20:32）21:49 终止：step:1 指标已取，step:2 只跑到 update 结束即停（reward 平线无继续价值）。
  - 运行里程碑（待填）：直载 materialize → engram → vLLM → gen#1 → update#1（看 trace6 峰值/OOM）→ sync#1（看 SYNC-PROFILE）→ step:1 指标（看 reward/clip_ratio/一致性）。
- **第 11 次 run（21:54-22:19）：resp 2048/B2 在 update#1 OOM，但收获一张关键反直觉证据**。原始行：`E1009 22:18:56.323693 ... NPU out of memory. Tried to allocate 22.00 MiB (NPU 5; 61.27 GiB total capacity; 54.89 GiB already allocated; 54.89 GiB current active; 34.25 MiB free; 55.72 GiB reserved in total by PyTorch)`（rank13=NPU5；小到 22MiB 的分配失败=贴顶+碎片）。sleep 已确证完成（`replica.sleep: all servers returned`×2）→ 纯 trainer update 撞墙。
  - **关键事实：512/B8（2613 tok/rank，峰值 60833MB@run10）与 1024/B2（1165 tok/rank，峰值 60856MB@run11）峰值几乎相同（~60.8GB）** —— 更新峰值**不随 response 长度缩放**！坡度也变了：run11 从 9GB→60.8GB 只用 ~23s（更少 token 更快到同一平台）→ 主导项是**每层近似固定、与 token 无关的驻留**（疑为逐层全参物化/驻留在 backward 前不释放——全模型 bf16 ≈ 128GB、40 层 ⇒ 若一半层驻留即 ~60GB 量级；待 instrument 实证）。推论：**"砍 response 保内存"路线作废**（它只砍 token 相关项，砍不到平台顶）；此前 run10 存活是 60.8GB 贴顶时的碎片运气（小分配恰好有连续块）。
  - 卡总量口径修正：npu-smi 65536MB 物理，torch 可用 ≈ 61.27GiB，update 时 torch 池 + 其他（sleep 后 vLLM 207MB + 驱动）≈ 61.2GiB 打满。
- **21:50→22:24 重启（第 12 次 run，`grpo_top64_steps50_20261009_222411.log`）**：`MAX_RESPONSE_LEN=1024`（保持 B2；预期峰 ~46GB 的线性外推已证伪，实际仍会到 ~60GB 平台——本次=**在相同平台下换碎片格局的"运气重掷"**；顺手拿到 1024 预算下模型的闭合情况）。待判决：① update#1 存活？② gen#1 响应长度/clip③ reward。若再 OOM ⇒ 转向"平台顶"的结构性拆解（instrument 定位 60GB 驻留物 + 选择 cpu_offload / num_to_backward_prefetch / 自定义 FSDP 实现等杠杆；MemoryConfig 四件套中 chunk_batch 在 1 seq/rank 下无用、recompute/swap 已判死）。
- **第 12 次也在 update#1 OOM（22:44:35，.124 NPU5，`+300.00 MiB ... 55.03 GiB already allocated`）** → 平台顶与 shape 无关实锤两次。**22:53 起用单机冒烟（`scripts/train_engine_smoke.sh --seq-len 1165 --batch-size 1`，FSDP8）做受控复现 + 进程内逐层内存探针**（新增 `FSDP_TURBO_MEM_DEBUG=1`，`fsdp_turbo_dsv41_impl.py::_build_dsv41_module` 在层/子模块注册 forward pre/post + full-backward hooks，打印 alloc/resv/max；rank 过滤 `FSDP_TURBO_MEM_DEBUG_RANKS`）。冒烟结果（train_batch 完成 173.5s、grad_norm 1.19、未 OOM，torch 峰 **54.93GB**、npu-smi 瞬态 ~62.5GB —— 比 2 节点多活的差额=碎片/通信缓冲的毫厘之差）：
  - **逐层轨迹（rank5，seq=1165）**：FWD 单调累积 **1.32→46.54GB（≈1.13GB/层，无中途释放）**；BWD 全程 alloc 稳在 **47.6–48.7GB（消耗后不释放！）**，BWD 起始瞬态把 max 冲到 54.93；直到 L0 BWD 结束仍持 48.68GB。
  - 结论：**update 峰值 ≈ Σ(逐层 saved-for-backward ≈45GB) + base(~10–15GB)**；这就是"平台顶"。下一步=子模块粒度（smoke2 进行中：扩展到 Block 直接子模块 `attn/ffn/engram/norm`）定位 1.13GB/层的大件，然后选**范围化 recompute**（避开 attention 自身的嵌套 checkpoint=上次整层 recompute 崩溃根因）或等价减法。已知大件参考：hc 残差流 [b,s,4,d]≈67MB/层@1165tok；[b,s,4,4,d]（=267MB/层）已在此前改写中消灭。
  - **smoke2（子模块 hooks，23:32-23:36，PASSED 242.4s）归属结果**：每层 ~0.6–1.2GB 里 —— `attn` ~0.2GB、`ffn`(MoE) 0.07–0.51GB（层间波动，异步分配噪声）、`*_norm` 各 ~0.05GB；**且 children 之间有 ~0.15–0.2GB/处的"缝隙"**（Block.forward 自身的 hc_pre/hc_mixes/hc_post 代码段，两处 ~0.38GB/层）。模型形状（config.json text_config）：hidden 5120、40 层、64 heads×head_dim 512、kv_heads 1、**index_topk 512 + sliding 128**（稀疏选择 KV [1165,640,512]≈763MB/层是"若被保存"的量级，但 recompute 路径实测只存 query/idx ≈90MB——flash-MLA 的 save 列表在只读的 npu 实现里确认了 query/result 各 76MB 级别）；MoE top-6/32 experts、inter 2304。
  - **smoke3（进行中）：`SMOKE_RECOMPUTE='model.layers.{*}.ffn'`（冒烟脚本新增 env）** —— 范围化 recompute 只包 ffn，验证 ①不触发上次的嵌套 checkpoint 崩溃 ②峰值下降量（预期 -12~-20GB）。通过则作为 2 节点正式配置的减载杠杆。
  - **smoke3 结果（23:47-00:05）：PASSED，峰值 54.93→50.21 GiB（-4.7），grad_norm=1.1893293857574463 与基线逐位一致**（recompute 数值零回归）；train_batch 242.3s（+69s = 重算开销）。注：ffn 重算的打印（`print_rank`）在冒烟日志中不可见但行为确证（时延+内存双证据）。
- **🎯 第 13 次 run（00:16:02，冲刺版 = resp1024/B2 + `memory.recompute`（ffn 范围）+ LOCAL_EXPERT_EXPORT + SYNC_PROFILE）——内存墙确认攻破、同步确认提速，但 1024 预算仍不够：**
  - Hydra 坑：`recompute_plan` 的覆盖也必须带 `+`（原始错误：`Could not override ... To append to your config use +...memory.recompute_plan=[...] / Key 'recompute_plan' is not in struct / object_type=dict`）；`+memory.recompute=true` 可建面，plan 键需 `+`。
  - **update#1 存活**：采样峰值 ~57GB（npu-smi）过关；指标行 `actor/perf/max_memory_allocated_gb:37.98 ... max_memory_reserved_gb:54.90`；OOM/Traceback=0。**ffn-recompute 的 -4.7GiB 恰是生死差**（run12 死于 55.03GiB allocated + 300MiB 小分配）。
  - **同步 876s→13s 实锤**：`SYNC-PROFILE step1 (revert) 0.0s, step2 (receive+load) 12.9s, step3 (post-load) 0.0s, total 13.0s`。
  - 训推一致最优：`rollout_actor_probs_pearson_corr:0.9785 / rollout_probs_diff_mean:0.0384 / rollout_corr/kl:0.2259 / actor/entropy:3.38`（对照第 10 次 0.9598/0.0550/0.2913/1.90——LOCAL_EXPORT + ffn-recompute 均无伤，且更优）。
  - **但 `response_length/mean:1024.0`（100% 顶格）+ `critic/rewards/mean:-1.0` + `grad_norm:0.0` 依旧** → 排除项穷举：EOS 配置正确（model/config `eos_token_id=1`＝`<｜end▁of▁sentence｜>`＝tokenizer 一致）、回落 builder 渲染逐字节正确（`<｜begin▁of▁sentence｜><｜User｜>…Answer:".<｜Assistant｜><think>`，185 tokens 实测）、`ignore_eos=False`。**结论：真实超长 CoT（DAPO 难题+思维链），1024 不够。** gen#1 仅 ~2 分钟（≈8.5 tok/s，KV 池 37.3GB 满档健康）。
- **第 14 次 run（00:46:54，最终探测版）：resp 2048/B2**（内存外推 alloc ~47GB、reserved 或逼近 57——存活概率较高；若 2048 仍 100% 截断 ⇒ 结论="需 >2k 预算"。deadline=机器回收 ~01:35）。
- **第 14 次 run 结局（01:14）+ 终止定案（用户 01:15 指示"不要再测试了"）**：update#1 OOM——`torch.OutOfMemoryError: NPU out of memory. Tried to allocate 1.23 GiB (NPU 4; 61.27 GiB total capacity; 53.91 GiB already allocated ...)`，峰值采样 60852MB（同一平台顶；-4.7GiB 不够 2048 用，2048 demand 仍比 1024 高 ~5-8GB）。双节点 ray 已停、卡已清。
- **🏁 综合结论（模型侧）**：pipeline 侧嫌疑全数排除（prompt 渲染逐字节✓、EOS 全链路✓(eos=1)、ignore_eos=False✓、reward 函数与 ground_truth✓、训推一致最优 pearson 0.9785/kl 0.226/diff 0.0384）；**512/1024/2048 三档预算全部 100% 顶格、零 EOS（三档共计几百条样本）**。用户假设"384→32 专家裁剪致生成退化/循环"与现象吻合。**唯一下一步判别（非 RL，~10 分钟）**：`scripts/ask_dsv41_serve.sh` 离线单样本生成看文本（循环/散乱 ⇒ 退化实锤，需 SFT 热启动教"终止+Answer"；连贯但超长 ⇒ 只需更大预算+内存优化）。未裁剪 top64/全量权重可作免费对照。
- **可复用资产（本日冲刺交付，双节点已同步）**：权重同步 876s→13s（LOCAL_EXPERT_EXPORT + SYNC-PROFILE 实测）；ffn 范围 recompute（-4.7GiB、grad_norm 逐位一致）；逐层内存探针 `FSDP_TURBO_MEM_DEBUG=1`（平台顶机理=每层 ~1.1GB saved-for-bwd 累积且 backward 不释放）；step 级 reward+一致性指标管线全通；内存安全区=resp≤1024（alloc 峰 38GB/res 54.9 存活），2048 尚差 5-10GB（剩余杠杆 cpu_offload / Ulysses SP=2）。

## 四、运行观察要诀（今日新得）

- **.124 的时钟比 .131 慢约 4 分钟**（2026-10-09 10:37 确认：同 job `41a85ebf` 的 .124 worker 行时间戳早于 driver 启动时刻约 3.5–4 分钟；driver 10:35:14 启动而 .124 worker 记 10:31:58）。判读 .124 来源行（`ip=80.5.25.124` 前缀）时按其本地时钟 -4min 换算。
- 本容器 **`ss` 命令输出恒为空（坏）**，查连接用 `/proc/net/tcp`（本文用其把到 .124 的 80 条 ESTABLISHED 映射到 ray::WorkerDict）。
- **rank0 的 `print` 经 Ray 转发有明显延迟**（数分钟~十几分钟）；第一现场是各节点本机的 `/tmp/ray/session_latest/logs/worker-*.out`。
- 阶段判读：engram 段唯一无日志，靠 py-spy 栈区分——`adapter.py:350`=建图参数初始化 / `host_offload_embedding.py:280`+`shard_/pre_fsdp_hook`=engram 填表 / `broadcast`=广播 / `_read_direct_shards`=直载。

## 五、待办

- [x] **单机冒烟验证（2026-10-09 10:22 通过）**：`TRAIN-ENGINE SMOKE PASSED` —— 直载（9.1s）→ engram（12.6s）→ 首 forward（CPU-offload 修复生效）→ backward/optimizer（grad_norm 1.85）→ 导出 payload（1364 tensors/0.8s）全绿。0 个 Traceback。
- [ ] 同步最新 `fsdp_turbo_dsv41_impl.py` 到 `.124`（**含今天全部 5 处修复**），重启双机，验证 `parameters materialized in Xs (direct shard load)`（预期 1–3 分钟）
- [ ] `run_grpo_2node.sh`/`run_grpo_top64.sh` 已默认导出直载 + 性能开关；`.124` 同步后直接起
- [ ] `.124` 取一次 raylet 死因实锤（dmesg OOM）—— 该机现被其他租户占用，等释放
- [ ] 本节点卡是共享的：起任务前 `pick_devices()` 已内置，但若运行中被外容器抢卡仍会 OOM（误报签名见 §一 09:33 行）
