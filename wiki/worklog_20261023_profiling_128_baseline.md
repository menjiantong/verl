# worklog 2026-10-23 采集 baseline profiling（最终 128 单机 16 卡；119 启动后按用户指示切换）

## 1. 目标与方案
- 任务：用用户给定的 `my_scripts/pref_dsv41/_launch_profling_1023.sh` 采集一次基线 profiling。
- 拓扑：单机 16 卡（80.5.25.119，`single` 模式，FSDP16/EP8），不启用任何性能优化开关。
- 模型/数据：`DeepSeek-V4.1-Flash-top32-bf16-noengram`；countbenchqa_lite train/val；
  max_prompt=1152、max_response=512、USE_KL=0、rollout mem util=0.6。
- 采集范围：actor / ref / rollout 三个角色的 rank0；`global_profiler.steps=[2]`，
  `profile_continuous_steps=False`（每步独立 trace）；NPU contents=`npu,cpu,memory,shapes`，level1。
- 设计理由：
  1. 选第 2 步：第 1 步含权重加载/编译/首步 Adam 惰性建状态等一次性开销（见 E38），
     第 2 步更接近稳态；3 步总长度可保证 profile step 完整执行。
  2. `profile_continuous_steps=False`：把 actor update / ref logprob / rollout 分开落盘，
     便于分别定位瓶颈，避免一个超大 trace。
  3. level1 + shapes/memory：需要 kernel 级耗时、算子 shape 和显存事件；level2 额外开
     AICore 采样，开销更大且首轮采集不需要。
  4. actor/ref `analysis=True` 让框架落盘后直接解析；rollout 的 vLLM 引擎侧开 `discrete=True`
     并关闭 analysis，避免引擎进程内解析阻塞采样。
- 资源预算：权重 ~126 GiB / 16 rank ≈ 7.9 GiB/rank；64 GiB HBM/卡，rollout util 0.6。
  宿主机 2.4 TiB 内存，当前 available 2.0 TiB。

## 2. 启动前检查（09:05-09:10 UTC）
- 脚本默认步数陷阱（证据）：`_launch_profling_1023.sh` 第 8 行 `STEPS=${1:-500}`，
  第 34 行 `STEPS=${STEPS:-3}` 永远不会生效（STEPS 已非空）→ 不带参数执行会跑 500 步。
  **本次显式传参 3 步**，不修改用户脚本。
- 119 残留进程（证据，npu-smi/ps，09:06 UTC）：
  - `VLLM::EngineCore` ×2（PID 275491/275830，PPID=1，已孤儿化，启动于 Oct22 11:32，
    elapsed 19:35:48），`VLLM::Worker_TP0..7_EP0..7` ×16（PID 276555-276594）。
  - 16 张卡 HBM 6.2–6.7 GB/卡，AICore 0% → 无有效计算，属于 Oct22 11:32 的
    `/tmp/ray_dsv41_perf_copy` 实验残留（与 W3 worklog 记录的 “119 被 W2 遗留 vLLM 占用” 一致）。
  - 3 个 `ray start` 进程（PID 2674/3514/4308，STAT=Tl 已停止，temp-dir=/tmp/ray_dsv41_perf_copy）。
  - 结论：均为本机历史实验的孤儿进程，可安全清理；否则新作业无法拿到干净 HBM。
- 磁盘：`/` overlay 12T 用 89%（余 1.4T）；`/mnt/share` 180T 用 95%（余 9.7T）。
  profile 输出目录 `/mnt/share/m00899630/workspace/dsv41-verl/profiles` 存在且可写。
- 配置键校验：`global_profiler.tool/steps/profile_continuous_steps/save_path` 与
  `NPUToolConfig.contents/level/analysis/discrete` 均在 `verl/utils/profiler/config.py` 与
  `verl/trainer/config/ppo_trainer.yaml` 中存在，参数名无误。

## 3. 执行记录
（见下方追加）

## 3. 执行记录
- 09:09:55 UTC：脚本参数键、磁盘、NPU 预检完成；119 存在 Oct22 11:32 的孤儿进程
  （2×`VLLM::EngineCore` + 16×`VLLM::Worker_TP*`，PPID=1，AICore 0%，HBM 6.2–6.7 GB/卡）。
- 09:10:00 清理残留：
  - `pkill -9 -f "VLLM::"` 清理 18 个孤儿；注意该模式也会匹配调用者自身的命令行，
    首次调用导致 SSH 会话被一并终止（工具侧误伤，不影响清理结果）。
  - `kill -9 2674 3514 4308` 清理 3 个 STAT=Tl 的旧 `ray start`（temp-dir=/tmp/ray_dsv41_perf_copy）。
  - 清理后 npu-smi：NPU0-7 均 "No running processes found"；HBM 降至空闲基线
    3.0–3.4 GB/卡（与 120 空闲卡一致），AICore 0%。
- 09:10:54 启动采集：
  `setsid nohup bash my_scripts/pref_dsv41/_launch_profling_1023.sh 3 > /tmp/profiling_119_20261023.out 2>&1`
  - launcher PID 2386418；外层日志 `logs/W3_longrun_20261023_091054.outer.log`；
  - STEPS=3、PROFILE_STEP=2；
  - SAVE_PATH=`/mnt/share/m00899630/workspace/dsv41-verl/profiles/run_baseline_profile_20261023_091054`。
- 09:13:52 框架日志：`worker group kwargs: {'profile_steps': [2], 'device_name': 'npu'}`
  → profiler 配置已正确注入 worker group。
- 09:14:37 actor 侧开始构建（AscendC attention/KDA fallback 为既有正常告警）。
- 资源监控：`my_scripts/pref_dsv41/1023_monitor_profiling_119.sh` 每 30s 采样
  NPU HBM/AICore 与主机内存 → `logs/profiling_119_1023_monitor.log`。
- 参数缺陷说明：`_launch_profling_1023.sh` 第 8 行 `STEPS=${1:-500}` 使第 34 行
  `STEPS=${STEPS:-3}` 永不生效；本次显式传参 `3` 规避，未修改用户脚本。

## 4. 切换容器 119 -> 128（用户指示，09:25 UTC）
- 用户要求改用 128 容器，随即终止 119 上的 3-step 采集。
- 119 处置：09:22 对 launcher 进程组执行 `kill -TERM -- -2386418`（当时仍在权重加载阶段，
  未进入训练步），随后 `pkill -9 -f "VLL[M]::"`、`pkill -9 -f "verl.trainer.main_pp[o]"`、`ray stop --force`。
  - 验证：PGID 2386418 无幸存者；ps 无 VLLM/main_ppo。
  - 注意：清理后 npu-smi 出现外部租户进程（host PID 144160+ / container PID 351+，
    HBM ~20 GB/卡，AICore 0%），与本次作业无关，未触碰。
- 128 容器检查（09:26 UTC，证据）：
  - NPU：16×Ascend910，AICore 全 0%；进程列表 NPU0-7 均 "No running processes found"；
    HBM 空闲基线 2.9–3.2 GB/卡（与 119/120 空闲卡一致）。
  - 主机：mem_total 2454G，used 30G，available 2393G；overlay 3.4T 用 71%（958G 可用）。
  - 网络：`enp48s3u1u1` 存在（MAC c8:a3:62:b4:24:ec），与启动脚本 NIC 一致。
- 新脚本：`my_scripts/pref_dsv41/1023_launch_profling_use128_v1.sh`
  （基于 `_launch_profling_1023.sh`，`diff -u` 核对仅两处差异）：
  1. `export MY_IP=80.5.25.128`；
  2. `STEPS=${1:-3}` 并删除永不生效的 `STEPS=${STEPS:-3}`（原默认 500 会误跑 500 步）。
  `bash -n` 通过。

## 5. 采集过程指标（128）
- step1（无 profiling）完成：`timing_s/step=503.23`（gen 210.61 / old_log_prob 28.76 /
  update_actor 218.78 / update_weights 24.68），`perf/throughput=6.04`，
  `response_length/mean=483.42`（clip_ratio 0.906），`critic/rewards/mean=0.225`。
- step2（profile step）09:43:57 开始：rollout 阶段持续运行，至 10:35 仍未结束，
  profile 目录从 0 增长到 35 GB（~0.7 GB/min）。对比 step1 gen 210.6 s，说明
  `contents=[npu,cpu,memory,shapes]` + level1 在 vLLM rollout 上的记录开销极大
  （trace 写入与 per-step synchronize 的 CPU 路径占主导，npu-smi AICore 多数采样为 0%，
  但 vLLM worker CPU 合计约 1700%）。
- 资源（采集期间稳定）：HBM 52.9–53.2 GB/卡（16 卡），主机 used 1211G / avail 1385G；
  与预估（rollout util 0.6 → ~38GB KV + 权重 + 激活）不超 64GB，符合预期。
- profile 目录：`/mnt/share/m00899630/workspace/dsv41-verl/profiles/run_baseline_profile_20261023_092633`
  （train/、agent_loop_rollout_replica_0/）。

## 6. 后续长跑准备（120，用户指定：采集结束后启动）
- 目标脚本：`my_scripts/_launch_W3_longrun_1023_change_score_and_len_thinking.sh`
  （基于 `_launch_W3_longrun_1022.sh`：MAX_RESPONSE_LEN 512→2048、countbenchqa2 打分、
  enable_thinking=true；用于验证训练中途指标恶化）。
- 120 预检（10:26 UTC）：脚本 `bash -n` 通过；16 卡全部空闲（HBM 2.9–3.2 GB/卡，
  npu-smi 8 个 NPU 均无进程）；无 VLLM/main_ppo/ray 残留；
  `verl/utils/reward_score/countbenchqa2.py` 存在；NIC enp48s3u1u1 存在；
  根盘 15T 用 97%（余 467G，注意 Ray 临时目录），/mnt/share 余 9.1T；mem avail 1700G。
- 对比基线 `my_scripts/_launch_W3_longrun_1022_log.txt`（110 step）：
  reward mean 0.169 → 0.9，response_length mean 433.5 → ~280，entropy 1.42 → ~0.09~0.11，
  pg_loss ~0.056 后基本不再下降，throughput ~4.3 → ~6.5；即长跑中 response 变短、
  entropy 塌缩、reward 饱和，是本次新脚本要验证/对比的“指标恶化”。

## 7. 采集完成（12:17 UTC）
- 3 步全部完成（log 出现 `training/global_step:3`），launcher 退出；128 清理干净：
  npu-smi 8 个 NPU 均无进程，HBM 回到 3.0–3.2 GB/卡基线，host used 30G。
- 三步指标对比（同一 3-step 任务，仅 step2 开 profiling）：

  | step | gen_s | old_log_prob_s | update_actor_s | update_weights_s | step_s | throughput | resp_len | clip | reward | entropy |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 210.61 | 28.76 | 218.78 | 24.68 | 503.23 | 6.036 | 483.42 | 0.906 | 0.225 | 1.227 |
  | 2 (profile) | 7644.17 | 27.21 | 194.96 | 24.98 | 8564.76 | 0.376 | 494.59 | 0.938 | 0.211 | 1.407 |
  | 3 | 155.73 | 28.68 | 209.78 | 24.54 | 419.03 | 7.596 | 436.52 | 0.781 | 0.450 | 1.018 |

- 结论（evidence）：
  1. profiling 开销几乎全部落在 rollout generation：gen 210.61 s → 7644.17 s = **36.3×**；
     old_log_prob / update_actor / update_weights 均无劣化（27.2/195.0/25.0 s）。
  2. 机制：vLLM 每个 decode step 在 `AsyncGPUModelRunnerOutput.get_output`
     → `Event.synchronize`（vllm/v1/worker/gpu_model_runner.py:357）；torch_npu profiler
     对 `Event.synchronize` 等做了 `record_function` 包装
     （torch_npu/profiler/profiler_interface.py:52）。contents 含 memory+shapes、level1 时
     CPU 侧事件记录成为瓶颈；采样期间多数时刻 npu-smi AICore=0%，但 device/host trace slice
     持续增长（每秒新增），说明不是死锁，而是 profiling 串行化/记录开销。
  3. 训练侧 trace 记录没有明显变慢，说明该开销主要来自 vLLM 的 per-step synchronize 路径。
- 产物（共 91 GB）：`profiles/run_baseline_profile_20261023_092633`
  - `train/node-29-128_1931412_.../PROF_000001_...`：9.7 GB，含
    `ASCEND_PROFILER_OUTPUT/`、`logs/`（actor/ref 的 analysis 已完成）；
  - `agent_loop_rollout_replica_0/dp0_pp0_tp{0..7}_.../PROF_000001_...`：82 GB，8 个 TP rank。
- 资源：全程 HBM 峰值 53.2 GB/卡（16 卡，余量约 11 GB/卡），host used 峰值 1240G / 2454G，
  未 OOM；/mnt/share 采集后余 9.1T。
- 后续可用 `ASCEND_PROFILER_OUTPUT/*.csv`（op_summary/op_statistic 等）做 kernel/算子分析；
  rollout 侧 `PROF_*/device_0/data` 与 `host/data` 可再用 msprof 解析。
