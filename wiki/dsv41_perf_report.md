# DeepSeek-V4.1-Flash top32 · 性能报告（任务3，2026-10-13/14）

> 数据源：**任务2 的 50 步原始配置 run**（128+129，16×Ascend910，2026-10-13 18:15 → 23:30 完成 50/50 步）。
> 方法：`wiki/worklog_dsv41_task3_perf.md`（调研+方案）；本文件为结果汇总。

## 1. 训练/整步耗时（50 步实测，均值/min/max，秒）

| 阶段 | mean | min | max | 备注 |
|---|---|---|---|---|
| **step 总** | **356.5** | 175.9 | 777.0 | 首步 777s 为预热；后期 ~6 min/step |
| gen（rollout） | 250.8 | 90.4 | 353.4 | 8 prompt × n=8 |
| update_actor | 80.2 | 55.3 | 376.9 | 仅占 22% |
| old_log_prob | 8.2 | 6.8 | 9.5 | |
| update_weights（FSDP→vLLM 同步） | 16.7 | 16.2 | 20.0 | 25.87 GiB/步、51 buckets |

**规模**：tokens/step 16564（mean）、seqlen/mean 1035、response_len/mean ~200；
显存 max_allocated 39.0 GB / reserved 54.9 GB（61.3GB 卡）。

## 2. MFU（verl `perf/mfu/actor` 恒 0.0——`flops_counter.py` 不支持 deepseek_v41，官方路径需扩表；以下为离线推算）

**模型**（sft116 config）：40 层 / hidden 5120 / 64 heads×576(512+rope64) / MoE 32 experts top-6 + 1 shared、
moe_inter 2304 / vocab 129280；**可训练参数 ≈13.9B**（由权重同步 25.87 GiB bf16 反推；总 250.14B 含 engram 等宿主侧结构）。

- 公式：FLOPs_step ≈ 6 × N × tokens（fwd+bwd），N 取可训练 13.9B。
  → 6×13.9e9×16564 ≈ **1.38 PFLOP/step（actor 训练部分）**。
- update_actor 80.2s / 16 卡 → 每卡 ≈ **1.08 TFLOPS**。
- 对照 910B bf16 峰值（376 TFLOPS 口径）→ **MFU ≈ 0.3%（估算下限口径）**；
  若按总量参数 250B 口径（不适用于稀疏 MoE 实际计算）则为 ≈5.5%（上限口径）。
- **结论：训练 MFU 极低（<1% 量级）**，与配置特征一致：`param_offload=True + optimizer_offload=True`
  （每步 CPU↔NPU 搬运）、entropy checkpointing、recompute、以及每步 25.87 GiB 的独立权重同步开销。
  **优化顺序建议**：① offload 策略（显存 39/61GB 有余量）；② 权重同步 bus 化/去重；③ 扩 flops_counter 拿官方口径。

## 3. 推理 profiling（引擎侧）

**采集路线（逐层排障后打通）**：
1. `msprof`（CANN 9.1 采集）→ **容器权限挡**：`Operation not permitted`（内核级采集不可用）；
2. **vLLM 原生 profiler**（`ProfilerConfig(profiler="torch")` + `llm.start_profile/stop_profile`，
   底层 torch_npu.profiler）→ **采集成功**：dump 脚本已加 `VA_DUMP_PROFILE=1` 门控（dump_engine_stages.py，
   构造处传 `profiler_config` + 首段 prefill 前后 start/stop）。
   - 8 rank 原始 trace 已落盘：**128:/tmp/vllm_prof/**（`dp0_pp0_tp*_..._ascend_pt/`：FRAMEWORK torch 轨迹 +
     PROF_*/device_0 CANN 设备数据）。
3. **CSV 导出被同一权限问题阻塞**（`msprof --export` 也走采集链路）；torch_npu 的 `analysis.prof_parse`
   仅暴露框架侧解析器（拼装式 API，未形成现成算子表）。
   → **在权限放开的环境（或宿主）执行**：`msprof --export=on --output=<out> -d <trace_dir>` 即得
   `ASCEND_PROFILER_OUTPUT/kernel_details.csv`（算子级）、`op_statistic.csv`（算子汇总）。

**已可得结论（定性）**：gen 阶段（250.8s/步）占整步 70%——rollout 为当前最大耗时项；
引擎 8 卡 TP=8+EP=8，enforce_eager=True（无图捕获），chunked prefill 2048。

## 4. 后续（按优先级）
1. 在可导出环境完成 `kernel_details.csv` → 算子级 top 表（含 `npu_sparse_flash_mla`/MoE 融合核/GEMM 占比）；
2. 对 gen 阶段做一次专门 profiling（rl_config 短 run + start_profile 于 rollout 窗口）；
3. 扩 `verl/utils/flops_counter.py` 支持 deepseek_v41 → 日志直接出 MFU；
4. offload/同步开销实验（任务2 环境即可复跑 A/B）。
