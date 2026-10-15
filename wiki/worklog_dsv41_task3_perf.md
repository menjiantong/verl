# worklog · 任务3：性能数据采集（训练 MFU + 推理 profiling）（2026-10-13 立项，任务2 完成后执行）

> 用户任务书（原文摘要）：任务2 跑完后收集性能数据——训练的 MFU、推理的 profiling；**先看 verl 的指导，
> 如果没有先设计方案，然后执行，输出 wiki**。

## 状态
- [x] 步骤1：调研 verl 自带指导（2026-10-13 晚，**有现成指导**）——
  - **MFU**：`verl/utils/flops_counter.py` 提供模型 FLOPs 估算（**不支持 deepseek_v41** → 日志警告
    "MFU will always be zero"）；`ray_trainer.py:1373` 把 actor 侧的 `actor/mfu` 映射成 `perf/mfu/actor`。
    → 官方路径 = 给 flops_counter 加 deepseek_v41（稀疏 attention top-k + MoE 激活专家口径）；
    备选 = 离线推算（见方案）。
  - **Profiling**：`docs/perf/verl_profiler_system.md`（global_profiler 配置：tool=nsys/torch/torch_memory、
    steps、save_path；role 级 enable/rank；**NPU 支持 torch_memory 快照（torch-npu==2.10.0.post4）**）；
    `docs/perf/torch_profiling.md`、`docs/perf/nsight_profiling.md`（N/A）；
    `docs/ascend_tutorial/zh/dev_guide/performance/ascend_performance_analysis_guide.md`（昇腾性能分析指南：
    使能方法、长耗时/空泡、负载均衡、算子初诊、shape 亲和）。
- [ ] 步骤2：设计采集方案（定稿见下）；[ ] 步骤3：执行采集；[ ] 步骤4：输出 wiki（`wiki/dsv41_perf_report.md`）。

## 方案（2026-10-13 晚定稿；任务2 run 完成后执行）
- **训练 MFU（两条腿）**：
  ① 离线推算（快、零风险）：用任务2 日志的 `timing_s/update_actor`、tokens/step(=global_seqlen)、FSDP16 卡数、
     模型配置（总参 250.14B、激活参数按专家路由估）算 6ND FLOPs → MFU = 实测算力/峰值；
  ② verl 原生化（选做）：扩展 `flops_counter.py` 支持 deepseek_v41（注意力按 top-k 选中数计、MoE 按激活专家计），
     让 `perf/mfu/actor` 直接出数。
- **推理 profiling**：
  ① 首选 vLLM 自带 torch profiler 采集器（`VLLM_TORCH_PROFILER_DIR` / start_profile API，torch_npu.profiler 后端）
     采集一个 rollout step 的 decode+prefill timeline → 算子级耗时 top 表；
  ② 备选/交叉验证：`msprof`（CANN 自带，PROF 环境变量）单进程采集；
  ③ 产出：gen 阶段耗时分布（gen 353s/step 的构成）、算子热点、与训练侧比对。
- **数据来源**：任务2 的 50 步 run（128+129）已完成 step1 基线：gen 353s、update_actor 377s、step 777s、
  tokens 16148、throughput 1.30 steps…（见任务2 worklog 记录区）。

## 记录区（追加式）
- **[00:0x 汇总] 性能报告已产出 → `wiki/dsv41_perf_report.md`**（50 步实测 timing 全表 / MFU 双口径推算
  ≈0.3–5.5%（offload 主导）/ 推理 profiling：msprof 采集被容器权限挡→ vLLM torch_npu 路线**采集成功**
  （8 rank 原始 trace @128:/tmp/vllm_prof），CSV 导出受同一权限阻塞、已文档化导出命令）。
- **[23:2x 启动] 数据源**：任务2 的 50 步 run（128+129）已完成 → 全程 timing 已提取：
  gen 均值 250.8s / update_actor 80.2s / old_log_prob 8.2s / update_weights 16.7s / **step 均值 356.5s（min 176）**；
  **tokens/step 16564、seqlen/mean 1035**；导出权重 25.87GiB（≈13.9B bf16 可训练参数）；max_mem 39.0/54.9GB。
  模型配置（sft116 config.json）：40 层 / hidden 5120 / 64 头 / head_dim 512（+rope 64）/ **32 专家 top-6 + 1 shared**、
  moe_inter 2304 / vocab 129280 / sliding 128 / index_topk 512。
- **[23:3x 执行] 引擎侧 profiling 路线**：① `msprof`（CANN 9.1 自带）→ **被容器权限挡**
  （`Operation not permitted`，内核级采集不可用）；② 改走 **vLLM 原生 profiler**（`llm.start_profile/stop_profile`
  + `VLLM_TORCH_PROFILER_DIR`，底层 torch_npu.profiler）→ dump 脚本已加 `VA_DUMP_PROFILE=1` 门控（2 处）
  → 在 **128**（8 卡空闲）跑 len200 离线引擎前向采集，输出 `/tmp/vllm_prof`（每 worker 一份 trace）。
  首跑因 ssh 缺 CANN env 夭折（F-51），带 env 重跑中。解析目标：`*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
  → 算子级耗时 top 表。
