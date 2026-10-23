# 123 dummy 5-step A/B（2026-10-22）

## 1. 目的

在 **80.5.25.123** 上用 `load_format=dummy` 跑 5 step，与 **80.5.25.119** 上正在跑的 `load_format=auto` 长跑前 5 step 对比，验证 dummy 是否影响 `training/rollout_probs_diff_mean` / `pearson` 的寻推一致性。

控制变量：同一 25 段、同一 checkpoint、同一 countbenchqa_lite 数据、同一拓扑/bsz/prompt/response/KL/util/agent/thinking/sampler；**唯一差异是 load_format**。

## 2. 配置对比

| 项 | 119 auto（baseline） | 123 dummy（A/B） |
|---|---|---|
| 节点 | 80.5.25.119 | 80.5.25.123 |
| 实验名 | `GRPO-DSV41-countbenchqa-W2-longrun-20261022_132541` | `GRPO-DSV41-countbenchqa-W2-dummy5-123-20261022_144123` |
| `load_format` | **auto** | **dummy** |
| 模型 | `DeepSeek-V4.1-Flash-top32-bf16-noengram` | 同左 |
| 数据 | countbenchqa_lite train/val | 同左 |
| 拓扑 | TP8×DP1×EP8；FSDP16/EP8/EFSDP1/TP1 | 同左 |
| batch | 8 prompts × 8 samples | 同左 |
| prompt/response | 1152 / 512 | 同左 |
| KL / util | KL0 / 0.6 | 同左 |
| vision / freeze | 1 / 1 | 同左 |
| agent / thinking | `deepseek_v41_vl_single_turn` / false | 同左 |
| sampler | `sync_refill_failed_groups=True` | 同左 |
| steps | 500 | **5** |
| save/test | -1 / -1 | 同左 |

## 3. 启动

- 脚本：`verl/my_scripts/1022_dsv41_grpo_2node_new_dummy5_123.sh`
- 启动方式：在 .123 上 `nohup bash my_scripts/1022_dsv41_grpo_2node_new_dummy5_123.sh 5 > logs/dummy5_123_20261022_144123.nohup.log 2>&1 < /dev/null &`
- 启动时间：2026-10-22 14:41:23 UTC
- PID：wrapper 564548；`main_ppo` 564789
- 日志：
  - outer：`verl/logs/W2_dummy5_123_20261022_144123.outer.log`
  - inner：`verl/logs/W2_dummy5_123_20261022_144123.inner.log`
  - nohup：`verl/logs/dummy5_123_20261022_144123.nohup.log`
- 启动前检查：.123 `main_ppo` 为空、NIC `enp48s3u1u1` 存在、repo 存在、NPU HBM 0。
- 配置核验：命令含 `actor_rollout_ref.rollout.load_format=dummy`；resolved config 里 rollout `'load_format': 'dummy'`；`Total training steps: 5`；train dataset 450。

## 4. 初始状态（2026-10-22 14:43 UTC）

- `main_ppo` PID 564789，状态 `Sl`，elapsed 1:55。
- 0 step；日志显示 `create worker group`、`Total training steps: 5`，engine 初始化中。
- NPU HBM 采样 0（模型尚未加载），Power ~160 W，Temp 31-33 °C。
- 无 Traceback/OOM/HCCL 错误。

## 5. 119 auto 基线（steps 1-5）

| step | diff_mean | pearson | reward | resp_len |
|---|---|---|---|---|
| 1 | 0.007824 | 0.998856 | 0.1688 | 433.5 |
| 2 | 0.010621 | 0.997587 | 0.3375 | 389.7 |
| 3 | 0.009938 | 0.997756 | 0.4641 | 338.9 |
| 4 | 0.010098 | 0.997951 | 0.5766 | 260.2 |
| 5 | 0.010893 | 0.997295 | 0.5625 | 274.8 |

## 6. 判读标准

- 如果 dummy 在首次全量同步后与 auto 等价：123 的 step1-5 应与 119 同 step 在 run-to-run 噪声内，step1 尤其应接近（`ray_trainer.py:1428-1430` 在第一次 rollout 前已 `update_weights`；非 LoRA 同步完整 `state_dict()`，含 vision）。
- 如果 dummy 的 step1 reward/diff 明显更差：说明存在未同步的 vision/非持久 buffer/Engram 状态，需要查同步 payload/tensor names；不要继续假设“dummy 更稳”。
- 不重启被杀的 run；报错先分析；不改配置；5 step、无 checkpoint。
- 重点指标：`training/rollout_probs_diff_mean`（目标 <0.01）、`training/rollout_actor_probs_pearson_corr`（目标 >0.99）、`critic/rewards/mean`、`response_length/mean`；附 HBM/CPU/grad/pg_loss/diff_std。

## 7. 巡检

- 每 15 分钟检查 119 和 123；123 结束后汇总 5-step 对比，写入本文件。
- 119 上另有一个 GSM8K `perf_baseline_profile_20261022_114823`（PID 6300，状态 `Tl` stopped，非本 run），不要 kill；当前 countbenchqa run 是 PID 237305。
- 2026-10-22 14:53 UTC：123 仍 0 step；`Total training steps: 5` 已于 14:43:27 打印；outer log 124,138 B，最后写入 14:51:42（NFS 时钟，比 123 本机快约 7 min）。wrapper PID 564548 存活、dummy5 main_ppo 在；NPU 每 chip HBM 2.9-3.3 GB（整机 4.7%）、AICore 0%，仍处 engine 初始化；无 Traceback/OOM/HCCL 错误。119 同期 step9 diff 0.008693 通过。


## 8. 巡检记录

| UTC 时间 | 状态 | step | diff_mean | pearson | reward | resp_len | npu-smi HBM GB | main_ppo PID | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-22 14:52 | INIT | 0 | - | - | - | - | 7.6-8.0 | 564789 | engine init（vLLM EngineCore），日志持续增长，无错误 |
| 2026-10-22 15:00 | INIT/RUN | 0 | - | - | - | - | 48.4-48.8 | 564789 | 首次全量同步完成（54 buckets/1664 tensors，load 9.1s）；Training Progress 0/5，step1 rollout running:8 finished:0；无报错 |
| 2026-10-22 15:03 | RUN(step1) | 0 | - | - | - | - | 60.0-61.1（92.0%） | 564789 | rollout 已结束（sleep returned），进入 old_log_prob/update；日志 1.55MB、无报错；HBM 92% 需关注 |
| 2026-10-22 15:16 | step1 | 1 | 0.007530 | 0.998850 | 0.1266 | 488.20 | 52.9-53.3（81%） | 564789 | alloc/reserved 26.06/48.00 GB；CPU 1486.25 GB；grad 1.89；pg_loss 0.0334；diff_std 0.01687；step 607s；与 119 auto step1（0.007824/0.998856）基本一致；无报错 |
| 2026-10-22 15:26 | step2 | 2 | 0.010729 | 0.997567 | 0.2250 | 456.05 | 24.3-29.3（37.9%） | 564789 | alloc/reserved 26.46/48.05 GB；CPU 1484.74 GB；grad 1.62；pg_loss 0.0769；diff_std 0.02481；step 457s；diff 与 auto step2（0.010621/0.997587）几乎一致；dummy 连续两步 reward 偏低、resp 偏长（0.225 vs 0.3375，456 vs 390），待 5 步判断；无报错 |
| 2026-10-22 15:35 | step3 | 3 | 0.010532 | 0.997096 | 0.4078 | 424.14 | 17.5-17.9（26.9%，步间） | 564789 | alloc/reserved 30.44/48.09 GB；CPU 1487.68 GB；grad 2.14；pg_loss 0.0731；diff_std 0.02626；step 430s；3/5；dummy 三步 reward 均低于 auto、resp 均更长（0.4078 vs 0.4641；424 vs 339），一致性指标接近；无报错 |


### 8. 123 dummy 5-step 完成与对比（2026-10-22 15:39 UTC）

- 5/5 完成：`Training Progress: 100%|██████████| 5/5 [38:46<00:00, 436.72s/it]`；`Final validation metrics: None`；wrapper 564548 / main_ppo 564789 均已退出。
- 退出阶段 traceback（非训练失败）：5/5 与 metrics 打印完后，atexit 回调 `_exit_function` join DataLoader worker 时抛 `RuntimeError: DataLoader worker (pid 619094) is killed by signal: Killed.`；发生在训练完成之后，5 个 step 指标完整，A/B 数据可用。

| step | dummy diff | auto diff | Δdiff | dummy pearson | auto pearson | dummy reward | auto reward | dummy resp | auto resp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.007530 | 0.007824 | -0.000294 | 0.998850 | 0.998856 | 0.1266 | 0.1688 | 488.2 | 433.5 |
| 2 | 0.010729 | 0.010621 | +0.000108 | 0.997567 | 0.997587 | 0.2250 | 0.3375 | 456.0 | 389.7 |
| 3 | 0.010532 | 0.009938 | +0.000594 | 0.997096 | 0.997756 | 0.4078 | 0.4641 | 424.1 | 338.9 |
| 4 | 0.010882 | 0.010098 | +0.000784 | 0.997169 | 0.997951 | 0.3937 | 0.5766 | 430.0 | 260.2 |
| 5 | 0.010411 | 0.010893 | -0.000482 | 0.996171 | 0.997295 | 0.3656 | 0.5625 | 391.9 | 274.8 |

- 一致性结论：5 步 Δdiff 均值 +0.00014、Δpearson 均值 -0.00052，均在 ±0.001 判据内；step1 Δdiff -0.00029、Δpearson -6e-6，几乎重合。**load_format=dummy 未改变寻推一致性**，与第 10 节代码分析（首次 rollout 前全量同步、非 LoRA 走完整 state_dict、dummy 不影响非持久 buffer）一致。
- 仅 step5 pearson 差 -0.00112，略超 ±0.001 噪声带；dummy 0.99617 仍 >0.99，整体不构成退化证据。
- reward 与 resp 存在系统性差异（dummy 5 步 reward 低 0.04-0.20，resp 长 55-170 tokens）。load_format 不能解释该差异（SYNC-PROFILE 两边均为 1664 tensors / 26.19 GiB），更可能是 123 vs 119 的节点/采样轨迹差异；若要归因需同节点、同 seed 重跑 auto。
- 资源：HBM alloc 26.06→42.44 GB（reserved 48.00-48.33 GB），无 OOM；CPU mem 1484-1490 GB 是 123 host 额外占用（free -g: used 1658 vs 119 1291 GB），非 dummy 导致。
| 2026-10-22 15:36 | step4 | 4 | 0.010882 | 0.997169 | 0.3937 | 430.03 | —（步间低点） | 564789 | alloc/reserved 42.44/48.27 GB；CPU 1488.41 GB；grad 1.61；pg_loss 0.0974；diff_std 0.02675；step 449s；无报错 |
| 2026-10-22 15:37 | step5 | 5 | 0.010411 | 0.996171 | 0.3656 | 391.88 | 3.0-3.3（4.6%，进程退出） | 564789 | alloc/reserved 42.44/48.33 GB；CPU 1490.39 GB；grad 2.24；pg_loss 0.1581；diff_std 0.02655；step 383s；5/5 完成 |
| 2026-10-22 15:37 | DONE | 5 | - | - | - | - | 3.0-3.3（4.6%，procs=0） | — | `Training Progress: 100%|██████████| 5/5 [38:46]`；末尾 `DataLoader worker ... killed by signal: Killed` 位于 `Exception ignored in atexit callback`、`Final validation metrics: None` 之后，属退出清理噪声；正常完成，不重启 |


## 9. 5-step 单变量对比结论（2026-10-22 15:37 UTC）

123 dummy 已跑满 5/5（`Training Progress: 100%|██████████| 5/5 [38:46]`），进程正常退出。末尾 `RuntimeError: DataLoader worker ... killed by signal: Killed` 出现在 `Exception ignored in atexit callback` 中，且在 `Final validation metrics: None` 之后，属于解释器退出时的 DataLoader 清理噪声，不是训练崩溃；按规则不重启。

### 9.1 逐步对比

| step | 119 auto diff | 123 dummy diff | 119 pearson | 123 pearson | 119 reward | 123 reward | 119 resp | 123 resp |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.007824 | 0.007530 | 0.998856 | 0.998850 | 0.1688 | 0.1266 | 433.5 | 488.2 |
| 2 | 0.010621 | 0.010729 | 0.997587 | 0.997567 | 0.3375 | 0.2250 | 389.7 | 456.0 |
| 3 | 0.009938 | 0.010532 | 0.997756 | 0.997096 | 0.4641 | 0.4078 | 338.9 | 424.1 |
| 4 | 0.010098 | 0.010882 | 0.997951 | 0.997169 | 0.5766 | 0.3938 | 260.2 | 430.0 |
| 5 | 0.010893 | 0.010411 | 0.997295 | 0.996171 | 0.5625 | 0.3656 | 274.8 | 391.9 |
| 均值 | **0.009875** | **0.010017** | **0.997889** | **0.997371** | **0.4219** | **0.3038** | **339.4** | **438.1** |

### 9.2 结论

1. **load_format 不是一致性的决定因素**：dummy 5-step 平均 diff `0.010017` vs auto `0.009875`（+0.000142，+1.4%），pearson `0.997371` vs `0.997889`（-0.00052）；两者都在 step1 后从 ~0.0075 跳到 ~0.0105，随后在 0.01 附近波动。dummy 没有改善，也没有明显恶化。代码路径也支持：首次 rollout 前已 `update_weights(0)` 全量同步（`ray_trainer.py:1428-1430`），非 LoRA `state_dict()` 含 vision/aligner（`transformer_impl.py:1049-1050`），`base_sync_done` 只影响 LoRA（`engine_workers.py:673,790-806`）。
2. **共同模式是 update 后的分布漂移**：两个 run 的 step1 diff 都在 0.0075-0.0078、pearson ~0.9989；step2 起 diff 升至 0.0105 左右、pearson 降至 0.996-0.9978。这与 actor update 幅度/方向、以及 FSDP 无 router replay 下 MoE 路由翻转的机制一致，而不是 load_format。
3. **dummy 的次要差异**：5/5 步 reward 都低于 auto（均值 0.3038 vs 0.4219，低 0.118），response 都更长（438.1 vs 339.4，长 98.7 token）；step 时间几乎相同（465.1 vs 456.7 s，+1.8%）。因为 step1 的 diff/pearson 几乎相同，这不像“权重未同步”的特征（那会先抬高 diff），更像 dummy 加载路径带来的采样 RNG/轨迹差异或 run-to-run 方差；5 步单 seed 不足以区分，需多 seed 或更长 A/B 才能定论。
4. **资源**：actor alloc/reserved 峰值 42.44/48.33 GB（dummy）vs 42.44/48.49 GB（auto），基本一致；npu-smi 峰值 dummy 60.0-61.1 GB/chip（92.0%），auto 53-60.4 GB/chip（最高单卡 98.7%）。dummy 的 `perf/cpu_memory_used_gb` 约 1487 GB，比 auto 的 ~1043 GB 高约 444 GB，需确认是否节点/容器统计口径差异，暂不作为 load_format 结论。
5. **对当前 119 长跑的含义**：不要为了 diff_mean 回退 `load_format=dummy`。当前 119 的连续超标应通过单变量 A/B（LR 5e-6→3e-6 或 batch 8→16）验证，不要同时改 load_format。
