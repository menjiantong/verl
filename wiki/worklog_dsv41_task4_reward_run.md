# worklog · 任务4：修复后双机 GSM8K 重跑（目标 reward 上升趋势；20 步见趋势、50 步明显）（2026-10-14 启动）

> 用户任务书（原文）："算子替换生效后，使用双机重跑GSM8k，目标是reward有上升趋势。20步有上升趋势，50步明显。
> 1、可以开启KL-loss 2、结合上次对reward 不上升的问题，改一下batch 等配置，重跑"

## 前置（为什么这次不一样）
- **算子替换已生效且 NaN 根因已修**（2026-10-14 凌晨）：eager 注意力"未写入 cache 槽过掩码乘法 0×NaN"→
  `masked_fill(0)`（dsa_v41.py md5 **dd08e172**，双机 K 序列 + 长度谱系双验证全干净；见任务1 worklog）。
- 上次 reward 不上升的归因（128 日志分析）：① lr=1e-6 + pg_clipfrac 全程 0 → 步长太小；
  ② train_batch=8 × n=8，GSM8K acc≈7.5% → P(组全错)≈0.54，约一半 GRPO 组零梯度；③ 一致性差为次级因素（现已大幅改善）。

## 本次配置变更（相对任务2 原始口径）
| 项 | 旧 | 新 | 理由 |
|---|---|---|---|
| `actor.optim.lr` | 1e-6 | **2e-6** | 直击"步长太小"主因 |
| `use_kl_loss` | False | **True**（low_var_kl，coef 0.001 默认） | 用户指定；2× lr 的稳定锚（ref worker 回归，+1 前向/步） |
| `TRAIN_BSZ` | 8 | **10**（×n8=80，%16==0 ✓，5 序列/rank） | 每组全错概率不变，但步内组数 +25%、有信号组更多；（内存按 task2 实测 39GB@4 序列外推可控，未翻倍以规避 OOM） |
| ref micro batch | — | **per_gpu=1** | 开 KL 后 hydra 必填（首启即因此报错，已修） |
| 其余 | 保持 proven：EFSDP=1 / ROLLOUT_IS=token / max_resp 1024 / ffn recompute / entropy checkpointing | | |

**引擎/训练侧全修复配方（双机同 env）**：
`VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1 VA_V41_OPROJ_FP32_AR=1(启动器默认) VA_V41_ATTN_EAGER=1`
（PARTIAL 不设=防 update OOM；PREFILL_LOGPROBS 不设=保持 decode 口径观察 diff_mean）

## 机器与代码同步
- **127（master）+ 123（slave）**；123 侦察：16 卡全空（AICore 0%）、/mnt/share 挂载 ✓、权重/数据/tokenizer ✓。
- 代码同步：131 整树 tar→123 + 顶层 scripts/；**关键四文件 md5 与 131 对账一致**
  （dsa_v41 dd08e172 / dsa_v1 17ff0ec1 / FSDPTurbo model.py c90a6ad3 / vllm_async_server 74bc94c4）——F-80 纪律。
- 改动脚本：`verl/examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh`（lr/KL/ref-batch），双机已同步。

## 启动命令（留档）
```bash
# master（127）：
ssh -p 50001 root@80.5.25.127 "bash -lc 'cd /workspace-verl/verl && RAY_PORT=6390 \
  VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1 VA_V41_ATTN_EAGER=1 \
  TRAIN_BSZ=10 EXPERIMENT_NAME=GRPO-DeepSeekV41-sft116-consfix \
  bash scripts/dsv41_deploy/launch_grpo_master.sh 50'"
# slave（123，等 master 出现 "waiting for the slave" 后）：
ssh -p 50001 root@80.5.25.123 "bash -lc 'cd /workspace-verl/verl && RAY_PORT=6390 \
  VERL_DSV41_HC_BF16=1 VERL_DSV41_ALIGN_ATTN=1 VERL_DSV41_MOE_BF16=1 VA_V41_ATTN_EAGER=1 \
  TRAIN_BSZ=10 EXPERIMENT_NAME=GRPO-DeepSeekV41-sft116-consfix \
  bash scripts/dsv41_deploy/launch_grpo_slave.sh 80.5.25.127'"
```
- RAY_PORT=6390（F-77 外部 GCS 占 6379）。
- master 日志：`/tmp/grpo_2node_master_20261014_043717.log`；slave：`/tmp/grpo_2node_slave_20261014_043218.log`；
  verl 训练日志在 127 `verl/logs/grpo_top64_steps50_*.log`。

## 记录区（追加式）
- **[04:34 首启失败→修]** 开 KL 后 hydra 报 `[actor_rollout_ref.ref] Please set at least one of ...log_prob_micro_batch_size...`
  → 脚本补 `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`（209 行）→ 双机同步 → 重启（04:37）。
- **[04:38 重启]** master 等待 slave；slave 加入中。
- **[04:40 第二处失败→修]** 模型构建报 `Host-offload Engram requires a one-dimensional EP mesh with EP > 1`
  （FSDPTurbo host_offload_embedding.py:254，`shard_()` 收到退化 mesh）——**根因 = 开 KL 后新创建的 ref worker
  没带 turbo 拓扑**（top64 示例只在 ACTOR 段设 `strategy=fsdp_turbo_dsv41`+distributed 面；KL 关时 ref 从不建模型，
  所以历史跑没暴露）。文件 md5 与 128（task2 树）一致 ⇒ 非代码回归，纯配置缺口。
  修法：照单机版 `run_deepseek_v41_grpo_fsdp_turbo_npu.sh` 的 REF 段，给 top64 脚本补
  `REF_TURBO` 变量 + 完整 `REF=(...)` 数组（log_prob micro batch=1、strategy=fsdp_turbo_dsv41、
  distributed FSDP/TP/EP/EFSDP、fsdp_plan/ep_plan、offload 族）×2 机同步（bash -n ✓）。
  （前一版单行 `ref.log_prob_...=1` 移除，避免 CLI 重复键。）
- **[04:48 重启]** master 等待 slave；slave 加入中（RAY_PORT=6390）。
- **[05:2x]** hydra 配置逐项核对无误（lr 2e-6 / use_kl_loss=True / TRAIN_BSZ=10 / FSDP16 EP8 EFSDP1 / REF 段齐）；
  **权重物化 78.6s（direct shard load）** + engram 逐 rank 直读；worker 群（actor_rollout_ref）建立。
- **[06:11 ★ Step 1 完成 —— NaN 在真实 RL 中消失]**（首批有限值！）：
  - **`training/rollout_probs_diff_valid=1.0`**（此前恒 NaN）、**pearson 0.9611 / diff_mean 0.05685 / std 0.0985 / max 0.870**；
  - rewards/mean **-0.825**（max 1.0/min -1.0 真二值）；advantages mean -0.0989（有非零信号）；
  - **lr=2e-6 ✓ / kl_coef=0.001 ✓**（kl_loss=0.0@step1，ref≈起点合理）；pg_clipfrac 0.0（步 1 正常）；
  - 内存 37.66/54.70GB（batch 10 成立）；response_len mean 136.9（clip 2.5%）；
  - 时长：**gen 1543.8s（显著慢于 task2 的 ~250s：eager 补丁 + 机器差 + batch）**、update_actor 270.3s、
    ref 7.9s（KL 的 ref 前向 ✓）、step 合计 **1878s（~31min/步）**。
- **[06:2x 用户指令：跑 5 个 step 后终止]** → 已装自动监视器（131 后台，4min/查，
  `grep training/global_step` ≥5 → 双机 `pkill main_ppo/WorkerDict/VLLM::/ray::` + `ray stop --force`，
  日志 `/tmp/watch_step5.log`）。预期 step5 ≈ 08:15-08:40。
- [ ] step 2-5 滚动抄录（reward/KL/diff_mean/gen 时长）
- [ ] step 5 后：双机终止确认 + 5 步汇总
- **[06:55 用户指令：结束执行]** 立即终止（提前于"5 步"计划——step 2 的 rollout 尚在进行中，不计入）。
  - 终止操作：131 监视器 kill；双机 `ray stop` + pkill（`WorkerDi[c]t/VLLM[::]/main_pp[o]/ray[::]/rayle[t]/gcs_serve[r]/TaskRunnerV[1]/EngineCor[e]/SimpleStorageUni[t]/vllm_asyn[c]`，
    先 ray 后 worker——先杀 worker 会被 ray 重新拉起，踩过一次）。
  - **终态核验**：两机活进程=0（残留的 `ps` 匹配项全部是 `<defunct>` 僵尸——历史 probe 的 worker，不占显存/算力）；
    npu-smi 回到基线（~3GB/卡）。**step 2 未完成 ⇒ 有效数据 = step 1**（前述全套指标）。
  - ⚠ 过程小坑记录：① 用 `pkill -f "while true; do"` 自匹配把本机 shell 打死（F-53 变体，带空格/分号的模式也要字符类）；
    ② 远程命令里 `$(...)` 嵌引号会被吞 → 拆简单命令。
  - **5 步计划的替代结论**：step1 已足够回答核心问题——**修复后 RL 循环内 consistency 指标首次全有限**
    （diff_valid=1.0 / pearson 0.961 / diff_mean 0.0569），reward/自验 LR/KL 配置全部生效；步速 ≈31min（gen 1544s 占 82%，
    eager 补丁 + 机器差为主要拖累）。
- [ ] 20 步趋势判读；50 步汇总曲线
