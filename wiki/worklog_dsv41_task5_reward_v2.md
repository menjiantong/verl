# worklog · 任务5：reward 不上涨攻坚 v2 —— 关算子替换 + GSM8K 打分器修正 + 109/110 跑 50 step（2026-10-14 启动）

> 用户指令（原文）："当前任务还是解决reward不上涨的问题，1、之前做的算子替换，没有什么提升，可以都关掉了。
> 2、之前的gsm8k转化的脚本不对，导致使用的dapo的打分器，改成训练集，使用gsm8k的打分器。然后109 110重跑50 step。
> 注意同步下代码和环境。"

## 变更 A：算子替换全部关闭（回退到原始口径）
- 引擎侧：`VA_V41_OPROJ_FP32_AR=**0**`（显式覆盖启动器默认 1）、`VA_V41_ATTN_EAGER` 不设（=0，融合核）、`FP32_PARTIAL` 不设（此前已因 OOM 停用）。
- 训练侧：`VERL_DSV41_HC_BF16 / ALIGN_ATTN / MOE_BF16` 全部不设（=0）。
- 代码中的修复**保留**（eager 的 masked_fill NaN 修复、wait 镜像——开关关掉即休眠）。
- 理由：用户判定算子替换无收益；关掉后 gen 更快（融合核比 eager 快 ~50%）、更少变量。
- 预期一致性略降（回到 ~0.96 带宽内含噪）；本任务以 reward 趋势为先。

## 变更 B：GSM8K 数据打分器修正（根因 + 实证）
- **根因**：`scripts/prepare_gsm8k_rl.py` 旧版刻意抄了 dapo 模式——`data_source="math_dapo"` +
  `style="rule-lighteval/MATH_v2"` ⇒ `default_compute_score` 把全部 GSM8K rollout 送进 **DAPO/MATH 打分器**。
- **实证差异**（131 上直接对跑两打分器）：
  | 输出形态 | dapo（旧） | gsm8k strict | gsm8k flexible（新） |
  |---|---|---|---|
  | `Answer: 18` | +1 | 0 | +1 |
  | `Answer: $18`（模板字面量） | +1 | 0 | +1 |
  | **`Answer: 18.`（句号尾）** | **-1**（pred='18.' 不去尾点） | 0 | **+1**（补 rstrip） |
  | `\boxed{18}` | +1 | 0 | +1 |
  | `#### 18` | -1 | +1 | +1 |
  ⇒ dapo 对本协议的两个系统性误判：**尾点**与 **#### 形态**；SFT 模型按 "Answer:" 协议输出 ⇒ 旧跑 reward 被系统性压低。
- **修法（三处）**：
  1. `prepare_gsm8k_rl.py`：`data_source="openai/gsm8k"`、`style="rule"`；已重生成两 parquet
     （train 7473 / test 1319，旧文件存 `*.parquet.dapo.bak`）；schema 与 dapo 文件逐列一致。
  2. `verl/utils/reward_score/__init__.py`：gsm8k 分支以 `method="flexible"` 调用（默认 strict 只认 `#### N`，
     会全零）；注释说明。
  3. `verl/utils/reward_score/gsm8k.py`：flexible 提取补 `rstrip(".")`（对齐 `eval_gsm8k_serve.norm_num`）。
- 端到端验证（default_compute_score 走 data_source="openai/gsm8k"）：5/5 形态全 1.0（旧 dapo 路径在
  period/hash 两例为 -1）。**奖励域从 {+1,-1} 变为 {0,1}**（GRPO 组内归一化无感，曲线读法需换尺度）。
- 保留：instruction wrapper（Answer: 协议，与 SFT 训练一致）、TRAIN/BSZ=10、lr 2e-6、use_kl_loss=True
  （task4 既定，用户未推翻）。

## 机器与同步
- **master=109 / slave=110**（两机 16 卡全空 ~3GB 基线、/mnt/share 挂载、权重/数据/tokenizer ✓）。
- 同步：131 四棵树（含今晚修复版 dsa_v41 dd08e172）+ 顶层 `scripts/`（含新 prepare 脚本）tar-over-ssh →
  双机；随后 md5 对账（4+3 文件）。

## 启动命令（留档）
```bash
# master（109）：
ssh -p 50001 root@80.5.25.109 "bash -lc 'cd /workspace-verl/verl && RAY_PORT=6390 \
  VA_V41_OPROJ_FP32_AR=0 TRAIN_BSZ=10 \
  EXPERIMENT_NAME=GRPO-DeepSeekV41-sft116-gsm8k-scorer \
  bash scripts/dsv41_deploy/launch_grpo_master.sh 50'"
# slave（110，等 master 出现 waiting for the slave 后）：
ssh -p 50001 root@80.5.25.110 "bash -lc 'cd /workspace-verl/verl && RAY_PORT=6390 \
  VA_V41_OPROJ_FP32_AR=0 TRAIN_BSZ=10 \
  EXPERIMENT_NAME=GRPO-DeepSeekV41-sft116-gsm8k-scorer \
  bash scripts/dsv41_deploy/launch_grpo_slave.sh 80.5.25.109'"
```

## 记录区（追加式）
- [ ] 同步完成 + md5 对账
- [ ] 启动 + 首步 reward 形态（期望不再是 -0.8 量级；含 KL/ref、无算子开关）
- [ ] step 1-50 reward 曲线（20 步趋势 / 50 步汇总）
- **[10:17 启动]** 同步完成：**四文件三机 md5 全一致**（dsa_v41 dd08e172 / reward_score/__init__ 9c4c884d /
  gsm8k.py cb5564f0 / prepare_gsm8k_rl.py b4c94029）。master=109 日志
  `/tmp/grpo_2node_master_20261014_101737.log`；slave=110 `/tmp/grpo_2node_slave_20261014_101833.log`；
  verl 日志 `/workspace-verl/verl/logs/grpo_top64_steps50_20261014_101916.log`。
- **[10:20 集群就位]** "slave present (2 active nodes, 16 NPU)"，driver 起。hydra 核对：
  `lr=2e-6 ✓ / use_kl_loss=True ✓ / ref.strategy=fsdp_turbo_dsv41 ✓ / train_files=gsm8k_main_train_rl.parquet ✓ / 50 steps ✓`；
  算子开关均未导出（EAGER/AR/PARTIAL/HC_BF16/ALIGN_ATTN/MOE_BF16 全关；AR 显式 =0 覆盖启动器默认）。
- **[10:25 采集器]** 131 后台每 5min 抓 step 指标 → `/tmp/run109_metrics.log`（采至 step50）。
- 时间预期：engram/init ~40-55min（无缓存路径）→ step1 约 11:15-11:30；每步 ~30min（gen 主导）。
- **[10:46 ★ 首跑失败：引擎 engram 初始化 pinned 分配被拒]** master 日志报
  `RuntimeError: Engine core initialization failed`；根因行（110 侧 worker）：
  **`RuntimeError: aclrtMallocHost failed: rc=207001 size=24576974336`**（24.58GB = 一张 engram 表的 rank 切片，
  引擎侧 host-registered pinned 分配）。仅 110 出现（2 条），109 无。
  - **根因判定**：`vm.overcommit_memory` 三机不一致——**109=1（总是允许）、110=0（启发式拒绝）、123=0**；
    失败发生在 init 峰值（trainer engram 391GB + 引擎 engram 连载 + trainer 读盘产生的数百 GB NFS 页缓存），
    mode=0 的启发式在此水位拒绝 24.6GB 不可回收 pinned 分配。127/123 凌晨同拓扑能过=运气（页缓存水位侥幸）。
  - **修复**：两机 `sysctl -w vm.overcommit_memory=1` + `drop_caches`（页缓存 535GB→2GB，free 2.4TB）。
  - 失败库立项（附：24.6GB aclrtMallocHost + overcommit 配方）。
- **[12:09 重启]** master(109) `/tmp/grpo_2node_master_20261014_120911.log`；slave(110)
  `/tmp/grpo_2node_slave_20261014_121002.log`；新 verl 日志 `…_121044.log`；集群 16 NPU 就位。
  采集器 v2（整行抓取，已修正旧版 ssh 引号问题）→ `/tmp/run109_metrics.log`。
  预期 step1 ≈ 13:00-13:30。
