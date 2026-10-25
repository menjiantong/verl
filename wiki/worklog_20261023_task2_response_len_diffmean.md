# worklog 2026-10-23 任务二：response length 与 rollout/train diff_mean 恶化

## 1. 问题与证据
- 现象：`_launch_W3_longrun_1022_log.txt`（110 step，512 response、旧 scorer、thinking=false）
  中 `training/rollout_probs_diff_mean` 在长跑中途明显恶化：
  | step | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | diff_mean | 0.0078 | 0.0100 | 0.0112 | 0.0152 | 0.0110 | 0.0310 | 0.0377 | 0.0349 | 0.0267 | 0.0256 | 0.0136 | 0.0079 |
  → step50–90 升到 0.026–0.038（约为初期 3–5 倍），step100 后回落。
- 用户假设：与 response length 有关。需要实验验证。

## 2. 实验一：短 response（max_response_len=50）
- 设计：基于 `_launch_W3_longrun_1022.sh`（512、countbenchqa、thinking=false、500 step 默认），
  仅把 `MAX_RESPONSE_LEN` 限制为 50（用户允许 20/50；取 50，给最终 boxed 答案留余量，避免 20
  时全部截断导致 reward 恒 0、实验失去区分度）。
- 预期与判读：
  - 若 diff_mean 不再在 step50–90 升到 0.03+，支持“response length 相关”假设；
  - 若仍恶化，说明与 response length 无关（更可能是 sampler/off-policy/权重同步等）。
  - 注意混杂因素：50 token 下若 reward 长期为 0（无有效学习），diff_mean 低不能单独作为证据；
    需同时看 `critic/rewards/mean`、`response_length/mean`、`actor/entropy`。
- 步数：150 step（覆盖基线恶化窗口 step50–90 并留观察余量）。
- 节点：80.5.25.122（16 卡全空闲，mem avail 2407G，/ 9.8T）。
- 配置核对（/proc cmdline 实测）：max_response_length=50、max_model_len=1202、
  enable_thinking=false、total_training_steps=150；未注入 custom_reward_function（沿用旧 scorer）。
- 启动：15:59:20 UTC，`setsid nohup bash my_scripts/_launch_W3_longrun_1022_maxresp50_v1.sh`；
  outer log `logs/W3_longrun_20261023_155920.outer.log`。

## 3. 实验二（条件触发）：长跑 + 定期存权重，复跑恶化步权重
- 触发条件：实验一看不出结论（diff_mean 仍恶化或 reward 恒 0 无法判读）。
- 设计草案：
  - 基线配置（512 response、旧 scorer、thinking=false）长跑，`trainer.save_freq=30`
    （用户允许 30/50，取 30，覆盖 step50–90 窗口：30/60/90 三个点）；
  - `trainer.max_actor_ckpt_to_keep=2` 控制磁盘（单 checkpoint 为模型+优化器，体积待实测）；
  - 运行中按 step 记录 diff_mean，标记恶化步（预期 step50–90）；保留该步 checkpoint，
    其余按用户要求删除；
  - 从该权重继续/复跑若干 step，观察 diff_mean 是否立即偏高或继续恶化。
- 待确认：checkpoint 实际大小与保存耗时（当前 /mnt/share 余 6.5T）；resume 方式
  （`trainer.resume_mode=auto` + `default_local_dir` 或 `resume_from_path`）。
- 节点预留：119/120/124（当前均 16 卡空闲）。

## 4. 并行任务与节点分配（15:58 UTC 状态）
- 118：1024 response 长跑（原 120 上因外部租户占用 OOM，迁移至 118 重跑）。
- 122：任务二实验一（maxresp50，150 step）。
- 119/120/124：空闲，供实验二使用。

## 6. 实验一 step1 结果（16:29 UTC，122）
- `critic/rewards/mean=0.0422`、`response_length/mean=50.0`、`clip_ratio=1.0`（全部截断）、
  `actor/entropy=2.058`、`grad_norm=4.401`、`training/rollout_probs_diff_mean=0.02022`、
  `pearson=0.9950`、`timing_s/gen=233.2`、`update_actor=172.1`、`step=474.2`、`throughput=2.53`。
- 对照基线（512 response）step1：diff_mean=0.0078、entropy=1.423、reward=0.169。
- 初步判读：**把 response 限制到 50 并没有降低 diff_mean，step1 反而更高（0.0202 vs 0.0078）**；
  但 50 token 下 100% 截断、reward 仅 0.042，learning 很弱，存在混杂（强制截断把模型推离
  自身 EOS 分布）。需要继续看 step50–90 的趋势再下结论。

## 7. 实验一最终结果（150/150 step 完成，122；日志在共享盘）
- 122 节点在跑完后失联（SSH FAIL），但日志完整保留：
  `logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261023_160511.log`。
- diff_mean 曲线（节选）：
  | step | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | diff_mean | .0202 | .0347 | .0449 | .0371 | .0330 | .0376 | .0248 | .0227 | .0289 | .0339 | .0228 | .0212 | .0291 | .0226 | .0278 | .0267 |
  | reward | .042 | .323 | .478 | .773 | .844 | .900 | .886 | .830 | .788 | .816 | .689 | .900 | .745 | .872 | .773 | .675 |
  | resp_len | 50 | 47.5 | 47.7 | 45.8 | 39.6 | 30.1 | 12.8 | 41.3 | 35.1 | 21.5 | 23.0 | 21.3 | 19.4 | 17.3 | 20.0 | 18.7 |
  | clip | 1.0 | .859 | .875 | .766 | .406 | .172 | .016 | .594 | .422 | .016 | .016 | .016 | .016 | .016 | .016 | .016 |
  | entropy | 2.06 | 1.09 | .707 | .659 | .361 | .427 | .154 | .186 | .299 | .180 | .109 | .092 | .112 | .110 | .097 | .122 |
- 结论：**把 response 限制到 50 并没有阻止 diff_mean 恶化**。step10–90 仍升到 0.025–0.045
  （基线 512 同期 0.010–0.038）；后期 response 已缩到 ~19 token、reward 0.9、entropy 0.1，
  diff_mean 仍在 0.021–0.029 振荡。→ response length 不是 diff_mean 恶化的主因。
- 副作用证据：50 token 时 100% 截断、reward 仅 0.042；模型后来学会在 ~19 token 内作答，
  entropy 从 2.06 塌到 0.09–0.12（与基线同样的 entropy 塌缩）。
- 因此实验二（定期存权重复跑）在“验证 response len 假设”上已无必要；若仍要做，
  目的应改为验证“恶化步权重本身是否可复现高 diff_mean”。
