# DSV4.1 countbenchqa · 单机长跑（一致性拓扑 + 已验证性能优化）

- 日期：2026-10-22（容器内为 UTC 2026-10-21 16:49 起）
- 工作目录：`/mnt/share/m00899630/workspace/dsv41-verl`
- 运行节点：**80.5.25.120**（单机 16 chip，SSH 进入即容器内部，未使用 `docker exec`）
- 状态：**长跑进行中**；step 1 已完成，step 2 已完成（进度条 459.44s），step 3 进行中
- 目的：把一致性工作的拓扑 + 性能工作中**已验证有效**的优化 + FSDPTurbo `pref` 提交合并成一套可长期运行的配置，并给出首步数据与同配置基线对比。

---

## 0. 本轮约束

1. **只使用 80.5.25 段机器**；本轮不再使用 21 段。
   - 此前在 21 段 `.102` 上的临时验证 run 已 `pkill`，`npu-smi -t proc-mem` 确认 `No process in device`；后续不再向 21 段提交任务。
2. 80.5.25 的机器 SSH 进去就是 docker 内部，**不需要二次 `docker exec`**；本轮直接在容器内启动。
3. 性能优化**只保留已验证有效的项**；已证明无收益或不可行的项不加入（见 §1.1）。
4. 长跑不存 checkpoint：`save_freq=-1`、`test_freq=-1`；不会写 ckpt。

---

## 1. 合并来源与决策

| 来源 | 内容 | 本轮采用 |
|---|---|---|
| `wiki/worklog_consistenty_analysis_running_smaller_20261020.md` | 一致性最终拓扑与配方：engine TP8×DP1×EP8、trainer FSDP16/EP8/EFSDP1/TP1、bsz 8×8、prompt 1152 / response 512、LR 5e-6、KL0、util 0.6、`processed_logprobs` | 全部保留 |
| `wiki/run_and_improve_speed_20261020.md` | `expandable_segments`、AdamWSwap、`load_format=auto`（VLM 正确性） | 保留前两项 + `load_format=auto` |
| `FSDPTurbo` commit `4af4cbd`（"pref"） | `torch.npu.Stream/current_device` 适配，使 AdamWSwap 可在 NPU 上跑 | 主工作树已包含，直接使用 |

### 1.1 只保留已验证有效的优化

| 优化 | 性能工作中的证据 | 结论 | 本轮 |
|---|---|---|---|
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | §8.1/8.9：steps 2–3 均值 2.958→2.647 s/1k，**-10.5%** | 有效 | ✅ 启用 |
| `AdamWSwap`（`fsdp_turbo.memory.swap_optimizer`） | §8.10：step3 update_actor 198.2→180.1s，**-8.7%**；§8.14 与 expandable 组合 **-16.6%** | 有效 | ✅ 启用 |
| `VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=0` | §8.1/8.2：2.740 vs 2.769 s/1k（**≈0**） | 无效 | ❌ 不加 |
| `VA_V41_PREFILL_LOGPROBS=1`（prefill rescore） | §8.17(h)：3-step 均值 +1.2%，在噪声内 | 无稳定收益 | ❌ 不加 |
| `VERL_DISABLE_ACTOR_AUTOCAST=1` | §8.17(f)：3-step 仅 -1.7%，step4 backward **OOM** | 不可部署 | ❌ 不加 |
| `micro_batch_per_gpu=2/8` | §8.11/8.2：micro=2 OOM，micro=8 配置断言失败 | 不可行 | ❌ 不加 |
| 全无 offload / 参数常驻 + 仅 optimizer offload | §8.12/8.13：expert dispatch `HcclAllreduce` **OOM** | 不可行 | ❌ 不加 |
| `HCCL_OP_EXPANSION_MODE=AIV` | §8.6：11.5MB 消息无收益；AIV_ONLY 初始化失败 | 无效 | ❌ 不加 |

> `load_format=auto` 不是性能优化，而是 VLM 正确性开关：`dummy` + 权重同步不会恢复冻结的 vision tower，会产生随机图像 embedding、乱码 rollout、reward≈0（性能工作 §8.17）。本轮保留。

---

## 2. 最终配置

| 项 | 值 |
|---|---|
| 节点 | 80.5.25.120，8 卡 × 2 chip = 16 Ascend910 |
| engine (vLLM) | **TP8 × DP1 × EP8**（16 chip / TP8 → 2 个 replica，每个 replica EP8） |
| trainer (FSDPTurbo) | **FSDP16 / EP8 / EFSDP1 / TP1**，`strategy=fsdp_turbo_dsv41` |
| 数据 | countbenchqa_lite train/val（450/50），VLM，chat 模式 `enable_thinking=false` |
| batch | `TRAIN_BSZ=8` × `SAMPLE_N=8`，`ppo_mini_batch_size=8`，`micro=1` |
| 长度 | prompt 1152 / response 512，`max_model_len=1664` |
| 优化器 | LR 5e-6，KL0，`optimizer=AdamWSwap`，`optimizer_impl=fsdp_turbo.memory.swap_optimizer`，`optimizer_offload=False` |
| offload | `param_offload=True`、`offload_policy=True`（与一致性配方一致；AdamWSwap 只替换 optimizer 路径） |
| rollout | `gpu_memory_utilization=0.6`，`logprobs_mode=processed_logprobs`，`load_format=auto` |
| 精度 | bf16，vision 冻结（`VERL_DSV41_VISION=1`、`VERL_DSV41_VISION_FREEZE=1`） |
| 速度开关 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| checkpoint | `save_freq=-1`，`test_freq=-1`，500 step |

---

## 3. 代码改动（全部有备份）

### 3.1 恢复被误删的 `transformer_impl.py`（阻塞性 bug）

- 现象：新 run 在 worker 初始化时报
  `ModuleNotFoundError: No module named 'verl.workers.engine.fsdp.transformer_impl'`。
- 根因：一致性提交 `a5c45ed9 test consistency 1021` **删除了**
  `verl/verl/workers/engine/fsdp/transformer_impl.py`（1673 行），
  但同目录 `fsdp_turbo_impl.py:20` 仍然 `from .transformer_impl import FSDPEngineWithLMHead`。
  该提交之后新启动的 FSDPTurbo run 都会失败（已在跑的旧进程不受影响）。
- 修复：从父提交 `57b5535a` 恢复该文件，内容与 `dsv41-verl-copy` 中的同文件逐字节一致。
  - 路径：`verl/verl/workers/engine/fsdp/transformer_impl.py`
  - 行数 1673，78261 bytes；`git status` 显示为未跟踪文件（HEAD 中已无此文件）。
- 证据：恢复后 `IMPORT_OK`，长跑通过 worker 初始化。

### 3.2 AdamWSwap 兼容性补丁

文件：`FSDPTurbo/fsdp_turbo/memory/swap_optimizer/swap_optimizer.py`
备份：`swap_optimizer.py.bak_20261022_cons_speed`

问题链（每一步都有实际 traceback）：

1. `AttributeError: 'Tensor' object has no attribute 'to_local'`
   （`opt_states_initialization` 里对非 DTensor 参数调用 `.to_local()`）
2. 加通用 helper 后，又在 optimizer step 报
   `RuntimeError: Output 0 of _ToTorchTensorBackward is a view and is being modified inplace`
   （`.to_local()` 是 autograd view，`copy_` 原地修改被 PyTorch 禁止）。

最终修复：新增 `_local_tensor()`，DTensor 优先取原始本地张量 `t._local_tensor`，
没有该属性时退回 `to_local()`，普通 Tensor 直接返回；替换 11 处 `.to_local()` 调用。

```python
def _local_tensor(t):
    if hasattr(t, "_local_tensor"):
        return t._local_tensor
    if hasattr(t, "to_local"):
        return t.to_local()
    return t
```

效果：`[opt-state-debug][rank0] optimizer=AdamWSwap groups=1 params=1184`，
step1 正常完成；**无该补丁时 AdamWSwap 无法在本拓扑启动**。

> 注意：这是为当前 FSDPTurbo + DTensor 混合参数列表做的兼容修复，不改变 AdamWSwap 算法；
> 性能收益仍来自性能工作 §8.10/8.14 已验证的 AdamWSwap。

### 3.3 基础启动脚本透传 `TRAIN_EXTRA_ARGS`

文件：`verl/my_scripts/dsv41_grpo_countbenchqa_1node.sh`
备份：`dsv41_grpo_countbenchqa_1node.sh.bak_20261022_cons_speed`

在 hydra 命令末尾增加一行，空值时无副作用：

```bash
    actor_rollout_ref.rollout.logprobs_mode=${LOGPROBS_MODE:-processed_logprobs} \
    ${TRAIN_EXTRA_ARGS:-} \
    2>&1 | tee "${RUN_LOG}"
```

### 3.4 新增长跑脚本

`verl/my_scripts/dsv41_grpo_countbenchqa_cons_speed.sh`（7057 bytes，`chmod 755`）

在 `dsv41_grpo_countbenchqa_cons_loc.sh` 基础上只加：

```bash
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_EXTRA_ARGS="actor_rollout_ref.rollout.load_format=auto ${TRAIN_EXTRA_ARGS:-}"
export ENABLE_ADAMW_SWAP="${ENABLE_ADAMW_SWAP:-1}"
if [ "$ENABLE_ADAMW_SWAP" = "1" ]; then
  export TRAIN_EXTRA_ARGS="actor_rollout_ref.actor.optim.optimizer_impl=fsdp_turbo.memory.swap_optimizer \
actor_rollout_ref.actor.optim.optimizer=AdamWSwap \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False ${TRAIN_EXTRA_ARGS}"
fi
```

同时把 `MY_IP` 自动探测从只匹配 `80.5.21.x` 改为匹配 `80.5.21.x / 80.5.25.x`；
`ENABLE_ADAMW_SWAP=0` 可一键回退到一致性原配方（只保留 expandable_segments）。

---

## 4. 启动命令

80.5.25.120 的 SSH 进入即容器内部（端口 50002），无需 `docker exec`：

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
STEPS=500 bash my_scripts/dsv41_grpo_countbenchqa_cons_speed.sh
```

实际启动输出（容器 UTC 2026-10-21 16:49:48）：

```
[cons-speed] host=node-29-120 ip=80.5.25.120 nic=enp48s3u1u1 steps=500
[cons-speed] engine=TP8xDP1xEP8 (2 replicas on 16 chips)
[cons-speed] trainer=FSDP16/EP8/EFSDP1/TP1 bsz=8x8
[cons-speed] prompt=1152 response=512 lr=5e-6 kl=0 thinking=false util=0.6
[cons-speed] speed: alloc_conf=expandable_segments:True adamw_swap=1 load_format=auto
[cons-speed] experiment=GRPO-DSV41-countbenchqa-cons-speed-20261021_1649
[cons-speed] log=logs/cons_speed_20261021_164948.nohup.log
[cons-speed] started pid=1195347
```

- 外层日志：`logs/cons_speed_20261021_164948.nohup.log`
- 内层 tee：`logs/grpo_countbenchqa_steps500_20261021_164948.log`
- 模型日志：`logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261021_165002.log`

---

## 5. 运行时间线（容器 UTC / 宿主机 CST）

| 时间（UTC） | 事件 |
|---|---|
| 16:49:48 | 启动 |
| 16:52:41 | `create worker group` |
| 16:57 | vLLM engine 初始化（`load_format=auto` 读 checkpoint） |
| 16:59:27 | `all initialize finished, ready to fit` |
| 17:07:42 | **step 1 完成**，495.28s |
| 17:15:16 | **step 2 完成**，459.44s（进度条 `2/500`） |
| 17:20+ | step 3 进行中 |

宿主机时间为 UTC+8：16:49 UTC = 00:49 CST 2026-10-22。

---

## 6. 首步结果（80.5.25.120，step 1）

来源：`logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261021_165002.log` 的 `step:1 - ...` 行。

| 指标 | 值 |
|---|---:|
| global_seqlen/mean | 2773.5 |
| actor/entropy | **1.4631** |
| training/rollout_probs_diff_mean | **0.007471** |
| training/rollout_actor_probs_pearson_corr | **0.998889** |
| critic/score/mean (reward) | 0.12656 |
| response_length/mean | 443.38 |
| response_length/clip_ratio | 0.78125 |
| timing_s/gen | 248.32 s |
| timing_s/old_log_prob | 32.70 s |
| timing_s/update_actor | **166.55 s** |
| timing_s/update_weights | 26.88 s |
| timing_s/step | **495.02 s** |
| perf/total_num_tokens | 44,376 |
| actor/perf/max_memory_reserved_gb | 50.39 GB |
| actor/perf/cpu_memory_used_gb | 1364.29 GB |

判定：
- **diff_mean = 0.00747 < 0.01**，pearson 0.99889，达到一致性目标。
- reward 0.1266、response 443、entropy 1.463，属于正常 VLM 训练状态（非 dummy/乱码）。
- `load_format=auto` 在 resolved config 中确认（3 处 `'load_format': 'auto'`）。
- `[opt-state-debug] optimizer=AdamWSwap groups=1 params=1184`，确认 AdamWSwap 生效。
- NPU HBM 在生成阶段最高约 64.8 GB/chip（util 0.6 + 训练侧），未 OOM；回到 update 阶段约 43 GB/chip。

---

## 7. 与同配置 W2 基线的对比

W2 基线：`logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261020_160333.log`
（同为 512/KL0/bsz8×8/DP1/EP8，**未加** expandable_segments / AdamWSwap）。

| step | run | gen s | old_log s | **update_actor s** | update_weights s | step s | update_actor s/1k tok |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | W2 baseline | 230.27 | 26.85 | **201.90** | 23.36 | 503.47 | 4.357 |
| 1 | **本 run（25.120）** | 248.32 | 32.70 | **166.55** | 26.88 | 495.02 | **3.753（-13.9%）** |
| 2 | W2 baseline | 175.04 | 25.70 | **187.76** | 22.90 | 411.99 | 3.783 |
| 3 | W2 baseline | 143.13 | 27.31 | **206.00** | 23.48 | 400.47 | 3.884 |

结论（有数据支撑）：
1. **update_actor 的优化复现**：同配置 step1 从 201.90s 降到 166.55s（**-17.5%**），
   按 token 归一 4.357→3.753 s/1k（**-13.9%**），与性能工作 §8.14 的 -16.6% 同量级。
2. **整步时间只降 1.7%**：gen 从 230.3 升到 248.3s（+7.8%），old_log 从 26.8 升到 32.7s（+21.8%），
   把 update_actor 省下的 35.4s 抵消了。gen/old_log 的逐 step 波动远大于 optimizer 收益，
   **不能**用单步 step 时间宣称整体加速；需要 5–10 step 的均值。
3. 21 段临时 pilot（已停止，仅 1 step）update_actor=135.2s（2.89 s/1k）、step=415.3s；
   该数字优于 25.120（166.5s），说明节点/负载差异明显，长跑结论以 25.120 为准。

---

## 8. 失败尝试与负结果（可复现）

| # | 现象 | 根因 | 处理 |
|---:|---|---|---|
| 1 | `ModuleNotFoundError: verl.workers.engine.fsdp.transformer_impl` | 一致性提交 `a5c45ed9` 删除该文件，`fsdp_turbo_impl.py` 仍 import | 从父提交 `57b5535a` 恢复 |
| 2 | `AttributeError: 'Tensor' object has no attribute 'to_local'` | AdamWSwap 假设所有参数都是 DTensor | 加 `_local_tensor` helper |
| 3 | `RuntimeError: Output 0 of _ToTorchTensorBackward is a view and is being modified inplace` | `.to_local()` 返回 autograd view，optimizer `copy_` 原地改 | helper 改为优先 `t._local_tensor` |
| 4 | `/tmp/ray` 95% full 警告 | ray 临时目录水位高；当前仍有 682 GB 可用 | 观察；不影响本 run |
| 5 | 生成阶段 HBM ~64.8/65.5 GB | util 0.6 + 训练侧共存 | 未 OOM；继续监控，异常时降 util/关 AdamWSwap |

---

## 9. 当前状态与巡检命令

```bash
# 进程
pgrep -af 'main[_]ppo' | head
# 进度与首步指标
tail -f /mnt/share/m00899630/workspace/dsv41-verl/verl/logs/cons_speed_20261021_164948.nohup.log
grep -a 'Training Progress' logs/cons_speed_20261021_164948.nohup.log | tail
grep -a 'step:[0-9]* -' logs/DeepSeek-V4.1-Flash-top32-bf16-noengram-20261021_165002.log | tail
# NPU
LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH \
  /usr/local/sbin/npu-smi info
```

停止（如需）：

```bash
pkill -9 -f 'main[_]ppo'
pkill -9 -f 'VLLM::'
pkill -9 -f 'vllm serve'
```

ETA：前两步平均 (495+459)/2 ≈ 477s/step，500 step 约 **66 小时**（随 response 长度/生成速度波动）。

---

## 10. 结论

1. 长跑已在 **80.5.25.120** 启动，配置 = 一致性最终拓扑 + `expandable_segments` + AdamWSwap + `load_format=auto`，不存 checkpoint。
2. 首步 diff_mean **0.00747**、pearson **0.99889**，在 0.01 目标内；AdamWSwap 已确认生效。
3. 只启用了性能工作中已验证有效的两项优化；无效/失败项均未加入（§1.1）。
4. 同配置 step1 的 update_actor **-17.5%（按 token -13.9%）**；整步时间被 gen/old_log 波动掩盖，需继续积累 5–10 step 数据。
5. 待办：继续巡检 step2+ 指标、5/10 step 的 diff 趋势、entropy/response 是否稳定、HBM 峰值；若 OOM 则回退 `ENABLE_ADAMW_SWAP=0` 或降低 `ROLLOUT_GPU_MEM_UTIL`。

---

## 11. 2026-10-22 追加：56 step 结果、epoch 边界停止与根因（修正 §10 的单步结论）

> 本节优先于 §10 的“首步成功”结论：首步只代表 step1，不能代表 56 step。

### 11.1 run 在 step 56 停止，不是 OOM

- 进度：`56/500`，最后一步 `step:56` 正常出数；随后日志出现
  `RuntimeError: DataLoader worker (pid 1250457) is killed by signal: Killed.`
- 根因：**`trainer.total_epochs=1`**。trainer 的主循环是
  `while current_epoch < total_epochs and global_steps <= total_training_steps`；
  countbenchqa_lite train 450 条、batch 8，`steps_per_epoch = 450//8 = 56`。
  `total_training_steps=500` 虽然传了，但 epoch 上限先到，run 在 56 step 正常结束；
  DataLoader worker 的 SIGKILL 是退出阶段 atexit 清理触发的噪声，不是训练中途崩溃。
- 排除 OOM：容器 cgroup `memory.failcnt=0`、`oom_kill=0`；host 内存仍有 2.2 TB 可用；
  dmesg 无 OOM kill 记录；NPU HBM 在 run 结束后回落到基线 ~3 GB/chip。
- 这也解释了 W2 为何同样死在 step56：同一个 epoch 上限，不是随机 SIGKILL。

**修复**：启动脚本已加 `trainer.total_epochs=1000`（`total_training_steps=500` 仍然是最终上限），
下一个 run 不会再在 56 step 停止。修复前已上传到
`my_scripts/dsv41_grpo_countbenchqa_cons_speed.sh`。

### 11.2 56 step 汇总（25.120，AdamWSwap + expandable_segments）

| 指标 | 值 |
|---|---:|
| steps | 56 |
| diff_mean mean / std / min / max | **0.010530 / 0.001558 / 0.007471 / 0.014328** |
| diff_mean > 0.01 的步数 | **32 / 56** |
| pearson mean / min | 0.99331 / 0.98711 |
| entropy first / last / min | 1.463 / 0.269 / 0.163 |
| reward mean / max | 0.6855 / 0.8719 |
| response mean / min | 332.6 / 210.9 |
| update_actor mean / min / max | **165.46 / 138.20 / 218.42 s** |
| step mean / min / max | **385.75 / 344.32 / 495.02 s** |

趋势：
- 前 5 step diff：0.00747 / 0.00952 / 0.00897 / 0.01007 / 0.01018，均值 **0.00924**；
  后 5 step：0.00921 / 0.00868 / 0.00811 / 0.00899 / 0.00973，均值 **0.00894**。
- 中途多个 step 超过 0.01，最大 0.01433；**不是“始终在 0.01 水平”**。
- entropy 从 1.46 降到 0.16，response 从 step1 的 443 降到 min 211；
  与一致性工作 §12 的结论一致：KL0 下 response 缩短会把 full-response diff_mean 抬高。
- 后 5 step 又回到 0.01 以下，说明不是“训练越远越差”的单调恶化，而是与 response 长度/entropy 同步波动。

### 11.3 与 W2 同配置 56 step 对照

| run | diff_mean 56-step mean | >0.01 步数 | entropy 末值 | 备注 |
|---|---:|---:|---:|---|
| W2（21 段，vanilla optimizer） | **0.007128** | **0 / 56** | 0.199 | 一致性工作 §(i) |
| 本 run（25.120，AdamWSwap+expandable） | **0.010530** | **32 / 56** | 0.269 | 本节 |

差异显著（+48%）。可能原因（尚未证明，不能下结论）：
1. **25 段 vs 21 段的驱动/内核/硬件差异**（两台机器不是同一批）；
2. **AdamWSwap 的 fused AdamW 数值路径**与 CPU single-tensor AdamW 不同，
   可能改变权重轨迹/entropy/response 长度；
3. `load_format=auto` 让 vision 权重真实加载（W2 的加载方式需核对），
   但两者 reward/pearson 均正常，不像 dummy 乱码。

### 11.4 下一步（必须先做，不能直接宣称长跑达标）

1. **5–10 step 同节点 A/B**：在 25.120 用 `ENABLE_ADAMW_SWAP=0` 跑 5–10 step，
   其余与本节完全一致（`trainer.total_epochs=1000`、512/KL0、bsz8×8、load_format auto、expandable）。
   对照本 run 的 step1–5（0.00747/0.00952/0.00897/0.01007/0.01018）。
2. 若关掉 AdamWSwap 后 diff 回到 ~0.007–0.008，则 **AdamWSwap 是一致性回归源**，
   长跑应禁用 AdamWSwap（只保留 expandable_segments）；若仍 ~0.0105，则差异来自 25 段节点/驱动，
   需要换节点或按节点重新标定。
3. A/B 通过后再启动 500 step 长跑；因为 `save_freq=-1`，本次 56 step 无法续跑，只能从 step0 重来。

## 12. `_launch_baseline_prof_1022.sh` 审查（2026-10-22）

结论：**既不是 W2 长跑，也不是原始 perf baseline，而是两者混合**。
性能优化确实已删除；但 W2 的任务配置没有搬全，且 `TRAIN_EXTRA_ARGS` 未被转发，
脚本里的 `load_format=auto` / `total_epochs=1000` 实际不生效。

### 12.1 实际生效的配置

入口：`dsv41-verl-copy/verl/my_scripts/dsv41_grpo_2node_new.sh single`
→ `examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh`

- 数据：**GSM8K** `gsm8k_main_train_rl.parquet`（7473 行），不是 countbenchqa。
- 模型：noengram（launcher 覆盖，OK）。
- 拓扑：GEN_TP=8（example 默认）×GEN_DP=1×GEN_EP=8；FSDP16/EP8/EFSDP1/TP1（OK）。
- batch：8×8（OK）；KL=0（OK）；rollout util=0.6（OK）。
- prompt/response：**1024/512**（example 默认 1024；W2 是 1152/512）。
- 无 vision、无 agent loop、thinking 未设置、无 `VERL_DSV41_VISION(_FREEZE)`。
- `save_freq=-1` / `test_freq=-1`（example 硬编码，OK）。
- `load_format`：默认 `dummy`；脚本写的 `auto` 没传进去。
- `total_epochs`：example 硬编码 1；脚本写的 1000 没传进去。
- 性能优化：没有 `expandable_segments`、没有 `AdamWSwap`/`optimizer_impl`、
  没有 `optimizer_offload=False`（已删干净）。

### 12.2 关键断点：`TRAIN_EXTRA_ARGS` 没人接收

- W2 base `dsv41_grpo_countbenchqa_1node.sh:133` 有 `${TRAIN_EXTRA_ARGS:-}`。
- `dsv41_grpo_2node_new.sh` 和 example 脚本中 `TRAIN_EXTRA_ARGS` 均 **0 处引用**。
- 因此 launcher 里的 `trainer.total_epochs=1000`、`actor_rollout_ref.rollout.load_format=auto`
  都是空操作。GSM8K 7473//8=934 step/epoch，跑 500 step 时 epoch 上限不触发；
  但换成 countbenchqa（450//8=56）会重演 56 step 停止；VLM 则会回到 dummy 加载。

### 12.3 其他问题

- `cd .../dsv41-verl-copy/verl/my_scripts`，但 base 脚本 `MIRROR_ROOT` 固定为
  `/mnt/.../dsv41-verl`，实际代码/FSDPTurbo 走主树；两份 2node 脚本 md5 相同，
  所以目前只影响 `RUN_LOG` 落点（主树 `verl/logs`）与外层 log（copy 树 `verl/logs`）。
- 所有 profiler 参数被注释，`PROFILE_STEP`/`ANALYSIS`/`SAVE_PATH` 未设置；
  脚本名仍是 `prof`，但不会产生 profile。
- 顶部注释写 “GSM8K, KL=1, resp 1024, util=0.5”，实际 export 是 KL0/resp512/util0.6，
  注释与代码不一致；原始 baseline log（`perf_profile_20261020_230432.outer.log:133`）
  确实是 KL1/resp1024/util0.5 且 profiler 开启。
- `MY_IP=80.5.25.119`：该地址在 50002 无监听（本会话不可达）；在别的节点直接跑会
  因 HCCL_IF_IP 不匹配失败。需改成实际节点（如 25.120）或由 .119 本机执行。
- 外层 `> "$LOG"` 没有 `2>&1`/nohup；长跑会占住终端，外层错误不进 log。

### 12.4 建议

- 若要 **W2 配置但不带性能优化**：用主树 `dsv41_grpo_countbenchqa_1node.sh`，
  显式 `GEN_DP=1 GEN_EP=8 ROLLOUT_GPU_MEM_UTIL=0.6 USE_KL=0 MAX_RESPONSE_LEN=512`，
  `TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 actor_rollout_ref.rollout.load_format=auto"`；
  不要 `ENABLE_ADAMW_SWAP`，不要 `PYTORCH_NPU_ALLOC_CONF=expandable_segments`。
- 若要 **原始 perf baseline**：恢复 KL1/resp1024/util0.5 和 profiler 参数；
  当前的 KL0/resp512/util0.6 已经改了 workload，不能和优化 run 直接 A/B。
- 若坚持用 `_launch_baseline_prof_1022.sh`：先在 `dsv41_grpo_2node_new.sh` 的 single 分支
  末尾加 `${TRAIN_EXTRA_ARGS:-}`，再补 countbenchqa/vision/agent/thinking/prompt1152，
  并把 `MY_IP` 改成实际节点。

## 13. W2 长跑脚本重写（基于 2node_new，2026-10-22）

新增：`verl/my_scripts/_launch_W2_longrun_1022.sh`；本地副本
`D:\remote_workplace\worklog\launch_W2_longrun_1022.sh`。

基座：`my_scripts/dsv41_grpo_2node_new.sh single`。该基座已打两个补丁
（备份 `dsv41_grpo_2node_new.sh.bak_20261022_w2_longrun`）：

- master/single 的 hydra 命令末尾增加 `${TRAIN_EXTRA_ARGS:-}`，让额外覆盖真正透传；
- `RUN_LOG` 改为 `${RUN_LOG:-...}`，允许外层脚本指定 inner log。

W2 配置（全部显式）：

- 数据：countbenchqa_lite train/val；模型：noengram。
- 引擎：TP8×DP1×EP8（16 卡 2 replica）；训练：FSDP16/EP8/EFSDP1/TP1。
- batch 8×8；prompt 1152 / response 512；LR 5e-6；KL0；rollout util 0.6。
- vision=1/freeze=1；agent=`deepseek_v41_vl_single_turn`；thinking=false；
  logprobs=`processed_logprobs`；save/test=-1；offload=True/True/True。
- 无 `expandable_segments`、无 `AdamWSwap`/`optimizer_impl`/`optimizer_offload=False`。

今天的问题修复：

- `trainer.total_epochs=1000`：countbenchqa 450//8=56 步/epoch，默认 1 会停在第 56 步。
- `actor_rollout_ref.rollout.load_format=auto`：VLM vision tower 必须从 checkpoint 读。
- `TRAIN_EXTRA_ARGS` 从零构建，不再继承旧 shell 里可能残留的 AdamWSwap。
- `MY_IP` 自动取本机 `80.5.25.x`，不再硬编码 `80.5.25.119`。
- 外层/内层日志分离，`nohup` + `2>&1`，默认后台；启动前 `pgrep main_ppo` 防叠加。

配置验证（只做 Hydra `--cfg job`，未训练）：

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
EXTRA_ARGS="--cfg job" DETACH=0 STEPS=1 \
  bash my_scripts/_launch_W2_longrun_1022.sh
```

结果 `EXIT=0`，解析值：

| 项 | 值 |
|---|---|
| `trainer.total_epochs` | **1000** |
| `actor_rollout_ref.rollout.load_format` | **auto** |
| `data.max_prompt_length` / `max_response_length` | **1152 / 512** |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | **0.6** |
| `default_agent_loop` | `deepseek_v41_vl_single_turn` |
| `enable_thinking` | `false` |
| `logprobs_mode` | `processed_logprobs` |

验证后无 `main_ppo`、无 Ray 残留。

启动：

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
bash my_scripts/_launch_W2_longrun_1022.sh            # 500 步，后台
STEPS=10 bash my_scripts/_launch_W2_longrun_1022.sh   # 10 步冒烟
DRY_RUN=1 bash my_scripts/_launch_W2_longrun_1022.sh  # 只打印配置
```

日志：`verl/logs/W2_longrun_<TS>.outer.log` 与 `.inner.log`。

### 13.1 与 W2 的逐项 diff

用 W2 日志 `long_W2_20261021_000307.nohup.log:120` 的 `python3 -m verl.trainer.main_ppo`
命令行，与新脚本 `--cfg job` 的命令行逐项比较（84 项 key）：

**完全相同的部分**：数据、模型、TP8×DP1×EP8、FSDP16/EP8/EFSDP1/TP1、bsz8×8、
prompt1152/response512、LR5e-6、KL0、util0.6、vision/freeze、agent loop、
thinking=false、logprobs、offload、save/test、FFN recompute、`rollout_is=null`。

**仅新脚本有**：

- `actor_rollout_ref.rollout.load_format=auto`。W2 命令行没有这个覆盖，实际 vLLM
  启动参数是 `--load_format dummy`（W2 日志第 1201 行）。这是 VLM vision tower
  的加载差异，不只是性能差异；新脚本启动时会多读一遍完整 checkpoint。

**有意不同**：

- `trainer.total_epochs`：W2=1，新=1000。W2 因此在 450//8=56 步停；新脚本可跑满 500。
- experiment name / `default_local_dir`：仅命名不同。

**不是差异**：

- `trainer.total_training_steps`：W2=500；新脚本实际启动 500，`--cfg job` 校验时用
  `STEPS=1`，所以 diff 里显示 1，不代表正式运行值。

**刚补上的差异**：

- `trainer.v1.sampler.sync_refill_failed_groups`：W2 显式 True；2node example 默认
  false。已在 wrapper 的 `TRAIN_EXTRA_ARGS` 补 `=True`，`--cfg job` 解析为 true。

**非 hydra 配置差异**：

- 节点：W2=80.5.21.104；新脚本自动检测 80.5.25.x（当前 80.5.25.120）。这是目前
  一致性 56 step 差异（0.0105 vs 0.0071）最可疑的来源，尚未通过同节点 A/B 排除。
- 基座：W2=`dsv41_grpo_countbenchqa_1node.sh`；新=`dsv41_grpo_2node_new.sh single`
  + `TRAIN_EXTRA_ARGS`/`RUN_LOG` 补丁。解析后的 hydra 参数除上面几项外一致。
- 性能优化：两边都没有 `expandable_segments` / `AdamWSwap`。

### 13.2 简化版（2026-10-22 12:31，最终采用）

按要求把 wrapper 简化到 42 行：

- 直接 `cd /mnt/share/m00899630/workspace/dsv41-verl/verl`，不再动态算 ROOT；
- 直接 `MY_IP=80.5.25.120`、`NIC=enp48s3u1u1`，删除 hostname/NIC 自动探测；
- 只保留 W2 相对 base 默认值的 delta：model、countbenchqa 数据、prompt1152、response512、
  KL0、util0.6、vision/freeze、experiment；拓扑、batch、LR、offload、save/test 全部
  交给 `dsv41_grpo_2node_new.sh` 的默认值；
- `TRAIN_EXTRA_ARGS` 保留 `total_epochs=1000`、`load_format=auto`、agent loop、
  thinking=false、logprobs、`sync_refill_failed_groups=True`；
- 删除 DETACH/FORCE/DRY_RUN/EXTRA_ARGS/pgrep 和长串 unset，只保留
  `unset PYTORCH_NPU_ALLOC_CONF ENABLE_ADAMW_SWAP`；
- 保留 base patch 检查：`grep -Fq '${TRAIN_EXTRA_ARGS:-}'`。

`--cfg job` 重新校验通过：`total_epochs=1000`、`load_format=auto`、prompt1152、
response512、util0.6、agent loop、thinking=false、logprobs=processed、
`sync_refill_failed_groups=true`。与 W2 的 diff 仍只有 load_format / total_epochs /
experiment 三类（`total_training_steps` 校验时为 1，正式运行为 500）。

### 13.3 去重最终版（40 行）

按“base 里已经完全相同的值不要重复”再删三项：

- `NIC=enp48s3u1u1`：`dsv41_grpo_2node_new.sh:28` 的默认值就是它；
- `VERL_DSV41_VISION_FREEZE=1`：`fsdp_turbo_dsv41_impl.py:1010` 的默认值就是 1；
- `actor_rollout_ref.rollout.logprobs_mode=processed_logprobs`：rollout 配置默认就是
  `processed_logprobs`（`rollout/rollout.yaml:93`）。

最终 wrapper 为 40 行。`--cfg job` 重新校验仍为：`total_epochs=1000`、`load_format=auto`、
prompt1152、response512、util0.6、agent loop、thinking=false、logprobs=processed（默认）、
`sync_refill_failed_groups=true`。

命令行 diff 里会看到 `ONLY_W2 logprobs_mode=processed_logprobs`，这是因为 W2 显式写了、
新脚本靠默认值；解析后的实际值相同，不是行为差异。

### 13.4 TRAIN_EXTRA_ARGS 逐行书写（44 行）

为可读性，`TRAIN_EXTRA_ARGS` 改为反斜杠续行，每个 override 一行：

```bash
export TRAIN_EXTRA_ARGS="trainer.total_epochs=1000 \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.agent.default_agent_loop=deepseek_v41_vl_single_turn \
+data.apply_chat_template_kwargs.enable_thinking=false \
trainer.v1.sampler.sync_refill_failed_groups=True"
```

双引号内的 `\` 换行会被 shell 去掉，变量展开后仍是空格分隔的多个 hydra 参数；
base 脚本的 `${TRAIN_EXTRA_ARGS:-}` 未加引号，能正常拆分。

`--cfg job` 重新校验通过，解析值与 §13.3 完全一致：`total_epochs=1000`、
`load_format=auto`、prompt1152、response512、util0.6、agent loop、thinking=false、
logprobs=processed（默认）、`sync_refill_failed_groups=true`。

### 13.5 前台输出修复（2026-10-22）

现象：删除 `nohup` 后执行 `bash _launch_W2_longrun_1022.sh`，终端没有训练输出。

原因：脚本最后一行仍是

```bash
bash "$BASE" single "$STEPS" > "$LOG" 2>&1 < /dev/null &
```

`&` 仍然把任务放到后台，`> "$LOG" 2>&1` 又把所有输出重定向到日志；
删除 `nohup` 只是去掉了对 SIGHUP 的保护，并不会变成前台，反而更容易在
shell/SSH 退出时被带走。最新一次后台尝试的日志停在
`Failed to register worker to Raylet: IOError: Failed to read data from the socket: End of file`，
之后没有 `main_ppo` 残留，和后台进程被挂断的现象一致。

修复：

```bash
set -eo pipefail
...
bash "$BASE" single "$STEPS" 2>&1 | tee "$LOG"
```

- 前台运行，输出同时进终端和 `$LOG`；
- base 自己的 `tee "$RUN_LOG"` 仍写 inner log；
- 需要后台时在脚本外使用：
  `nohup bash my_scripts/_launch_W2_longrun_1022.sh > /tmp/w2.out 2>&1 &`

另外清理了编辑残留的单独 `nohup` 行，`bash -n` 通过。

### 13.6 12:52 长跑失败：HCCL_SOCKET_IFNAME 被设成 IP（2026-10-22）

日志：`verl/logs/W2_longrun_20261022_125210.outer.log`

失败点：vLLM EngineCore 初始化，训练还没开始。

根因：

```text
Config_Error_Invalid_Environment_Variable(EI0001):
Value 80.5.25.120 for environment variable HCCL_SOCKET_IFNAME is invalid.
```

`dsv41_grpo_2node_new.sh:97` 是：

```bash
export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
```

仓库里没有任何脚本写 `NIC=80.5.25.120`（全文 grep 只在本工作记录里出现）。
12:52 使用的 wrapper 在 12:36 的“去重”编辑中删掉了 `export NIC=enp48s3u1u1`，
只保留 `MY_IP=80.5.25.120`；base 的 `NIC=${NIC:-enp48s3u1u1}` 只在 `NIC`
未设置时生效，所以启动 shell 里已有的 `NIC`（值与 `MY_IP` 相同）被继承，
最终导出成 `HCCL_SOCKET_IFNAME=80.5.25.120`。这不是仓库脚本写死的值。

另外，报错里的 expect 文案 “an ip address that exists in the local network
interfaces list” 来自 `libhccl_plf.so` 的 IP 校验分支；同文件里网卡名校验的
文案是 “a valid network interface name (e.g., eth0, bound0) present on this
host”。节点实际是 `.119` 时，`HCCL_IF_IP=80.5.25.120` 同样不成立。

另一个不一致：

- 脚本 `MY_IP` 是 `80.5.25.120`；
- 日志里的 Ray actor 是 `ip=80.5.25.119`；
- 也就是实际运行节点/容器是 `.119`，`HCCL_IF_IP=.120` 同样不对。

修复：

```bash
export MY_IP=80.5.25.119     # 按实际节点；如果换到 .120 就改这里
export NIC=enp48s3u1u1
[ -d "/sys/class/net/$NIC" ] || { echo "NIC not found"; exit 1; }
```

修复后 base 会得到：

```text
GLOO_SOCKET_IFNAME=enp48s3u1u1
HCCL_SOCKET_IFNAME=enp48s3u1u1
HCCL_IF_IP=80.5.25.119
```

`createHCCLCommOrigin ... hcclGetRootInfo ... error code 6` 只是上层表现；
真正根因是 `HCCL_SOCKET_IFNAME` 的变量值错误。日志里的 W2 参数
（`total_epochs=1000`、`load_format=auto`、agent loop、sync_refill）都正确，
失败与模型/训练配置无关。
