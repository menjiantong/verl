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
