# worklog：DeepSeek-V4.1 top32 单机 16 卡 EP16 SFT 3 epoch（2026-10-17）

> 任务：用 `verl/my_scripts/run_sft_dsv41_1node16_ep16.sh` 从原始 top32 权重重新训练
> **3 个 epoch**（GSM8K main train，7473 条，gbs=16 → 467 步/epoch，共 1401 步），
> 在 epoch1/epoch2/epoch3 结束时分别保存并导出 vLLM 可加载权重到
> `/mnt/share/m00899630/weights/`（目录名以 `epoch1/2/3` 结尾），导出成功后删除临时 ckpt。
>
> 机器：单机 16 卡 NPU（Ascend910），共享盘 `/mnt/share`（180T，当前 3.5T 可用）。
> 代码：`/mnt/share/m00899630/workspace/dsv41-verl/verl`（共享盘，非本地 `/workspace-verl`）。

## 1. 关键配置（本次实际启动参数）

| 项 | 值 | 说明 |
|---|---|---|
| 脚本 | `my_scripts/run_sft_dsv41_1node16_ep16.sh 1401` | 单机 16 卡 |
| FSDP / EP / efsdp / TP | **16 / 16 / 1 / 1** | dense FSDP16；experts EP16（32/16=2 experts/rank）；无 edp 副本 |
| micro / global batch | **1 / 16** | 每卡 1 条样本，DP=16，无梯度累积 |
| `trainer.total_epochs` | **3** | 脚本硬编码 |
| `trainer.total_training_steps` | **1401** | 3 × 467 |
| `save_freq` | **467** | 在 step 467 / 934 / 1401 保存（step 从 1 开始，`global_step % save_freq == 0`） |
| `max_ckpt_to_keep` | **3** | 保留 3 个 epoch ckpt |
| `CKPT_DIR` | `/mnt/share/m00899630/dsv41/sft_out_ep16_3ep` | 新目录，避免与旧 run 混 |
| `checkpoint.save_contents` | `['model']` | 只存模型，不存 optimizer |
| `param_offload` / `optimizer_offload` | **False / True** | 复现已验证的 EP16 运行配置 |
| `pad_to_length_bucket` | 512 | 数据 token_len 最大 424，所有微批形状固定 512 |
| attention recompute / ffn recompute | `1` / `chunk=64`；`recompute_plan=["model.layers.{*}.ffn"]` | 已开启 |
| 预取 | forward 1 / backward 1 | 脚本默认 |

### 1.1 启动命令（实际执行）

```bash
cd /mnt/share/m00899630/workspace/dsv41-verl/verl
export CKPT_DIR=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep
export EXPERIMENT=gsm8k-top32-3ep
export SAVE_FREQ=467
export MAX_KEEP=3
mkdir -p "$CKPT_DIR"
bash my_scripts/run_sft_dsv41_1node16_ep16.sh 1401
```

后台任务 ID：`bkc1pqhw3`
训练日志：`logs/sft_dsv41_steps1401_20261017_103001.log`
后台 stdout：`/tmp/claude-0/-mnt-share-m00899630-workspace-dsv41-verl/42377611-cd11-4b34-bdad-fc68f6bbcea0/tasks/bkc1pqhw3.output`

## 2. 启动前清理（解决上一次终止的直接原因）

清理前机器上有 **3 个 EP16 launcher**（09:47:52 / 09:48:18 / 09:50:58 启动）以及旧 EP8 launcher，
其中只有 09:50:58 那个日志在正常训练；其余停在初始化阶段并占用 Ray/host 资源。
多个 launcher 同时争抢同一组 16 卡，是上一次运行不可控/被终止的直接环境因素。

实际清理命令：

```bash
pkill -9 -f "[s]ft_trainer_ray"
pkill -9 -f "[r]un_sft_dsv41_1node16_ep16.sh"
pkill -9 -f "[r]un_sft_dsv41_1node16.sh"
ray stop --force
```

清理后证据（2026-10-17 10:28 左右）：

- `ps -eo pid,cmd | grep -E "[s]ft_trainer_ray|[r]un_sft_dsv41"` → 空。
- `npu-smi info` → 16 个 chip 的 HBM 均为 **~2.9–3.3 GB / 65536 MB**（仅驱动基线），AICore 0%。
- `free -g` → `available 1979 GB`。
- `ls /mnt/share/m00899630/dsv41/sft_out` → 空（旧 116 步 run 未落 ckpt）。

## 3. 显存预算（基于已实测的 EP16 run）

每卡参数（53.21B 全模型无副本 1/16 切分）：

- experts：45.298B / 16 = 2.831B × 2B = **5.274 GiB**
- dense：7.912B / 16 = 0.4945B × 2B = **0.921 GiB**
- 合计 **6.195 GiB**（checkpoint 直读实测 6.20 GiB）

峰值公式：

- step 1（optimizer step）：**5 × 6.20 = 30.98 GiB**（params + grads + 2×Adam + sqrt 临时量）
- step ≥2（Adam 状态在 `train_mode` 期间常驻）：**≈ 3 × 6.20 + 17.6 ≈ 36.18 GiB**
  （17.6 GiB ≈ 逐层 dense all-gather 驻留 ~12.5 + seq512 saved activations ~5 + expert fp32 reduce 瞬态 ~0.8）

历史 EP16 实测（09:50:58 run，log `sft_dsv41_steps116_20261017_095058.log`）：

| 指标 | 实测 |
|---|---|
| checkpoint read | 6.20 GiB/rank |
| After FSDP | 6.21 GiB alloc / 14.85 reserved |
| step 1 | **30.981867790222168 GiB alloc / 33.43 reserved** |
| step 2 | 32.865 GiB / 41.87 |
| step ≥3 | **36.18345499038696 GiB alloc / 45.30 reserved** |
| CPU memory | ~940–960 GB |
| 步速 | ~7.1–7.2 s/it |

本次 3 epoch 预计：1401 × 7.2 s ≈ **2.8 h** + 3 次 ckpt 保存/导出时间。

## 4. 状态时间线

| 时间 | 事件/证据 |
|---|---|
| 10:28 | 清理旧 launcher/ray；NPU 空闲（~3 GB/卡），host available 1979 GB |
| 10:30:01 | 启动 3 epoch run（`logs/sft_dsv41_steps1401_20261017_103001.log`） |
| 10:30–10:33 | 16 个 `ray::TrainingWorker.reset` 初始化（CPU ~96%，RSS ~13.6→15.6 GB/worker）；NPU 仍为基线显存 |
| 10:33:14 | 模型装载完成：`checkpoint read: 958 tensors slice-read directly (6.20 GiB this rank)`；`parameters materialized in 2.8s`；`After FSDP: 6.21 GiB alloc / 14.85 reserved`；`Total steps: 1401` |
| 10:33:30 | **step1 = 30.981867790222168 GiB alloc / 33.4336 reserved**；loss 0.7708839178085327；grad_norm 8.11634635925293 |
| 10:33:47 | step2 = 32.865 GiB / 41.873；step3 = **36.18345499038696 GiB / 45.25**，进入稳态 |
| 10:35:13 | step16/467，6.4–6.5 s/it；NPU HBM 45–50 GB/chip（约 50.3 GB 最高卡），AICore 0–93%；host CPU 936 GB |
| 10:56:03 | **step 186/467（epoch1 40%）**，平均 7.35 s/it；max_alloc 36.1835 GiB / reserved 45.2949 GiB；CPU 940.08 GB；NPU 最高 HBM 50351/65536 MB；ckpt 目录尚空 |
| 预计 11:32 | step467（epoch1 末）保存 `global_step_467` |
| 预计 12:20 | step934（epoch2 末）保存 `global_step_934` |
| 预计 13:10 | step1401（epoch3 末）保存 `global_step_1401`，训练退出 |

> 注：10:56 实测步速 7.35 s/it，比早期 6.4–6.5 s/it 慢，1401 步预计 ~2.9 h；epoch 边界时间按 7.4 s/it 重估。

### 4.1 本次运行关键日志证据（`logs/sft_dsv41_steps1401_20261017_103001.log`）

```
[fsdp_turbo_dsv41] checkpoint read: 958 tensors slice-read directly (6.20 GiB this rank; no cross-rank broadcast)
[fsdp_turbo_dsv41] parameters materialized in 2.8s (direct shard load)
[fsdp_turbo_dsv41] moved 134 buffers to npu (...)
After FSDP, memory allocated (GB): 6.21, memory reserved (GB): 14.85, device memory used/total (GB): 15.38/61.27
Total steps: 1401, num_warmup_steps: 0
step:1 - perf/max_memory_allocated_gb:30.981867790222168 - perf/max_memory_reserved_gb:33.43359375 - perf/cpu_memory_used_gb:926.36 - train/loss:0.7708839178085327 - train/grad_norm:8.11634635925293
step:2 - perf/max_memory_allocated_gb:32.86505317687988 - perf/max_memory_reserved_gb:41.873046875 - train/loss:0.6638268828392029 - train/grad_norm:11.876434326171875
step:3 - perf/max_memory_allocated_gb:36.18345499038696 - perf/max_memory_reserved_gb:45.25 - perf/cpu_memory_used_gb:934.86 - train/loss:0.8606873750686646 - train/grad_norm:9.86506462097168
```

与 2026-10-17 09:50 的 EP16 单步 run（`sft_dsv41_steps116_20261017_095058.log`）对比：
step1/2/3 的 `max_memory_allocated_gb` 和 loss 逐位相同，证明本次重启的数据顺序、模型装载和 EP16 拓扑一致。

### 4.2 监控方式

- 后台训练任务：`bkc1pqhw3`（Bash background，退出时自动通知）。
- 会话级 cron：`c59febe5`，每 10 分钟（每小时 **:04/:14/:24/:34/:44/:54**）检查一次日志（不写入磁盘，会话结束即失效；训练/导出完成后删除）。
- 每次检查记录：当前 step/epoch、`max_memory_allocated_gb`、`max_memory_reserved_gb`、`cpu_memory_used_gb`、步速、`global_step_{467,934,1401}` 是否出现。
- 计划：训练完成后按 §5 导出 epoch1/2/3，验收后删除 `sft_out_ep16_3ep/global_step_*` 临时 ckpt。

监控记录：

| 时间 (UTC) | 进度 | max_alloc / reserved (GiB) | CPU (GB) | 步速 | ckpt |
|---|---|---|---|---|---|
| 10:56:03 | epoch1 186/467 | 36.1835 / 45.2949 | 940.08 | 7.35 s/it | 无 |
| 10:58:37 | epoch1 207/467 | 36.1835 / 45.2949 | 940.22 | 7.11 s/it | 无 |
| 11:07:55 | epoch1 283/467 | 36.1835 / 45.2949 | 940.63 | 7.43 s/it | 无 |
| 11:17:54 | epoch1 364/467 | 36.1835 / 45.3027 | 940.92 | 7.61 s/it | 无 |
| 11:27:55 | epoch1 446/467 | 36.1835 / 45.3027 | 941.08 | 7.59 s/it | 无（预计 ~11:30 保存） |
| 11:37:55 | epoch2 54/467（global 521） | 36.1835 / 45.3027 | 941.58 | 6.99 s/it | **global_step_467 已存在** |
| 11:47:55 | epoch2 135/467（global 602） | 36.1840 / 45.3223 | 942.42 | 7.08–7.61 s/it | global_step_467 |
| 11:57:55 | epoch2 216/467（global 683） | 36.1840 / 45.3223 | 942.75 | 7.34 s/it | global_step_467 |
| 12:07:54 | epoch2 299/467（global 766） | 36.1840 / 45.3223 | 942.88 | 7.09–7.24 s/it | global_step_467 |
| 12:17:55 | epoch2 380/467（global 847） | 36.1840 / 45.3223 | 942.95 | 7.12 s/it | global_step_467 |
| 12:27:55 | epoch2 461/467（global 928） | 36.1840 / 45.3223 | 943.13 | 7.46–7.51 s/it | global_step_467（934 即将保存） |
| 12:37:55 | epoch3 69/467（global 1003） | 36.1840 / 45.3223 | 943.50 | 7.34 s/it | **global_step_467 + 934** |
| 12:47:55 | epoch3 150/467（global 1084） | 36.1840 / 45.3223 | 943.70 | 7.20–7.38 s/it | global_step_467 + 934 |
| 12:57:56 | epoch3 231/467（global 1165） | 36.1840 / 45.3223 | 943.81 | 7.31 s/it | global_step_467 + 934 |
| 13:07:55 | epoch3 312/467（global 1246） | 36.1841 / 45.3223 | 944.22 | 7.09–7.33 s/it | global_step_467 + 934 |
| 13:17:55 | epoch3 391/467（global 1325） | 36.1841 / 45.3223 | 944.41 | 7.14–7.42 s/it | global_step_467 + 934 |
| 13:27:55 | epoch3 467/467（global 1401） | 36.1841 / 45.3223 | 944.54 | 7.23 s/it（末步） | **global_step_467 + 934 + 1401** |

### 4.5 epoch3 checkpoint / 训练完成证据（`global_step_1401`）

```
step:1401 - perf/max_memory_allocated_gb:36.18409729003906 - perf/max_memory_reserved_gb:45.322265625 - perf/cpu_memory_used_gb:944.5374717712402 - train/loss:0.24097497761249542 - train/grad_norm:4.480372905731201
13:27:44 [Rank 3]  Saved model to .../global_step_1401/model_world_size_16_rank_3.pt
13:27:49 [Rank 6]  Saved model to .../global_step_1401/model_world_size_16_rank_6.pt
Saving checkpoint to: /mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_1401
Saved dataloader state to: .../global_step_1401/data_0.pt
Updated checkpoint tracker: .../latest_checkpointed_iteration.txt
Total time for train steps: 0.00s
Final validation metrics: None
```

- `model_world_size_16_rank_*.pt`：**16/16 个**，合计 **100 GB**。
- `latest_checkpointed_iteration.txt` = `1401`。
- 后台训练任务 `bkc1pqhw3` **exit code 0**；训练总耗时约 **2h57m**（10:30:01–13:27:55）。
- 全程峰值：**36.1841 GiB alloc / 45.3223 GiB reserved**；CPU ~944 GB；无 OOM/异常。
- 三个 ckpt 各 100 GB，共 300 GB；共享盘剩余 3.3 TB。

## 4.6 训练完成 → 导出阶段

训练已完成（epoch1/2/3 = `global_step_467/934/1401`），开始按 §5 导出三个 epoch 权重。
导出日志将写入 `logs/export_epoch{1,2,3}_*.log`，每个 epoch 验收通过后删除对应临时 ckpt。

### 4.6.1 epoch1 导出前体检（`global_step_467`）

```
[export] 16 shard file(s) in .../global_step_467
[export] entries: 958
[export] DTensor entries: 918 / plain: 40
[export] engram-named entries in state_dict: 0
[export] fused-expert entries: 80
[export] train params (rank0): 958 -> mapped 958, unmapped 0
[export] checkpoint tensors covered: 4718 / 6256; leftover 1538
```

- 16 个 shard 全部可读；958 个训练参数（918 DTensor + 40 buffer）全部映射，0 unmapped。
- 无 engram 条目（noengram 模型），fused expert 80 个（40 层 × gate_up/down）。
- leftover 1538 = 原 checkpoint 中未参与训练的 dspark/MTP/vision 等张量，按工具设计从 `--orig-dir` 保留。
- 导出命令：
  ```bash
  CKPT=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_467
  ORIG=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-noengram
  OUT=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1
  python3 my_scripts/export_sft_hf.py export --ckpt-dir "$CKPT" --orig-dir "$ORIG" \
      --out-dir "$OUT" --verify-probe layers.0.attn.wq_b.weight
  ```
- 后台导出任务：`bb48rtq2i`，日志 `logs/export_epoch1_export.log`。

### 4.6.2 epoch1 导出完成（验收通过）

```
[export] index rebuilt: 6256 tensors (4718 new + 1538 original-derived)
[export] side files symlinked/copied
[export] probe layers.0.attn.wq_b.weight: shape (32768, 1280) equal_to_original=False max_abs_diff=0.0014238357543945312
[export] EXPORT DONE: /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1
```

- 输出目录：`/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1`，**100 GB**。
- `model.safetensors.index.json`：**6256 tensors / 46 files**，缺失文件 **0**。
- `config.json` 为 noengram 配置（`engram_layer_ids: []`），symlink 到 `...-noengram/config.json`。
- probe 与原始值不同（max_abs_diff ≈ 1.42e-3），证明导出的是训练后的权重，不是原值。
- epoch1 导出任务 `bb48rtq2i` exit code 0。

### 4.6.3 epoch2 导出（`global_step_934` → `...-epoch2`）

- 体检：`958 -> mapped 958, unmapped 0`；`4718 / 6256` covered；16 shard。
- 命令：
  ```bash
  CKPT=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_934
  ORIG=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-noengram
  OUT=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch2
  python3 my_scripts/export_sft_hf.py export --ckpt-dir "$CKPT" --orig-dir "$ORIG" \
      --out-dir "$OUT" --verify-probe layers.0.attn.wq_b.weight
  ```
- 后台任务：`bz4laxopx`，日志 `logs/export_epoch2_export.log`。

### 4.6.4 epoch2 导出完成（验收通过）

```
[export] index rebuilt: 6256 tensors (4718 new + 1538 original-derived)
[export] probe layers.0.attn.wq_b.weight: shape (32768, 1280) equal_to_original=False max_abs_diff=0.0017480850219726562
[export] EXPORT DONE: /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch2
```

- 输出目录 **100 GB**；index **6256 tensors / 46 files**，缺失文件 **0**；config `engram_layer_ids: []`。
- probe max_abs_diff ≈ **1.75e-3**（与 epoch1 的 1.42e-3 不同，说明两个 epoch 的权重确实不同）。
- epoch2 导出任务 `bz4laxopx` exit code 0。

### 4.6.5 epoch3 导出（`global_step_1401` → `...-epoch3`）

- 体检：`958 -> mapped 958, unmapped 0`；`4718 / 6256` covered；16 shard。
- 命令：
  ```bash
  CKPT=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_1401
  ORIG=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-noengram
  OUT=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3
  python3 my_scripts/export_sft_hf.py export --ckpt-dir "$CKPT" --orig-dir "$ORIG" \
      --out-dir "$OUT" --verify-probe layers.0.attn.wq_b.weight
  ```
- 后台任务：`bp94hifhn`，日志 `logs/export_epoch3_export.log`。

### 4.6.6 epoch3 导出完成（验收通过）

```
[export] index rebuilt: 6256 tensors (4718 new + 1538 original-derived)
[export] probe layers.0.attn.wq_b.weight: shape (32768, 1280) equal_to_original=False max_abs_diff=0.00199127197265625
[export] EXPORT DONE: /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3
```

- 输出目录 **100 GB**；index **6256 tensors / 46 files**，缺失文件 **0**；config `engram_layer_ids: []`。
- probe max_abs_diff ≈ **1.99e-3**（epoch1 1.42e-3、epoch2 1.75e-3、epoch3 1.99e-3，三者不同）。
- epoch3 导出任务 `bp94hifhn` exit code 0。

## 4.7 清理临时数据（已完成）

```bash
C=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep
rm -rf "$C/global_step_467" "$C/global_step_934" "$C/global_step_1401"
rm -f  "$C/latest_checkpointed_iteration.txt"
rmdir  "$C"
```

- 删除前 `sft_out_ep16_3ep` = **298 GB**（3 个 FSDP ckpt）；删除后目录已移除。
- 共享盘可用空间：3.3 TB → **2.9 TB**。
- 保留：三份导出权重（各 100 GB）、训练日志、导出日志、本 worklog。

## 4.8 最终产物

| epoch | 目录 | 大小 | 验收 |
|---|---|---:|---|
| 1 | `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1` | 100 GB | 6256 tensors / 46 files / missing 0；probe diff 1.42e-3 |
| 2 | `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch2` | 100 GB | 6256 tensors / 46 files / missing 0；probe diff 1.75e-3 |
| 3 | `/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch3` | 100 GB | 6256 tensors / 46 files / missing 0；probe diff 1.99e-3 |

三份目录的 `config.json` 均为 noengram 配置（`engram_layer_ids: []`），可直接作为 vLLM/verl 的 `MODEL_PATH`。

## 5. 结论

- EP16/FSDP16/efsdp1 单机 16 卡配置在 3 epoch（1401 步）全程稳定：峰值 **36.1841 GiB alloc / 45.3223 GiB reserved**，无 OOM；步速约 7.2–7.4 s/it；总耗时约 2h57m。
- epoch1/2/3 权重均已导出并逐项验收；临时 FSDP ckpt 已删除。
- 全程未修改训练代码或启动脚本，只新增本 worklog。

## 6. 3 epoch 权重评测（GSM8K main test，50 条）

### 6.1 测试集与脚本

- 源测试集：`/mnt/share/m00899630/dsv41/sft_data/gsm8k_main_test_sft.parquet`（1319 条，messages 格式）。
- 从 messages 提取 `question`（user）与 `answer`（assistant 最后一条 `Answer:`），固定随机抽 **50 条**（`random_state=42`），保存为
  `/mnt/share/m00899630/dsv41/sft_data/gsm8k_main_test_50qa.parquet`，三个 epoch 使用完全相同的 50 条。
- vLLM 启动脚本：**`my_scripts/run_vllm_top32_epoch_eval.sh`**（基于 `run_vllm_top32 epoch123.sh` + `run_vllm_top32_dp2.sh` 改）：
  - 模型目录由 `MODEL_PATH` 指定；DP2 × TP8，16 卡；
  - 仓库本地 `vllm`/`vllm-ascend` 路径；`HCCL_IF_IP=80.5.25.123`；
  - **去掉 `--engram-config` / `--enable-return-routed-experts`**：三个 epoch 模型都是 noengram（`engram_layer_ids: []`），显式 EngramConfig 会被 vLLM 拒绝；
  - `--max-model-len 4096`、greedy、`--max-num-seqs 16`。
- 评测命令（每个 epoch）：
  ```bash
  python3 my_scripts/eval_gsm8k_serve.py --port 20007 --n 50 --max-tokens 1024 \
    --test-file /mnt/share/m00899630/dsv41/sft_data/gsm8k_main_test_50qa.parquet \
    --out /mnt/share/m00899630/dsv41/sft_data/eval_3ep_epochN_50.jsonl
  ```
- 每次 serve 加载约 **2.5–3 分钟**；评测后停止 server、确认 16 卡回到基线显存，再切下一个 epoch。

### 6.2 结果

| epoch | correct | 准确率 | closed(stop) | has_answer | hit_cap |
|---|---:|---:|---:|---:|---:|
| epoch1 | 20/50 | **40%** | 50/50 | 50/50 | 0 |
| epoch2 | 20/50 | **40%** | 50/50 | 50/50 | 0 |
| epoch3 | 22/50 | **44%** | 49/50 | 49/50 | 1 |

- 50 条随机样本的 95% 置信区间约 ±14%，epoch3 的 44% 与 epoch1/2 的 40% **差异不显著**；三个 epoch 总体在同一水平。
- 三个 epoch 都对（16 条）：idx `[1,2,9,17,18,21,22,26,28,31,33,35,36,41,42,46]`。
- 三个 epoch 都错（23 条）：idx `[3,4,11,12,13,14,15,16,19,20,23,24,25,30,32,37,38,39,40,43,44,45,47]`。
- 各 epoch 独有正确：epoch1 `{6,10}`；epoch2 `{7,34,49}`；epoch3 `{5,8,48}`。
- epoch3 唯一未闭合样本：idx 12（Kelly 5 quarters + 2 dimes 买 55 分 pop），输出 `200` 反复循环到 1024 token cap；其余 49 条均正常 `stop` 且给出 `Answer:`。
- 失败样本绝大多数 **已经正常闭合但算错**（算术/单位/多步推理错误），不是 never-EOS；SFT 已基本治好循环收尾问题。

### 6.3 与历史基线对比（仅参考，样本不同）

- 历史 before-SFT：32 条中 **0 条正确**，大量 never-EOS/循环（`eval_before_top32_20261010_1753.jsonl`）。
- 历史 1 epoch SFT（gbs64，116 步）：32 条中 **8 条正确（25%）**（`eval_after_sft116_20261011_0501.jsonl`）。
- 本次 3 epoch：同一固定 50 条上 **40–44%**；样本与历史不同，不能直接做显著性比较，但方向一致（收尾修复 + 正确率提升）。

### 6.4 产物

- `sft_data/gsm8k_main_test_50qa.parquet`（固定 50 条，含 orig_idx）
- `sft_data/eval_3ep_epoch{1,2,3}_50.jsonl`（逐样本 question/pred/gt/finish/text_tail）
- `logs/eval_3ep_epoch{1,2,3}_50.log`
- `logs/vllm_DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch{1,2,3}_*.log`
- 评测结束后已停止全部 vLLM server，16 卡回到基线显存。

### 4.4 epoch2 checkpoint 证据（`global_step_934`）

```
12:29:07 [Rank 6]  Saved model to .../global_step_934/model_world_size_16_rank_6.pt
12:29:08 [Rank 5]  Saved model to .../global_step_934/model_world_size_16_rank_5.pt
12:29:13 [Rank 4]  Saved model to .../global_step_934/model_world_size_16_rank_4.pt
12:29:13 [Rank 11] Saved model to .../global_step_934/model_world_size_16_rank_11.pt
Saving checkpoint to: /mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_934
```

- `model_world_size_16_rank_*.pt`：**16/16 个**，合计 **100 GB**。
- `latest_checkpointed_iteration.txt` = `934`。
- epoch2 进度条总计 58:00（含末尾存盘）；保存窗口约 12:29:07–12:29:13。
- 共享盘剩余 **3.3 TB**（两个 ckpt 已占 200 GB）。

### 4.3 epoch1 checkpoint 证据（`global_step_467`）

```
11:31:06 [Rank 14] Saved model to .../global_step_467/model_world_size_16_rank_14.pt
11:31:11 [Rank 10] Saved model to .../global_step_467/model_world_size_16_rank_10.pt
11:31:11 [Rank 0]  Saved model config and tokenizer class to .../global_step_467/huggingface
11:31:13 [Rank 4]  Saved model to .../global_step_467/model_world_size_16_rank_4.pt
Saving checkpoint to: /mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_467
Saved dataloader state to: .../global_step_467/data_0.pt
```

- `model_world_size_16_rank_*.pt`：**16/16 个**，每个 6,653,610,373 B（~6.20 GiB），合计 **100 GB**。
- `latest_checkpointed_iteration.txt` = `467`。
- 目录还含 `fsdp_config.json`、`data_0.pt`、`huggingface/`（config/tokenizer）。
- 保存耗时约 7 s（11:31:06–11:31:13），训练随后进入 epoch2。

## 5. 导出计划（训练完成后执行）

临时 ckpt 目录：`/mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_{467,934,1401}/`
（每 rank `model_world_size_16_rank_<r>.pt`，16 个文件）。

导出工具：`my_scripts/export_sft_hf.py`（离线 CPU 汇编，glob `model_world_size_*_rank_*.pt`，支持任意 world size）。

每个 epoch 执行：

```bash
CKPT=/mnt/share/m00899630/dsv41/sft_out_ep16_3ep/global_step_467
ORIG=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-sft116-noengram   # 与训练 MODEL_PATH 一致（noengram config）
OUT=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-gsm8k-sft-3ep-epoch1

python3 my_scripts/export_sft_hf.py inspect  --ckpt-dir "$CKPT"
python3 my_scripts/export_sft_hf.py dry-map  --ckpt-dir "$CKPT" --orig-dir "$ORIG"
python3 my_scripts/export_sft_hf.py export   --ckpt-dir "$CKPT" --orig-dir "$ORIG" \
        --out-dir "$OUT" --verify-probe layers.0.attn.wq_b.weight
```

epoch2/3 分别用 `global_step_934`/`global_step_1401` 和
`...-epoch2`/`...-epoch3` 输出目录。

导出验收：

- `dry-map`：958 train params → mapped，0 unmapped；
- `export` 打印 `EXPORT DONE`，`model.safetensors.index.json` 可解析，6256 张量（4718 新 + 1538 原值）；
- probe `layers.0.attn.wq_b.weight` 与原始值不同（训练生效）；
- 输出目录 `config.json` 为 noengram 配置（`engram_layer_ids: []`）。

清理：每个 epoch 导出并验收后删除对应的 `global_step_N` 临时 ckpt；三个都完成后删除
`/mnt/share/m00899630/dsv41/sft_out_ep16_3ep` 空目录。保留 `weights/...epoch{1,2,3}` 与训练/导出日志。

## 6. 已知风险/注意事项

- **不要与其它 SFT launcher 并发**：本机共享 16 卡，启动前必须 `pkill` + `ray stop --force` 并核对 `npu-smi`。
- `optimizer_offload=True` 不降低 in-step 峰值（Adam 在 train_batch 期间常驻），只增加 host 副本；本次为复现已验证配置保留。
- ckpt 写在共享盘 `/mnt/share`（3.5T 可用）；3 个 ckpt 约 300 GB，3 个导出目录约 330 GB，导出后及时删 ckpt。
- 导出 `--orig-dir` 必须用训练同款 `...-noengram`，否则 config 会带回 engram 层。
- 导出脚本按 DTensor 的 mesh/placement 汇编，EP16 的 1×1×16 expert mesh 与 FSDP16 的 dense mesh 均可处理；首次 `dry-map` 失败时先查 `nrows`/`placements`。

## 7. 错误记录

（运行中遇到错误时在此追加：时间、原始报错、根因、处置、证据路径。）
