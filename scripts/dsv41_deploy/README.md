# dsv41_deploy —— DeepSeek-V4.1-Flash top32：部署 + 启动包（GRPO/RL 与 SFT）

> **用途**：把一个新容器/新机器从零部署到能跑**双机 GRPO** 与**双机 SFT**；本目录是自包含的脚本+文档快照。
> **配套文档**：失败库 [`../../wiki/dsv41_run_failures_catalog.md`](../../wiki/dsv41_run_failures_catalog.md)（出问题先查它）；
> 一致性手册 [`../../wiki/dsv41_consistency_playbook.md`](../../wiki/dsv41_consistency_playbook.md)（训推一致性排查）；
> SFT 细节：`/workspace-verl/.claude/skills/dsv41-sft/`。
> **版本快照（当前生产栈）**：FSDPTurbo `7714bef` + vllm-ascend `b2da8fb47` + vllm `ced6857af`（`VLLM_VERSION=0.30.0`）。

## 0. 包内容与部署约定

| 文件 | 作用 |
|---|---|
| `preflight.sh` | 单机体检（**部署验收 + 每次启动前**；FAIL 行对照失败库编号） |
| `launch_grpo_master.sh` / `launch_grpo_slave.sh` | GRPO 双机一键启动（包装 `run_grpo_2node.sh`，已带 proven env） |
| `launch_sft_master.sh` / `launch_sft_slave.sh` | SFT 双机一键启动（包装 `run_sft_dsv41_2node.sh`） |
| `run_grpo_2node.sh` | GRPO 双机主启动器：自动挑卡、precheck、Ray 集群管理（master/slave 两态） |
| `run_grpo_top64.sh` | GRPO 单机启动器（被 2node 的 master 以 `scripts/run_grpo_top64.sh` 相对路径调用） |
| `run_sft_dsv41_2node.sh` | SFT 双机主启动器 |

**部署约定（重要）**
- 目标机工作目录约定为 **`/workspace-verl`**（代码树 `vllm/`、`vllm-ascend-v41-private/`、`FSDPTurbo/`、`verl/` 都在其下）。
- **全新机器部署** = 把本目录 `*.sh` 拷到目标机 `/workspace-verl/scripts/`（runner 之间按这个相对路径互相调用）：
  ```bash
  scp -P 50001 dsv41_deploy/*.sh root@<host>:/workspace-verl/scripts/
  ```
- 已有机器（119/120/131…）的 `/workspace-verl/scripts/` 已含同名 runner（与本包同一份快照）；
  `launch_*.sh` 会用**自身所在目录**解析同目录 runner，所以把整个目录放哪都能就地跑。

## 1. 新机部署步骤

### 1.0 前置
- 容器可访问网卡 `enp48s3u1u1`（HCCL/GLOO 都绑它；节点 IP 形如 `80.5.25.x`）；
- `/mnt/share` 已挂载（权重/数据/词表全在 `/mnt/share/m00899630/` 下）；
- 本环境 ssh 约定：`ssh -p 50001 root@80.5.25.x`。

### 1.1 同步代码树（三选一）
1. **整树镜像（推荐，131 为母本）**：`bash scripts/sync_to_host.sh <目标IP>`（tar over ssh，末尾逐文件 md5 验收）；
2. 从已有同类容器推（同上，母本换成该机）；
3. 全新构建 → 按下面"**不入 git 生成物清单**"逐项核对，缺一不可。

**新机最常漏的 3 个生成物**（失败库 F-01/F-02/F-03；都会跑到很深阶段才爆）：

| 文件 | 缺失症状 |
|---|---|
| `vllm/vllm/_version.py` | 启动即 `InvalidVersion: Invalid version: 'dev'` |
| `vllm_ascend/libvllm_ascend_kernels.so`（顶层 + `lib/` 两份） | 引擎 `WorkerProc initialization failed` |
| `vllm_ascend/_cann_ops_custom/**`（~1304 文件）+ `include/**` + `vllm_ascend/_version.py` | `aclnnHcPre ... not in libopapi.so` |

### 1.2 数据与权重（`/mnt/share/m00899630` 下，两机共享）
- 模型：`weights/DeepSeek-V4.1-Flash-top32-bf16`（原始）、`weights/DeepSeek-V4.1-Flash-top32-sft116`（SFT 热启动后，**RL 用这个**）；
- 词表：`dsv41/rl_tokenizer`；
- RL 数据：`dsv41/rl_data/gsm8k_main_{train,test}_rl.parquet`（生成脚本 `scripts/prepare_gsm8k_rl.py`）；
- SFT 数据：`dsv41/sft_data/gsm8k_main_train_sft.parquet`（生成脚本 `scripts/prepare_gsm8k_sft.py`）。

### 1.3 体检（每台）
```bash
bash preflight.sh          # 期望最后一行 ALL OK
```
> `preflight.sh` 与 4 个 `launch_*.sh` 均已内置 **CANN 环境自举**（F-51：非 login shell 下自动 source
> `ascend-toolkit/set_env.sh`）与 **fd 提升**（F-64）：从普通 ssh 直跑即可，无需 `bash -lc` 包装。
> 2026-10-13 实跑验收：119 上 `ALL OK`。

## 2. 环境变量速查（启动器已内置默认；一般不用改）

| 变量 | 默认（启动器） | 说明 |
|---|---|---|
| `MODEL_PATH` | `.../DeepSeek-V4.1-Flash-top32-sft116` | **RL 别用 bf16 原始权重**（never-EOS，F-31） |
| `TRAIN_FILE` / `TEST_FILE` | gsm8k rl parquet | |
| `EFSDP_SIZE` | **1** | RL 必须 1（F-30）。⚠ 主启动器 `run_grpo_2node.sh` 内历史默认是 2——两个 `launch_grpo_*.sh` 已显式覆盖为 1，**不要裸跑主启动器** |
| `ROLLOUT_IS` | `token` | TIS 截断重要性采样（threshold 2.0）；`null` 关闭 |
| `VA_V41_OPROJ_FP32_AR` | `1` | 引擎 o-proj 段 fp32 all-reduce（一致性优化；**两节点必须同值**） |
| `FSDP_SIZE` / `EP_SIZE` / `GEN_TP` | 16 / 8 / 8 | 拓扑固定；EP 必须 =8（否则触发 F-42） |
| `SAMPLE_N` / `MAX_RESPONSE_LEN` | 8 / 1024 | train_batch_size × n 需被 16 整除 |
| `FSDP_TURBO_ATTN_RECOMPUTE` / `_CHUNK` | 1 / 64 | **别关**（关了 update OOM，F-20）；改值两节点同值 |
| `VERL_DSV41_DIRECT_SHARD_LOAD` | 1 | 每 rank 直读自己权重分片（替代 ~30MB/s 跨机广播） |
| `VERL_DSV41_LOCAL_EXPERT_EXPORT` | 1 | 权重同步只导本 rank 专家分片（日志 `SYNC-PROFILE sender:` 可见） |
| `VLLM_VERSION` | `0.30.0` | 旧栈 pin；换 rebase 新栈时需联动（见脚本注释） |
| `RAY_memory_usage_threshold` | 0.99 | 宿主内存 kill 线（≈2.43TB×0.99） |
| `ASCEND_RT_VISIBLE_DEVICES` | 自动挑前 8 张空闲 die | 需固定时自行 export（如避开邻居占用的 die） |
| `HCCL_IF_IP` / `GLOO_SOCKET_IFNAME` | 自动探测 NIC | 只有单机 `run_grpo_top64.sh` 需要手工覆盖（默认写死 131） |

## 3. 启动 GRPO（RL，双机；每台取 8 die，共 16 die）

```bash
# ① master（例：80.5.25.119）：
bash launch_grpo_master.sh 5        # 5 步；冒烟用 1。支持 STEPS env 覆盖
# ② 等 master 日志出现 "[2node] waiting for the slave"
# ③ slave（例：80.5.25.120）：
bash launch_grpo_slave.sh 80.5.25.119
```

**验收里程碑**（master 包装器日志按序 grep）：
```
[2node] precheck ok
[2node] slave present (2 active nodes, 16 NPU total)
[fsdp_turbo_dsv41] parameters materialized in Xs   ← 权重装载（direct shard load）
host-offload Engram tables loaded …                ← engram 表上主机
SYNC-PROFILE sender: …                             ← 每个 step 后权重同步的证据行
step: N - … training/rollout_actor_probs_pearson_corr: …   ← 训练指标
Training Progress: 100%
```
- 训练日志：`verl/logs/grpo_top64_steps<N>_<stamp>.log`（容器重启不丢）。
- **停止/清场**：见 §6。

## 4. 启动 SFT（双机；每台取 16 die，共 32 die）

关键事实（全部血泪实测，详见 skill）：
- `expert_fully_shard_parallel_size=2`（脚本默认；efsdp=1 → 每卡专家三件套 ~44GB，step2 必 OOM，F-21）；
- **不要开任何 offload**（不降峰值、步速 4× 慢）；micro=1 / gbs=64；
- **116 步 = 1 epoch**（稳态 ~12s/步 ≈ 25 分钟；加载 ~15 分钟）。

```bash
# ① master：
bash launch_sft_master.sh 116
# ② 等 "[sft2] waiting for the slave"
# ③ slave：
bash launch_sft_slave.sh <MASTER_IP>
```
里程碑：`[sft2] slave present, NPU total=32` → `materialized in Xs` → `host-offload Engram tables loaded` → `step:1 … loss:1.8x` → `Epoch 1/1: 116/116 → Saved model to …/global_step_116/`。

后续（ckpt 合并 → 导出 → 评测）要点：
- ckpt 双机各写本地盘（每卡 ~55GiB）→ 合并到 `/mnt/share/m00899630/dsv41/sft_ckpt_top32_gsm8k/global_step_116/`（32 分片；**128 侧复制要 nohup + 完成标记**防 ssh 超时截断）；
- 导出：`python3 scripts/export_sft_hf.py inspect/dry-map/export`（验收：958 mapped / 0 unmapped）；
- 评测：`MODEL_PATH=<导出目录> bash scripts/run_vllm_top32_dp2.sh` + `eval_gsm8k_serve.py`。

## 5. 监控（常用一行）

```bash
L=verl/logs/grpo_top64_steps<N>_<stamp>.log
grep -E "pearson|diff_mean|rewards/mean|clip_ratio|grad_norm" $L | tail          # 训练指标
grep -E "SYNC-PROFILE|timing_s/" $L | tail                                       # 同步与耗时
grep -c "Traceback" $L                                                           # 0 = 干净（收尾噪声见 F-32）
```
指标含义：`rollout_actor_probs_pearson_corr` / `rollout_probs_diff_mean` 即训推一致性（目标 ≥0.995 / <0.01，现状与路线见一致性 worklog §6.4）；
`rollout_corr/rollout_is_*` 为 TIS 字段；`reward mean` 正常应离开 -1 常数区。

## 6. 停止与清场（每台各跑一遍）

```bash
ray stop --force; sleep 2
pkill -9 -f "VLLM::"; pkill -9 -f "main_ppo"; pkill -9 -f "vLLMHttpServer"
pkill -9 -f "ray::"; pkill -9 -f "raylet"; sleep 3
ps -eo stat,comm | grep -E "VLLM::|WorkerDict|main_ppo|raylet" | grep -cv "^Z"   # 期望 0
bash -lc "npu-smi info" | grep -c "Process id"                                   # 期望 0（须 login shell，F-50）
```
⚠ `pkill "ray::"` **只能在集群已停或启动前**执行（活着时会把 Ray 代理连带杀掉，F-57）；`<defunct>` 僵尸无害（F-60）。

## 7. 排障速查

全量分类库：[`../../wiki/dsv41_run_failures_catalog.md`](../../wiki/dsv41_run_failures_catalog.md)。高频：

| 症状 | 查 |
|---|---|
| 启动即挂 / ImportError / .so 找不到 | F-01/02/03（生成物同步） |
| 引擎 `WorkerProc initialization failed` | F-02 |
| `aclnnHcPre ... not in libopapi.so` | F-03 |
| `rows are not tiled across ranks at 0` | F-04（两台都要带修复） |
| update OOM | F-20/22/23 |
| reward 恒 -1 | F-30（输出乱码）vs F-31（response 顶格） |
| slave 加入失败 / 心跳丢失 / 节点被杀 | F-62/F-63 |
| 卡在 "Connected to Ray cluster" 不动 | F-64（fd 耗尽） |
| 日志尾 Traceback 但指标已出 | F-32（收尾噪声，无害） |
| npu-smi 空输出/假 0 | F-50 |

## 8. 单机模式（可选）

- GRPO 单机 8 卡：`bash run_grpo_top64.sh 1`（注意脚本默认 `HCCL_IF_IP=80.5.25.131`，换机须覆盖为本站 IP）；
- SFT 单机 16 卡历史变体：`scripts/run_sft_dsv41_1node16.sh`（未随本包，见工作区 scripts/）。
