# worklog：dsv41 部署启动包（verl/SFT）+ 运行失败库补全（2026-10-13 晚）

> 用户指令两条：① "先整理下脚本吧，给我一份完整可用的脚本，和 readme 文档，放到 verl 仓库中，我用来部署启动 verl 和 SFT"；
> ② "上次的错误记录不充分，修改的不全，错误日志也不全，补齐"。

## 1. 交付物 ①：部署+启动包 `verl/scripts/dsv41_deploy/`（仓库内）

| 文件 | 说明 |
|---|---|
| `README.md` | 部署+启动总文档：包内容/部署约定、新机部署 4 步（同步树→生成物核对→preflight→冒烟）、env 速查表（16 项）、GRPO 双机启动 3 步+里程碑、SFT 启动（含 efsdp=2/无 offload/116 步关键事实）、监控、清场、排障表、单机模式 |
| `preflight.sh` | 单机体检（fd 软限 / python+PYTHONPATH+import / vllm 版本解析 / vllm_ascend_C.so ldd / `_cann_ops_custom` 文件数 / engram 复本感知修复 / `/mnt/share` 数据面 / NPU 空闲 die 数 / 宿主内存），FAIL 行直指失败库编号 |
| `launch_grpo_master.sh` / `launch_grpo_slave.sh` | GRPO 双机一键启动。**已带 proven env**：`EFSDP_SIZE=1`（显式覆盖主启动器内的历史默认 2，见失败库 F-30）、`MODEL_PATH=…-sft116`（F-31）、`ROLLOUT_IS=token`、`VA_V41_OPROJ_FP32_AR=1`（两节点同值）；`ulimit -n 65536`（F-64）；用脚本自身目录解析 runner（就地可跑） |
| `launch_sft_master.sh` / `launch_sft_slave.sh` | SFT 双机一键启动（STEPS 默认 116=1 epoch） |
| `run_grpo_2node.sh` / `run_grpo_top64.sh` / `run_sft_dsv41_2node.sh` | 三份主启动器**快照**（与工作区 `scripts/` 同源，cp 自 `/workspace-verl/scripts/`，byte 一致） |

**设计决定**：
- 部署约定 = 全新机器把 `*.sh` 铺到 `/workspace-verl/scripts/`；已有机器（119/120/131…）工作区已有同名 runner，包放哪都可就地运行（wrapper 用 `SCRIPT_DIR` 解析）。
- `run_grpo_2node.sh` 必须与 `run_grpo_top64.sh` 同目录（master 以相对路径调它）——README §0 已写明。
- 双机算力口径差异已在 README 写清：**GRPO 每机取 8 die（共 16）；SFT 每机取 16 die（共 32）**（历史常被混淆）。

**已同步**：`/workspace-verl/scripts/dsv41_deploy/`（9 文件）已推送 **119 + 120**；全部 `bash -n` 通过。

## 2. 交付物 ②：失败库补全（`wiki/dsv41_run_failures_catalog.md`）

- **规模**：23 条 → **41 条**（旧编号 F-xx 全保留，充实原文日志与修复命令）。
- **新增 11 条**：
  - F-04 engram tiling 误报（复本感知修复、两侧都要带）；
  - F-24 engram 缓存 `expanded size (4) vs (8)`（VERL_DSV41_MAX_BATCH_SIZE）；F-25 本地盘满（ray spill）；
  - F-32 DataLoader SIGKILL + `ERR99999` 收尾噪声（**含 2026-10-13 1 步跑原文栈**，教如何判定良性）；F-33 SFT 保存 `'…Adapter' object has no attribute 'config'`（权重已存成功+守卫）；
  - F-42 `No backend type associated with device type cpu`（两个出处：optimizer clip_grads 修在树内 / EP≠8 回退 all-gather）；
  - F-56 `/etc/profile` CANN 9.0.0 路径告警（无害，忽略）；F-57 集群建好后 `pkill "ray::"` 连带杀 ray 代理；
  - F-62 从机整树暴毙（121 邻居 `pkill python` 击杀 / 126 宿主被邻居压满）——两种根因+排查；F-63 slave 加入失败（`nodes=1/2` → ray stop 重跑）；F-64 head raylet fd 耗尽（`File descriptor limit reached`）。
- **新增 §8 一致性 harness 类**（F-70…F-76，7 条速查：切片前缀 / 钩子类名 / router 层标识 / TP 切 token / 压缩位编号 / cast 口径 / dump 生产加载三件套），深度指向 `dsv41_consistency_playbook.md` §7。
- **补齐方式**：原文日志全部取自实盘（本 run 日志、`verl/logs/*.log`、taskA worklog §2 排障全史、sft/top64 worklog、skill）并有"来源"标注；未留原文的条目如实写"来源"。
- **索引表/速查顺序**同步更新（新增 §7 节点消失、§8 harness、F-32 假报错识别条目）。
- **已同步**：119 + 120 `/workspace-verl/verl/wiki/`（md5 `8d91a7d0a9b0ae55c9511593842279c7` 三处一致）。

## 3. 验证（含实跑中发现并修复的 3 个自检缺陷）

- `bash -n` 全部通过；**双机（119/120）实跑验收通过**：119 `preflight` 全 OK（ulimit WARN 属预期——ssh 原壳 fd 1024，启动器会提）；
  三处 md5 一致（`preflight 1648d59c` / `launch_grpo_master 9fc3f4e4` / `launch_grpo_slave 6e05a48f`；131 仓库=119=120）。
- **实跑修出的 3 个自检缺陷**（"实跑验收"的价值）：
  1. **版本检查假失败**：`import vllm` 把 INFO 打到 stdout 污染 `$V` → `version.parse` 失败（131/119 均复现）→ `tail -1` 修复；
  2. **ldd 假阳性**：plain `ldd` 对 `libtorch/libc10/libtorch_npu/libascendcl/...` 报 "not found"（运行时经 torch RPATH/CANN 环境解析）→ 改为 **`ctypes.CDLL` 真实 dlopen** 判定（先 `import torch_npu`）；
  3. **非 login shell 缺 CANN**（F-51 在 preflight 自身复发）→ **preflight + 4 个 `launch_*.sh` 全部内置 CANN 自举**（`import torch_npu` 失败时 source `ascend-toolkit/set_env.sh`），非 login shell 直跑可过。
  另：NPU 检查从 per-die 循环（16 次 login-shell 调用，131 上 3.5 分钟）改为**单次 `npu-smi info` + HBM<4GB 口径**（~40 秒）；131 实读 7/16（8 die 邻居 12–13GB + 1 die 46GB），119 实读 16/16。
- **遗留/待确认**：① `run_grpo_2node.sh` 内 `EFSDP_SIZE` 历史默认=2 与新结论（RL 必须=1）冲突——启动器已显式覆盖 + README/失败库双处警示，**是否直接改主启动器默认值待定**（改动前需复核 2-node 场景 efsdp=2 是否真的坏，或干脆默认 1）；② harness §8 为速查版，全量 14 坑仍在 playbook §7（两份保持分工）；③ 131 逢邻居作业时 NPU 计数会波动（共享机，报告口径已注"<4GB"）。
