# DeepSeek-V4.1-Flash启动说明

## 1. 单机A3


拉取镜像：
```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.11
```

下载代码
```
git clone -b dev_dsv41_1027  https://gitcode.com/menjiantong/FSDPTurbo
git clone -b from_9a38d36_1030 https://github.com/menjiantong/vllm-ascend/
git clone -b from_a98247a_1030 https://github.com/menjiantong/vllm/
git clone -b dev_dsv41_rebase1030_cp_engram https://github.com/menjiantong/verl

cd FSDPTurbo
git reset --hard cc3cd1eba9466a03be0f85922eae2e644bda0b5a
cd ..

cd vllm-ascend/
git reset --hard b0901f799fa1a178254304d44269fe352dbce315
cd ..

cd vllm
git reset --hard fd46ca5adaee11331a4c2f1b74483fef491f03cf
cd ..

cd verl 
git reset --hard 6302c3f682cf159299b90bdeeea22f1286d70e11
cd ..
```

安装环境
```
bash build.sh
bash 1110_run_dsv41_188_longrun_v1.sh
```
## 2. 性能优化版（1114 验证：启动 1043 s → ~390 s，长跑 0 OOM）

### 2.1 最新 commit（只更新 FSDPTurbo 和 verl，其余与第 1 节相同）

```
cd FSDPTurbo
git reset --hard 1852e93064c8e679bb0f731206ad0e549ccde1a2
cd ..

cd verl
git reset --hard 7352d013929d7571a8c98df8aa2ec2720d555a93
cd ..

cd vllm-ascend/
git reset --hard b0901f799fa1a178254304d44269fe352dbce315
cd ..

cd vllm
git reset --hard fd46ca5adaee11331a4c2f1b74483fef491f03cf
cd ..
```

- `FSDPTurbo 1852e93`：新增 `MY_ENV_DSV41_MODEL_INIT` / `MY_ENV_DSV41_ENGRAM_INIT` 初始化开关（默认关，行为不变）。

### 2.2 启动脚本


```
cd verl
bash my_scripts/1114_run_dsv41_188_longrun_perf_v3.sh                        # 默认 200 步
MY_ENV_DSV41_STEPS=1 bash my_scripts/1114_run_dsv41_188_longrun_perf_v3.sh   # 1 步冒烟
```
日志：`verl/logs/DeepSeek-V4.1-Flash-top32-bf16-<YYYYmmdd_HHMMSS>.log`（训练入口自身 tee）。

### 2.3 验证日志（80.5.25.124 / 80.5.27.147）

日志目录：`/mnt/share/m00899630/workspace/dsv41-verl-rebase/verl/logs/`

| 日志文件 | 脚本/机器 | 结果 |
|---|---|---|
| `1110_188_baseline124_20261114_045636.log` | 第 1 节基线口径 / 124 | 2 步 rc=0，启动 ≈1043 s |
| `1110_188_perf124_rep_20261114_054555.log` | v1（含显式 KV 的旧版）/ 124 | 2 步 rc=0，pearson 0.99921/0.99907，min free 19.0 GiB |
| `1110_188_perf124_rep2_20261114_060550.log` | v1 / 124 | 2 步 rc=0，pearson 0.99928/0.99926 |
| `1110_188_perf124_step2_20261114_062602.log` | v1 + PERF / 124 | 1 步 rc=0，启动 385.8 s |
| `1110_188_perf124_revert_20261114_072755.log` | 1110 common 回退后 / 124 | 1 步 rc=0，pearson 0.99904 |
| `1110_188_perf124_clean_20261114_074249.log` | 清理后最终树 / 124 | 1 步 rc=0，pearson 0.99918，step 302.0 s |
| `1110_188_longrun_perf_20261114_080933.log` | **v2 简化版（no-KV）/ 124，200 步** | step1 完成：pearson 0.99923、step 342.1 s、0 OOM；08:24:50 起整个进程组被 SIGSTOP（T 状态），

## 3. 图模式+mini batch size




### 3.1 最新 commit（只更新 FSDPTurbo 和 verl，其余与第 1 节相同）

```
cd FSDPTurbo
git reset --hard 1852e93064c8e679bb0f731206ad0e549ccde1a2
cd ..

cd verl
git reset --hard 9deab56a4436c8b15d903c4044f8853c209f5949
cd ..

cd vllm-ascend/
git reset --hard b0901f799fa1a178254304d44269fe352dbce315
cd ..

cd vllm
git reset --hard fd46ca5adaee11331a4c2f1b74483fef491f03cf
cd ..
```

verl新增启动脚本

### 3.2 启动脚本


`verl/my_scripts/final/`：`env_dsv41.sh`（公共环境）+ 3 个启动脚本

数据 countbenchqa_lite，bsz 8×n8，FSDP16/EP16/CP4，推理 TP8×DP2/EP16，32 专家。

| 脚本 | 形状 | 差异 | 实测（单机 16 卡，满长口径） |
|---|---|---|---|
| `1116_dsv41_1k2k_graph_v1.sh` | prompt≤1k，resp 2k | 图模式，mb 1/1/1 | 410.1 s/步 |
| `1116_dsv41_1k2k_graph_mb2_v1.sh` | 同上 | + actor mb2 / ref·logprob mb4 | 382.1 s/步（同机比 v1 快 10%） |
| `1116_dsv41_2k6k_graph_mb1_v1.sh` | prompt≤2k，resp 6k | 图模式，actor mb1 / ref·logprob mb2 | 1329.2 s/步 |

启动：
```
cd /mnt/share/m00899630/workspace/dsv41-verl-rebase/verl
STEPS=200 bash my_scripts/final/1116_dsv41_1k2k_graph_mb2_v1.sh
```

