# Engram 加载逻辑分析与优化设计方案（2026-10-14 · 设计稿 v1，未实施）

> 用户课题：分析 engram 加载逻辑、为何这么久、给出优化方案；**先设计方案**（本文即设计稿）。
> 证据来源：两条加载路径代码精读 + 2026-10-14 凌晨 127/123 实跑时序 + 131 磁盘/网络微基准。

---

## 1. 现状逻辑（代码级）

### 1.1 数据本体（sft116 checkpoint）
- engram 层 = 1/14；每层一张 embed 表：逻辑 **384.0M 行 × 256 维 × bf16 = 183.1 GiB**，
  **各独占一个 safetensors 文件**（`model-00008-of-00268` / `model-00151-of-00268`）。
- 其余 engram 张量（wkv/q/k weights）小文件合计 ~4 GiB；checkpoint 全量 493.4 GiB（engram 占 75%）。

### 1.2 训练侧现役路径：`_load_host_engram_tables`
（`verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py:1188`）
1. 模型构建时 FSDPTurbo `host_offload_embedding.shard_()` 已按 **1-D EP mesh 行分片**：每 rank
   `384M/EP` 行（FSDP16/EP8 → 48M 行/rank/表 = pinned host **24.6 GiB**）；
2. 加载：每 rank `safe_open(shard).get_slice(name)[start:end]` —— **一次性物化自己行段的 24.6 GiB 临时张量**
   （safetensors 普通 buffered read）→ `copy_` 进 pinned 表 → 抽检 → `requires_grad_(False)` 冻结；
3. **两表串行**；rank 间并行（同节点 8 rank 各读不同行段 = 节点内合计全表一份）；
4. 收尾：layout `all_gather_object`（元数据级校验，廉价）→ 打印 "loaded ... in Xs (frozen)"。
- 旧路径（rank0 全量 393GB + DCP 广播 / 逐文件 fallback）已被 `VERL_DSV41_DIRECT_SHARD_LOAD=1` 替代（fallback 保留）。

### 1.3 引擎侧路径：`EngramEmbedding.load_checkpoint`
（`vllm-ascend-v41-private/vllm_ascend/models/deepseek_v41/engram/embedding.py:215`）
- EDP/EP 行分片（8 rank × 48M 行）；`get_slice` + **`chunk_rows=65536`（≈32 MB/块）的串行 chunk 循环**
  read→copy；共享 slice 由 group rank0 单写 + CPU all_gather_object 同步。
- 引擎在**同一节点再读一遍全表**（trainer 与 engine 各持有 384 GB host 常驻）。

### 1.4 冗余账（每次双机启动）
| 读者 | 每节点读取量 | 来源 |
|---|---|---|
| trainer（8 rank 合计） | ~370 GiB | NFS（同文件） |
| engine（8 rank 合计） | ~370 GiB | NFS（同文件） |
| **每节点合计** | **~740 GiB** | 两节点 ≈ 1.5 TiB/次启动 |

---

## 2. 为什么这么久（量化）

**131 实测（同一 NFS：172.27.1.1:/Share，nconnect=8，O_DIRECT 64M bs）**：

| 路径 | 带宽 | 备注 |
|---|---|---|
| NFS **buffered read**（safetensors/torch 实际走的路） | **~442 MB/s** | 9.7s / 4 GiB |
| NFS **O_DIRECT read** | **~3.7 GB/s** | 8.4× 于 buffered |
| 本地盘 write（direct） | ~7.3 GB/s | overlay 12T（123 剩 8.6T） |
| 本地盘 read（direct） | ~11.9 GB/s | |

**结论**：
- 740 GiB/节点 ÷ 0.44 GB/s ≈ **28 分钟/节点**（与历史"加载 20–45 分钟"、今晚 05:05→05:20+ 的 engram 段吻合；
  两节点并发 + 邻居读 NFS 会再恶化）；
- **瓶颈 = NFS 普通读路径带宽**（非 CPU、非 copy、非文件布局——大文件各自连续、读法已是顺序行段）；
- 数据量本身不可压缩（bf16 稠密表；量化会改动两栈数值一致性，禁用）。
- 待复核项：buffered 442MB/s 是否被 131 的 955GB 页缓存压力/readahead 参数拖累（P0 的第一步就是复测这一点，
  127/123 恢复后在目标机上重测）。

---

## 3. 方案空间（对比）

| # | 方案 | 机制 | 预期收益 | 成本/风险 | 改动面 |
|---|---|---|---|---|---|
| **P0** | fadvise 预热 / 并行读 / 栈间错峰 | (a) `posix_fadvise(WILLNEED)` 或按 O_DIRECT 速率复制式预取入页缓存；(b) 训练两表并行；(c) trainer 与 engine 错峰 | 若 buffered 提到 ~1.5GB/s：2–3×（~28→10min） | 低 | 训练侧 1 处 + 引擎侧可选 |
| **P1★** | **节点本地 checkpoint 缓存（主方案）** | 2 个 183GiB 大文件（+3 个 engram 小文件）一次性**用 O_DIRECT 拷贝**到本地盘（370GiB ≈ 100s @3.7GB/s）；`MODEL_PATH` 指向本地目录，其余文件/索引 **symlink 回 NFS**；两栈零代码改动 | **740GiB 从 0.44GB/s → ~10GB/s：28min → ~2–3min（≈10–15×）** | 中：磁盘 370GiB/节点、manifest 校验、缓存的建/校/失效流程 | 零代码（脚本+路径） |
| P2 | 栈间共享单份（shm/memfd 只读映射） | 一读两栈用；节点内只读共享同一份 384GB | 再省 ~1× NFS + 省 384GB host RAM/节点 | 高（两栈 loader 改造） | 中–大 |
| P3 | 读循环并行化（引擎 chunk / 训练 rank 内线程） | 线程池并发 ranged reads | NFS 服务端瓶颈时有限 | 低–中 | 两处代码 |

### P1 目录结构设计（零代码接入的关键）
```
/mnt/share/.../deepseek-v41-sft116  （源，NFS）
本地: /data/ckpt_cache/sft116/  (本地盘)
├── model-00008-of-00268.safetensors   ← 真身（O_DIRECT 拷贝）
├── model-00151-of-00268.safetensors   ← 真身
├── model-sft-extra-*-00009/00152/00267*.safetensors  ← 真身（小 engram 张量）
├── model.safetensors.index.json       ← symlink → NFS（名字不变，索引无需改）
└── 其余 263 个权重文件                 ← symlink → NFS（params 加载速度已 78.6s 可接受）
```
- **manifest**：每个本地真身记录 {name, size, mtime_ns, md5(首/中/尾 各 64MB)}；启动前 10s 级校验；
  源变更（size/mtime 变）→ 标记失效并可后台重刷。
- **回退**：本地缓存缺失/校验失败/磁盘水位不足 → 自动回退直接读 NFS（现行为），无硬依赖。
- **两栈共用**：trainer 与 vLLM engine 都吃 `MODEL_PATH`/同目录索引 → 一份缓存同时服务两栈（本方案最大杠杆）。

---

## 4. 推荐路线（分阶段）

1. **P0 验证（0.5 天）**：在 127/123（恢复后）复测 buffered vs O_DIRECT（用真实 safetensors 读法 + dd 两种口径），
   并试 `fadvise(WILLNEED)` 预取；确认"442MB/s 是 NFS 路径属性"而非 131 特有现象。顺带量训练"两表并行"的收益。
2. **P1 主方案落地（1–2 天）**：
   - 写 `scripts/dsv41_deploy/engram_cache.sh`（build/verify/refresh 三动作，幂等，O_DIRECT 拷贝 + manifest）；
   - 在 127/123 建缓存，`MODEL_PATH` 切本地目录（两栈同切）；
   - **对照验收**：冷缓存 vs 热缓存各起一次（同配置、只到 step 1），对比
     ①engram 段墙钟 ②init→step1 总时长 ③NFS 读流量 ④一致性指标不回归（diff_valid/pearson/diff_mean 同水平）。
3. **P2/P3 备选（视 P1 后残值）**：若 init 其余段（引擎启动、权重同步首轮）成为新瓶颈再评估。

**预期总账**：双机 init ≈54min（今晚实测）中 engram 段 ~20–40min → P1 后 ~2–3min；
净收益 **~20–35 分钟/次启动**，且对 NFS 邻居争用免疫。

---

## 5. 风险与验证指标

| 风险 | 缓解 |
|---|---|
| 缓存副本与源不一致 | manifest（size+mtime+抽样 md5）启动前强制校验；不一致回退 NFS |
| 磁盘水位（370GiB/节点） | 建缓存前检查 ≥500GiB 余量（123: 8.6T ✓）；不足则禁用（回退） |
| 多机缓存漂移 | 每机幂等脚本 + 校验；重启作业不依赖缓存存在性 |
| 一致性回归 | 对照跑用同一 checkpoint；engram 表内容抽检对拍（trainer vs engine 同 hash id 同值已有 layout 校验） |
| 改动影响面 | P1 零代码（只换路径）；P0/P3 的代码改动全部 env 门控、默认关 |

**验证指标**：engram 段墙钟 / init→step1 总时长 / NFS 吞吐（dstat）/ 两栈一致性四指标（NaN 已修，应平稳）。
