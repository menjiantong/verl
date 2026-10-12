# `wiki/learn/` —— 代码学习笔记(与工作记录分开)

这个文件夹放**"代码是怎么工作的"**类文档:接口契约、调用链、内部机制、跨后端对比。
它和 `wiki/` 根目录下的 `worklog_*.md` 是两类东西,刻意分开:

| | 位置 | 回答的问题 | 写法 |
|---|---|---|---|
| **学习笔记** | `wiki/learn/` | "这段代码为什么这么写?输入输出是什么?别的实现差在哪?" | 以**代码**为纲,行号为准;结论应是可复现的读码结果,不随时间失效 |
| **工作记录** | `wiki/worklog_*.md` | "那天我们做了什么?踩了什么坑?为什么改?改完什么结果?" | 以**时间**为纲,含命令/日志/失败;结论会随实验推进被修订(用删除线+指向修正节) |

两者互相引用:学习笔记在末尾指向相关 worklog,worklog 在需要解释机制时指向学习笔记。
学习笔记里的行号会随上游代码变动而漂移 —— 引用时**一律带 `文件:行号` 并写下当时的代码片段**,
便于日后核对(行号漂了但片段还在)。

## 索引

| # | 文档 | 主题 | 相关 worklog |
|---|---|---|---|
| 01 | [`01_weight_sync_get_per_tensor_param.md`](01_weight_sync_get_per_tensor_param.md) | `BaseEngine.get_per_tensor_param` 及其姊妹方法:调用链、I/O 契约、五个后端的物化策略对比(重点 `FSDPEngine`) | [`../worklog_dsv41_weight_sync_export.md`](../worklog_dsv41_weight_sync_export.md)(export 段的实测与快慢路径) |
| 03 | [`03_verl_weight_load_offload_memory.md`](03_verl_weight_load_offload_memory.md) | 训练引擎(fsdp_turbo_dsv41)的**权重加载/offload/显存账**:初始化调用链、三条装载路径(直载分片/rank0+广播/resume)及实测字节与耗时、offload 三个开关的真实语义(FSDP2 CPUOffloadPolicy vs 手动逐参搬运 vs engram host 表)、逐阶段显存公式与全配置实测对照、8 条实测坑 | [`../dsv41_memory_and_parallel_archive.md`](../dsv41_memory_and_parallel_archive.md)(逐阶段数字全表) · [`../worklog_dsv41_sft.md`](../worklog_dsv41_sft.md)(SFT 显存战时间线) |
| 02 | [`02_dsv41_indexer_scatter_marker_e9.md`](02_dsv41_indexer_scatter_marker_e9.md) | V4.1 索引器写缓存的整条链(`forward`→`update_keys`→`scatter_cache_sk`→`npu_scatter_nd_update_sk` 的 aclnn/tiling/kernel),以及 `[-1,-1]` marker 为什么被当成真实坐标(E9 的根因、地址算术、复现工具与"改了源码没改二进制"的现状) | [`../worklog_dsv41_top64_grpo.md`](../worklog_dsv41_top64_grpo.md) §5.7.8(E9 的时间线) |

> 新增一篇就往这张表里加一行。命名:`NN_topic.md`,`NN` 只表示阅读顺序,不表示依赖。