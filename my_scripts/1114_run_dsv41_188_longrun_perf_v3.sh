#!/usr/bin/env bash
# [1114] DSV4.1 单机 16 卡长跑（自包含版，不再 source 1030_ray_dsv41_env.sh）
# 基线 1110_run_dsv41_188_longrun_v1.sh；已验证优化：MODEL_INIT=zero / ENGRAM_INIT=zero
# （只省启动时的 CPU 随机初始化，结果随后被 checkpoint 整片覆盖，计算逻辑不变）。
# 直接调用训练配置入口 examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh。
# 日志：verl/logs/DeepSeek-V4.1-Flash-top32-bf16-<YYYYmmdd_HHMMSS>.log
# 成功日志：20261114_092300（147，26.2.rc1，1 步 334.7 s，0 OOM）
# commit：verl fc14cd02（v3 自包含版）+ 本修复提交
#         FSDPTurbo 1852e93 / vllm-ascend b0901f79 / vllm fd46ca5a
# 分支：dev_dsv41_rebase1030_cp_engram / dev_dsv41_1027 / from_9a38d36_1030 / from_a98247a_1030
# 注意：26.2.rc1（80.5.27.x）rollout 数值 NaN，长跑必须用 80.5.25.x。
set -eo pipefail
R=/mnt/share/m00899630/workspace/dsv41-verl-rebase
cd "$R/verl"

# ================= 模型与数据 =================
export MODEL_PATH=/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16   # top32-bf16 权重目录
export TRAIN_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_train_512.parquet # 训练集 parquet
export TEST_FILE=/mnt/share/m00899630/dsv41/rl_data/dapo_math_val_64.parquet     # 验证集 parquet

# ================= 并行 / CP =================
export NDEVICES_PER_NODE=16 NNODES=1        # 单机 16 卡
export FSDP_SIZE=16 EP_SIZE=16              # 训练 FSDP16 / EP16
export GEN_TP=8 GEN_DP=2 GEN_EP=16          # 推理 TP8×DP2 / GEN_EP16
export CP_SIZE=4                            # 训练上下文并行 4
export MY_ENV_DSV41_CP_PAD_MULTIPLE=8       # CP 对齐倍数：padded len%4==0 且 /4%2==0（CP=4 必须 8）
export MY_ENV_DSV41_CP_KEEP_TRAIN_MODE=1    # CP 时保持训练模式，不切推理模式：0 | 1
export MY_ENV_DSV41_CP_GATHER_OUTPUT=1      # CP 输出 gather 回全序列再算 loss：0 | 1

# ================= engram / KL / sleep =================
export USE_KL=1                             # 启用 ref 模型与 KL 项：0 | 1
export MY_ENV_DSV41_ENGRAM_SHARE_MODEL=1    # actor/ref 共用一份 frozen engram 表，省 366 GiB：0 | 1
export VLLM_ENGRAM_DP_SHARED_MEMORY=1       # 推理 DP2 共享 TP 分片表（否则 2×366 GiB 撑爆主机内存）：0 | 1
export MY_ENV_VLLM_SLEEP_PIN_MEMORY=1       # sleep 备份用 pinned 内存：0=pageable（曾在 camem.unmap 卡死）
export FREE_CACHE_ENGINE=False              # rollout 后不销毁 KV cache：False | True

# ================= 自实现开关（默认值  # 作用：可选枚举） =================
# 注：上文 CP / engram / vision 小节里的 MY_ENV_* 也是自实现开关；此处列其余开关。
export MY_ENV_DSV41_MODEL_INIT=zero          # 参数初始化填充：normal=RNG | zero=清零（本脚本用 zero）
export MY_ENV_DSV41_ENGRAM_INIT=zero         # engram 主机表填充：normal | zero | skip（skip 有 NaN 风险，仅实验）
export MY_ENV_DSV41_DISABLE_THP=1            # 训练/vLLM 进程关闭 THP：0 | 1（thpoff 档）
export MY_ENV_VLLM_VIT_ATTN_OP=fa            # 推理视觉注意力算子：fa=与训练同算子 | fia=vllm-ascend 默认
export MY_ENV_TRAIN_FUSED_VISION_ATTN=1      # 训练视觉注意力融合：0 | 1
export MY_ENV_DSV41_STEPS=${MY_ENV_DSV41_STEPS:-200}  # 训练步数（hydra total_training_steps）
# —— 已验证但长跑不用（会留 132 s 尾）或仅诊断，默认注释 ——
# export MY_ENV_DSV41_ENGRAM_SPREAD=1        # engram 表按 rank 钉 NUMA 做 first-touch：0 | 1（spread 档）
# export MY_ENV_DSV41_PIN_PROCESS=1          # 整个 worker 进程钉 NUMA：0 | 1（pinproc 档）
# export MY_ENV_DSV41_NUMA_NODES=0,2,4,5,6,7 # SPREAD/PIN_PROCESS 使用的 NUMA 节点表
# export MY_ENV_DSV41_NUMA_RANK_MAP=0,2,3,4,5,6,7,0,2,3,4,5,6,7,6,7  # 训练 16 rank 容量加权落点
# export MY_ENV_DSV41_NUMA_INTERLEAVE=1      # 优先 set_mempolicy(MPOL_INTERLEAVE)：0 | 1
# export MY_ENV_DSV41_ENGRAM_FETCH_PROBE=1   # 打印 engram fetch 分段耗时：0 | 1（诊断）
# export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_CALLS=64  # 只采样前 N 次调用
# export MY_ENV_DSV41_ENGRAM_FETCH_PROBE_SYNC=1    # 计时前做 device 同步（会改变时序）
# export MY_ENV_DSV41_ENGRAM_HOST_BUF_MAX_MB=256   # engram host buffer 上限（MB）
# export MY_ENV_DSV41_ENGRAM_PIN_MEMORY=0    # engram host buffer 使用 pinned：0 | 1
# export MY_ENV_DSV41_ENGRAM_REUSE_HOST_BUF=1      # 复用 host buffer：0 | 1
# export MY_ENV_DSV41_ENGRAM_REUSE_RECOMPUTE=1     # 重算复用：0 | 1
# export MY_ENV_DSV41_SLEEP_LEVEL=0          # vLLM sleep 级别：0 | 1 | 2（2 已否决：丢 CaMem 映射→507057）
# export MY_ENV_DSV41_GATE_BIAS_DEBUG=1      # 记录 gate bias 的 dtype/hash：0 | 1（诊断）
# export MY_ENV_DSV41_OPT_STATE_PROBE=1      # 优化器状态创建探针：0 | 1
# export MY_ENV_DSV41_OPT_STATE_PROBE_DIR=/tmp/opt_state_probe  # 探针输出目录
# export MY_ENV_DSV41_OPT_STATE_DUMP_SEC=30  # 探针心跳间隔（秒）
# export MY_ENV_ENGINE_HCCL_HOST_RANGE=60200-60350  # vLLM 引擎 HCCL 主机端口段
# export MY_ENV_ENGINE_HCCL_NPU_RANGE=61200-61350   # vLLM 引擎 HCCL NPU 端口段
# export MY_ENV_TRAIN_SPARSE_FLASH_ATTN=1    # 训练稀疏 flash attention：0 | 1
# export MY_ENV_TRAIN_TRITON_INPLACE_ROPE=1  # Triton inplace RoPE：0 | 1
# export MY_ENV_TRAIN_FUSED_CLIPPED_SWIGLU=1 # 融合算子（COMPRESSOR/LIGHTNING_INDEXER/MHC 同理）：0 | 1
# —— 已回退、当前代码中不存在（留档勿设）：KV_ZERO_GAP_AFTER_WRITE / ZERO_KV_STORAGE /
#    HEAD_DTYPE / DEBUG_WEIGHT_HASH / PERF；KV_CACHE_BYTES 仅旧 v2 包装脚本使用，本脚本沿用 gmu=0.6。

# ================= 环境：CANN / Python =================
# ssh 非登录 shell 必须显式 source CANN；nnal/atb 提供 libatb.so，缺失时 vLLM worker 报错
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  if [ -f "$_env" ]; then source "$_env" >/dev/null 2>&1; fi
done
export PATH=/usr/local/python3.11.15/bin:$PATH   # CANN 配套 Python 3.11.15，优先于系统 python
# rebase 仓库必须排在 PYTHONPATH 最前，避免命中镜像里的同名包
export MY_ENV_COMPAT_ROOT=/mnt/share/m00899630/envcompat   # sitecustomize：TMPDIR 清理 ENOTEMPTY 降级
export PYTHONPATH="$MY_ENV_COMPAT_ROOT:$R/vllm:$R/vllm-ascend:$R/FSDPTurbo:$R/verl:${PYTHONPATH:-}"
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true   # Ray gRPC 需要大 fd 上限
# transformers 5.10.4（80.5.25.x 镜像）缺 qwen4_exp；FSDPTurbo models/__init__.py 会无条件导入
# qwen/modeling_qwen4_exp.py，后者需要 transformers.models.qwen4_exp，缺失时建模型直接 ModuleNotFoundError
# （124 实测）。/mnt/share/m00899630/env/py521 是 transformers 5.21.0 全量副本，仅在本机缺失 qwen4_exp 时前置；
# 80.5.27.x 自带 5.21.0，条件为假、不覆盖。
TF_DIR=$(python3 -c 'import os, transformers; print(os.path.dirname(transformers.__file__))' 2>/dev/null || true)
if [ -n "$TF_DIR" ] && [ ! -d "$TF_DIR/models/qwen4_exp" ]; then
  export PYTHONPATH="/mnt/share/m00899630/env/py521:${PYTHONPATH:-}"
fi

# ================= 缓存 / 临时目录（根盘满，全部落共享盘） =================
export MY_ENV_CACHE_ROOT=${MY_ENV_CACHE_ROOT:-/mnt/share/m00899630/cache}       # 缓存根目录
export HF_HOME=${HF_HOME:-$MY_ENV_CACHE_ROOT/huggingface}                       # HF 主目录
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}                # datasets 缓存
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}                               # HF hub 缓存
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}          # transformers 缓存
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$MY_ENV_CACHE_ROOT/torch_extensions}  # torch 扩展编译缓存
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$MY_ENV_CACHE_ROOT/xdg}                 # XDG 通用缓存
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$MY_ENV_CACHE_ROOT/triton}          # Triton 缓存（需可执行/dlopen）
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$MY_ENV_CACHE_ROOT/torchinductor}  # inductor 缓存
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$MY_ENV_CACHE_ROOT/vllm}              # vLLM 编译缓存
export MY_ENV_LOCAL_TMP=/dev/shm/m00899630                                      # 本机 tmp 根（noexec）
export TMPDIR=${TMPDIR:-/mnt/share/m00899630/tmp}                               # 通用临时目录放 NFS
export RAY_TMPDIR=${RAY_TMPDIR:-$MY_ENV_LOCAL_TMP/ray_tmp}                      # Ray 会话目录放 /dev/shm
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$RAY_TMPDIR" "$TORCH_EXTENSIONS_DIR" \
         "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

# ================= 网络（取 80.5.x 业务网，不用 172.x 容器网） =================
export MY_IP=${MY_IP:-$(hostname -I | tr ' ' '\n' | grep -m1 '^80\.5\.' || true)}  # 本机业务网 IP
export NIC=${NIC:-$(python3 -c 'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(i for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address==ip))' 2>/dev/null || echo enp48s3u1u1)}  # MY_IP 对应网卡
export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP   # Gloo/HCCL 都走业务网卡
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0  # 建链 1500s/执行 3600s，长跑不因异步错误退出
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60150 HCCL_NPU_SOCKET_PORT_RANGE=61000-61150  # HCCL 端口段，与引擎段不重叠

# ================= Ray =================
export RAY_PORT=${RAY_PORT:-6390}           # head 端口，避开默认 6379
export RAY_AUTH_MODE=disabled               # 内网集群，关闭 token 认证
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=1000  # 放宽 OOM 阈值/1 s 刷新
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  # 不让 Ray 改写可见卡列表
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}  # 暴露 16 卡

# ================= vLLM / Ascend 引擎 =================
export VLLM_VERSION=0.30.0                  # vllm-ascend 用该变量选择 0.30 API 分支
export HCCL_BUFFSIZE=1024                   # EP MoE mc2 通信窗口（过小引擎 warmup 失败）
export VLLM_USE_V2_MODEL_RUNNER=0           # 用旧版 model runner（V2 在 DSV41 未验证）
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000  # 引擎就绪/单次执行超时（秒）
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000               # 设备建链/传输超时（ms）
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1               # 线程绑定交给 runtime；开启算子队列下发

# ================= verl / FSDPTurbo 策略 =================
export VERL_DSV41_OFFLOAD_PIN_MEMORY=0      # 本驱动大 pinned 分配失败(207001)，offload 改用 pageable
export VERL_DSV41_DIRECT_SHARD_LOAD=1       # 每个 rank 直接读自己的 checkpoint 分片
export VERL_DSV41_LOCAL_EXPERT_EXPORT=1     # weight sync 只导出本 rank 专家
export VERL_DSV41_VISION=1                  # countbenchqa 带图，冻结 vision tower
export VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=1 VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1  # 建模型/优化器前清设备缓存
export VERL_DSV41_OPT_STATE_DEBUG=1         # 预建 Adam 状态，规避首步惰性建状态长卡顿
export VERL_DSV41_OPT_STATE_DEBUG_DIR=$R/verl/logs/opt_state_debug   # 优化器状态探针日志目录
export FSDP_TURBO_ATTN_RECOMPUTE=1 FSDP_TURBO_ATTN_CHUNK=64   # 注意力整层重算 + chunk 64

# ================= 启动 =================
ray stop --force >/dev/null 2>&1 || true    # 先停旧 Ray，保证 actor 继承本次环境
ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
bash examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh \
  trainer.total_training_steps="${MY_ENV_DSV41_STEPS}" \
  '++actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]' \
  "$@"
