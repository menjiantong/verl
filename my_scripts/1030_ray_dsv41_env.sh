#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1 RL 环境变量 + Ray 启停（1030 rebase 基线，128 单机默认）
#
# 用法：
#   source my_scripts/1030_ray_dsv41_env.sh               # 只导出环境变量
#   bash my_scripts/1030_ray_dsv41_env.sh start           # 单机 head
#   bash my_scripts/1030_ray_dsv41_env.sh join <head_ip>  # 多机从节点加入
#
# 设计：环境变量全部集中在本文件；训练配置全部在
#   examples/grpo_trainer/run_deepseek_v41_top64_grpo_fsdp_turbo_npu.sh
# 避免控制台 export 和多层脚本嵌套。
# =============================================================================
# 注意：不要在此脚本启用 `set -u`。/usr/local/Ascend/nnal/atb/set_env.sh 的
# get_cxx_abi_option 会读取未定义的 $1，nounset 下会让整个 shell 直接退出。

# ---------- 仓库路径 ----------
export MIRROR_ROOT=/mnt/share/m00899630/workspace/dsv41-verl-rebase

# ---------- 昇腾 CANN 环境 ----------
# ssh 非登录 shell 必须显式 source；nnal/atb 提供 libatb.so，缺它 vLLM worker 报
# "OSError: libatb.so: cannot open shared object file"。
for _env in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann-9.1.0/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$_env" ] && source "$_env" >/dev/null 2>&1
done
export PATH=/usr/local/python3.11.15/bin:$PATH
# vllm/vllm-ascend/FSDPTurbo/verl 全部用 rebase 仓库内 checkout，且必须排在最前，
# 避免命中镜像里的同名包。
export PYTHONPATH="$MIRROR_ROOT/vllm:$MIRROR_ROOT/vllm-ascend:$MIRROR_ROOT/FSDPTurbo:$MIRROR_ROOT/verl:${PYTHONPATH:-}"
# ssh 非交互 shell 默认 ulimit -n=1024，Ray gRPC 会 "Too many open files"。
ulimit -n 1048576 2>/dev/null || ulimit -n 655350 2>/dev/null || true

# ---------- 网络 ----------
# hostname -I 首地址可能是 172.x（容器网），必须显式取 80.5.x 业务网。
# [1031 换容器] 原逻辑只匹配 80.5.25.*；新容器为 80.5.27.164，放宽到 80.5.*。
export MY_IP=${MY_IP:-$(hostname -I | tr ' ' '\n' | grep -m1 '^80\.5\.')}
# 由 MY_IP 反查业务网卡名（128 机为 enp48s3u1u1）；可用 NIC 覆盖。
export NIC=${NIC:-$(python3 -c 'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(i for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address==ip))' 2>/dev/null || echo enp48s3u1u1)}
export GLOO_SOCKET_IFNAME=$NIC HCCL_SOCKET_IFNAME=$NIC HCCL_IF_IP=$MY_IP
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=3600 HCCL_ASYNC_ERROR_HANDLING=0
# HCCL 通信端口段；与 vllm_async_server 固定的 60200-60350/61200-61350 不重叠。
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60150 HCCL_NPU_SOCKET_PORT_RANGE=61000-61150

# ---------- Ray ----------
export RAY_PORT=${RAY_PORT:-6390}      # 避开外部 GCS 默认 6379
export RAY_AUTH_MODE=disabled          # 内网集群，关闭 Ray token 认证
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=1000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ---------- vLLM / Ascend 引擎 ----------
export VLLM_VERSION=0.30.0             # vllm-ascend 用该变量选择 0.30 API 分支
export HCCL_BUFFSIZE=1024              # EP MoE mc2 通信窗口，过小引擎 warmup 失败
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1

# ---------- verl 训练/显存策略（只保留新仓库仍实现的开关） ----------
# 本驱动在共享 NUMA 节点上超大 pinned 分配会失败(aclrtMallocHostWithCfg 207001)，
# offload 改用 pageable 同步拷贝，语义不变。
export VERL_DSV41_OFFLOAD_PIN_MEMORY=0
export VERL_DSV41_DIRECT_SHARD_LOAD=1    # 每个 rank 直接读自己的 checkpoint 分片
export VERL_DSV41_LOCAL_EXPERT_EXPORT=1  # weight sync 只导出本 rank 专家
export VERL_DSV41_VISION=1               # countbenchqa 带图；同时冻结 vision tower
# [1031 A/B 已验证] 视觉塔是 train/infer 漂移起点：训练 SDPA 与推理 FIA 单算子
# 差 0.244%，训练融合 FlashAttentionScore 只差 0.158%；164 实测图像特征误差
# 1.589%→1.464%，最终 hidden 7.51%→6.18%，默认启用融合视觉注意力。
export MY_ENV_TRAIN_FUSED_VISION_ATTN=${MY_ENV_TRAIN_FUSED_VISION_ATTN:-1}
export VERL_DSV41_EMPTY_CACHE_AFTER_BUILD=1
export VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1
export VERL_DSV41_OPT_STATE_DEBUG=1      # 预建 Adam 状态，规避首步惰性建状态长卡顿
export VERL_DSV41_OPT_STATE_DEBUG_DIR=$MIRROR_ROOT/verl/logs/opt_state_debug
export FSDP_TURBO_ATTN_RECOMPUTE=1 FSDP_TURBO_ATTN_CHUNK=64

# ---------- Ray 启停 ----------
dsv41_ray_start() {
  # 幂等：本机已有可用 Ray 集群时直接复用，避免重启训练时 `ray start --head` 报错。
  if ray status >/dev/null 2>&1; then
    echo "[ray] already running on $MY_IP:$RAY_PORT"
    ray status
    return 0
  fi
  echo "[ray] head ip=$MY_IP nic=$NIC port=$RAY_PORT"
  ray start --head --port="$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  ray status
}

dsv41_ray_join() {
  local head_ip=${1:?usage: $0 join <head_ip>}
  echo "[ray] join $head_ip:$RAY_PORT ip=$MY_IP nic=$NIC"
  ray start --address="$head_ip:$RAY_PORT" --node-ip-address="$MY_IP" --disable-usage-stats
  ray status --address="$head_ip:$RAY_PORT" | tail -8
}

# 直接执行时按参数启停；被 source 时只导出环境，由调用方决定何时启动。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-start}" in
    start) dsv41_ray_start ;;
    join)  dsv41_ray_join "${2:-}" ;;
    *) echo "usage: $0 [start|join <head_ip>]" >&2; exit 2 ;;
  esac
fi
