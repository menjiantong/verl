#!/usr/bin/env bash
# [1120 9.3 镜像适配] 相对已验证 env_dsv41_yundao.sh 的五处增量：
#   1) python 路径 3.11.15 -> 3.11.18（新镜像实测路径）
#   2) source CANN set_env.sh：新镜像非登录 shell 不自动带 libhccl.so 等
#      （2026-11-20 13:00 基线 smoke 报 ImportError: libhccl.so，证据见 worklog §14）
#   3) export RAY_ADDRESS=local：ray.init(address=None) 会读
#      $RAY_TMPDIR/ray/ray_current_cluster；该文件残留 2026-11-17 的死集群地址
#      7.150.15.242:6390（实测无 raylet/gcs 进程、6390 未监听），
#      13:01 smoke 因此卡在 "Connecting to existing Ray cluster" + GCS TimedOut。
#      单机强制起本地集群（证据见 worklog §15）。
#   4) 预先算好 NIC：env_dsv41_yundao.sh 用 python3+psutil 探测业务网卡，但它在 PATH
#      前置的 /usr/local/python3.11.15 在 9.3 镜像不存在 → python3 落到 /usr/bin/python3
#      （无 psutil）→ 回退硬编码 enp48s3u1u1（80.5.25.x 网卡名）→ Gloo 报
#      "Unable to find address for: enp48s3u1u1"（13:11 基线 smoke，worklog §16）。
#      这里用 3.11.18 先探测并 export，基础脚本的 ${NIC:-...} 即不再回退。
#   5) NNAL ATB 库路径：vllm_ascend worker 初始化要加载 libatb.so，9.3 镜像非登录
#      shell 未带其路径 → 13:25 smoke 报 "libatb.so: cannot open shared object file"
#      （worklog §17）。按 /usr/local/Ascend/nnal/atb/set_env.sh 的规则显式设置
#      cxx_abi=1（torch 2.10.0 wheel 实测 compiled_with_cxx11_abi=True）。
# 其余全部沿用 env_dsv41_yundao.sh
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export MY_IP=${MY_IP:-$(hostname -I | tr ' ' '\n' | grep -m1 '^7\.' || true)}
export NIC=${NIC:-$(/usr/local/python3.11.18/bin/python3 -c 'import os,psutil,socket; ip=os.environ["MY_IP"]; print(next(i for i,a in psutil.net_if_addrs().items() for x in a if x.family==socket.AF_INET and x.address==ip))' 2>/dev/null || true)}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "$here/env_dsv41_yundao.sh"
export PATH=/usr/local/python3.11.18/bin:$PATH
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=$ATB_HOME_PATH/lib:$LD_LIBRARY_PATH
export RAY_ADDRESS=local
