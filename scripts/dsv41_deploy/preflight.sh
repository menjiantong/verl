#!/usr/bin/env bash
# 单机环境体检（部署验收 / 每次启动前各节点跑一遍）。
#     bash preflight.sh [MODEL_PATH]
# 退出码 0 = 全部通过；输出里出现 FAIL 行即有问题（对照失败库编号排查）。
# 注意：npu-smi 必须走 login shell（F-50），本脚本内部已包 bash -lc。
set -u
MODEL_PATH=${1:-/mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16}
fail=0
ok(){  echo "  OK   $*"; }
bad(){ echo "  FAIL $*"; fail=1; }
warn(){ echo "  WARN $*"; }

echo "[preflight] $(hostname -I 2>/dev/null | awk '{print $1}') $(date '+%F %T')"

# 0) fd 软限（F-64）
n=$(ulimit -n)
[ "$n" -ge 65536 ] 2>/dev/null && ok "ulimit -n=$n" || warn "ulimit -n=$n（<65536；启动器会自行提升，但手工起进程时注意 F-64）"

# 1) python 与代码树
export PATH=/usr/local/python3.11.15/bin:$PATH
# CANN 环境自举（F-51/F-52：ssh 非 login shell 下缺 LD_LIBRARY_PATH —— import/dlopen/npu-smi 会全挂）
if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  for f in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/cann-9.1.0/set_env.sh; do
    [ -f "$f" ] && source "$f" >/dev/null 2>&1 || true
  done
fi
python3 -c 'import sys; assert sys.version_info[:2]==(3,11)' 2>/dev/null && ok "python3.11 ($(python3 -V 2>&1))" || bad "python3.11 不可用（PATH？）"

export PYTHONPATH="/workspace-verl/vllm:/workspace-verl/vllm-ascend-v41-private:/workspace-verl/FSDPTurbo:${PYTHONPATH:-}"
python3 -c "import torch_npu, vllm_ascend" >/dev/null 2>&1 && ok "import torch_npu + vllm_ascend" \
  || bad "torch_npu/vllm_ascend import 失败（F-51/F-52：需 login shell / PYTHONPATH 前缀被覆盖）"
V=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1)   # vllm import 会往 stdout 打 INFO，取最后一行
python3 -c "from packaging import version; version.parse('$V')" 2>/dev/null && ok "vllm.__version__='$V' 可解析" \
  || bad "vllm 版本解析失败（F-01：vllm/vllm/_version.py 缺失会 fallback 'dev'；实得 V='$V'）"

# 2) 不入 git 的生成物 / .so（F-02/F-03，新机最常漏）
SO=/workspace-verl/vllm-ascend-v41-private/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
SO_OK=0
if [ -f "$SO" ]; then
  # 真实 dlopen 判定（ldd 在缺 LD_LIBRARY_PATH 时会对 libtorch/libpython 等报假 "not found"）
  python3 -c "import torch_npu, ctypes; ctypes.CDLL('$SO')" >/dev/null 2>&1 && { ok "vllm_ascend_C.so dlopen 成功"; SO_OK=1; } \
    || { bad "vllm_ascend_C.so dlopen 失败（F-02）"; ldd "$SO" 2>/dev/null | grep "not found" | sed 's/^/       /'; }
else bad "vllm_ascend_C.so 不存在（整树未同步？）"; fi
K1=/workspace-verl/vllm-ascend-v41-private/vllm_ascend/libvllm_ascend_kernels.so
K2=/workspace-verl/vllm-ascend-v41-private/vllm_ascend/lib/libvllm_ascend_kernels.so
KLIB=""; [ -f "$K1" ] && KLIB=$K1; [ -z "$KLIB" ] && [ -f "$K2" ] && KLIB=$K2
if [ -n "$KLIB" ]; then
  python3 -c "import torch_npu, ctypes; ctypes.CDLL('$KLIB')" >/dev/null 2>&1 \
    && ok "libvllm_ascend_kernels.so 在位且可加载（顶层:$([ -f "$K1" ] && echo Y || echo N) lib/:$([ -f "$K2" ] && echo Y || echo N)）" \
    || bad "libvllm_ascend_kernels.so 存在但 dlopen 失败（F-02）"
else bad "libvllm_ascend_kernels.so 缺失（F-02：至少顶层或 lib/ 一份）"; fi
cnt=$(find /workspace-verl/vllm-ascend-v41-private/vllm_ascend/_cann_ops_custom -type f 2>/dev/null | wc -l)
[ "${cnt:-0}" -ge 1000 ] && ok "_cann_ops_custom ($cnt 文件)" || bad "_cann_ops_custom 文件数=$cnt（F-03：应为 ~1304，含 libcust_opapi.so）"

# 3) 训练引擎关键修复在位（F-04）
grep -q "copies_per_range" /workspace-verl/verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py 2>/dev/null \
  && ok "engram 复本感知校验在位（F-04）" || bad "缺 copies_per_range 修复（F-04，两台都要有）"

# 4) /mnt/share 数据面
ls "$MODEL_PATH/config.json" >/dev/null 2>&1 && ok "MODEL_PATH 可达: $MODEL_PATH" || bad "MODEL_PATH 不可达（/mnt/share 未挂载？）"
ls /mnt/share/m00899630/dsv41/rl_tokenizer >/dev/null 2>&1 && ok "tokenizer 可达" || bad "rl_tokenizer 不可达"

# 5) NPU 空闲（单次 npu-smi + HBM<4GB ≈ 空闲口径；精确 per-die 判定在启动器 pick_devices 里做）
#    须 login shell（F-50：非 login shell 下 npu-smi 静默失败）。
txt=$(bash -lc "npu-smi info" 2>/dev/null)
total=$(printf '%s\n' "$txt" | grep -cE '[0-9]+ */ *65536')
nfree=$(printf '%s\n' "$txt" | grep -oE '[0-9]+ */ *65536' | awk -F'/' '{gsub(/ /,"",$1); if ($1+0<4000) c+=1} END{print c+0}')
if [ "${total:-0}" -gt 0 ]; then
  if [ "$nfree" -ge 8 ]; then ok "NPU 空闲 die: $nfree/$total（HBM<4GB 口径）"
  elif [ "$nfree" -gt 0 ]; then warn "NPU 空闲 die: $nfree/$total（<8，启动器会报 only N free chips）"
  else bad "NPU 全忙（F-62 邻居？）"; fi
else warn "未见 NPU（不在容器内 / npu-smi 失败？）"; fi

# 6) 宿主内存（邻居压入的观察项）
used=$(free -g | awk 'NR==2{print $3}')
[ "${used:-0}" -lt 1600 ] && ok "host mem used=${used}G" || warn "host mem used=${used}G（>1.6T，邻居/自身占用高；启动有风险）"

echo "[preflight] $( [ $fail = 0 ] && echo 'ALL OK' || echo '有 FAIL 项，先修复（对照 wiki/dsv41_run_failures_catalog.md）')"
exit $fail
