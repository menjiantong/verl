# [1110] 主机侧 NUMA / THP 控制工具。
#
# 背景（云道单机 + 80.5.25.126 反向实验）：engram host-offload 表在训练侧（FSDPTurbo）与
# 推理侧（vLLM）各分配数百 GiB pinned 主机内存。默认由首次触达的线程落位，会把单个
# NUMA 节点抽干；该节点上 THP 直接压缩（compact_stall≈compact_fail）会退化成
# 单 rank 86~311 s 的慢路径（900 次 optimizer state 创建的 P0 探针）。
#
# 三个开关（默认关闭，行为不变）：
#   MY_ENV_DSV41_DISABLE_THP=1       prctl(PR_SET_THP_DISABLE)，关掉本进程 THP，
#                                    靶向规避“低水位节点上的直接压缩”慢路径。
#   MY_ENV_DSV41_ENGRAM_SPREAD=1     按 rank 把 engram 大表的“分配 + 首次填充”钉到
#                                    对应 NUMA 节点的 CPU 上（first-touch 落位），
#                                    16 rank 均摊到所有有内存节点。
#   MY_ENV_DSV41_NUMA_INTERLEAVE=1   set_mempolicy(MPOL_INTERLEAVE)（本容器 seccomp
#                                    禁止该 syscall 时回退并打日志；能用则优先级更高）。
#   MY_ENV_DSV41_NUMA_NODES=0,2,4   可选，显式指定节点；默认自动取 online 且有内存的节点。
#
# 用法：Worker.__init__ 里、任何大块主机分配之前调用 apply_host_numa_env()；
# 表格分配再用 host_alloc_spread_ctx() 包住“分配 + 首次填充”
# （见 FSDPTurbo HostOffloadEngram / vllm-ascend engram embedding）。

import contextlib
import ctypes
import os
import platform
import re

PR_SET_THP_DISABLE = 41

MPOL_DEFAULT = 0
MPOL_INTERLEAVE = 3

_SYS_SET_MEMPOLICY = {
    "aarch64": 237,
    "x86_64": 238,
}

# 最近一次由本模块设置的线程策略；interleaved() 退出时恢复到它。
_POLICY = {"mode": MPOL_DEFAULT, "nodes": ()}


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _libc():
    return ctypes.CDLL("libc.so.6", use_errno=True)


def _read_text(path, default=""):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return default


def parse_node_list(text):
    """解析 '0-3,5,7' 形式，返回升序节点列表。"""
    nodes = set()
    for part in str(text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            nodes.update(range(int(start), int(end) + 1))
        else:
            nodes.add(int(part))
    return sorted(nodes)


def online_nodes():
    return parse_node_list(_read_text("/sys/devices/system/node/online", "0"))


def _node_has_memory(node):
    text = _read_text("/sys/devices/system/node/node%d/meminfo" % node)
    match = re.search(r"MemTotal:\s*(\d+)\s*kB", text)
    return bool(match) and int(match.group(1)) > 0


def memory_nodes():
    """online 且有内存的节点（跳过 MemTotal=0 的无内存节点，如 126/188 的 node1）。"""
    return [node for node in online_nodes() if _node_has_memory(node)]


def resolve_nodes(spec=None):
    if spec:
        nodes = parse_node_list(spec)
        with_mem = set(memory_nodes())
        return [node for node in nodes if node in with_mem or not with_mem]
    return memory_nodes()


def disable_thp():
    """关闭本进程 THP；返回 prctl 是否成功。"""
    libc = _libc()
    ret = libc.prctl(
        ctypes.c_int(PR_SET_THP_DISABLE),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    return ret == 0


def cpu_list_for_node(node):
    """节点上的 CPU 列表（解析 nodeN/cpulist，如 '0-39'）。"""
    text = _read_text("/sys/devices/system/node/node%d/cpulist" % node, "")
    cpus = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


_AFFINITY_BYTES = 128  # 1024 CPU 足够本机型（320）


def get_affinity():
    """返回当前线程允许的 CPU 列表。"""
    buffer = ctypes.create_string_buffer(_AFFINITY_BYTES)
    _libc().sched_getaffinity(ctypes.c_int(0), ctypes.c_size_t(_AFFINITY_BYTES), buffer)
    allowed = []
    for byte_index, byte in enumerate(buffer.raw):
        for bit in range(8):
            if byte & (1 << bit):
                allowed.append(byte_index * 8 + bit)
    return allowed


def set_affinity(cpus):
    """把当前线程钉到指定 CPU 列表；返回是否成功。"""
    mask = (ctypes.c_ubyte * _AFFINITY_BYTES)()
    valid = 0
    for cpu in cpus:
        if 0 <= cpu < _AFFINITY_BYTES * 8:
            mask[cpu // 8] |= 1 << (cpu % 8)
            valid += 1
    if not valid:
        return False
    return _libc().sched_setaffinity(ctypes.c_int(0), ctypes.c_size_t(_AFFINITY_BYTES), mask) == 0


def current_rank():
    """进程身份：优先 RANK（verl worker），其次 LOCAL_RANK，再退化为 pid。"""
    for key in ("RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None and str(value).strip().isdigit():
            return int(value)
    return os.getpid()


def rank_node(nodes=None, rank=None):
    """rank -> 节点：优先显式映射表，否则均摊到有内存节点。

    MY_ENV_DSV41_NUMA_RANK_MAP 可按节点容量加权（如 188 的 node3 只有 189 GiB，
    只放 2 个 rank；满容量 node6/7 各放 3 个）。
    """
    spec = os.getenv("MY_ENV_DSV41_NUMA_RANK_MAP", "").strip()
    if spec:
        mapping = [int(part) for part in spec.replace(" ", "").split(",") if part]
        if mapping:
            index = current_rank() if rank is None else int(rank)
            return mapping[index % len(mapping)]
    candidates = list(nodes) if nodes else memory_nodes()
    if not candidates:
        raise OSError("no NUMA node with memory")
    index = current_rank() if rank is None else int(rank)
    return candidates[index % len(candidates)]


@contextlib.contextmanager
def rank_pinned_host_alloc(nodes=None, rank=None):
    """在 with 块内把线程钉到 rank 对应的 NUMA 节点，first-touch 的页落在该节点。

    退出时恢复进入前的 CPU 亲和，不影响后续计算的调度自由度。
    """
    allowed = get_affinity()
    target = rank_node(nodes, rank)
    cpus = [cpu for cpu in cpu_list_for_node(target) if cpu in allowed] or allowed
    saved = allowed
    set_affinity(cpus)
    try:
        yield target
    finally:
        set_affinity(saved)


def _set_mempolicy(mode, nodes):
    machine = platform.machine()
    number = _SYS_SET_MEMPOLICY.get(machine)
    if number is None:
        raise OSError("set_mempolicy syscall number unknown on %s" % machine)
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    if mode == MPOL_DEFAULT or not nodes:
        ret = libc.syscall(ctypes.c_long(number), ctypes.c_int(mode),
                           ctypes.c_void_p(None), ctypes.c_ulong(0))
    else:
        maxnode = max(nodes) + 1
        nbytes = (maxnode + 7) // 8
        mask = (ctypes.c_ubyte * nbytes)()
        for node in nodes:
            mask[node // 8] |= 1 << (node % 8)
        ret = libc.syscall(ctypes.c_long(number), ctypes.c_int(mode),
                           ctypes.byref(mask), ctypes.c_ulong(maxnode))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "set_mempolicy(mode=%d, nodes=%s) failed: %s"
                      % (mode, list(nodes), os.strerror(errno)))


def set_interleave(nodes=None):
    """把当前线程设为 MPOL_INTERLEAVE；返回设置前的 (mode, nodes)。"""
    global _POLICY
    previous = dict(_POLICY)
    target = list(nodes) if nodes else memory_nodes()
    if not target:
        raise OSError("no NUMA node with memory for interleave")
    _set_mempolicy(MPOL_INTERLEAVE, target)
    _POLICY = {"mode": MPOL_INTERLEAVE, "nodes": tuple(target)}
    return previous["mode"], previous["nodes"]


def restore_policy(policy):
    _set_mempolicy(policy[0], policy[1])
    _POLICY.update(mode=policy[0], nodes=tuple(policy[1]))


@contextlib.contextmanager
def interleaved(nodes=None):
    """在 with 块内把当前线程设为交错分配，退出时恢复进入前的策略。

    覆盖“分配 + 首次填充”整段，确保页在交错策略下落位。
    """
    previous = set_interleave(nodes)
    try:
        yield
    finally:
        restore_policy(previous)


@contextlib.contextmanager
def host_alloc_spread_ctx(nodes=None, rank=None):
    """engram 大表“分配 + 首次填充”的 NUMA 摊开上下文（默认 no-op）。

    优先级：MY_ENV_DSV41_NUMA_INTERLEAVE（mempolicy，若内核/容器允许）
          > MY_ENV_DSV41_ENGRAM_SPREAD（按 rank 钉节点做 first-touch）。
    188/126 容器 CapEff 无 CAP_SYS_NICE，set_mempolicy/mbind 被 seccomp 拒绝（EPERM），
    因此实际走 first-touch；日志里会打印回退原因。
    """
    spread = _truthy(os.getenv("MY_ENV_DSV41_ENGRAM_SPREAD", "0"))
    interleave = _truthy(os.getenv("MY_ENV_DSV41_NUMA_INTERLEAVE", "0"))
    if not spread and not interleave:
        yield None
        return
    target_nodes = resolve_nodes(nodes)
    if interleave:
        try:
            previous = set_interleave(target_nodes)
        except OSError as exc:
            print(
                "[MY_DEBUG_LOG] host_alloc_interleave_failed pid=%s nodes=%s error=%s "
                "(fallback to first-touch spread=%s)"
                % (os.getpid(), list(target_nodes), exc, int(spread)),
                flush=True,
            )
        else:
            try:
                yield None
            finally:
                restore_policy(previous)
            return
    if spread:
        with rank_pinned_host_alloc(target_nodes, rank) as node:
            yield node
    else:
        yield None


def apply_host_numa_env():
    """按环境变量一次性应用 THP / 交错 / 进程钉节点；返回结果字典（便于日志与断言）。"""
    result = {
        "pid": os.getpid(),
        "thp_off_requested": _truthy(os.getenv("MY_ENV_DSV41_DISABLE_THP", "0")),
        "interleave_requested": _truthy(os.getenv("MY_ENV_DSV41_NUMA_INTERLEAVE", "0")),
        "spread_requested": _truthy(os.getenv("MY_ENV_DSV41_ENGRAM_SPREAD", "0")),
        "pin_process_requested": _truthy(os.getenv("MY_ENV_DSV41_PIN_PROCESS", "0")),
        "rank": current_rank(),
        "thp_off_ok": None,
        "interleave_ok": None,
        "pin_node": None,
        "pin_ok": None,
        "nodes": (),
        "error": "",
    }
    if not any(result[key] for key in (
            "thp_off_requested", "interleave_requested", "spread_requested",
            "pin_process_requested")):
        return result
    if result["thp_off_requested"]:
        result["thp_off_ok"] = disable_thp()
    if result["interleave_requested"]:
        try:
            nodes = resolve_nodes(os.getenv("MY_ENV_DSV41_NUMA_NODES"))
            result["nodes"] = tuple(nodes)
            set_interleave(nodes)
            result["interleave_ok"] = True
        except OSError as exc:
            result["error"] = str(exc)
    if result["pin_process_requested"]:
        # 进程级 first-touch：把本 worker 的所有主机分配（engram 表 + FSDP 参数/
        # 优化器 offload）钉到 rank 对应节点，避免只有表摊开、其余 1 TB 级分配仍集中。
        try:
            nodes = resolve_nodes(os.getenv("MY_ENV_DSV41_NUMA_NODES"))
            target = rank_node(nodes)
            allowed = get_affinity()
            cpus = [cpu for cpu in cpu_list_for_node(target) if cpu in allowed] or allowed
            result["pin_node"] = target
            result["pin_ok"] = set_affinity(cpus)
        except OSError as exc:
            result["error"] = (result["error"] + "; " + str(exc)).strip("; ")
    # 统一日志：一行打全，便于和 P0 的 WorkerDict pid 对齐（已验证过该 pid 即模型构建进程）。
    print(
        "[MY_DEBUG_LOG] host_numa_applied pid=%s rank=%s thp_off=%s thp_ok=%s "
        "interleave=%s interleave_ok=%s spread=%s pin_process=%s pin_node=%s pin_ok=%s "
        "nodes=%s error=%s"
        % (
            result["pid"],
            result["rank"],
            int(result["thp_off_requested"]),
            result["thp_off_ok"],
            int(result["interleave_requested"]),
            result["interleave_ok"],
            int(result["spread_requested"]),
            int(result["pin_process_requested"]),
            result["pin_node"],
            result["pin_ok"],
            list(result["nodes"]),
            result["error"] or "-",
        ),
        flush=True,
    )
    return result
