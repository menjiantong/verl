"""NUMA interleave for DSV41 RL workers (opt-in: VERL_DSV41_NUMA_INTERLEAVE=1).

The 2-node GRPO runs allocate multi-hundred-GB pinned host regions per rank
(trainer engram ~366 GiB/node, engine tables + sleep/backup pools, FSDP grad
host-offload). With the default local-allocation policy each rank binds to one
of the 8 NUMA nodes (~350 GB each), whose free memory is shared with other
tenants on these boxes; a rank whose local node is nearly full makes the NPU
driver's pinned allocation (aclrtMallocHostWithCfg) fail with "driver error:
out of memory" even when the machine has >1.5 TB free *in total* (observed
2026-10-15, task 3.1b/3.1c/3.1d/3.1e).

MPOL_INTERLEAVE spreads allocations round-robin across all nodes so no single
node saturates. Best-effort: any failure here is ignored -- a memory-policy
tweak must never break the training stack. Gated by the env var so other
users of this shared tree are unaffected.
"""
import os

if os.environ.get("VERL_DSV41_NUMA_INTERLEAVE") == "1":
    try:
        import ctypes

        _numa = ctypes.CDLL("libnuma.so.1")
        _numa.numa_available.restype = ctypes.c_int
        if _numa.numa_available() >= 0:
            _numa.numa_allocate_nodemask.restype = ctypes.c_void_p
            _numa.numa_bitmask_clearall.argtypes = [ctypes.c_void_p]
            _numa.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            _numa.numa_bitmask_free.argtypes = [ctypes.c_void_p]
            _numa.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
            _numa.numa_set_interleave_mask.restype = ctypes.c_int

            mask = _numa.numa_allocate_nodemask()
            _numa.numa_bitmask_clearall(mask)
            for node in range(_numa.numa_max_node() + 1):
                _numa.numa_bitmask_setbit(mask, node)
            rc = _numa.numa_set_interleave_mask(mask)
            _numa.numa_bitmask_free(mask)
            if rc != 0:
                raise RuntimeError(f"numa_set_interleave_mask rc={rc}")
    except Exception:
        pass

# [2026-10-17 E41] THP opt-out for DSV41 RL workers (opt-in: VERL_DSV41_DISABLE_THP=1).
# 证据（worklog E41）：节点 THP=[always] 且内存碎片化严重时，几十~几百 MB 的匿名分配
# （优化器状态 zeros_like 等）在缺页路径上触发 >9 分钟级 compaction/reclaim stall
# （task55b：128 节点 compact_fail 6.4M、单次 zeros_like 565s；同代码 127 节点 41s 完成）。
# PR_SET_THP_DISABLE 让本进程只用 4K 页，绕开 2M 巨页分配/compaction。任何失败都忽略。
if os.environ.get("VERL_DSV41_DISABLE_THP") == "1":
    try:
        import ctypes

        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_THP_DISABLE = 41
        rc = _libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
        if rc != 0:
            raise RuntimeError(f"prctl(PR_SET_THP_DISABLE) rc={rc}")
    except Exception:
        pass
